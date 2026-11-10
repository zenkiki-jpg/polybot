"""
Расширенный логгер: все монеты × все интервалы Up/Down + сделки.

Пишет в data/multi/:
  ticks_YYYYMMDD_HH.csv.gz      — снимки стаканов (5 уровней бидов и асков по обеим
                                  сторонам) + цены Chainlink spot/TWAP30/TWAP60 и Binance
                                  для монеты рынка. Частота зависит от интервала
                                  (5m — каждую секунду, 1d — раз в 30 с; см. --sample).
  trades_ws_YYYYMMDD_HH.csv.gz  — сделки в реальном времени (market-канал CLOB):
                                  что увидел бы бот в момент сделки.
  trades_api2_YYYYMMDD_HH.csv.gz — полная история сделок каждого окна из Data API
                                  через минуту после закрытия (с кошельками и сырым JSON).
  windows.csv                   — одна строка на окно сразу после закрытия: источник
                                  резолва, price to beat, цены на старте и в конце.
  outcomes.csv                  — исходы: официальный (когда появится) и, для рынков
                                  на свечах Binance, исход по свече.
Пропущенные исходы (например, после перезапуска) докачивает
    py analysis/backfill_outcomes.py

Запуск (из корня проекта):
    py logger/multi_logger.py
    py logger/multi_logger.py --coins btc,eth,sol,xrp,doge --intervals 5m,15m,1h,4h,1d
    py logger/multi_logger.py --sample 5m=1,15m=1,1h=5 --no-ws-trades

Можно запускать вместе со старым логгером и пейпером — они не мешают друг другу.
"""
import argparse
import csv
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pm import Resolver, fetch_outcome, _pool, log  # noqa: F401
import json
import queue

from core.multi import (COINS, INTERVALS, DEFAULT_SAMPLE, MultiFeed, MultiMarketCache,
                        TradeStream, RotatingGzCsv, fetch_books_batch, fetch_all_trades,
                        normalize_api_trade, binance_kline_outcome)

TICK_FIELDS = [
    "ts_unix", "key", "coin", "interval", "slug", "window_start", "window_end",
    "seconds_to_close", "ref_kind", "price_to_beat", "ptb_source",
    "cl_spot", "cl_twap30", "cl_twap60", "binance_mid",
    "up_bid", "up_bid_size", "up_ask", "up_ask_size",
    "down_bid", "down_bid_size", "down_ask", "down_ask_size",
    "up_bids", "up_asks", "down_bids", "down_asks",
]
WS_TRADE_FIELDS = ["recv_ts", "key", "slug", "outcome", "asset_id", "price", "size", "side", "exchange_ts"]
API_TRADE_FIELDS = ["key", "slug", "window_start", "window_end", "winner", "exchange_ts", "asset_id",
                    "outcome", "side", "price", "size", "usdc", "wallet", "tx", "raw"]
OUTCOME_FIELDS = ["key", "slug", "window_start", "window_end", "outcome", "outcome_source",
                  "outcome_kline", "kline_open", "kline_close", "logged_at"]
# сколько ждать официальный резолв, прежде чем сдаться (потом — backfill_outcomes.py)
GIVE_UP_S = {"5m": 3600, "15m": 3600, "1h": 6 * 3600, "4h": 6 * 3600, "1d": 12 * 3600}
WINDOW_FIELDS = [
    "key", "coin", "interval", "slug", "window_start", "window_end", "condition_id",
    "resolution_source", "ref_kind", "ptb_gamma",
    "ptb_rtds_spot", "ptb_rtds_twap30", "ptb_rtds_twap60", "binance_at_start",
    "close_rtds_spot", "close_rtds_twap60", "binance_at_end",
    "outcome", "outcome_source", "api_trades", "api_trades_src", "resolved_logged_at",
]


def parse_sample(s: str) -> dict:
    out = dict(DEFAULT_SAMPLE)
    for part in filter(None, (s or "").split(",")):
        k, v = part.split("=")
        out[k.strip()] = max(1, int(v))
    return out


def ptb_for(m, feed):
    if m.price_to_beat is not None:
        return m.price_to_beat, "gamma"
    if m.ref_kind in ("spot", "twap30", "twap60"):
        v = feed.at_exact(m.coin, m.ref_kind, m.start * 1000)
        if v is not None:
            return v, "rtds_exact"
    return None, None


def run(a):
    coins = [c.strip() for c in a.coins.split(",") if c.strip()]
    intervals = [i.strip() for i in a.intervals.split(",") if i.strip()]
    bad = [c for c in coins if c not in COINS] + [i for i in intervals if i not in INTERVALS]
    if bad:
        raise SystemExit(f"Неизвестно: {bad}. Монеты: {list(COINS)}, интервалы: {list(INTERVALS)}")
    sample = parse_sample(a.sample)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ticks_w = RotatingGzCsv(out, "ticks", TICK_FIELDS)
    ws_w = RotatingGzCsv(out, "trades_ws", WS_TRADE_FIELDS)
    api_w = RotatingGzCsv(out, "trades_api2", API_TRADE_FIELDS)

    def plain_csv(path, fields):
        new = not path.exists() or path.stat().st_size == 0
        fh = open(path, "a", newline="", encoding="utf-8")
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
            fh.flush()
        return fh, w

    win_fh, win_w = plain_csv(out / "windows.csv", WINDOW_FIELDS)
    out_fh, out_w = plain_csv(out / "outcomes.csv", OUTCOME_FIELDS)
    csv_lock = threading.Lock()

    def write_outcome(m, **kw):
        with csv_lock:
            out_w.writerow({"key": m.key, "slug": m.slug, "window_start": m.start, "window_end": m.end,
                            "logged_at": datetime.now(timezone.utc).isoformat(), **kw})
            out_fh.flush()

    feed = MultiFeed(coins)
    feed.start(use_binance=not a.no_binance)
    cache = MultiMarketCache()

    token_info: dict[str, tuple] = {}   # token -> (key, slug, "Up"/"Down", end)
    counters = {"ws_trades": 0, "api_trades": 0, "windows": 0, "outcomes": 0}

    def on_ws_trade(t):
        info = token_info.get(t["asset_id"])
        if info is None:
            return
        counters["ws_trades"] += 1
        ws_w.write({"recv_ts": round(time.time(), 3), "key": info[0], "slug": info[1],
                    "outcome": info[2], **t})

    trades = TradeStream(on_ws_trade)
    if not a.no_ws_trades:
        trades.start()

    # --- фоновые задачи (история сделок, свечи Binance), чтобы не тормозить резолвы ---
    jobs: "queue.PriorityQueue" = queue.PriorityQueue()
    job_seq = [0]

    def schedule(due, kind, m):
        job_seq[0] += 1
        jobs.put((due, job_seq[0], kind, m))

    def worker():
        while True:
            due, _, kind, m = jobs.get()
            wait = due - time.time()
            if wait > 0:
                jobs.put((due, _, kind, m))
                time.sleep(min(wait, 1.0))
                continue
            try:
                if kind == "trades" and m.condition_id:
                    rows, src = fetch_all_trades(m.condition_id)
                    for r in rows:
                        api_w.write({"key": m.key, "slug": m.slug, "window_start": m.start,
                                     "window_end": m.end, **normalize_api_trade(r),
                                     "raw": json.dumps(r, separators=(",", ":"))}, ts=m.end)
                    counters["api_trades"] += len(rows)
                elif kind == "kline":
                    oc, o, c = binance_kline_outcome(m.coin, m.start, m.end)
                    if oc is None and time.time() < m.end + 600:
                        schedule(time.time() + 30, "kline", m)   # свеча ещё не готова
                    else:
                        write_outcome(m, outcome_kline=oc, kline_open=o, kline_close=c,
                                      outcome_source="kline_only")
            except Exception as e:
                log(f"[worker] {kind} {m.slug}: {e}")

    threading.Thread(target=worker, daemon=True).start()

    def on_resolved(m, outcome, source):
        write_outcome(m, outcome=outcome, outcome_source=source)
        counters["outcomes"] += 1

    resolver = Resolver(on_resolved, give_up_after_s=6 * 3600)
    resolver.start()

    # --- снимки цен на старте и в конце окна (пока они ещё есть в памяти) ---
    snaps: dict[str, dict] = {}
    open_markets: dict[str, object] = {}

    def snap_start(m, now):
        sn = snaps.setdefault(m.slug, {})
        if "start_done" in sn or now - m.start > 600:
            return
        s_ms = m.start * 1000
        for kind, col in (("spot", "ptb_rtds_spot"), ("twap30", "ptb_rtds_twap30"),
                          ("twap60", "ptb_rtds_twap60")):
            if sn.get(col) is None:
                sn[col] = feed.at_exact(m.coin, kind, s_ms)
        if sn.get("binance_at_start") is None:
            sn["binance_at_start"] = feed.at_exact(m.coin, "binance", s_ms, tol_ms=1500)
        if all(sn.get(c) is not None for c in ("ptb_rtds_spot", "ptb_rtds_twap60")) or now - m.start > 60:
            sn["start_done"] = True

    def snap_close(m):
        sn = snaps.setdefault(m.slug, {})
        e_ms = m.end * 1000
        row = {
            "key": m.key, "coin": m.coin, "interval": m.interval, "slug": m.slug,
            "window_start": m.start, "window_end": m.end, "condition_id": m.condition_id,
            "resolution_source": m.resolution_source, "ref_kind": m.ref_kind,
            "ptb_gamma": m.price_to_beat,
            "ptb_rtds_spot": sn.get("ptb_rtds_spot"), "ptb_rtds_twap30": sn.get("ptb_rtds_twap30"),
            "ptb_rtds_twap60": sn.get("ptb_rtds_twap60"), "binance_at_start": sn.get("binance_at_start"),
            "close_rtds_spot": feed.at_exact(m.coin, "spot", e_ms),
            "close_rtds_twap60": feed.at_exact(m.coin, "twap60", e_ms),
            "binance_at_end": feed.at_exact(m.coin, "binance", e_ms, tol_ms=1500),
            "outcome_source": "pending",
            "resolved_logged_at": datetime.now(timezone.utc).isoformat(),
        }
        with csv_lock:
            win_w.writerow(row)
            win_fh.flush()
        counters["windows"] += 1
        snaps.pop(m.slug, None)
        if not a.no_api_trades:
            schedule(m.end + 60, "trades", m)
        if m.ref_kind == "binance":
            # минутная свеча 12:00 ET закрывается в 12:00:59 — для 1d ждём дольше
            schedule(m.end + (90 if m.interval == "1d" else 15), "kline", m)

    log(f"Мульти-логгер: монеты {coins}, интервалы {intervals}, частота {sample}, папка {out}")
    time.sleep(3)

    keys = [(c, i) for c in coins for i in intervals]
    next_due = {k: 0.0 for k in keys}
    seen = set()
    last_status = 0.0
    try:
        while True:
            loop_t = time.time()
            now = loop_t
            # 1) текущие рынки всех ключей (параллельно; из кэша — мгновенно)
            futs = {k: _pool.submit(cache.get, k[0], k[1], now) for k in keys}
            markets = {k: f.result() for k, f in futs.items()}
            active_tokens = []
            for k, m in markets.items():
                if m is None:
                    continue
                if m.slug not in seen:
                    seen.add(m.slug)
                    m.give_up_s = GIVE_UP_S.get(m.interval)
                    resolver.add(m)
                    open_markets[m.slug] = m
                    token_info[m.up_token] = (m.key, m.slug, "Up", m.end)
                    token_info[m.down_token] = (m.key, m.slug, "Down", m.end)
                active_tokens += [m.up_token, m.down_token]
            trades.set_tokens(active_tokens)

            # снимки: старт — пока окно свежее, конец — через 4 с после закрытия
            for slug, m in list(open_markets.items()):
                if now < m.end:
                    snap_start(m, now)
                elif now >= m.end + 4:
                    open_markets.pop(slug, None)
                    snap_close(m)

            # 2) каким ключам пора писать тик
            due = [k for k in keys if markets.get(k) is not None and now >= next_due[k]]
            for k in due:
                next_due[k] = now - (now % sample[k[1]]) + sample[k[1]]
            if due:
                toks = [t for k in due for t in (markets[k].up_token, markets[k].down_token)]
                books = fetch_books_batch(toks)
                for k in due:
                    m = markets[k]
                    ptb, src = ptb_for(m, feed)
                    row = {
                        "ts_unix": round(now, 3), "key": m.key, "coin": m.coin, "interval": m.interval,
                        "slug": m.slug, "window_start": m.start, "window_end": m.end,
                        "seconds_to_close": round(m.end - now, 2), "ref_kind": m.ref_kind,
                        "price_to_beat": ptb, "ptb_source": src,
                        "cl_spot": feed.latest(m.coin, "spot"), "cl_twap30": feed.latest(m.coin, "twap30"),
                        "cl_twap60": feed.latest(m.coin, "twap60"),
                        "binance_mid": feed.latest(m.coin, "binance"),
                    }
                    for name, tok in (("up", m.up_token), ("down", m.down_token)):
                        b = books.get(tok)
                        if b is None:
                            continue
                        row.update({
                            f"{name}_bid": b.best_bid, f"{name}_bid_size": b.best_bid_size,
                            f"{name}_ask": b.best_ask, f"{name}_ask_size": b.best_ask_size,
                            f"{name}_bids": b.depth_str("bids", 5), f"{name}_asks": b.depth_str("asks", 5),
                        })
                    ticks_w.write(row, ts=now)

            # 3) статус раз в 30 с
            if now - last_status > 30:
                last_status = now
                found = sum(1 for m in markets.values() if m is not None)
                missing = [f"{c}-{i}" for (c, i), m in markets.items() if m is None]
                ws_age = time.time() - trades.last_msg if trades.last_msg else None
                def age(c):
                    t = feed.last_recv(c, "twap60")
                    return "нет" if t is None else f"{time.time() - t:.0f}с"
                prices = " ".join(f"{c}={feed.latest(c, 'twap60')} ({age(c)})" for c in coins)
                print(f"[{datetime.now():%H:%M:%S}] рынков {found}/{len(keys)} | тиков {ticks_w.rows} | "
                      f"сделок ws {counters['ws_trades']} api {counters['api_trades']} | "
                      f"окон {counters['windows']}, исходов {counters['outcomes']}, ждут резолва {resolver.pending_count()} | "
                      f"ws-сделки: {'нет данных' if ws_age is None else f'{ws_age:.0f}с назад'}", flush=True)
                print(f"           chainlink twap60 (возраст): {prices}", flush=True)
                if missing:
                    print(f"           не найдены: {', '.join(missing)}", flush=True)
                cache.prune(now)
                for t in [t for t, v in token_info.items() if v[3] < now - 3 * 3600]:
                    token_info.pop(t, None)

            time.sleep(max(0.05, 1.0 - (time.time() - loop_t)))
    except KeyboardInterrupt:
        log("Остановлено.")
    finally:
        for w in (ticks_w, ws_w, api_w):
            w.close()
        win_fh.close()
        out_fh.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Мульти-логгер Polymarket Up/Down")
    ap.add_argument("--coins", default="btc,eth,sol,xrp,doge")
    ap.add_argument("--intervals", default="5m,15m,1h,4h,1d")
    ap.add_argument("--sample", default="", help="частота тиков, напр. 5m=1,15m=2,1h=5")
    ap.add_argument("--out", default="data/multi")
    ap.add_argument("--no-ws-trades", action="store_true", help="не слушать сделки в реальном времени")
    ap.add_argument("--no-api-trades", action="store_true", help="не выкачивать историю сделок окна")
    ap.add_argument("--no-binance", action="store_true")
    run(ap.parse_args())
