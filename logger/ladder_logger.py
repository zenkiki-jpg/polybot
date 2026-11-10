"""
Логгер крипто-лесенок Polymarket: "Bitcoin above ___ on <дата>?" и "Bitcoin price on <дата>?"
+ опционы Deribit + цена Binance.

Пишет в data/ladder/:
  books_YYYYMMDD_HH.csv.gz   — снимки стакана каждого страйка (Yes и No: лучшие цены,
                               объёмы, 3 уровня бидов/асков Yes) + цена Binance.
                               Частота растёт к экспирации: >24ч — раз в 5 мин,
                               2–24ч — раз в минуту, <2ч — раз в 15 с.
  deribit_YYYYMMDD_HH.csv.gz — опционы BTC/ETH с экспирацией в ближайшие 10 дней, раз в минуту.
  events.csv                 — найденные события: время конца, источник резолва, правила.
  outcomes.csv               — исход каждого страйка после резолва + цена Binance в момент конца.
  trades_YYYYMMDD_HH.csv.gz  — все сделки по дневным лесенкам после резолва.

Запуск:
    py logger/ladder_logger.py
    py logger/ladder_logger.py --coins bitcoin,ethereum --days 7 --hourly
"""
import argparse
import csv
import json
import queue
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pm import log  # noqa: E402
from core.multi import ET, MONTHS, RotatingGzCsv, fetch_books_batch, fetch_all_trades, normalize_api_trade  # noqa: E402
from core.events import EventCache, EventResolver, discover_by_tag, parse_strike  # noqa: E402
from core.external import deribit_options, binance_mids, binance_close_at  # noqa: E402

COINS = {"bitcoin": ("BTC", "BTCUSDT"), "ethereum": ("ETH", "ETHUSDT"),
         "solana": ("SOL", "SOLUSDT"), "xrp": ("XRP", "XRPUSDT")}
BOOK_FIELDS = ["ts", "event_slug", "kind", "coin", "end", "tte_s", "market_slug", "title", "lo", "hi",
               "yes_bid", "yes_bid_size", "yes_ask", "yes_ask_size", "no_bid", "no_bid_size", "no_ask",
               "no_ask_size", "yes_bids", "yes_asks", "spot"]
DERIBIT_FIELDS = ["ts", "instrument", "currency", "expiry", "strike", "type", "mark_iv", "mark_price",
                  "bid", "ask", "underlying", "index", "oi"]
EVENT_FIELDS = ["first_seen", "slug", "kind", "coin", "title", "end", "resolution_source", "n_markets",
                "description"]
OUTCOME_FIELDS = ["event_slug", "kind", "coin", "end", "market_slug", "title", "lo", "hi", "winner",
                  "source", "binance_close_at_end", "logged_at"]
TRADE_FIELDS = ["event_slug", "market_slug", "title", "exchange_ts", "asset_id", "outcome", "side",
                "price", "size", "wallet", "tx"]


def candidates(coins, days, hourly, now):
    t = datetime.fromtimestamp(now, ET)
    out = []
    for name in coins:
        for k in range(-1, days + 1):
            d = (t + timedelta(days=k)).date()
            m, dd, y = MONTHS[d.month - 1], d.day, d.year
            out += [(f"{name}-above-on-{m}-{dd}", "above", name),
                    (f"{name}-above-on-{m}-{dd}-{y}", "above", name),
                    (f"{name}-price-on-{m}-{dd}", "range", name),
                    (f"{name}-price-on-{m}-{dd}-{y}", "range", name)]
        if hourly:
            for h in range(0, 4):
                x = (t + timedelta(hours=h)).replace(minute=0)
                h12 = x.hour % 12 or 12
                ap = "am" if x.hour < 12 else "pm"
                m, dd, y = MONTHS[x.month - 1], x.day, x.year
                out += [(f"{name}-above-on-{m}-{dd}-{y}-{h12}{ap}-et", "above_hourly", name),
                        (f"{name}-above-on-{m}-{dd}-{h12}{ap}-et", "above_hourly", name)]
    return out


def sample_every(tte):
    if tte > 24 * 3600:
        return 300
    if tte > 2 * 3600:
        return 60
    return 15


def run(a):
    coins = [c.strip() for c in a.coins.split(",") if c.strip() in COINS]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    books_w = RotatingGzCsv(out, "books", BOOK_FIELDS)
    der_w = RotatingGzCsv(out, "deribit", DERIBIT_FIELDS)
    trades_w = RotatingGzCsv(out, "trades", TRADE_FIELDS)

    def plain(name, fields):
        p = out / name
        new = not p.exists() or p.stat().st_size == 0
        fh = open(p, "a", newline="", encoding="utf-8")
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
            fh.flush()
        return fh, w
    ev_fh, ev_w = plain("events.csv", EVENT_FIELDS)
    oc_fh, oc_w = plain("outcomes.csv", OUTCOME_FIELDS)
    lock = threading.Lock()

    cache = EventCache(refresh_s=1800, miss_retry_s=1800)
    meta: dict[str, tuple] = {}         # event slug -> (kind, coin)
    seen_events: set = set()
    last_sample: dict[str, float] = {}
    jobs: "queue.Queue" = queue.Queue()
    counters = {"rows": 0, "deribit": 0, "resolved": 0, "trades": 0}

    def on_done(ev, res):
        kind, coin = meta.get(ev.slug, ("?", "?"))
        close = binance_close_at(COINS[coin][1], ev.end) if coin in COINS and ev.end else None
        with lock:
            for sm in ev.markets:
                w, src = res.get(sm.slug, (None, "timeout"))
                oc_w.writerow({"event_slug": ev.slug, "kind": kind, "coin": coin, "end": ev.end,
                               "market_slug": sm.slug, "title": sm.title, "lo": sm.lo, "hi": sm.hi,
                               "winner": w, "source": src, "binance_close_at_end": close,
                               "logged_at": datetime.now(timezone.utc).isoformat()})
            oc_fh.flush()
        counters["resolved"] += 1
        if not a.no_trades and kind != "above_hourly":
            jobs.put(ev)

    resolver = EventResolver(on_done)
    resolver.start()

    def trades_worker():
        while True:
            ev = jobs.get()
            for sm in ev.markets:
                if not sm.condition_id:
                    continue
                rows, _ = fetch_all_trades(sm.condition_id)
                for r in rows:
                    trades_w.write({"event_slug": ev.slug, "market_slug": sm.slug, "title": sm.title,
                                    **normalize_api_trade(r)}, ts=ev.end or time.time())
                counters["trades"] += len(rows)
    threading.Thread(target=trades_worker, daemon=True).start()

    def deribit_loop():
        cur = sorted({COINS[c][0] for c in coins} & {"BTC", "ETH"})
        while True:
            for c in cur:
                try:
                    ts = time.time()
                    for row in deribit_options(c):
                        der_w.write({"ts": round(ts, 1), **row}, ts=ts)
                        counters["deribit"] += 1
                except Exception as e:
                    log(f"[deribit] {c}: {e}")
            time.sleep(a.deribit_every)
    if not a.no_deribit:
        threading.Thread(target=deribit_loop, daemon=True).start()

    log(f"Логгер лесенок: {coins}, дней вперёд {a.days}, часовые: {a.hourly}, папка {out}")
    last_disc = 0.0
    last_status = 0.0
    try:
        while True:
            now = time.time()
            # 1) поиск событий раз в 10 минут
            if now - last_disc > 600:
                last_disc = now
                cands = candidates(coins, a.days, a.hourly, now)
                tagged = discover_by_tag("crypto", r"^(Bitcoin|Ethereum|Solana|XRP) (above|price on)")
                for s in tagged:
                    name = s.split("-")[0]
                    if name in coins:
                        cands.append((s, "above_hourly" if s.endswith("-et") else
                                      ("above" if "-above-" in s else "range"), name))
                for slug, kind, name in cands:
                    if kind == "above_hourly" and not a.hourly:
                        continue
                    ev = cache.ensure(slug, parse_strike)
                    if ev is None:
                        continue
                    meta[slug] = (kind, name)
                    if slug not in seen_events:
                        seen_events.add(slug)
                        resolver.add(ev)
                        with lock:
                            ev_w.writerow({"first_seen": datetime.now(timezone.utc).isoformat(),
                                           "slug": slug, "kind": kind, "coin": name, "title": ev.title,
                                           "end": ev.end, "resolution_source": ev.resolution_source,
                                           "n_markets": len(ev.markets),
                                           "description": (ev.description or "")[:1500]})
                            ev_fh.flush()
                        log(f"[+] {slug}: {len(ev.markets)} страйков, конец "
                            f"{datetime.fromtimestamp(ev.end, timezone.utc):%Y-%m-%d %H:%M} UTC" if ev.end else "")

            # 2) снимки стаканов
            due = []
            for ev in cache.active(now):
                tte = (ev.end - now) if ev.end else 10 ** 9
                if now - last_sample.get(ev.slug, 0) >= sample_every(tte):
                    last_sample[ev.slug] = now
                    due.append(ev)
            if due:
                toks = [t for ev in due for sm in ev.markets for t in (sm.yes_token, sm.no_token)]
                books = fetch_books_batch(toks)
                try:
                    spots = binance_mids([COINS[c][1] for c in coins])
                except Exception:
                    spots = {}
                for ev in due:
                    kind, name = meta.get(ev.slug, ("?", "?"))
                    spot = spots.get(COINS.get(name, ("", ""))[1])
                    for sm in ev.markets:
                        y, n = books.get(sm.yes_token), books.get(sm.no_token)
                        row = {"ts": round(now, 1), "event_slug": ev.slug, "kind": kind, "coin": name,
                               "end": ev.end, "tte_s": round(ev.end - now) if ev.end else None,
                               "market_slug": sm.slug, "title": sm.title, "lo": sm.lo, "hi": sm.hi,
                               "spot": spot}
                        if y:
                            row.update(yes_bid=y.best_bid, yes_bid_size=y.best_bid_size, yes_ask=y.best_ask,
                                       yes_ask_size=y.best_ask_size, yes_bids=y.depth_str("bids", 3),
                                       yes_asks=y.depth_str("asks", 3))
                        if n:
                            row.update(no_bid=n.best_bid, no_bid_size=n.best_bid_size, no_ask=n.best_ask,
                                       no_ask_size=n.best_ask_size)
                        books_w.write(row, ts=now)
                        counters["rows"] += 1

            if now - last_status > 60:
                last_status = now
                act = cache.active(now)
                print(f"[{datetime.now():%H:%M:%S}] событий активно {len(act)} "
                      f"(страйков {sum(len(e.markets) for e in act)}) | строк стакана {counters['rows']} | "
                      f"deribit {counters['deribit']} | ждут резолва {resolver.count()}, "
                      f"зарезолвлено {counters['resolved']} | сделок {counters['trades']}", flush=True)
                if not act:
                    print("           ни одного события не найдено — пришли этот вывод, проверю формат slug",
                          flush=True)
            time.sleep(max(0.2, 5 - (time.time() - now)))
    except KeyboardInterrupt:
        log("Остановлено.")
    finally:
        for w in (books_w, der_w, trades_w):
            w.close()
        ev_fh.close()
        oc_fh.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Логгер крипто-лесенок Polymarket + Deribit")
    ap.add_argument("--coins", default="bitcoin,ethereum")
    ap.add_argument("--days", type=int, default=7, help="на сколько дней вперёд искать события")
    ap.add_argument("--hourly", action="store_true", help="ещё и часовые лесенки (там очень мало денег)")
    ap.add_argument("--deribit-every", type=int, default=60)
    ap.add_argument("--no-deribit", action="store_true")
    ap.add_argument("--no-trades", action="store_true")
    ap.add_argument("--out", default="data/ladder")
    run(ap.parse_args())
