"""
Логгер v2 для 5-минутных BTC up/down рынков Polymarket.

Пишет два файла:
  data/ticks.csv    — каждый тик (по умолчанию раз в секунду): лучшие цены и
                      объёмы обеих сторон из /book, глубина асков (5 уровней),
                      все потоки цены (Chainlink spot / TWAP30 / TWAP60, Binance),
                      price to beat.
  data/windows.csv  — одна строка на окно после официального резолва:
                      price to beat из Gamma и из RTDS, источник резолва,
                      ИСХОД ИЗ ОФИЦИАЛЬНОГО API.

Этого достаточно, чтобы потом честно проверить любое правило входа
скриптом analysis/edge_report.py.

Запуск:
    py logger/polymarket_logger.py
    py logger/polymarket_logger.py --interval 1 --ticks data/ticks.csv --windows data/windows.csv
"""
import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pm import (PriceFeed, MarketCache, Resolver, fetch_books, window_start,
                     slug_for, price_to_beat, WINDOW, log)

TICK_FIELDS = [
    "ts_utc", "ts_unix", "window_start", "slug", "seconds_to_close",
    "ref_kind", "ref_price", "price_to_beat", "ptb_source",
    "cl_spot", "cl_twap30", "cl_twap60", "binance_mid",
    "up_bid", "up_bid_size", "up_ask", "up_ask_size",
    "down_bid", "down_bid_size", "down_ask", "down_ask_size",
    "up_asks", "down_asks", "book_latency_ms",
]

WINDOW_FIELDS = [
    "slug", "window_start", "condition_id", "resolution_source", "ref_kind",
    "ptb_gamma", "ptb_rtds_twap60", "ptb_rtds_twap30", "ptb_rtds_spot",
    "close_rtds_twap60", "close_rtds_spot",
    "outcome", "outcome_source", "resolved_logged_at",
]


def open_csv(path: Path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    if new:
        w.writeheader()
        f.flush()
    return f, w


def run(interval: float, ticks_path: Path, windows_path: Path):
    feed = PriceFeed()
    feed.start()
    cache = MarketCache()

    wf, ww = open_csv(windows_path, WINDOW_FIELDS)

    def on_resolved(m, outcome, source):
        s_ms, e_ms = m.start * 1000, m.end * 1000
        ww.writerow({
            "slug": m.slug, "window_start": m.start, "condition_id": m.condition_id,
            "resolution_source": m.resolution_source, "ref_kind": m.ref_kind,
            "ptb_gamma": m.price_to_beat,
            "ptb_rtds_twap60": feed.at_exact("twap60", s_ms),
            "ptb_rtds_twap30": feed.at_exact("twap30", s_ms),
            "ptb_rtds_spot": feed.at_exact("spot", s_ms),
            "close_rtds_twap60": feed.at_exact("twap60", e_ms),
            "close_rtds_spot": feed.at_exact("spot", e_ms),
            "outcome": outcome, "outcome_source": source,
            "resolved_logged_at": datetime.now(timezone.utc).isoformat(),
        })
        wf.flush()
        log(f"[резолв] {m.slug}: {outcome} ({source})")

    resolver = Resolver(on_resolved)
    resolver.start()

    tf, tw = open_csv(ticks_path, TICK_FIELDS)
    log(f"Логгер v2: тики -> {ticks_path}, окна -> {windows_path}, интервал {interval}с")
    log("Жду 3с, пока подключатся потоки цен…")
    time.sleep(3)

    seen_windows = set()
    next_tick = time.time()
    try:
        while True:
            now = time.time()
            ws = window_start(now)
            slug = slug_for(ws)
            stc = ws + WINDOW - now

            m = cache.get(slug)
            if m is None:
                log(f"[warn] рынок {slug} не найден (пока)")
            else:
                if slug not in seen_windows:
                    seen_windows.add(slug)
                    resolver.add(m)
                    cache.prune(keep_after=ws - 2 * WINDOW)

                t0 = time.time()
                up, down = fetch_books(m)
                lat = (time.time() - t0) * 1000
                ptb, ptb_src = price_to_beat(m, feed)
                ref, _ = feed.latest(m.ref_kind)

                row = {
                    "ts_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                    "ts_unix": round(now, 3), "window_start": ws, "slug": slug,
                    "seconds_to_close": round(stc, 2),
                    "ref_kind": m.ref_kind, "ref_price": ref,
                    "price_to_beat": ptb, "ptb_source": ptb_src,
                    "cl_spot": feed.latest("spot")[0], "cl_twap30": feed.latest("twap30")[0],
                    "cl_twap60": feed.latest("twap60")[0], "binance_mid": feed.latest("binance")[0],
                    "book_latency_ms": round(lat),
                }
                for name, b in (("up", up), ("down", down)):
                    if b is not None:
                        row.update({
                            f"{name}_bid": b.best_bid, f"{name}_bid_size": b.best_bid_size,
                            f"{name}_ask": b.best_ask, f"{name}_ask_size": b.best_ask_size,
                            f"{name}_asks": b.depth_str("asks", 5),
                        })
                tw.writerow(row)
                tf.flush()

                d = (ref - ptb) if (ref is not None and ptb is not None) else None
                print(f"{datetime.now():%H:%M:%S} {slug} до закрытия {stc:5.1f}с | "
                      f"{m.ref_kind}={ref} ptb={ptb} ({ptb_src}) Δ={d if d is None else round(d, 2)} | "
                      f"UP {row.get('up_bid')}/{row.get('up_ask')}  DOWN {row.get('down_bid')}/{row.get('down_ask')} "
                      f"| книги {lat:.0f}мс | ждут резолва: {resolver.pending_count()}", flush=True)

            next_tick += interval
            time.sleep(max(0.0, next_tick - time.time()))
            if time.time() - next_tick > interval * 5:
                next_tick = time.time()  # сильно отстали — не пытаемся догонять
    except KeyboardInterrupt:
        log("Остановлено. Окна, ещё не получившие резолв, в windows.csv не попадут.")
    finally:
        tf.close()
        wf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Логгер v2 Polymarket BTC 5m")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--ticks", type=str, default="data/ticks.csv")
    ap.add_argument("--windows", type=str, default="data/windows.csv")
    a = ap.parse_args()
    run(a.interval, Path(a.ticks), Path(a.windows))
