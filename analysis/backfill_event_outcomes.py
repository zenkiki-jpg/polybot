"""
Докачивает исходы лесенок и погодных событий, которые логгер не смог узнать
(в outcomes.csv они записаны с пустым winner / source=timeout).

    py analysis/backfill_event_outcomes.py --dir data/weather
    py analysis/backfill_event_outcomes.py --dir data/ladder
    py analysis/backfill_event_outcomes.py --dir data/weather --debug     # показать сырые ответы API

Можно запускать, не останавливая логгеры, и сколько угодно раз.
"""
import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd  # noqa: E402

from core.events import fetch_event, fetch_winner, parse_strike, parse_bucket  # noqa: E402

DONE = {"Yes", "No", "SPLIT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="data/weather или data/ladder")
    ap.add_argument("--debug", action="store_true", help="показать сырые ответы API по первому рынку")
    ap.add_argument("--limit", type=int, default=400, help="максимум событий за запуск")
    a = ap.parse_args()
    d = Path(a.dir)
    is_weather = "weather" in d.name.lower()
    E = pd.read_csv(d / "events.csv").drop_duplicates("slug", keep="last")
    op = d / "outcomes.csv"
    O = pd.read_csv(op) if op.exists() and op.stat().st_size > 0 else pd.DataFrame(columns=["market_slug", "winner"])
    fields = list(pd.read_csv(op, nrows=0).columns) if op.exists() and op.stat().st_size > 0 else None
    if fields is None:
        from logger.weather_logger import OUTCOME_FIELDS as WF
        from logger.ladder_logger import OUTCOME_FIELDS as LF
        fields = WF if is_weather else LF
    done = set(O.loc[O.winner.isin(DONE), "market_slug"])
    now = time.time()
    todo = E[pd.to_numeric(E.end, errors="coerce") < now - 3600].sort_values("end")
    print(f"Событий закончилось: {len(todo)}. Рынков с известным исходом: {len(done)}.")

    new = not op.exists() or op.stat().st_size == 0
    fh = open(op, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if new:
        w.writeheader()
    found = missing = events = 0
    debug_left = 1 if a.debug else 0
    for _, e in todo.head(a.limit).iterrows():
        ev = fetch_event(e.slug)
        if ev is None:
            continue
        pf = parse_bucket if is_weather else parse_strike
        need = [sm for sm in ev.markets if sm.slug not in done]
        if not need:
            continue
        events += 1
        close = None
        if not is_weather and str(e.get("coin")) in ("bitcoin", "ethereum"):
            from core.external import binance_close_at
            close = binance_close_at("BTCUSDT" if e.coin == "bitcoin" else "ETHUSDT", int(e.end))
        for sm in need:
            sm.lo, sm.hi = pf(sm.title, ev)
            win, src = fetch_winner(sm, debug=debug_left > 0)
            debug_left = 0
            if not win:
                missing += 1
                continue
            found += 1
            row = {"event_slug": ev.slug, "market_slug": sm.slug, "title": sm.title, "lo": sm.lo, "hi": sm.hi,
                   "winner": win, "source": src, "logged_at": datetime.now(timezone.utc).isoformat(),
                   "end": e.end}
            for c in ("kind", "coin", "city", "station", "unit", "date"):
                if c in e:
                    row[c] = e[c]
            if close is not None:
                row["binance_close_at_end"] = close
            w.writerow(row)
            fh.flush()
        if events % 10 == 0:
            print(f"  … событий {events}: найдено исходов {found}, не найдено {missing}", flush=True)
    fh.close()
    print(f"Готово: найдено {found} исходов в {events} событиях, всё ещё неизвестно {missing}.")
    if missing and not a.debug:
        print("Если 'неизвестно' много — запусти с --debug и пришли вывод: покажу, что отвечает API.")


if __name__ == "__main__":
    main()
