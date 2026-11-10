"""
Докачивает исходы окон, которых нет в data/multi (часовые рынки резолвятся часами,
а после перезапуска логгера ожидающие окна теряются).

    py analysis/backfill_outcomes.py
    py analysis/backfill_outcomes.py --dir data/multi --limit 300

Дописывает строки в data/multi/outcomes.csv. Можно запускать, не останавливая логгер,
и сколько угодно раз: уже известные исходы повторно не запрашиваются.
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

from core.pm import fetch_outcome, gamma_market_by_slug  # noqa: E402
from core.multi import outcome_ref_by_slug, binance_kline_outcome  # noqa: E402
from logger.multi_logger import OUTCOME_FIELDS  # noqa: E402

OFFICIAL = {"UP", "DOWN", "SPLIT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/multi")
    ap.add_argument("--limit", type=int, default=200, help="максимум окон за запуск")
    a = ap.parse_args()
    d = Path(a.dir)
    W = pd.read_csv(d / "windows.csv").drop_duplicates("slug", keep="last")
    op = d / "outcomes.csv"
    O = pd.read_csv(op) if op.exists() and op.stat().st_size > 0 else pd.DataFrame(columns=OUTCOME_FIELDS)

    have_official = set(W.loc[W.outcome.isin(OFFICIAL), "slug"]) | set(O.loc[O.outcome.isin(OFFICIAL), "slug"])
    have_kline = set(O.loc[O.outcome_kline.isin(["UP", "DOWN"]), "slug"])
    now = time.time()
    todo = W[(W.window_end < now - 600) & ~W.slug.isin(have_official)].copy()
    # сначала окна вообще без исхода, потом те, где уже есть исход по свече Binance
    todo["has_kline"] = todo.slug.isin(have_kline)
    todo = todo.sort_values(["has_kline", "window_end"])
    print(f"Окон всего: {len(W)}, без официального исхода: {len(todo)} "
          f"(из них {int((~todo.has_kline).sum())} совсем без исхода). Обработаю до {a.limit}.")
    print("Каждое окно — несколько запросов к API, это займёт несколько минут. Ctrl+C можно жать в любой момент.")

    new = not op.exists() or op.stat().st_size == 0
    fh = open(op, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=OUTCOME_FIELDS, extrasaction="ignore")
    if new:
        w.writeheader()
    got = kl = still = 0
    debug_shown = 0
    for i, (_, r) in enumerate(todo.head(a.limit).iterrows(), 1):
        if i % 25 == 0:
            print(f"  … {i}/{min(a.limit, len(todo))}: официальных {got}, по свечам {kl}, без резолва {still}",
                  flush=True)
        row = {"key": r.key, "slug": r.slug, "window_start": r.window_start, "window_end": r.window_end,
               "logged_at": datetime.now(timezone.utc).isoformat()}
        ref = outcome_ref_by_slug(r.slug)
        out, src = (fetch_outcome(ref) if ref and ref.outcomes else (None, None))
        if out:
            row.update(outcome=out, outcome_source=src)
            got += 1
        else:
            still += 1
            if debug_shown < 3:  # покажем, что Gamma говорит про нерезолвнутый рынок
                debug_shown += 1
                try:
                    m = gamma_market_by_slug(r.slug) or {}
                    print(f"  нет резолва: {r.slug}: closed={m.get('closed')} "
                          f"outcomePrices={m.get('outcomePrices')} uma={m.get('umaResolutionStatus')} "
                          f"marketSlug={m.get('slug')}")
                except Exception as e:
                    print(f"  нет резолва: {r.slug}: {e}")
        if r.ref_kind == "binance" and r.slug not in have_kline:
            oc, o, c = binance_kline_outcome(r.coin, int(r.window_start), int(r.window_end))
            if oc:
                row.update(outcome_kline=oc, kline_open=o, kline_close=c)
                kl += 1
        if row.get("outcome") or row.get("outcome_kline"):
            row.setdefault("outcome_source", "kline_only")
            w.writerow(row)
            fh.flush()
    fh.close()
    print(f"Готово: официальных исходов найдено {got}, по свечам Binance {kl}, всё ещё без резолва {still}.")


if __name__ == "__main__":
    main()
