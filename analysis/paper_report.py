"""
Отчёт по data/paper_v2.csv (paper trader v2). Можно запускать, не останавливая бота.

    py analysis/paper_report.py --csv data/paper_v2.csv
    py analysis/paper_report.py --csv data/paper_v2.csv --since 2026-10-01

Что смотреть:
  * fill_%      — в какой доле сигналов вообще было что купить дешевле max_price.
  * ev_%/ci95   — доходность на $1 и нижняя граница 95% интервала. Пока ci95_low
                  отрицательная, "прибыль" неотличима от везения.
  * skip_disagree_won_% — как часто мы были бы правы, когда рынок был против нас.
                  Если это около 0–20%, значит sanity-фильтр спасает от ошибок данных.
"""
import argparse
import sys

import numpy as np
import pandas as pd

# Windows (cp1251): не падать на символах вроде ~/Δ при выводе в файл или пайп
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/paper_v2.csv")
    ap.add_argument("--since", help="учитывать окна начиная с даты (UTC), напр. 2026-10-01")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    if a.since:
        df = df[df.window_start >= pd.Timestamp(a.since, tz="UTC").timestamp()]
    if df.empty:
        print("Пока нет строк.")
        return
    first = pd.to_datetime(df.window_start.min(), unit="s", utc=True)
    last = pd.to_datetime(df.window_start.max(), unit="s", utc=True)
    print(f"Период: {first:%Y-%m-%d %H:%M} — {last:%Y-%m-%d %H:%M} UTC, окон: {df.slug.nunique()}")
    print("Источник исхода:", df.drop_duplicates("slug").outcome_source.value_counts(dropna=False).to_dict())
    print("Источник price to beat:", df.drop_duplicates("slug").ptb_source.value_counts(dropna=False).to_dict())

    rows = []
    for cfg, g in df.groupby("config"):
        traded = g[g.status.isin(["filled", "partial"]) & g.pnl.notna()]
        cost = (traded.fill_cost + traded.fee.fillna(0))
        ret = traded.pnl / cost
        n = len(traded)
        se = ret.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        dis = g[(g.status == "skip_disagree") & g.won.notna()]
        rows.append({
            "config": cfg, "threshold": g.threshold.iloc[0], "offset": g.offset.iloc[0],
            "signals": len(g), "trades": n,
            "fill_%": n / len(g) * 100,
            "no_liq": int((g.status == "no_liquidity").sum()),
            "price_cap": int((g.status == "price_cap").sum()),
            "avg_price": traded.fill_avg_price.mean(),
            "winrate_%": traded.won.astype(float).mean() * 100 if n else np.nan,
            "pnl_$": traded.pnl.sum(),
            "ev_%": ret.mean() * 100 if n else np.nan,
            "ci95_low_%": (ret.mean() - 1.96 * se) * 100 if n > 1 else np.nan,
            "worst_day_$": traded.groupby(pd.to_datetime(traded.window_start, unit="s").dt.date).pnl.sum().min()
            if n else np.nan,
            "skip_disagree": len(dis),
            "skip_disagree_won_%": dis.won.astype(float).mean() * 100 if len(dis) else np.nan,
        })
    t = pd.DataFrame(rows).sort_values(["threshold", "offset"])
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print()
    print(t.round(3).to_string(index=False))

    ok = t[(t["ci95_low_%"] > 0) & (t.trades >= 100)]
    print("\nИТОГ:", "статистически значимый плюс у: " + ", ".join(ok.config)
          if len(ok) else "ни у одного конфига нет значимого плюса (нужно ci95_low_% > 0 и >=100 сделок).")
    print(f"Проверено конфигов: {len(t)}. Чем их больше, тем выше шанс случайного 'победителя' — "
          f"подтверждай на следующей неделе данных с --since.")


if __name__ == "__main__":
    main()
