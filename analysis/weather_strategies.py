"""
Проверка простых правил на погодных рынках — эксплуатация перекоса "фаворит / лонгшот".

Калибровка показала: дешёвые корзины (Yes по 0.05–0.30) проигрывают 35–65% на доллар,
а фавориты по 0.85–0.95 в последние 12 часов дают +7–9%. Здесь это проверяется как
настоящие правила торговли:

  Yes-фаворит:  купить Yes корзины, если её аск в заданном диапазоне;
  No-лонгшот:   купить No корзины, если аск No в диапазоне (т.е. сама корзина дешёвая).

Для каждого правила: одна сделка на корзину — первый снимок, где условие выполнено;
доходность на $1 с комиссией тейкера; стандартная ошибка по событиям; сколько
сделок упиралось в тонкий стакан; результат отдельно в первой и второй половине
периода (если правило настоящее, плюс должен быть в обеих).

    py analysis/weather_strategies.py
    py analysis/weather_strategies.py --stake 20 --since 2026-10-14
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from multiout import load_gz, load_outcomes, ev_buy, clustered  # noqa: E402

COLS = ["ts", "event_slug", "city", "end", "tte_s", "market_slug", "yes_ask", "yes_ask_size",
        "no_ask", "no_ask_size"]
WINDOWS = [(0, 6), (6, 12), (12, 24), (24, 48)]
BANDS = {"Yes": [(0.70, 0.85), (0.85, 0.95), (0.95, 0.99)],
         "No": [(0.70, 0.85), (0.85, 0.95), (0.95, 0.99)]}


def wilson_low(k, n, z=1.96):
    if n == 0:
        return np.nan
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / den


def run_rule(B, side, lo, hi, h0, h1, stake):
    ask, size = f"{side.lower()}_ask", f"{side.lower()}_ask_size"
    x = B[(B.tte_s > h0 * 3600) & (B.tte_s <= h1 * 3600) & (B[ask] >= lo) & (B[ask] < hi)]
    if x.empty:
        return None, None
    x = x.sort_values("ts").groupby("market_slug", as_index=False).first()
    x["px"] = x[ask]
    x["won_side"] = x.won if side == "Yes" else 1 - x.won
    x["r"] = ev_buy(x.px, x.won_side)
    x["thin"] = (x[size] * x.px) < stake
    return x, x.r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/weather")
    ap.add_argument("--since")
    ap.add_argument("--stake", type=float, default=10.0, help="$ на сделку (для PnL и проверки глубины)")
    a = ap.parse_args()
    d = Path(a.dir)
    B = load_gz(d, "books", usecols=COLS)
    O = load_outcomes(d)
    if B.empty or O.empty:
        print("нет снимков или исходов")
        return
    win = O[O.winner.isin(["Yes", "No"])].set_index("market_slug").winner.map({"Yes": 1.0, "No": 0.0})
    B["won"] = B.market_slug.map(win)
    B = B.dropna(subset=["won"])
    B = B[B.tte_s > 0]
    if a.since:
        B = B[B.ts >= pd.Timestamp(a.since, tz="UTC").timestamp()]
    ends = B.groupby("event_slug").end.first()
    split = ends.median()
    print(f"Событий с исходом: {B.event_slug.nunique()}, корзин: {B.market_slug.nunique()}, "
          f"период {pd.to_datetime(ends.min(), unit='s'):%m-%d} — {pd.to_datetime(ends.max(), unit='s'):%m-%d}; "
          f"граница половин: {pd.to_datetime(split, unit='s'):%m-%d %H:%M}")

    rows, keep = [], {}
    for side, bands in BANDS.items():
        for lo, hi in bands:
            for h0, h1 in WINDOWS:
                x, r = run_rule(B, side, lo, hi, h0, h1, a.stake)
                if x is None or len(x) < 20:
                    continue
                m, se, n_ev = clustered(r, x.event_slug)
                first = x.end.map(lambda e: e <= split)
                m1 = x.r[first].mean() if first.any() else np.nan
                m2 = x.r[~first].mean() if (~first).any() else np.nan
                key = f"{side} {lo:.2f}–{hi:.2f}, {h0}–{h1}ч"
                keep[key] = x
                # пессимистичная оценка: нижняя граница winrate (Уилсон, по числу СОБЫТИЙ —
                # корзины одного события зависимы) против средней цены с комиссией.
                # Важна для правил "купить по 0.97": 100% побед на малой выборке ничего не доказывают.
                wl = wilson_low(x.won_side.mean() * n_ev, n_ev)
                ev_low = wl / x.px.mean() - 1 - 0.07 * (1 - x.px.mean())
                rows.append({"правило": key, "сделок": len(x), "событий": n_ev,
                             "цена": x.px.mean(), "winrate": x.won_side.mean(),
                             "EV_%": m * 100, "±": se * 100, "EV_пессим_%": ev_low * 100,
                             "1я_половина_%": m1 * 100, "2я_половина_%": m2 * 100,
                             "тонкий_стакан_%": x.thin.mean() * 100,
                             "PnL_$": (x.r * a.stake).sum()})
    R = pd.DataFrame(rows).sort_values("EV_пессим_%", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"\n=== Правила (ставка ${a.stake:g}, доходность с комиссией тейкера 0.07·p·(1−p)) ===")
    print("EV_пессим — доходность при нижней границе winrate (95%). Правило интересно, если\n"
          "EV_пессим > 0 И плюс в ОБЕИХ половинах периода.")
    print(R.round(2).to_string(index=False))

    good = R[(R["EV_пессим_%"] > 0) & (R["1я_половина_%"] > 0) & (R["2я_половина_%"] > 0)]
    if good.empty:
        print("\nНи одно правило не прошло оба фильтра (EV_пессим > 0 и плюс в обеих половинах).")
        return
    print("\nПрошли фильтры: " + "; ".join(good["правило"]))
    best = good.sort_values("PnL_$", ascending=False).iloc[0]["правило"]
    x = keep[best]
    print(f"\n=== Разбор лучшего правила: {best} ===")
    by_city = x.groupby("city").agg(сделок=("r", "size"), EV_pct=("r", lambda r: r.mean() * 100),
                                    PnL=("r", lambda r: (r * a.stake).sum())).sort_values("PnL")
    print("По городам (худшие и лучшие):")
    show = by_city if len(by_city) <= 10 else pd.concat([by_city.head(5), by_city.tail(5)])
    print(show.round(2).to_string())
    x["day"] = pd.to_datetime(x.end, unit="s").dt.date
    daily = x.groupby("day").r.apply(lambda r: (r * a.stake).sum())
    print(f"\nPnL по дням: {daily.round(2).to_dict()}")
    print(f"Дней в плюсе: {(daily > 0).sum()} из {len(daily)}; худшая сделка: {x.r.min()*a.stake:+.2f}$")
    print("\nПомни: правил проверено много, и одно из них могло 'выстрелить' случайно. "
          "Окончательный ответ — повторный прогон через неделю с --since <сегодняшняя дата>.")


if __name__ == "__main__":
    main()
