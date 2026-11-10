"""
Отчёт по данным погодного логгера (data/weather).

    py analysis/weather_report.py
    py analysis/weather_report.py --since 2026-10-20 --margin 0.05

Разделы:
  1. Покрытие: города, станции, единицы, сколько событий зарезолвлено.
  2. Арбитраж по сумме корзин: корзины покрывают все варианты и ровно одна выигрывает,
     значит "купить Yes во всех" стоит меньше $1 — гарантированная прибыль.
  3. Калибровка: доходность покупки Yes по аску по цене и времени до конца.
  4. "Мёртвые корзины": если станция уже показала максимум X, корзины целиком ниже X
     выиграть не могут. Сколько за их No ещё просят и сколько бы это принесло.
  5. Ансамблевый прогноз против рынка: вероятность корзины = доля членов ансамбля,
     попавших в неё, с поправкой на систематическую ошибку станции (считается только
     по ПРОШЛЫМ дням — без подглядывания). Сравнение точности и доходность покупки там,
     где прогноз выше цены больше, чем на --margin.
"""
from __future__ import annotations

import argparse
import math
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

from multiout import (load_gz, load_outcomes, ev_buy, fee_per_share, clustered, calibration,  # noqa: E402
                      bucket_sum_arb, summarize_arb)

TTE_BINS = [-1e9, 0, 6 * 3600, 12 * 3600, 24 * 3600, 48 * 3600, 30 * 86400]
TTE_LABELS = ["после конца", "<6ч", "6–12ч", "12–24ч", "1–2д", ">2д"]


def c_to(unit, c):
    return c * 9 / 5 + 32 if unit == "F" else c


def in_bucket(v, lo, hi):
    lo = -math.inf if pd.isna(lo) else lo
    hi = math.inf if pd.isna(hi) else hi
    return (v >= lo) & (v <= hi)


# ---------------------------------------------------------------------- #
def coverage(E, O):
    print("\n=== 1. Покрытие ===")
    if E.empty:
        print("событий нет")
        return
    E = E.drop_duplicates("slug", keep="last")
    g = E.groupby(["city", "station", "unit"], dropna=False).agg(событий=("slug", "nunique"))
    if len(O):
        res = O[O.winner.isin(["Yes", "No"])].groupby("event_slug").winner.apply(lambda w: (w == "Yes").sum())
        E["resolved"] = E.slug.map(res).fillna(0) == 1
        g["зарезолвлено"] = E.groupby(["city", "station", "unit"], dropna=False).resolved.sum()
    print(g.to_string())
    bad = E[E.buckets.astype(str).str.contains("=None")]
    if len(bad):
        print(f"\nВ {len(bad)} событиях есть корзины, которые не удалось разобрать — пришли events.csv.")
    if len(O):
        n_yes = O[O.winner == "Yes"].groupby("event_slug").size()
        weird = n_yes[n_yes != 1]
        if len(weird):
            print(f"Событий, где выиграло не ровно 1 корзина: {len(weird)} (проверь правила).")


def dead_buckets(B, M, E):
    print("\n=== 4. 'Мёртвые корзины': станция уже показала больше, а No ещё не стоит $1 ===")
    if M.empty or B.empty:
        print("нет данных METAR")
        return
    off = E.drop_duplicates("slug").set_index("slug")
    M = M.dropna(subset=["temp_c", "obs_time"]).drop_duplicates(["station", "obs_time"])
    rows = []
    for ev, g in B.dropna(subset=["station", "date"]).groupby("event_slug"):
        st, date = g.station.iloc[0], g.date.iloc[0]
        unit = off.unit.get(ev, g.unit.iloc[0]) if "unit" in off else g.unit.iloc[0]
        uo = off.utc_offset.get(ev) if "utc_offset" in off else None
        if uo is None or pd.isna(uo):
            continue
        day0 = pd.Timestamp(date, tz="UTC").timestamp() - uo
        obs = M[(M.station == st) & (M.obs_time >= day0) & (M.obs_time < day0 + 86400)].sort_values("obs_time")
        if obs.empty:
            continue
        ot = obs.obs_time.to_numpy()
        runmax = np.maximum.accumulate(c_to(unit, obs.temp_c.to_numpy()))
        for r in g.itertuples(index=False):
            i = np.searchsorted(ot, r.ts, side="right") - 1
            if i < 0 or pd.isna(r.hi) or not np.isfinite(r.hi):
                continue
            seen_max = math.floor(runmax[i] + 1e-9)
            # запас 1 градус: METAR и Wunderground могут округлять по-разному
            if r.hi < seen_max - 1 and pd.notna(r.no_ask) and r.no_ask < 0.999:
                rows.append({"event_slug": ev, "market_slug": r.market_slug, "ts": r.ts, "no_ask": r.no_ask,
                             "no_ask_size": r.no_ask_size, "hi": r.hi, "seen": seen_max, "won": r.won})
    D = pd.DataFrame(rows)
    if D.empty:
        print("Таких моментов не было (рынок быстро обнуляет мёртвые корзины).")
        return
    first = D.sort_values("ts").groupby("market_slug").first().reset_index()
    first = first.dropna(subset=["won"])
    first["no_won"] = 1 - first.won
    first["r"] = ev_buy(first.no_ask, first.no_won)
    print(f"Корзин, которые уже были мёртвыми, но No стоил < $0.999: {len(D.market_slug.unique())}")
    if len(first):
        m, se, n = clustered(first.r, first.event_slug)
        print(f"Покупка No в первый такой момент: средняя цена {first.no_ask.mean():.4f}, "
              f"No выиграл в {first.no_won.mean():.2%}, доходность {m*100:+.2f}% ± {se*100:.2f} "
              f"на $1, объём лучшего уровня в среднем {first.no_ask_size.mean():.0f} шаров.")
        lost = first[first.no_won == 0]
        if len(lost):
            print(f"ВНИМАНИЕ: {len(lost)} раз 'мёртвая' корзина всё-таки выиграла — METAR и Wunderground "
                  f"разошлись. Пример: {lost.market_slug.iloc[0]}")


def ensemble_model(B, F, O, E, margin):
    print("\n=== 5. Ансамблевый прогноз против рынка ===")
    if F.empty or O.empty:
        print("нет прогнозов или исходов")
        return
    ev_info = E.drop_duplicates("slug").set_index("slug")
    # победившая корзина -> "факт" = середина корзины (для хвостов — граница)
    win = O[O.winner == "Yes"].copy()

    def center(r):
        if np.isfinite(r.lo) and np.isfinite(r.hi):
            return (r.lo + r.hi) / 2
        return r.hi if np.isfinite(r.hi) else r.lo
    win["lo"] = win.lo.astype(float)
    win["hi"] = win.hi.astype(float)
    win["actual"] = win.apply(center, axis=1)
    actual = win.set_index("event_slug").actual

    F = F.dropna(subset=["value"]).copy()
    F["is_member"] = F.kind == "ens"
    # последний прогноз до момента снимка: берём прогнозы, сделанные не позже снимка
    F = F.sort_values("ts")
    snaps = B.dropna(subset=["won"]).copy()
    snaps["slot"] = (snaps.ts // 3600).astype("int64")          # одна точка на корзину в час
    snaps = snaps.sort_values("ts").drop_duplicates(["market_slug", "slot"])

    # ошибка станции по прошлым дням: actual - ensemble_mean (только завершённые ДО снимка события)
    ens_mean_by_ev = {}
    for ev, g in snaps.groupby("event_slug"):
        i = ev_info.loc[ev] if ev in ev_info.index else None
        if i is None:
            continue
        f = F[(F.station == i.station) & (F.date == i.date)] if pd.notna(i.station) else F[(F.city == i.city) & (F.date == i.date)]
        f = f[f.unit == i.unit]
        if f.empty:
            continue
        last_ts = f.ts.max()
        ff = f[f.ts == last_ts]
        mem = ff[ff.is_member].value if ff.is_member.any() else ff.value
        ens_mean_by_ev[ev] = (i.station if pd.notna(i.station) else i.city, i.end, mem.mean())
    bias_hist = pd.DataFrame([(ev, st, end, m, actual.get(ev)) for ev, (st, end, m) in ens_mean_by_ev.items()],
                             columns=["event_slug", "station", "end", "ens_mean", "actual"]).dropna()

    out = []
    for ev, g in snaps.groupby("event_slug"):
        i = ev_info.loc[ev] if ev in ev_info.index else None
        if i is None:
            continue
        key = i.station if pd.notna(i.station) else i.city
        f = F[(F.station == i.station) & (F.date == i.date)] if pd.notna(i.station) else F[(F.city == i.city) & (F.date == i.date)]
        f = f[f.unit == i.unit]
        if f.empty:
            continue
        ts_list = np.sort(f.ts.unique())
        for t, gg in g.groupby("ts"):
            j = np.searchsorted(ts_list, t, side="right") - 1
            if j < 0:
                continue
            ff = f[f.ts == ts_list[j]]
            mem = (ff[ff.is_member].value if ff.is_member.any() else ff.value).to_numpy(dtype=float)
            if len(mem) < 5:
                continue
            past = bias_hist[(bias_hist.station == key) & (bias_hist.end < t)]
            bias = (past.actual - past.ens_mean).mean() if len(past) >= 3 else 0.0
            vals = np.round(mem + bias)
            spread = max(np.std(vals), 0.8)
            # сглаживание: каждый член ансамбля — нормальное распределение со стандартной ошибкой spread/2
            for r in gg.itertuples(index=False):
                lo = -np.inf if pd.isna(r.lo) else r.lo - 0.5
                hi = np.inf if pd.isna(r.hi) else r.hi + 0.5
                z = lambda v: 0.5 * (1 + np.vectorize(math.erf)(v / math.sqrt(2)))
                p = np.mean(z((hi - vals) / (spread / 2)) - z((lo - vals) / (spread / 2)))
                out.append((ev, r.market_slug, t, r.tte_s, float(np.clip(p, 0.001, 0.999)), r.yes_bid,
                            r.yes_ask, r.no_ask, r.won, bias))
    X = pd.DataFrame(out, columns=["event_slug", "market_slug", "ts", "tte_s", "p", "yes_bid", "yes_ask",
                                   "no_ask", "won", "bias"])
    if X.empty:
        print("не удалось сопоставить прогнозы со снимками")
        return
    X["mid"] = (X.yes_bid + X.yes_ask) / 2
    X["tb"] = pd.cut(X.tte_s, TTE_BINS, labels=TTE_LABELS)
    ok = X.dropna(subset=["mid"])
    ok = ok[(ok.mid > 0.02) & (ok.mid < 0.98)]
    print("Ошибка прогноза (Brier, меньше = лучше), неочевидные корзины (цена 0.02–0.98):")
    print(ok.groupby("tb", observed=True).apply(lambda g: pd.Series({
        "точек": len(g), "модель": ((g.p - g.won) ** 2).mean(), "рынок": ((g.mid - g.won) ** 2).mean()}),
        include_groups=False).round(4).to_string())

    tr = []
    for r in X.sort_values("ts").itertuples(index=False):
        if r.tte_s is None or r.tte_s <= 0:
            continue
        if pd.notna(r.yes_ask) and 0 < r.yes_ask < 1 and r.p - r.yes_ask - fee_per_share(r.yes_ask) > margin:
            tr.append((r.event_slug, r.market_slug, "Yes", r.yes_ask, r.won, r.tte_s))
        if pd.notna(r.no_ask) and 0 < r.no_ask < 1 and (1 - r.p) - r.no_ask - fee_per_share(r.no_ask) > margin:
            tr.append((r.event_slug, r.market_slug, "No", r.no_ask, 1 - r.won, r.tte_s))
    T = pd.DataFrame(tr, columns=["event_slug", "market_slug", "side", "ask", "won", "tte"])
    if T.empty:
        print(f"Расхождений больше {margin:.0%} не было.")
        return
    T = T.drop_duplicates(["market_slug", "side"], keep="first")
    T["r"] = ev_buy(T.ask, T.won)
    m, se, n = clustered(T.r, T.event_slug)
    print(f"\nПокупка там, где прогноз выше цены на >{margin:.0%} (первая такая точка на корзину):")
    print(f"  сделок {len(T)} в {n} событиях, средняя цена {T.ask.mean():.3f}, winrate {T.won.mean():.3f}, "
          f"доходность {m*100:+.2f}% ± {se*100:.2f} на $1 с комиссией")
    print(T.groupby("side").apply(lambda g: f"{len(g)} сделок, цена {g.ask.mean():.3f}, {g.r.mean()*100:+.2f}%",
                                  include_groups=False).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/weather")
    ap.add_argument("--since")
    ap.add_argument("--margin", type=float, default=0.05)
    a = ap.parse_args()
    d = Path(a.dir)
    B = load_gz(d, "books")
    F = load_gz(d, "forecasts")
    M = load_gz(d, "metar")
    O = load_outcomes(d)
    E = pd.read_csv(d / "events.csv") if (d / "events.csv").exists() else pd.DataFrame()
    if len(F) and len(E):
        offs = F.dropna(subset=["utc_offset"]).groupby(["station", "date"]).utc_offset.last()
        E["utc_offset"] = [offs.get((s, dt)) for s, dt in zip(E.station, E.date)]
    if a.since and len(B):
        t0 = pd.Timestamp(a.since, tz="UTC").timestamp()
        B = B[B.ts >= t0]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"Снимков: {len(B)}, прогнозов: {len(F)}, METAR: {len(M)}, исходов (корзин): {len(O)}")
    coverage(E, O)
    if B.empty:
        return
    if len(O):
        B["won"] = B.market_slug.map(O.set_index("market_slug").winner).map({"Yes": 1.0, "No": 0.0})
    else:
        B["won"] = np.nan
    print("\n=== 2. Арбитраж по сумме корзин (все Yes дешевле $1 или все No дешевле n−1) ===")
    live = B[B.tte_s > 0]
    A = bucket_sum_arb(live)
    print(summarize_arb(A, live.groupby(["event_slug", "ts"]).ngroups))
    if len(A):
        print("Сами случаи (проверь глазами — одиночные огромные 'плюсы' обычно сбой стакана):")
        print(A.sort_values("edge", ascending=False).head(5).round(4).to_string(index=False))
    print("\n=== 3. Калибровка: доходность покупки Yes по аску, % на $1 (± — по событиям) ===")
    print(calibration(B[B.tte_s > 0], TTE_BINS, TTE_LABELS).to_string() if B.won.notna().any() else "исходов пока нет")
    dead_buckets(B, M, E)
    ensemble_model(B, F, O, E, a.margin)


if __name__ == "__main__":
    main()
