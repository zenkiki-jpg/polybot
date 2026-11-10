"""
Отчёт по данным логгера лесенок (data/ladder).

    py analysis/ladder_report.py
    py analysis/ladder_report.py --since 2026-10-20 --margin 0.03

Разделы:
  1. Покрытие и проверка правила резолва (по цене Binance в момент конца).
  2. Арбитраж внутри лесенки "above": вероятность "выше 82k" не может быть меньше,
     чем "выше 84k". Если Yes(82k) по аску + No(84k) по аску стоят меньше $1 —
     это гарантированная прибыль (платит минимум $1 при любом исходе).
  3. Арбитраж в диапазонных событиях ("price on"): сумма асков всех корзин < $1.
  4. Калибровка: доходность покупки Yes по аску по цене и времени до конца.
  5. Модель по опционам Deribit против рынка: вероятность "выше K" из подразумеваемой
     волатильности (Блэк–Шоулз, без учёта наклона улыбки) и доходность покупки
     там, где модель расходится с ценой больше, чем на --margin.
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

TTE_BINS = [0, 3600, 6 * 3600, 24 * 3600, 3 * 86400, 30 * 86400]
TTE_LABELS = ["<1ч", "1–6ч", "6–24ч", "1–3д", ">3д"]
CUR = {"bitcoin": "BTC", "ethereum": "ETH"}


def norm_cdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


# ---------------------------------------------------------------------- #
def coverage(B, O):
    print("\n=== 1. Покрытие и проверка правила резолва ===")
    if B.empty:
        print("нет снимков")
        return
    g = B.groupby(["kind", "coin"]).agg(событий=("event_slug", "nunique"), страйков=("market_slug", "nunique"),
                                        снимков=("ts", "nunique"))
    print(g.to_string())
    if O.empty:
        print("исходов пока нет")
        return
    o = O[O.winner.isin(["Yes", "No"])].copy()
    print(f"\nЗарезолвлено страйков: {len(o)} в {o.event_slug.nunique()} событиях; источники: "
          f"{O.source.value_counts().to_dict()}")
    a = o[(o.kind.astype(str).str.startswith("above")) & o.binance_close_at_end.notna()]
    if len(a):
        gt = (a.binance_close_at_end > a.lo) == (a.winner == "Yes")
        ge = (a.binance_close_at_end >= a.lo) == (a.winner == "Yes")
        print(f"Правило 'Yes, если Binance close > K' совпало в {gt.mean():.2%}, '>= K' — в {ge.mean():.2%} "
              f"({len(a)} страйков). ~100% значит, мы правильно понимаем, по какой цене резолвится.")


def ladder_arb(B, margin=0.0):
    print("\n=== 2. Арбитраж внутри лесенки 'above' (Yes нижнего страйка + No верхнего < $1) ===")
    x = B[B.kind.astype(str).str.startswith("above") & B.lo.notna() & np.isfinite(B.lo) & (B.tte_s > 0)]
    x = x.dropna(subset=["yes_ask", "no_ask"])
    if x.empty:
        print("нет данных")
        return
    found = []
    snaps = 0
    for (ev, ts), g in x.groupby(["event_slug", "ts"]):
        g = g.sort_values("lo")
        if len(g) < 2:
            continue
        snaps += 1
        k = g.lo.to_numpy()
        ya = g.yes_ask.to_numpy()
        na = g.no_ask.to_numpy()
        ys = g.yes_ask_size.to_numpy()
        ns = g.no_ask_size.to_numpy()
        cost = (ya + fee_per_share(ya))[:, None] + (na + fee_per_share(na))[None, :]
        i, j = np.where(np.triu(np.ones_like(cost, dtype=bool), 1) & (cost < 1 - margin))
        for a_, b_ in zip(i, j):
            found.append({"event_slug": ev, "ts": ts, "k_low": k[a_], "k_high": k[b_],
                          "cost": cost[a_, b_], "edge": 1 - cost[a_, b_], "max_sets": min(ys[a_], ns[b_]),
                          "type": "лесенка"})
    A = pd.DataFrame(found)
    print(summarize_arb(A, snaps))
    if len(A):
        top = A.sort_values("edge", ascending=False).head(8)
        print(top[["event_slug", "ts", "k_low", "k_high", "cost", "max_sets"]].round(4).to_string(index=False))


def range_arb(B):
    print("\n=== 3. Арбитраж в диапазонных событиях ('price on'): сумма корзин ===")
    x = B[(B.kind == "range") & (B.tte_s > 0)]
    if x.empty:
        print("диапазонных событий нет")
        return
    A = bucket_sum_arb(x)
    print(summarize_arb(A, x.groupby(["event_slug", "ts"]).ngroups))


def deribit_model(B, D, O, margin):
    print("\n=== 5. Модель по опционам Deribit против рынка (лесенки 'above') ===")
    if D.empty or O.empty:
        print("нет данных Deribit или исходов")
        return
    win = O.set_index("market_slug").winner
    x = B[(B.kind == "above") & B.lo.notna() & np.isfinite(B.lo)].copy()
    x["won"] = x.market_slug.map(win).map({"Yes": 1.0, "No": 0.0})
    x = x.dropna(subset=["won", "spot", "end"])
    x["cur"] = x.coin.map(CUR)
    x = x.dropna(subset=["cur"])
    if x.empty:
        print("нет зарезолвленных лесенок")
        return
    D = D.dropna(subset=["mark_iv", "strike", "expiry"]).copy()
    D["minute"] = (D.ts // 60).astype("int64")
    # одна кривая IV на (валюта, минута, экспирация, страйк): среднее колла и пута
    D = D.groupby(["currency", "minute", "expiry", "strike"], as_index=False).mark_iv.mean()
    surf = {}
    for (c, mn), g in D.groupby(["currency", "minute"]):
        surf[(c, mn)] = {e: (h.strike.to_numpy(), h.mark_iv.to_numpy()) for e, h in g.sort_values("strike").groupby("expiry")}
    # прореживаем: одна точка на страйк в 15 минут — этого достаточно и быстро
    x["slot"] = (x.ts // 900).astype("int64")
    x = x.sort_values("ts").drop_duplicates(["market_slug", "slot"])
    minutes = {c: np.array(sorted(m for (cc, m) in surf if cc == c)) for c in D.currency.unique()}

    def prob_above(cur, ts, T, K, F):
        ms = minutes.get(cur)
        if ms is None or not len(ms):
            return np.nan
        i = np.searchsorted(ms, ts // 60)
        cand = [ms[j] for j in (i - 1, i) if 0 <= j < len(ms)]
        m = min(cand, key=lambda v: abs(v - ts // 60))
        if abs(m - ts // 60) > 5:
            return np.nan
        s = surf[(cur, m)]
        exps = np.array(sorted(s))
        t = (T - ts) / (365.25 * 86400)
        if t <= 0:
            return np.nan

        def w_at(e):
            ks, ivs = s[e]
            iv = np.interp(K, ks, ivs) / 100.0
            return iv * iv * max((e - ts) / (365.25 * 86400), 1e-6)
        after = exps[exps >= T]
        before = exps[exps < T]
        if len(after) and len(before):
            e1, e2 = before[-1], after[0]
            w1, w2 = w_at(e1), w_at(e2)
            f = (T - e1) / (e2 - e1)
            w = w1 + f * (w2 - w1)
        elif len(after):
            e2 = after[0]
            w = w_at(e2) * t / max((e2 - ts) / (365.25 * 86400), 1e-6)   # та же IV, меньше времени
        else:
            e1 = before[-1]
            w = w_at(e1) * t / max((e1 - ts) / (365.25 * 86400), 1e-6)
        if w <= 0:
            return np.nan
        d2 = (math.log(F / K) - w / 2) / math.sqrt(w)
        return 0.5 * (1 + math.erf(d2 / math.sqrt(2)))

    x["p_model"] = [prob_above(c, int(ts), int(T), K, F)
                    for c, ts, T, K, F in zip(x.cur, x.ts, x.end, x.lo, x.spot)]
    x = x.dropna(subset=["p_model"])
    if x.empty:
        print("не удалось сопоставить снимки с Deribit")
        return
    x["mid"] = (x.yes_bid + x.yes_ask) / 2
    x["tb"] = pd.cut(x.end - x.ts, [0, 6 * 3600, 24 * 3600, 3 * 86400, 30 * 86400],
                     labels=["<6ч", "6–24ч", "1–3д", ">3д"])
    ok = x.dropna(subset=["mid"])
    ok = ok[(ok.mid > 0.02) & (ok.mid < 0.98)]
    print("Ошибка прогноза (Brier, меньше = лучше), только неочевидные страйки (цена 0.02–0.98):")
    print(ok.groupby("tb", observed=True).apply(lambda g: pd.Series({
        "снимков": len(g), "модель": ((g.p_model - g.won) ** 2).mean(), "рынок": ((g.mid - g.won) ** 2).mean()}),
        include_groups=False).round(4).to_string())

    # торговля: первая точка на страйк, где модель расходится с ценой больше margin
    tr = []
    for _, r in x.sort_values("ts").iterrows():
        if pd.notna(r.yes_ask) and 0 < r.yes_ask < 1 and r.p_model - r.yes_ask - fee_per_share(r.yes_ask) > margin:
            tr.append((r.event_slug, r.market_slug, "Yes", r.yes_ask, r.won, r.end - r.ts))
        if pd.notna(r.no_ask) and 0 < r.no_ask < 1 and (1 - r.p_model) - r.no_ask - fee_per_share(r.no_ask) > margin:
            tr.append((r.event_slug, r.market_slug, "No", r.no_ask, 1 - r.won, r.end - r.ts))
    T = pd.DataFrame(tr, columns=["event_slug", "market_slug", "side", "ask", "won", "tte"])
    if T.empty:
        print(f"Расхождений больше {margin:.0%} не было.")
        return
    T = T.drop_duplicates(["market_slug", "side"], keep="first")
    T["r"] = ev_buy(T.ask, T.won)
    m, se, n = clustered(T.r, T.event_slug)
    print(f"\nПокупка там, где модель выше цены на >{margin:.0%} (первая такая точка на страйк):")
    print(f"  сделок {len(T)} в {n} событиях, средняя цена {T.ask.mean():.3f}, winrate {T.won.mean():.3f}, "
          f"доходность {m*100:+.2f}% ± {se*100:.2f} (на $1, с комиссией)")
    print(T.groupby("side").apply(lambda g: f"{len(g)} сделок, {g.r.mean()*100:+.2f}%", include_groups=False).to_string())
    print("Модель не учитывает наклон улыбки волатильности и разницу Deribit/Binance — это первое приближение.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/ladder")
    ap.add_argument("--since")
    ap.add_argument("--margin", type=float, default=0.03, help="минимальное расхождение модели и цены")
    a = ap.parse_args()
    d = Path(a.dir)
    B = load_gz(d, "books")
    D = load_gz(d, "deribit")
    O = load_outcomes(d)
    if a.since and len(B):
        t0 = pd.Timestamp(a.since, tz="UTC").timestamp()
        B = B[B.ts >= t0]
        D = D[D.ts >= t0] if len(D) else D
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"Снимков стакана: {len(B)}, строк Deribit: {len(D)}, исходов: {len(O)}")
    coverage(B, O)
    if B.empty:
        return
    ladder_arb(B)
    range_arb(B)
    print("\n=== 4. Калибровка: доходность покупки Yes по аску, % на $1 (± — по событиям) ===")
    if len(O):
        B["won"] = B.market_slug.map(O.set_index("market_slug").winner).map({"Yes": 1.0, "No": 0.0})
        print(calibration(B, TTE_BINS, TTE_LABELS).to_string())
    else:
        print("исходов пока нет")
    deribit_model(B, D, O, a.margin)


if __name__ == "__main__":
    main()
