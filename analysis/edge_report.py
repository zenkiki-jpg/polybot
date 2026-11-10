"""
Честная проверка эджа по логу — без модели цены, прямо по реальным аскам.

Главная метрика: матожидание на $1 при покупке по аску с комиссией тейкера
    EV = won / ask - 1 - fee_rate * (1 - ask)
Считается по окнам (одна сделка на окно), со стандартной ошибкой.
Если EV не больше нуля с запасом в 2 стандартные ошибки — эджа нет.

Режимы:
  Новый лог (логгер v2):
      py analysis/edge_report.py --ticks data/ticks.csv --windows data/windows.csv
  Старый лог (v1, колонки bid/ask перепутаны, исход по рынку в конце окна):
      py analysis/edge_report.py --legacy data/live_log.csv

Старый режим — ПРИБЛИЖЕНИЕ: там нет официального исхода и настоящего
price to beat. Используй его только чтобы понять порядок величин.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Windows (cp1251): не падать на символах вроде ~/Δ при выводе в файл или пайп
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

FEE_RATE = 0.07
PRICE_BINS = [0, .1, .3, .5, .7, .85, .9, .95, .97, .98, .99, .999]
TIME_BINS = [0, 20, 60, 120, 300]


# ---------------------------------------------------------------------- #
# Загрузка
# ---------------------------------------------------------------------- #
def load_v2(ticks_path: str, windows_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.read_csv(ticks_path)
    w = pd.read_csv(windows_path)
    w = w.drop_duplicates("slug", keep="last")
    ok = w[w.outcome.isin(["UP", "DOWN"])][["slug", "outcome"]]
    t = t.merge(ok, on="slug", how="inner")
    df = pd.DataFrame({
        "slug": t.slug, "stc": t.seconds_to_close, "ref": t.ref_price,
        "ptb": t.price_to_beat, "up_ask": t.up_ask, "down_ask": t.down_ask,
        "up_bid": t.up_bid, "down_bid": t.down_bid,
        "up_ask_size": t.up_ask_size, "down_ask_size": t.down_ask_size,
        "outcome": t.outcome, "ts": t.ts_unix,
    })
    return df, w


def load_legacy(path: str) -> pd.DataFrame:
    OLD = ["logged_at_utc", "window_start_unix", "window_slug", "seconds_to_close",
           "btc_price_binance", "yes_best_bid", "yes_best_ask", "yes_mid",
           "no_best_bid", "no_best_ask", "no_mid"]
    NEW = OLD[:5] + ["btc_price_chainlink"] + OLD[5:]
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            if len(x) == 12:
                rows.append(dict(zip(NEW, x)))
            elif len(x) == 11:
                d = dict(zip(OLD, x))
                d["btc_price_chainlink"] = None
                rows.append(d)
    L = pd.DataFrame(rows)
    for c in NEW[1:]:
        if c != "window_slug":
            L[c] = pd.to_numeric(L[c], errors="coerce")
    L = L[~((L.yes_best_bid == 0.01) & (L.yes_best_ask == 0.99))]
    L["ts"] = pd.to_datetime(L.logged_at_utc, format="ISO8601", utc=True).astype("int64") / 1e9

    # v1 брал /price: side=SELL (записан как *_best_bid) на деле = лучший АСК
    inverted = (L.yes_best_bid > L.yes_best_ask).mean()
    print(f"[legacy] доля строк, где 'bid' > 'ask': {inverted:.1%} "
          f"-> {'меняю местами' if inverted > 0.5 else 'НЕ меняю'}")
    if inverted > 0.5:
        up_ask, up_bid, dn_ask, dn_bid = L.yes_best_bid, L.yes_best_ask, L.no_best_bid, L.no_best_ask
    else:
        up_ask, up_bid, dn_ask, dn_bid = L.yes_best_ask, L.yes_best_bid, L.no_best_ask, L.no_best_bid

    # исход: цена рынка в последние <=8с (приближение!)
    last = L[L.seconds_to_close <= 8].sort_values("seconds_to_close").groupby("window_slug").first()
    out = last.yes_mid.map(lambda m: "UP" if m >= 0.95 else ("DOWN" if m <= 0.05 else None))
    # price to beat: первая Chainlink-цена в окне (spot, а рынок резолвится по TWAP — приближение!)
    cl = L.dropna(subset=["btc_price_chainlink"])
    first = cl[cl.seconds_to_close >= 294].sort_values("seconds_to_close", ascending=False) \
        .groupby("window_slug").first().btc_price_chainlink

    df = pd.DataFrame({
        "slug": L.window_slug, "stc": L.seconds_to_close, "ref": L.btc_price_chainlink,
        "up_ask": up_ask, "down_ask": dn_ask, "up_bid": up_bid, "down_bid": dn_bid,
        "up_ask_size": np.nan, "down_ask_size": np.nan, "ts": L.ts,
    })
    df["outcome"] = df.slug.map(out)
    df["ptb"] = df.slug.map(first)
    return df.dropna(subset=["outcome"])


# ---------------------------------------------------------------------- #
# Анализ
# ---------------------------------------------------------------------- #
def ev_per_dollar(ask, won, fee_rate=FEE_RATE):
    return won / ask - 1.0 - fee_rate * (1.0 - ask)


def summarize(ret: pd.Series) -> dict:
    n = int(ret.count())
    m = ret.mean() if n else np.nan
    se = ret.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {"n": n, "ev_%": m * 100, "se_%": se * 100,
            "ci95_low_%": (m - 1.96 * se) * 100 if n > 1 else np.nan}


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Насколько рынок откалиброван: для каждой (время, цена) — первая точка в окне."""
    rows = []
    for side in ("UP", "DOWN"):
        a = df[f"{side.lower()}_ask"]
        s = pd.DataFrame({"slug": df.slug, "stc": df.stc, "ask": a,
                          "won": (df.outcome == side).astype(float)})
        rows.append(s)
    S = pd.concat(rows)
    S = S[(S.ask > 0.005) & (S.ask < 0.9995)]
    S["tb"] = pd.cut(S.stc, TIME_BINS)
    S["pb"] = pd.cut(S.ask, PRICE_BINS)
    S = S.sort_values("stc", ascending=False).groupby(["tb", "pb", "slug"], observed=True).first().reset_index()
    S["ret"] = ev_per_dollar(S.ask, S.won)
    out = S.groupby(["tb", "pb"], observed=True).apply(
        lambda g: pd.Series({"windows": len(g), "avg_ask": g.ask.mean(), "winrate": g.won.mean(),
                             **summarize(g.ret)}), include_groups=False)
    return out


def rule_grid(df: pd.DataFrame, thresholds, offsets, stake: float, max_price: float,
              require_agree: bool = True) -> pd.DataFrame:
    """Стратегия 'следуй за движением': первая точка в окне, где stc<=offset и |ref-ptb|>=thr,
    И есть аск < max_price (иначе ждём следующий тик). Покупка по лучшему аску."""
    d = df.dropna(subset=["ref", "ptb"]).copy()
    d["delta"] = d.ref - d.ptb
    d["side"] = np.where(d.delta >= 0, "UP", "DOWN")
    d["ask"] = np.where(d.side == "UP", d.up_ask, d.down_ask)
    d["bid"] = np.where(d.side == "UP", d.up_bid, d.down_bid)
    d["ask_size"] = np.where(d.side == "UP", d.up_ask_size, d.down_ask_size)
    d["won"] = (d.side == d.outcome).astype(float)
    d = d.sort_values(["slug", "stc"], ascending=[True, False])
    res = []
    for thr in thresholds:
        for off in offsets:
            sig = d[(d.stc <= off) & (d.stc > 0) & (d.delta.abs() >= thr)]
            n_sig = sig.slug.nunique()
            if require_agree:
                ref_px = sig.bid.fillna(sig.ask)
                sig = sig[~(ref_px < 0.5)]
            n_agree = sig.slug.nunique()
            buy = sig[(sig.ask > 0) & (sig.ask <= max_price) & (sig.ask < 1.0)]
            first = buy.groupby("slug").first()
            if first.empty:
                res.append({"threshold": thr, "offset": off, "windows_signal": n_sig,
                            "windows_filled": 0, "fill_rate_%": 0.0})
                continue
            ret = ev_per_dollar(first.ask, first.won)
            thin = (first.ask_size * first.ask < stake).mean() if first.ask_size.notna().any() else np.nan
            s = summarize(ret)
            res.append({"threshold": thr, "offset": off, "windows_signal": n_sig,
                        "disagree_skipped": n_sig - n_agree,
                        "windows_filled": len(first),
                        "fill_rate_%": len(first) / max(n_sig, 1) * 100,
                        "avg_ask": first.ask.mean(), "winrate_%": first.won.mean() * 100,
                        "breakeven_wr_%": (first.ask * (1 + FEE_RATE * (1 - first.ask))).mean() * 100,
                        **s, "thin_top_level_%": thin * 100 if thin == thin else np.nan,
                        "pnl_$_at_stake": ret.sum() * stake})
    return pd.DataFrame(res)


def ptb_checks(w: pd.DataFrame):
    print("\n--- Проверка price to beat и воспроизводимости резолва ---")
    print("Источники исхода:", w.outcome_source.value_counts(dropna=False).to_dict())
    both = w.dropna(subset=["ptb_gamma", "ptb_rtds_twap60"])
    if len(both):
        diff = (both.ptb_gamma - both.ptb_rtds_twap60).abs()
        print(f"|ptb Gamma - RTDS TWAP60 на старте|: n={len(both)}, медиана {diff.median():.4f}, "
              f"макс {diff.max():.4f}  (~0 значит, мы правильно понимаем price to beat)")
    rep = w.dropna(subset=["close_rtds_twap60"]).copy()
    rep["ptb"] = rep.ptb_gamma.fillna(rep.ptb_rtds_twap60)
    rep = rep.dropna(subset=["ptb"])
    rep = rep[rep.outcome.isin(["UP", "DOWN"])]
    if len(rep):
        mine = np.where(rep.close_rtds_twap60 >= rep.ptb, "UP", "DOWN")
        print(f"Исход по нашим данным (TWAP60 в конце >= ptb) совпал с официальным: "
              f"{(mine == rep.outcome).mean():.2%} из {len(rep)} окон")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks")
    ap.add_argument("--windows")
    ap.add_argument("--legacy", help="старый live_log.csv (v1)")
    ap.add_argument("--thresholds", default="5,10,15,20,30,50")
    ap.add_argument("--offsets", default="10,15,30,60,120")
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--max-price", type=float, default=0.99)
    ap.add_argument("--allow-disagree", action="store_true")
    ap.add_argument("--out", help="куда сохранить таблицу сетки (csv)")
    a = ap.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    if a.legacy:
        df = load_legacy(a.legacy)
        print("!!  LEGACY-режим: исход = цена рынка в последние секунды, ptb = первая "
              "Chainlink spot-цена в окне. Это приближения, не истина.")
        w = None
    elif a.ticks and a.windows:
        df, w = load_v2(a.ticks, a.windows)
    else:
        ap.error("нужно --ticks и --windows, или --legacy")

    print(f"\nОкон с известным исходом: {df.slug.nunique()}, точек: {len(df)}")
    if w is not None:
        ptb_checks(w)

    print("\n--- 1. Калибровка рынка (одна точка на окно; EV = покупка по аску с комиссией) ---")
    print(calibration(df).round(3).to_string())

    th = [float(x) for x in a.thresholds.split(",")]
    of = [float(x) for x in a.offsets.split(",")]
    g = rule_grid(df, th, of, a.stake, a.max_price, not a.allow_disagree)
    print(f"\n--- 2. Правило 'следуй за движением' (max_price={a.max_price}) ---")
    print("ev_% — средняя доходность на $1; ci95_low_% > 0 — только тогда есть повод верить в эдж")
    print(g.round(3).to_string(index=False))
    if a.out:
        g.to_csv(a.out, index=False)
        print(f"\nСохранено: {a.out}")

    good = g[(g.get("ci95_low_%", pd.Series(dtype=float)) > 0) & (g.windows_filled >= 100)] \
        if "ci95_low_%" in g else pd.DataFrame()
    print("\nИТОГ:", "есть конфиги с EV > 0 на 95% уровне и n>=100 — проверь их на НОВЫХ данных:\n"
          + good[["threshold", "offset", "windows_filled", "ev_%", "ci95_low_%"]].to_string(index=False)
          if len(good) else "ни один конфиг не показал статистически значимого положительного EV.")
    print("Помни: перебрав много конфигов, какой-то один обязательно 'выстрелит' случайно. "
          "Верить можно только результату на данных, которых не было при выборе.")


if __name__ == "__main__":
    main()
