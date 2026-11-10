"""
Первичный отчёт по данным мульти-логгера (data/multi).

    py analysis/multi_report.py
    py analysis/multi_report.py --dir data/multi --since 2026-10-01

Разделы:
  1. Покрытие: сколько окон по каждому рынку, источники резолва и price to beat.
  2. Калибровка каждого рынка: EV покупки по аску (с комиссией) по ценовым корзинам,
     одна точка на окно. Ищем рынки, которые откалиброваны хуже BTC 5m.
  3. Сделки: кто зарабатывает — доходность покупателя по цене сделки
     (взвешено по объёму). Здесь виден перекос в пользу лонгшотов.
  4. Опережение между рынками (5m): двигается ли цена ETH/SOL/XRP/DOGE-рынка
     вслед за BTC-рынком или за Binance с задержкой.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core.multi import load_gz_csvs  # noqa: E402

FEE = 0.07
PB = [0, .1, .3, .5, .7, .85, .95, .99]
TICK_COLS = ["ts_unix", "key", "coin", "interval", "slug", "window_start", "window_end",
             "seconds_to_close", "up_bid", "up_ask", "down_bid", "down_ask", "binance_mid"]


def load_windows(d: Path) -> pd.DataFrame:
    """windows.csv + outcomes.csv -> один исход на окно (outcome_final)."""
    W = pd.read_csv(d / "windows.csv").drop_duplicates("slug", keep="last")
    off = W.loc[W.outcome.isin(["UP", "DOWN"]), ["slug", "outcome", "outcome_source"]]
    op = d / "outcomes.csv"
    kline = pd.DataFrame(columns=["slug", "outcome_kline"])
    if op.exists() and op.stat().st_size > 0:
        O = pd.read_csv(op)
        off = pd.concat([off, O.loc[O.outcome.isin(["UP", "DOWN"]), ["slug", "outcome", "outcome_source"]]])
        kline = O.loc[O.outcome_kline.isin(["UP", "DOWN"]), ["slug", "outcome_kline"]].drop_duplicates("slug", keep="last")
    off = off.drop_duplicates("slug", keep="last").rename(columns={"outcome": "official", "outcome_source": "official_src"})
    W = W.drop(columns=[c for c in ("official", "official_src", "outcome_kline") if c in W]) \
        .merge(off, on="slug", how="left").merge(kline, on="slug", how="left")
    W["outcome_final"] = W.official
    use_k = W.outcome_final.isna() & W.outcome_kline.notna()
    W.loc[use_k, "outcome_final"] = W.loc[use_k, "outcome_kline"]
    W["final_src"] = W.official_src.where(W.official.notna(), "kline").where(W.outcome_final.notna(), None)
    return W


def ev(ask, won):
    return won / ask - 1 - FEE * (1 - ask)


def coverage(W, T, A):
    print("\n=== 1. Покрытие ===")
    g = W.groupby("key").agg(
        windows=("slug", "nunique"),
        official=("official", lambda x: x.isin(["UP", "DOWN"]).sum()),
        by_kline=("final_src", lambda x: (x == "kline").sum()),
        ref_kind=("ref_kind", lambda x: ",".join(sorted(x.dropna().unique()))),
        ptb_gamma_pct=("ptb_gamma", lambda x: x.notna().mean() * 100))
    g["ticks"] = T.groupby("key").size() if len(T) else 0
    g["api_trades"] = A.groupby("key").size() if len(A) else 0
    print(g.fillna(0).round(1).to_string())
    src = W.groupby("key").resolution_source.agg(lambda x: " | ".join(sorted(x.dropna().unique()))[:90])
    print("\nИсточники резолва:\n" + src.to_string())

    print("\n--- Совпадает ли исход, посчитанный по нашим ценам, с официальным ---")
    rows = []
    for key, d in W[W.official.isin(["UP", "DOWN"])].groupby("key"):
        ptb = d.ptb_gamma.fillna(d.ptb_rtds_twap60)
        ok = ptb.notna() & d.close_rtds_twap60.notna()
        agree_tw = ((d.close_rtds_twap60 >= ptb) == (d.official == "UP"))[ok]
        kk = d.outcome_kline.notna()
        rows.append({"key": key, "окон": len(d),
                     "проверено_TWAP": int(ok.sum()),
                     "совпало_TWAP_%": agree_tw.mean() * 100 if ok.any() else None,
                     "проверено_свечой": int(kk.sum()),
                     "совпало_свеча_%": (d.outcome_kline == d.official)[kk].mean() * 100 if kk.any() else None})
    print(pd.DataFrame(rows).round(2).to_string(index=False) if rows else "нет окон с официальным исходом")


def calibration(W, T):
    print("\n=== 2. Калибровка: EV покупки по аску, % на $1 (в скобках — окон) ===")
    print("Одна точка на окно в каждой корзине; время — доля окна до конца.")
    if T.empty:
        print("нет тиков")
        return
    out = W[W.outcome_final.isin(["UP", "DOWN"])][["slug", "outcome_final"]].rename(
        columns={"outcome_final": "outcome"})
    d = T.merge(out, on="slug")
    if d.empty:
        print("нет данных")
        return
    d["frac_left"] = d.seconds_to_close / (d.window_end - d.window_start)
    parts = []
    for side in ("UP", "DOWN"):
        a = d[f"{side.lower()}_ask"]
        parts.append(pd.DataFrame({"key": d.key, "slug": d.slug, "frac": d.frac_left, "ask": a,
                                   "won": (d.outcome == side).astype(float)}))
    S = pd.concat(parts)
    S = S[(S.ask > 0.005) & (S.ask < 0.995) & (S.frac > 0.02)]
    S["pb"] = pd.cut(S.ask, PB)
    S = S.sort_values("frac", ascending=False).groupby(["key", "pb", "slug"], observed=True).first().reset_index()
    S["r"] = ev(S.ask, S.won)

    def cell(g):
        n = len(g)
        if n < 20:
            return ""
        se = g.r.std() / np.sqrt(n)
        star = " *" if abs(g.r.mean()) > 2 * se else ""
        return f"{g.r.mean()*100:+.1f}±{se*100:.1f} ({n}){star}"
    tab = S.groupby(["key", "pb"], observed=True).apply(cell, include_groups=False).unstack("pb").fillna("")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(tab.to_string())
    print("* — отличие от нуля больше 2 стандартных ошибок (при многих ячейках часть звёздочек — случайность)")


def trades_section(A, W):
    print("\n=== 3. Сделки: доходность покупателя по цене сделки (взвешено по $) ===")
    if A.empty:
        print("нет данных trades_api")
        return
    A = A.copy()
    for c in ("price", "size"):
        A[c] = pd.to_numeric(A[c], errors="coerce")
    A = A.dropna(subset=["price", "size"])
    A["winner"] = A.slug.map(W.set_index("slug").outcome_final)
    A = A[A.winner.isin(["UP", "DOWN"]) & (A.price > 0) & (A.price < 1)]
    A["won"] = (A.outcome.astype(str).str.upper() == A.winner).astype(float)
    A["usd"] = A["price"] * A["size"]
    A["r"] = A.won / A.price - 1
    A["pb"] = pd.cut(A.price, PB)
    g = A.groupby(["interval_key", "pb"], observed=True).apply(
        lambda x: f"{np.average(x.r, weights=x.usd)*100:+.1f}% (${x.usd.sum():,.0f})" if x.usd.sum() > 500 else "",
        include_groups=False).unstack("pb").fillna("")
    print("Доходность ДО комиссии; в скобках — объём. Минус = покупатели по этим ценам переплачивают.")
    print(g.to_string())
    s = A.groupby(["interval_key", "side"]).usd.sum().unstack().fillna(0).round(0)
    print("\nОбъём по сторонам (как их отдаёт API):\n" + s.to_string())
    top = A.groupby("wallet").agg(usd=("usd", "sum"), n=("usd", "size"),
                                  pnl=("r", lambda r: (r * A.loc[r.index, "usd"]).sum()))
    top = top.sort_values("usd", ascending=False).head(10)
    top["roi_%"] = top.pnl / top.usd * 100
    print("\nТоп-10 кошельков по объёму (грубо: только покупательская сторона их сделок):")
    print(top.round(1).to_string())


def lead_lag(T):
    print("\n=== 4. Опережение между 5m-рынками (корреляция изменений цены Up за 1 с) ===")
    d = T[T.interval == "5m"].copy() if len(T) else T
    if d.empty:
        print("нет 5m данных")
        return
    d["sec"] = d.ts_unix.round().astype("int64")
    d["mid"] = (d.up_bid + d.up_ask) / 2
    mid = d.pivot_table(index="sec", columns="coin", values="mid", aggfunc="last")
    bn = d.pivot_table(index="sec", columns="coin", values="binance_mid", aggfunc="last")
    win = d.pivot_table(index="sec", columns="coin", values="window_start", aggfunc="last")
    full = np.arange(mid.index.min(), mid.index.max() + 1)
    mid, bn, win = mid.reindex(full), bn.reindex(full), win.reindex(full)
    dm = mid.diff().where(win.eq(win.shift()))      # не сравниваем через границу окна
    db = np.log(bn).diff()
    lags = [-2, -1, 0, 1, 2, 3, 5]
    rows = []
    if "btc" in dm:
        for c in [c for c in dm.columns if c != "btc"]:
            rows.append({"что -> куда": f"BTC-рынок -> {c.upper()}-рынок",
                         **{f"lag{k}": round(dm[c].corr(dm["btc"].shift(k)), 3) for k in lags}})
    for c in dm.columns:
        if c in db:
            rows.append({"что -> куда": f"Binance {c.upper()} -> {c.upper()}-рынок",
                         **{f"lag{k}": round(dm[c].corr(db[c].shift(k)), 3) for k in lags}})
    print(pd.DataFrame(rows).to_string(index=False))
    print("lagK = корреляция изменения цены рынка сейчас с изменением источника K секунд назад.\n"
          "Если максимум при lag1–lag3, рынок реагирует с задержкой — это потенциальный эдж,\n"
          "но только если задержка больше нашей (сейчас ~0.5–1.5 с на цикл).")


def maker_taker(A, T, W):
    print("\n=== 5. Кто зарабатывает: мейкеры против тейкеров (5m, покупки) ===")
    if A.empty or T.empty:
        print("нет данных")
        return
    x = A[A.key.astype(str).str.endswith("5m") & (A.side == "BUY")].copy()
    for c in ("price", "size", "exchange_ts"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["winner"] = x.slug.map(W.set_index("slug").outcome_final)
    x = x.dropna(subset=["price", "size", "exchange_ts"])
    x = x[x.winner.isin(["UP", "DOWN"])]
    x["won"] = (x.outcome.astype(str).str.upper() == x.winner).astype(float)
    x["usd"] = x["price"] * x["size"]
    x["pnl"] = (x.won - x.price) * x["size"]
    x["sec"] = x.exchange_ts.astype("int64")
    t = T[T.interval == "5m"][["ts_unix", "slug", "up_bid", "up_ask", "down_bid", "down_ask"]].copy()
    t["sec"] = t.ts_unix.round().astype("int64")
    t = t.sort_values("sec").drop_duplicates(["slug", "sec"], keep="last")
    m = pd.merge_asof(x.sort_values("sec"), t.drop(columns="ts_unix"), on="sec", by="slug",
                      direction="nearest", tolerance=1)
    up = m.outcome.astype(str).str.upper() == "UP"
    m["bid"] = np.where(up, m.up_bid, m.down_bid)
    m["ask"] = np.where(up, m.up_ask, m.down_ask)
    m = m.dropna(subset=["bid", "ask"])
    m["role"] = np.where(m.price >= m.ask - 1e-9, "тейкер (по аску)",
                         np.where(m.price <= m.bid + 1e-9, "мейкер (по биду)", "внутри спреда"))
    m["coin"] = m.key.astype(str).str.split("-").str[0]

    def stat(g):
        w = g.groupby("slug").agg(p=("pnl", "sum"), u=("usd", "sum"))
        r = w.p / w.u
        return f"{g.pnl.sum() / g.usd.sum() * 100:+.1f}% ±{r.std() / np.sqrt(len(r)) * 100:.1f} (${g.usd.sum() / 1000:,.0f}k)"
    print("Доходность ДО комиссии, ± — стандартная ошибка по окнам, в скобках — объём.")
    print("Роль определяется сравнением цены сделки со стаканом в ту же секунду.")
    print(m.groupby(["coin", "role"]).apply(stat, include_groups=False).unstack().to_string())
    m["до_конца"] = pd.cut(300 - (m.sec % 300), [0, 30, 60, 120, 200, 300])
    print(m.groupby(["до_конца", "role"], observed=True).apply(stat, include_groups=False).unstack().to_string())
    print("Внимание: это СРЕДНЕЕ по всем мейкерам. Новичок стоит в конце очереди и получает\n"
          "исполнения хуже среднего — реальную доходность покажет только симуляция с очередью.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/multi")
    ap.add_argument("--since", help="дата UTC, напр. 2026-10-01")
    a = ap.parse_args()
    d = Path(a.dir)
    W = load_windows(d)
    T = load_gz_csvs(d, "ticks", usecols=TICK_COLS)
    api_cols = ["key", "slug", "window_start", "window_end", "exchange_ts", "asset_id", "outcome",
                "side", "price", "size", "usdc", "wallet", "tx"]
    A = pd.concat([load_gz_csvs(d, "trades_api", usecols=api_cols),
                   load_gz_csvs(d, "trades_api2", usecols=api_cols)], ignore_index=True)
    if len(A):
        A = A.drop_duplicates(["tx", "asset_id", "wallet", "price", "size", "side"]) if "tx" in A else A
    if a.since:
        t0 = pd.Timestamp(a.since, tz="UTC").timestamp()
        W = W[W.window_start >= t0]
        if len(T):
            T = T[T.window_start >= t0]
        if len(A):
            A = A[pd.to_numeric(A.window_start, errors="coerce") >= t0]
    if len(A):
        A["interval_key"] = A.key
    print(f"Окон: {len(W)}, с исходом: {W.outcome_final.notna().sum()}, тиков: {len(T)}, сделок (api): {len(A)}")
    coverage(W, T, A)
    calibration(W, T)
    trades_section(A, W)
    lead_lag(T)
    maker_taker(A, T, W)


if __name__ == "__main__":
    main()
