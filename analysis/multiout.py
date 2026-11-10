"""
Общие куски анализа для событий с несколькими исходами (лесенки и погода).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.multi import read_gz_csv_tolerant  # noqa: E402

FEE = 0.07


def fee_per_share(p):
    return FEE * p * (1 - p)


def load_gz(d: Path, prefix: str, usecols=None) -> pd.DataFrame:
    parts = []
    for f in sorted(Path(d).glob(f"{prefix}_*.csv.gz")):
        txt = read_gz_csv_tolerant(f)
        if txt.count("\n") > 1:
            parts.append(pd.read_csv(io.StringIO(txt), low_memory=False,
                                     usecols=(lambda c: c in usecols) if usecols else None))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_outcomes(d: Path) -> pd.DataFrame:
    p = Path(d) / "outcomes.csv"
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=["market_slug", "winner"])
    o = pd.read_csv(p)
    return o.drop_duplicates("market_slug", keep="last")


def ev_buy(ask, won):
    """Доходность на $1 при покупке по ask с комиссией тейкера."""
    return won / ask - 1 - FEE * (1 - ask)


def clustered(r: pd.Series, groups: pd.Series):
    """Среднее и стандартная ошибка с кластеризацией по событию (точки внутри
    события зависимы, поэтому считаем одно среднее на событие)."""
    g = r.groupby(groups).mean()
    n = len(g)
    return r.mean(), (g.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan), n


def calibration(B: pd.DataFrame, tte_bins, tte_labels) -> pd.DataFrame:
    """B: снимки с колонками event_slug, market_slug, tte_s, yes_ask, won.
    Одна точка на (рынок, корзина времени) — первая по времени."""
    x = B.dropna(subset=["yes_ask", "won"]).copy()
    x = x[(x.yes_ask > 0.005) & (x.yes_ask < 0.995)]
    x["tb"] = pd.cut(x.tte_s, tte_bins, labels=tte_labels)
    x["pb"] = pd.cut(x.yes_ask, [0, .05, .15, .3, .5, .7, .85, .95, .995])
    x = x.sort_values("tte_s", ascending=False).groupby(["tb", "pb", "market_slug"], observed=True).first().reset_index()
    x["r"] = ev_buy(x.yes_ask, x.won)

    def cell(g):
        if len(g) < 15:
            return ""
        m, se, n = clustered(g.r, g.event_slug)
        star = " *" if se == se and abs(m) > 2 * se else ""
        return f"{m*100:+.1f}±{se*100:.1f} ({len(g)}){star}"
    return x.groupby(["tb", "pb"], observed=True).apply(cell, include_groups=False).unstack("pb").fillna("")


def bucket_sum_arb(B: pd.DataFrame, min_edge: float = 0.0) -> pd.DataFrame:
    """Для событий, где исходы ВЗАИМОИСКЛЮЧАЮЩИЕ и покрывают всё (корзины):
      * купить Yes во всех корзинах: стоит sum(yes_ask), платит ровно 1;
      * купить No во всех n корзинах: стоит sum(no_ask), платит ровно n-1.
    Возвращает снимки, где после комиссий остаётся плюс, и объём на этом."""
    rows = []
    for (ev, ts), g in B.groupby(["event_slug", "ts"]):
        if g.yes_ask.isna().any() and g.no_ask.isna().any():
            continue
        n = len(g)
        if g.yes_ask.notna().all():
            cost = (g.yes_ask + fee_per_share(g.yes_ask)).sum()
            size = g.yes_ask_size.min() if "yes_ask_size" in g else np.nan
            rows.append({"event_slug": ev, "ts": ts, "type": "все Yes", "n": n, "cost": cost,
                         "payout": 1.0, "edge": 1.0 - cost, "max_sets": size})
        if g.no_ask.notna().all():
            cost = (g.no_ask + fee_per_share(g.no_ask)).sum()
            size = g.no_ask_size.min() if "no_ask_size" in g else np.nan
            rows.append({"event_slug": ev, "ts": ts, "type": "все No", "n": n, "cost": cost,
                         "payout": n - 1.0, "edge": (n - 1.0) - cost, "max_sets": size})
    R = pd.DataFrame(rows)
    return R[R.edge > min_edge] if len(R) else R


def summarize_arb(A: pd.DataFrame, total_snaps: int) -> str:
    if A.empty:
        return f"Нарушений не найдено (проверено снимков: {total_snaps})."
    ep = A.sort_values("edge", ascending=False).groupby(["event_slug", "type"]).first().reset_index()
    ep["usd"] = ep.edge * ep.max_sets.fillna(0)
    lines = [f"Снимков с плюсом после комиссий: {len(A)} из {total_snaps} "
             f"({len(A) / max(total_snaps, 1) * 100:.2f}%), событий: {ep.event_slug.nunique()}.",
             f"Медианный плюс на комплект: {A.edge.median()*100:.2f}¢, максимум {A.edge.max()*100:.2f}¢.",
             f"Если брать лучший момент в каждом событии и весь объём лучшего уровня: "
             f"~${ep.usd.sum():,.2f} суммарно.",
             "Это верхняя граница: глубже лучшего уровня не смотрим, задержку не учитываем."]
    return "\n".join(lines)
