"""
Симулятор мейкерской стратегии на 5-минутных рынках — с честной очередью.

Идея: в момент T (за N секунд до конца окна) ставим лимитную заявку на ПОКУПКУ
токена Up или Down по цене p и держим до резолва. Вопрос — сколько бы нас
исполнили и сколько бы это принесло.

Как считается исполнение (консервативно):
  * Очередь перед нами = объём, который уже стоит на цене p в снимке стакана в
    секунду T (берётся из 5 уровней бидов). Отмены впереди нас НЕ учитываются —
    считаем, что все, кто стоял раньше, достоят до конца (это хуже реальности).
  * Каждая последующая сделка (строго после секунды T) по нашему токену по
    цене ровно p "съедает" очередь. Нас начинают исполнять, только когда съеден
    весь объём, стоявший перед нами.
  * Если прошла сделка по цене НИЖЕ p — уровень p пробит насквозь, значит нас
    исполнили полностью.
  * Заявка снимается за --cancel-before секунд до конца окна.
  * Мейкер не платит комиссию (ребейты не учитываем — ещё один запас).
Неблагоприятный отбор учитывается сам собой: исполнения берутся из реальных сделок.

Стратегии (сетка):
  side:  both  — по заявке на обе стороны (классический маркет-мейкинг)
         fav   — только на ту сторону, у которой бид выше (фаворит)
         dog   — только на дешёвую сторону
  price: join  — встать в очередь на лучший бид
         improve — на 1 тик выше лучшего бида (очереди перед нами нет), если это
                   всё ещё ниже аска
  post:  за сколько секунд до конца ставим заявку (каждая — отдельная заявка)
  hold:  сколько секунд заявка стоит, прежде чем мы её снимем (2, 5, 10, 30 или
         "full" — до конца окна). Это главный параметр: настоящие маркет-мейкеры
         снимают заявки за доли секунды, как только цена двинулась, а бот на
         домашнем компьютере с опросом раз в секунду — в лучшем случае за 1–2 с.

Для сравнения считается и тейкер: купить по аску в тот же момент (с комиссией).

Запуск:
    py analysis/maker_sim.py
    py analysis/maker_sim.py --size 50 --hold 2,5,full --since 2026-10-04
    py analysis/maker_sim.py --coins btc --out maker_results.csv
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

from core.multi import read_gz_csv_tolerant  # noqa: E402
from multi_report import load_windows  # noqa: E402

FEE = 0.07
TICK_COLS = ["ts_unix", "interval", "slug", "window_end", "up_bid", "up_ask", "down_bid", "down_ask",
             "up_bids", "down_bids"]
TRADE_COLS = ["key", "slug", "exchange_ts", "outcome", "side", "price", "size"]


# ---------------------------------------------------------------------- #
# Загрузка (по файлу, сразу фильтруя — чтобы не держать гигабайты в памяти)
# ---------------------------------------------------------------------- #
def _read(path, cols):
    import io
    txt = read_gz_csv_tolerant(path)
    if txt.count("\n") < 2:
        return pd.DataFrame(columns=cols)
    return pd.read_csv(io.StringIO(txt), usecols=lambda c: c in cols, low_memory=False)


def load_snapshots(d: Path, post_secs: list[int], coins, t0) -> pd.DataFrame:
    want = set(post_secs)
    parts = []
    files = sorted(d.glob("ticks_*.csv.gz"))
    for i, f in enumerate(files, 1):
        x = _read(f, TICK_COLS)
        if x.empty:
            continue
        x = x[x.interval == "5m"]
        if coins:
            x = x[x.slug.str.split("-").str[0].isin(coins)]
        if t0:
            x = x[x.window_end - 300 >= t0]
        x["stc"] = (x.window_end - x.ts_unix).round().astype(int)
        x = x[x.stc.isin(want)]
        parts.append(x)
        if i % 24 == 0:
            print(f"  тики: {i}/{len(files)} файлов", flush=True)
    s = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=TICK_COLS + ["stc"])
    # по одному снимку на (окно, секунда)
    return s.sort_values("ts_unix").drop_duplicates(["slug", "stc"], keep="first")


def load_trades(d: Path, coins, t0) -> pd.DataFrame:
    parts = []
    files = sorted(d.glob("trades_api2_*.csv.gz")) + sorted(d.glob("trades_api_*.csv.gz"))
    for i, f in enumerate(files, 1):
        x = _read(f, TRADE_COLS + ["window_start", "wallet", "tx"])
        if x.empty:
            continue
        x = x[x.key.astype(str).str.endswith("-5m") & (x.side == "BUY")]
        if coins:
            x = x[x.key.str.split("-").str[0].isin(coins)]
        if t0 and "window_start" in x:
            x = x[pd.to_numeric(x.window_start, errors="coerce") >= t0]
        x = x.drop_duplicates([c for c in ("tx", "wallet", "outcome", "price", "size") if c in x])
        parts.append(x[["slug", "exchange_ts", "outcome", "price", "size"]])
        if i % 24 == 0:
            print(f"  сделки: {i}/{len(files)} файлов", flush=True)
    t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=TRADE_COLS)
    t["exchange_ts"] = pd.to_numeric(t.exchange_ts, errors="coerce")
    t["price"] = pd.to_numeric(t.price, errors="coerce").round(4)
    t["size"] = pd.to_numeric(t["size"], errors="coerce")
    t["up"] = t.outcome.astype(str).str.lower() == "up"
    return t.dropna(subset=["exchange_ts", "price", "size"])


# ---------------------------------------------------------------------- #
# Логика исполнения
# ---------------------------------------------------------------------- #
def size_at(depth: str, price: float) -> float:
    """Объём на цене price в строке стакана вида '0.48:120|0.47:300|…'."""
    if not isinstance(depth, str) or not depth:
        return 0.0
    for part in depth.split("|"):
        try:
            p, s = part.split(":")
            if abs(float(p) - price) < 1e-6:
                return float(s)
        except ValueError:
            continue
    return 0.0


def tick_for(price: float) -> float:
    return 0.001 if (price < 0.04 or price > 0.96) else 0.01


def simulate_fill(tr_ts, tr_px, tr_sz, t_post, t_cancel, p, queue_ahead, size) -> float:
    """Сколько шаров исполнилось бы у заявки BUY по цене p.
    tr_* — сделки (покупки этого токена) в окне, отсортированные по времени."""
    m = (tr_ts > t_post) & (tr_ts <= t_cancel)
    if not m.any():
        return 0.0
    px, sz = tr_px[m], tr_sz[m]
    if (px < p - 1e-9).any():                 # уровень пробит насквозь
        return size
    at = sz[np.abs(px - p) < 1e-9].sum()      # сколько съели на нашей цене
    return float(np.clip(at - queue_ahead, 0.0, size))


def run(a):
    d = Path(a.dir)
    coins = [c.strip() for c in a.coins.split(",")] if a.coins else None
    t0 = pd.Timestamp(a.since, tz="UTC").timestamp() if a.since else None
    post = sorted({int(x) for x in a.post.split(",")}, reverse=True)
    holds = [10 ** 6 if h.strip() == "full" else int(h) for h in a.hold.split(",")]

    print("Загружаю исходы…", flush=True)
    W = load_windows(d)
    W = W[W.interval == "5m"] if "interval" in W else W
    winner = W.set_index("slug").outcome_final.to_dict()

    print("Загружаю снимки стакана…", flush=True)
    S = load_snapshots(d, post, coins, t0)
    S["winner"] = S.slug.map(winner)
    S = S[S.winner.isin(["UP", "DOWN"])]
    print(f"  снимков: {len(S)} в {S.slug.nunique()} окнах", flush=True)

    print("Загружаю сделки…", flush=True)
    T = load_trades(d, coins, t0)
    T = T[T.slug.isin(set(S.slug))].sort_values(["slug", "up", "exchange_ts"])
    print(f"  сделок (покупки, 5m): {len(T)}", flush=True)

    # индексы сделок по (окно, сторона)
    groups = {k: (g.exchange_ts.to_numpy(), g.price.to_numpy(), g["size"].to_numpy())
              for k, g in T.groupby(["slug", "up"], sort=False)}
    empty = (np.array([]), np.array([]), np.array([]))

    rows = []
    for r in S.itertuples(index=False):
        end = int(r.window_end)
        t_post = end - r.stc
        books = {}
        for side in ("UP", "DOWN"):
            pre = side.lower()
            bid, ask, depth = getattr(r, f"{pre}_bid"), getattr(r, f"{pre}_ask"), getattr(r, f"{pre}_bids")
            if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask >= 1 or bid >= ask:
                continue
            books[side] = (float(bid), float(ask), depth)
        if not books:
            continue
        fav = max(books, key=lambda s: books[s][0])
        for side, (bid, ask, depth) in books.items():
            if not (a.min_price <= bid <= a.max_price):
                continue
            role = "fav" if side == fav and len(books) == 2 else ("dog" if len(books) == 2 else "one")
            won = 1.0 if r.winner == side else 0.0
            ts, px, sz = groups.get((r.slug, side == "UP"), empty)
            for mode in ("join", "improve"):
                if mode == "join":
                    p, q = bid, size_at(depth, bid)
                else:
                    p = round(bid + tick_for(bid), 4)
                    if p >= ask - 1e-9:
                        continue
                    q = 0.0
                for hold in holds:
                    t_cancel = min(t_post + hold, end - a.cancel_before)
                    f = simulate_fill(ts, px, sz, t_post, t_cancel, p, q, a.size)
                    rows.append((r.slug, end, r.stc, side, role, mode, hold, p, q, f, f * p,
                                 f * (won - p), ask, won))
    R = pd.DataFrame(rows, columns=["slug", "end", "post", "side", "role", "mode", "hold", "price", "queue",
                                    "filled", "cost", "pnl", "ask", "won"])
    if R.empty:
        print("Нет данных для симуляции.")
        return
    R["coin"] = R.slug.str.split("-").str[0]
    R["day"] = pd.to_datetime(R.end, unit="s").dt.date
    # тейкер для сравнения: купить a.size по аску в тот же момент
    R["taker_pnl"] = a.size * (R.won - R.ask) - a.size * FEE * R.ask * (1 - R.ask)

    def summarize(g, side_filter):
        x = g[side_filter(g)]
        w = x.groupby("slug").agg(pnl=("pnl", "sum"), cost=("cost", "sum"))
        traded = w[w.cost > 0]
        n = len(traded)
        se = (traded.pnl / traded.cost).std() / np.sqrt(n) * 100 if n > 1 else np.nan
        daily = x.groupby("day").pnl.sum()
        return pd.Series({
            "заявок": len(x), "исполн_%": (x.filled > 0).mean() * 100 if len(x) else 0,
            "объём_$": x.cost.sum(), "PnL_$": x.pnl.sum(),
            "ROI_%": x.pnl.sum() / x.cost.sum() * 100 if x.cost.sum() > 0 else np.nan,
            "±": se, "дней_в_плюсе": f"{(daily > 0).sum()}/{len(daily)}",
            "худший_день_$": daily.min(),
            "тейкер_ROI_%": x.taker_pnl.sum() / (a.size * x.ask).sum() * 100,
        })

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    filters = {"both": lambda g: g.role.isin(["fav", "dog", "one"]),
               "fav": lambda g: g.role == "fav", "dog": lambda g: g.role == "dog"}
    out = []
    for (mode, hold), g in R.groupby(["mode", "hold"]):
        for name, flt in filters.items():
            s = summarize(g, flt)
            out.append({"цена": mode, "держим_с": "до конца" if hold >= 10 ** 6 else hold,
                        "сторона": name, **s.to_dict()})
    O = pd.DataFrame(out)
    print(f"\n=== Мейкер: заявка {a.size:g} шаров, ставим за {','.join(map(str, post))}с до конца, "
          f"цены {a.min_price}–{a.max_price} ===")
    print("ROI — на потраченные $, до ребейтов; ± — стандартная ошибка по окнам;\n"
          "тейкер_ROI — для сравнения: купить столько же по аску в тот же момент, с комиссией.")
    print(O.round(2).to_string(index=False))

    best = O.sort_values("PnL_$", ascending=False).iloc[0]
    bh = 10 ** 6 if best["держим_с"] == "до конца" else best["держим_с"]
    sel = R[(R["mode"] == best["цена"]) & (R.hold == bh) & filters[best["сторона"]](R)]
    print(f"\nЛучшая по PnL: {best['цена']} / {best['сторона']} / держим {best['держим_с']}. "
          f"По монетам, моменту входа и дням:")
    print(sel.groupby("coin").apply(lambda x: pd.Series({
        "PnL_$": x.pnl.sum(), "ROI_%": x.pnl.sum() / x.cost.sum() * 100 if x.cost.sum() else np.nan,
        "объём_$": x.cost.sum()}), include_groups=False).round(2).to_string())
    print(sel.groupby("post").apply(lambda x: pd.Series({
        "PnL_$": x.pnl.sum(), "ROI_%": x.pnl.sum() / x.cost.sum() * 100 if x.cost.sum() else np.nan}),
        include_groups=False).round(2).to_string())
    print(sel.groupby("day").pnl.sum().round(2).to_string())
    print("\nВнимание: лучшую строку выбрали ПО ЭТИМ ЖЕ данным. Верить можно, если она в плюсе\n"
          "и на днях, которых не было при выборе: перезапусти с --since <следующая дата>.")
    if a.out:
        O.to_csv(a.out, index=False)
        R.to_csv(Path(a.out).with_suffix(".orders.csv.gz"), index=False, compression="gzip")
        print(f"Сохранено: {a.out} и {Path(a.out).with_suffix('.orders.csv.gz')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Симулятор мейкера с очередью (5m рынки)")
    ap.add_argument("--dir", default="data/multi")
    ap.add_argument("--coins", default="", help="напр. btc,eth (по умолчанию все)")
    ap.add_argument("--since", help="дата UTC, напр. 2026-10-04")
    ap.add_argument("--size", type=float, default=20, help="размер заявки в шарах")
    ap.add_argument("--post", default="240,180,120,60,30", help="за сколько секунд до конца ставить")
    ap.add_argument("--hold", default="2,5,10,30,full",
                    help="сколько секунд держать заявку до отмены; full — до конца окна")
    ap.add_argument("--cancel-before", type=int, default=3, help="в любом случае снять за N секунд до конца")
    ap.add_argument("--min-price", type=float, default=0.05)
    ap.add_argument("--max-price", type=float, default=0.95)
    ap.add_argument("--out", help="сохранить таблицу (csv) и все заявки (.orders.csv.gz)")
    run(ap.parse_args())
