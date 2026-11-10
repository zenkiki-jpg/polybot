"""
Paper trader v2 — честная проверка стратегии "следуй за движением".

Отличия от v1:
  * Одним процессом гоняет СЕТКУ параметров (threshold × offset) на одних и
    тех же данных — больше не нужно запускать 12 окон.
  * Цена открытия = официальный price to beat (Gamma) или наблюдение Chainlink
    ровно на старте окна. Если его нет (бот запущен посреди окна) — окно
    пропускается.
  * Сигнал считается по тому потоку, по которому рынок резолвится
    (сейчас Chainlink TWAP 60s), либо по spot/binance, если задать --signal.
  * Покупка симулируется по РЕАЛЬНЫМ аскам из /book, уровень за уровнем, с
    комиссией тейкера shares*0.07*p*(1-p). Если аска ниже $1 нет — сделка
    не исполнена (и так и записывается), бот пробует на следующих тиках.
  * Если рынок оценивает твою сторону дешевле 0.5 — это почти наверняка
    ошибка твоих данных, а не подарок. Такие сигналы по умолчанию не
    торгуются, но записываются как skip_disagree с итоговым исходом, чтобы
    было видно, кто был прав.
  * Исход — только официальный (Data API / Gamma). Если за 30 минут резолва
    нет, сделка пишется с outcome пустым и в PnL не идёт.
  * Один CSV: одна строка = одна попытка (конфиг × окно), пишется после резолва.

Запуск:
    py live/paper_trader.py
    py live/paper_trader.py --thresholds 5,10,20,30,50 --offsets 10,20,30,60,120 --stake 10
Отчёт в любой момент:
    py analysis/paper_report.py --csv data/paper_v2.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pm import (Fill, Market, PriceFeed, MarketCache, Resolver, fetch_books,
                     simulate_buy, settle_pnl, price_to_beat, window_start, slug_for,
                     WINDOW, log)

CSV_FIELDS = [
    "config", "threshold", "offset", "slug", "window_start", "status", "attempts",
    "side", "signal_source", "signal_price", "price_to_beat", "ptb_source", "delta",
    "stc_at_decision", "decided_at_utc",
    "our_best_ask", "our_best_bid", "other_best_ask",
    "fill_shares", "fill_avg_price", "fill_worst_price", "fill_cost", "fee",
    "outcome", "outcome_source", "won", "pnl",
]


@dataclass(frozen=True)
class Config:
    threshold: float
    offset: float

    @property
    def name(self) -> str:
        return f"t{self.threshold:g}_o{self.offset:g}"


@dataclass
class Attempt:
    cfg: Config
    status: str = "pending"      # pending | filled | partial | no_liquidity | too_small |
                                 # price_cap | skip_disagree | no_signal
    attempts: int = 0
    side: str | None = None
    signal_price: float | None = None
    delta: float | None = None
    stc: float | None = None
    decided_at: float | None = None
    our_ask: float | None = None
    our_bid: float | None = None
    other_ask: float | None = None
    fill: Fill = field(default_factory=lambda: Fill("none"))

    @property
    def done(self) -> bool:
        return self.status in ("filled", "partial", "skip_disagree")


@dataclass
class WindowRun:
    market: Market
    ptb: float | None
    ptb_source: str | None
    attempts: dict  # cfg.name -> Attempt


class Engine:
    """Вся логика решений — без сети, чтобы её можно было тестировать."""

    def __init__(self, configs, stake: float, max_price: float, require_agree: bool,
                 signal_source: str, writer=None):
        self.configs = list(configs)
        self.stake = stake
        self.max_price = max_price
        self.require_agree = require_agree
        self.signal_source = signal_source
        self.writer = writer
        self.runs: dict[str, WindowRun] = {}
        self.lock = threading.Lock()
        self.totals = {c.name: {"n": 0, "wins": 0, "pnl": 0.0, "cost": 0.0,
                                "nofill": 0, "skip": 0} for c in self.configs}

    # ------------------------------------------------------------------ #
    def start_window(self, market: Market, ptb, ptb_src):
        with self.lock:
            if market.slug not in self.runs:
                self.runs[market.slug] = WindowRun(
                    market, ptb, ptb_src, {c.name: Attempt(c) for c in self.configs})
            return self.runs[market.slug]

    def on_tick(self, now: float, market: Market, signal_price: float | None,
                get_books) -> list[Attempt]:
        """get_books() -> (up_book, down_book); вызывается лениво, максимум раз за тик,
        уже ПОСЛЕ того как сигнал сработал (как у реального бота)."""
        run = self.runs.get(market.slug)
        if run is None or run.ptb is None or signal_price is None:
            return []
        stc = market.end - now
        if stc <= 0:
            return []
        delta = signal_price - run.ptb
        side = "UP" if delta >= 0 else "DOWN"   # ничья решается в пользу Up
        books = None
        changed = []
        for a in run.attempts.values():
            if a.done or stc > a.cfg.offset or abs(delta) < a.cfg.threshold:
                continue
            if books is None:
                books = get_books()
            up_b, down_b = books
            ours, other = (up_b, down_b) if side == "UP" else (down_b, up_b)
            a.attempts += 1
            a.side, a.signal_price, a.delta, a.stc, a.decided_at = side, signal_price, delta, stc, now
            a.our_ask = ours.best_ask if ours else None
            a.our_bid = ours.best_bid if ours else None
            a.other_ask = other.best_ask if other else None

            ref_px = a.our_bid if a.our_bid is not None else a.our_ask
            if self.require_agree and ref_px is not None and ref_px < 0.5:
                a.status = "skip_disagree"
                a.fill = Fill("skip_disagree", best_ask=a.our_ask)
            else:
                f = simulate_buy(ours, self.stake, self.max_price, market.fee_rate)
                a.fill = f
                if f.status in ("filled", "partial"):
                    a.status = f.status
                elif (f.status == "no_liquidity" and f.best_ask is not None
                      and self.max_price < f.best_ask < 1.0):
                    a.status = "price_cap"
                else:
                    a.status = f.status
            changed.append(a)
        return changed

    # ------------------------------------------------------------------ #
    def settle(self, market: Market, outcome: str | None, source: str | None) -> list[dict]:
        with self.lock:
            run = self.runs.pop(market.slug, None)
        if run is None:
            return []
        rows = []
        for a in run.attempts.values():
            if a.status == "pending":
                continue  # сигнала не было вовсе
            won = None
            pnl = None
            if outcome in ("UP", "DOWN") and a.side:
                won = (a.side == outcome)
            if a.status in ("filled", "partial"):
                if won is not None:
                    pnl = settle_pnl(a.fill, won)
                elif outcome == "SPLIT":
                    pnl = a.fill.shares * 0.5 - a.fill.cost - a.fill.fee
            t = self.totals[a.cfg.name]
            if pnl is not None:
                t["n"] += 1
                t["wins"] += int(bool(won))
                t["pnl"] += pnl
                t["cost"] += a.fill.cost + a.fill.fee
            elif a.status == "skip_disagree":
                t["skip"] += 1
            elif a.status not in ("filled", "partial"):
                t["nofill"] += 1
            f = a.fill
            row = {
                "config": a.cfg.name, "threshold": a.cfg.threshold, "offset": a.cfg.offset,
                "slug": market.slug, "window_start": market.start, "status": a.status,
                "attempts": a.attempts, "side": a.side, "signal_source": self.signal_source,
                "signal_price": a.signal_price, "price_to_beat": run.ptb,
                "ptb_source": run.ptb_source,
                "delta": round(a.delta, 4) if a.delta is not None else None,
                "stc_at_decision": round(a.stc, 2) if a.stc is not None else None,
                "decided_at_utc": datetime.fromtimestamp(a.decided_at, timezone.utc).isoformat()
                if a.decided_at else None,
                "our_best_ask": a.our_ask, "our_best_bid": a.our_bid, "other_best_ask": a.other_ask,
                "fill_shares": round(f.shares, 4) if f.shares else None,
                "fill_avg_price": round(f.avg_price, 5) if f.avg_price else None,
                "fill_worst_price": f.worst_price,
                "fill_cost": round(f.cost, 4) if f.cost else None,
                "fee": round(f.fee, 5) if f.fee else None,
                "outcome": outcome, "outcome_source": source, "won": won,
                "pnl": round(pnl, 4) if pnl is not None else None,
            }
            rows.append(row)
            if self.writer:
                self.writer(row)
        return rows

    def leaderboard(self, top: int = 8) -> str:
        items = sorted(self.totals.items(), key=lambda kv: -kv[1]["pnl"])
        lines = [f"{'конфиг':<14}{'сделок':>7}{'win%':>7}{'PnL $':>10}{'ROI %':>8}"
                 f"{'не исп.':>8}{'пропуск':>8}"]
        for name, t in items[:top]:
            wr = t["wins"] / t["n"] * 100 if t["n"] else 0
            roi = t["pnl"] / t["cost"] * 100 if t["cost"] else 0
            lines.append(f"{name:<14}{t['n']:>7}{wr:>7.1f}{t['pnl']:>10.2f}{roi:>8.2f}"
                         f"{t['nofill']:>8}{t['skip']:>8}")
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
def run(args):
    configs = [Config(float(t), float(o))
               for t in args.thresholds.split(",") for o in args.offsets.split(",")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists() or out.stat().st_size == 0
    fh = open(out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if new:
        w.writeheader()
    wlock = threading.Lock()

    def write_row(row):
        with wlock:
            w.writerow(row)
            fh.flush()

    engine = Engine(configs, args.stake, args.max_price, not args.allow_disagree,
                    args.signal, write_row)
    feed = PriceFeed()
    feed.start()
    cache = MarketCache()

    def on_resolved(m, outcome, source):
        rows = engine.settle(m, outcome, source)
        traded = [r for r in rows if r["status"] in ("filled", "partial")]
        print(f"\n=== {m.slug}: исход {outcome} ({source}); попыток {len(rows)}, "
              f"исполнено {len(traded)} ===", flush=True)
        print(engine.leaderboard(), flush=True)

    resolver = Resolver(on_resolved)
    resolver.start()

    log(f"Paper v2: {len(configs)} конфигов, ставка ${args.stake}, max_price {args.max_price}, "
        f"сигнал: {args.signal}, лог: {out}")
    log("Первое окно после запуска, скорее всего, будет пропущено — нужен price to beat.")
    time.sleep(3)

    started = set()
    next_tick = time.time()
    try:
        while True:
            now = time.time()
            ws = window_start(now)
            slug = slug_for(ws)
            m = cache.get(slug)
            if m is not None:
                ptb, ptb_src = price_to_beat(m, feed)
                if slug not in started:
                    started.add(slug)
                    resolver.add(m)
                    cache.prune(ws - 2 * WINDOW)
                run_ = engine.start_window(m, ptb, ptb_src)
                if run_.ptb is None and ptb is not None:
                    run_.ptb, run_.ptb_source = ptb, ptb_src

                kind = m.ref_kind if args.signal == "ref" else args.signal
                sig, _ = feed.latest(kind)
                changed = engine.on_tick(now, m, sig, lambda: fetch_books(m))
                for a in changed:
                    print(f"[{datetime.now():%H:%M:%S}] {a.cfg.name:<12} {a.side:<4} Δ={a.delta:+.2f} "
                          f"stc={a.stc:.1f}с ask={a.our_ask} -> {a.status}"
                          + (f" {a.fill.shares:.2f}sh @ {a.fill.avg_price:.4f}" if a.fill.shares else ""),
                          flush=True)
                if int(now) % 15 == 0:
                    d = None if (sig is None or run_.ptb is None) else sig - run_.ptb
                    print(f"[{datetime.now():%H:%M:%S}] {slug} до закрытия {m.end - now:5.1f}с | "
                          f"{kind}={sig} ptb={run_.ptb} ({run_.ptb_source}) "
                          f"Δ={None if d is None else round(d, 2)} | ждут резолва: {resolver.pending_count()}",
                          flush=True)
            next_tick += args.interval
            time.sleep(max(0.0, next_tick - time.time()))
            if time.time() - next_tick > args.interval * 5:
                next_tick = time.time()
    except KeyboardInterrupt:
        print("\nОстановлено. Итог по уже зарезолвленным окнам:")
        print(engine.leaderboard(top=100))
    finally:
        fh.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Paper trader v2 (сетка параметров, реальные аски, официальный исход)")
    ap.add_argument("--thresholds", default="5,10,15,20,30,50", help="пороги |Δ| в $ через запятую")
    ap.add_argument("--offsets", default="10,15,30,60,120", help="за сколько секунд до конца можно входить")
    ap.add_argument("--stake", type=float, default=10.0, help="$ на сделку (без комиссии)")
    ap.add_argument("--max-price", type=float, default=0.99, help="не покупать дороже этой цены")
    ap.add_argument("--signal", default="ref", choices=["ref", "spot", "twap30", "twap60", "binance"],
                    help="ref = тот поток, по которому резолвится рынок (рекомендуется)")
    ap.add_argument("--allow-disagree", action="store_true",
                    help="торговать даже если рынок оценивает нашу сторону < 0.5 (не рекомендуется)")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out", default="data/paper_v2.csv")
    run(ap.parse_args())
