"""
Офлайн-тесты (без сети). Запуск:
    py -m pytest tests -q
или просто
    py tests/test_core.py
"""
import csv
import os
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.pm import (parse_book, parse_market, simulate_buy, settle_pnl, taker_fee,
                     parse_resolution_payouts, parse_gamma_outcome, PriceFeed,
                     reference_kind, price_to_beat)
from live.paper_trader import Engine, Config, CSV_FIELDS
from logger.polymarket_logger import TICK_FIELDS, WINDOW_FIELDS

START = 1790400000


def make_market(outcomes=("Up", "Down"), ptb=None, src="https://data.chain.link/streams/btc-usd-twap-60s-streams"):
    ev = {"id": 123}
    if ptb is not None:
        ev["eventMetadata"] = json.dumps({"priceToBeat": ptb})
    tokens = ["TOK_" + o.upper() for o in outcomes]
    mkt = {"outcomes": json.dumps(list(outcomes)), "clobTokenIds": json.dumps(tokens),
           "conditionId": "0xabc", "resolutionSource": src}
    return parse_market(f"btc-updown-5m-{START}", ev, mkt)


def book(token, bids, asks):
    # как отдаёт Polymarket: от худшего к лучшему
    return parse_book(token, {
        "bids": [{"price": str(p), "size": str(s)} for p, s in sorted(bids)],
        "asks": [{"price": str(p), "size": str(s)} for p, s in sorted(asks, reverse=True)],
    })


def test_book_best_prices_ignore_order():
    b = book("x", [(0.01, 100), (0.5, 10), (0.97, 5)], [(0.99, 100), (0.98, 3)])
    assert b.best_bid == 0.97 and b.best_ask == 0.98
    assert b.best_ask_size == 3


def test_market_maps_tokens_by_label_and_reads_meta():
    m = make_market(outcomes=("Down", "Up"), ptb="83791.9")
    assert m.up_token == "TOK_UP" and m.down_token == "TOK_DOWN"
    assert m.price_to_beat == 83791.9
    assert m.ref_kind == "twap60"
    assert reference_kind("https://data.chain.link/streams/btc-usd") == "spot"


def test_simulate_buy_walks_levels_with_fee():
    b = book("x", [], [(0.90, 5), (0.92, 100)])
    f = simulate_buy(b, usd=10, max_price=0.99, fee_rate=0.07)
    assert f.status == "filled"
    exp_shares = 5 + (10 - 4.5) / 0.92
    assert abs(f.shares - exp_shares) < 1e-9
    assert abs(f.cost - 10) < 1e-9
    exp_fee = taker_fee(5, 0.90, 0.07) + taker_fee(5.5 / 0.92, 0.92, 0.07)
    assert abs(f.fee - exp_fee) < 1e-12
    assert abs(settle_pnl(f, True) - (exp_shares - 10 - exp_fee)) < 1e-9
    assert abs(settle_pnl(f, False) - (-10 - exp_fee)) < 1e-9


def test_simulate_buy_respects_cap_and_empty():
    b = book("x", [(0.99, 50)], [(0.999, 5), (1.0, 100)])
    assert simulate_buy(b, 10, 0.99, 0.07).status == "no_liquidity"
    assert simulate_buy(None, 10, 0.99, 0.07).status == "no_liquidity"
    b2 = book("x", [], [(0.95, 2), (0.99, 100)])
    f = simulate_buy(b2, 10, 0.96, 0.07)
    assert f.status == "partial" and abs(f.shares - 2) < 1e-9


def test_resolution_parsing_uses_outcome_labels():
    m = make_market(outcomes=("Down", "Up"))
    assert parse_resolution_payouts(m, [{"status": "resolved", "payouts": [0, 1000000]}]) == "UP"
    assert parse_resolution_payouts(m, [{"status": "proposed", "payouts": [0, 1000000]}]) is None
    assert parse_gamma_outcome(m, {"closed": True, "outcomePrices": '["1", "0"]'}) == "DOWN"
    assert parse_gamma_outcome(m, {"closed": False, "outcomePrices": '["1", "0"]'}) is None


def test_feed_parses_twap_and_exact_lookup():
    f = PriceFeed(use_binance=False)
    f.handle_rtds_message(json.dumps({
        "topic": "crypto_prices_twap_sixty", "type": "update", "timestamp": START * 1000 + 120,
        "payload": {"symbol": "btc/usd", "value": 65000.5,
                    "full_accuracy_value": "65000512345000000000000", "timestamp": START * 1000,
                    "window_s": 60}}))
    f.handle_rtds_message(json.dumps({
        "topic": "crypto_prices_chainlink", "type": "update",
        "payload": {"symbol": "btc/usd", "data": [{"timestamp": START * 1000 - 1000, "value": 64990},
                                                   {"timestamp": START * 1000, "value": 64991}]}}))
    f.handle_rtds_message("PONG")
    assert abs(f.at_exact("twap60", START * 1000) - 65000.512345) < 1e-9
    assert f.at_exact("twap60", START * 1000 + 1000) is None
    assert f.at_exact("spot", START * 1000) == 64991
    m = make_market()
    assert price_to_beat(m, f) == (f.at_exact("twap60", START * 1000), "rtds_exact")


def _engine(rows, **kw):
    cfgs = [Config(10, 30), Config(10, 10), Config(50, 30)]
    return Engine(cfgs, stake=10, max_price=kw.get("max_price", 0.99),
                  require_agree=kw.get("require_agree", True), signal_source="ref",
                  writer=rows.append)


def test_engine_full_cycle_and_retry():
    rows = []
    e = _engine(rows)
    m = make_market(ptb=100000.0)
    e.start_window(m, m.price_to_beat, "gamma")
    empty = (book("u", [(0.97, 10)], [(1.0, 100)]), book("d", [(0.01, 100)], [(0.03, 100)]))
    good = (book("u", [(0.95, 10)], [(0.96, 100)]), book("d", [(0.03, 100)], [(0.05, 100)]))
    t_open = START
    # 60с до конца, Δ=+20 — рано для offset 30
    assert e.on_tick(t_open + 240, m, 100020, lambda: good) == []
    # 25с до конца: t10_o30 срабатывает, но асков < $1 нет -> retry
    ch = e.on_tick(t_open + 275, m, 100020, lambda: empty)
    assert [a.status for a in ch] == ["no_liquidity"]
    # 20с до конца: теперь аск есть
    ch = e.on_tick(t_open + 280, m, 100020, lambda: good)
    assert [(a.cfg.name, a.status) for a in ch] == [("t10_o30", "filled")]
    # 8с: t10_o10 срабатывает; t50 так и не сработает
    ch = e.on_tick(t_open + 292, m, 100020, lambda: good)
    assert [(a.cfg.name, a.status, a.attempts) for a in ch] == [("t10_o10", "filled", 1)]
    out = e.settle(m, "UP", "data_api")
    assert len(out) == 2 and len(rows) == 2
    r = {x["config"]: x for x in out}
    assert r["t10_o30"]["attempts"] == 2 and r["t10_o30"]["won"] is True
    shares = 10 / 0.96
    exp = shares - 10 - taker_fee(shares, 0.96, 0.07)
    assert abs(r["t10_o30"]["pnl"] - round(exp, 4)) < 1e-4
    assert set(out[0]) <= set(CSV_FIELDS)


def test_engine_skips_when_market_disagrees_and_without_ptb():
    rows = []
    e = _engine(rows)
    m = make_market()  # нет price to beat
    e.start_window(m, None, None)
    assert e.on_tick(START + 290, m, 100020, lambda: (None, None)) == []
    m2 = make_market(ptb=100000.0)
    m2.slug = "other"
    e.start_window(m2, 100000.0, "gamma")
    disagree = (book("u", [(0.02, 10)], [(0.03, 100)]), book("d", [(0.97, 100)], [(0.98, 100)]))
    ch = e.on_tick(START + 290, m2, 100020, lambda: disagree)
    assert {a.status for a in ch} == {"skip_disagree"}
    out = e.settle(m2, "DOWN", "data_api")
    assert all(r["pnl"] is None and r["won"] is False for r in out)


def test_reports_run_on_synthetic_data():
    """Генерим маленький лог в формате v2 и прогоняем оба отчёта целиком."""
    import random
    random.seed(1)
    tmp = Path(tempfile.mkdtemp())
    ticks, wins, paper = tmp / "ticks.csv", tmp / "windows.csv", tmp / "paper.csv"
    with open(ticks, "w", newline="") as ft, open(wins, "w", newline="") as fw:
        tw = csv.DictWriter(ft, TICK_FIELDS); tw.writeheader()
        ww = csv.DictWriter(fw, WINDOW_FIELDS); ww.writeheader()
        for k in range(60):
            s = START + 300 * k
            ptb = 100000 + random.gauss(0, 50)
            px = ptb
            for i in range(0, 300, 2):
                px += random.gauss(0, 4)
                up_mid = min(max(0.5 + (px - ptb) / 60, 0.01), 0.99)
                tw.writerow({"slug": f"s{k}", "window_start": s, "ts_unix": s + i,
                             "seconds_to_close": 300 - i, "ref_kind": "twap60", "ref_price": px,
                             "price_to_beat": ptb, "up_bid": round(up_mid - 0.01, 2),
                             "up_ask": round(up_mid + 0.01, 2), "down_bid": round(0.99 - up_mid, 2),
                             "down_ask": round(1.01 - up_mid, 2), "up_ask_size": 50, "down_ask_size": 50})
            ww.writerow({"slug": f"s{k}", "window_start": s, "ptb_gamma": ptb,
                         "ptb_rtds_twap60": ptb, "close_rtds_twap60": px,
                         "outcome": "UP" if px >= ptb else "DOWN", "outcome_source": "data_api"})
    r = subprocess.run([sys.executable, str(ROOT / "analysis/edge_report.py"), "--ticks", str(ticks),
                        "--windows", str(wins)], capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stderr
    assert "совпал с официальным: 100.00%" in r.stdout

    with open(paper, "w", newline="") as fp:
        w = csv.DictWriter(fp, CSV_FIELDS); w.writeheader()
        for k in range(30):
            won = k % 10 != 0
            w.writerow({"config": "t10_o30", "threshold": 10, "offset": 30, "slug": f"s{k}",
                        "window_start": START + 300 * k, "status": "filled", "side": "UP",
                        "fill_cost": 10, "fee": 0.03, "fill_avg_price": 0.9, "won": won,
                        "pnl": (10 / 0.9 - 10.03) if won else -10.03,
                        "outcome_source": "data_api", "ptb_source": "gamma"})
    r = subprocess.run([sys.executable, str(ROOT / "analysis/paper_report.py"), "--csv", str(paper)],
                       capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stderr
    assert "t10_o30" in r.stdout


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"OK   {name}")
            except Exception as ex:
                fails += 1
                print(f"FAIL {name}: {ex!r}")
    sys.exit(1 if fails else 0)
