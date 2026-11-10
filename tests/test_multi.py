"""
Офлайн-тесты мульти-логгера. Запуск:  py tests/test_multi.py
"""
import gzip
import json
import os
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.multi import (window_for, parse_updown, MultiFeed, parse_clob_ws_trades,
                        normalize_api_trade, RotatingGzCsv, read_gz_csv_tolerant,
                        load_gz_csvs, ref_kind_of)


def ts(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def test_short_intervals():
    now = ts("2026-09-29T16:07:12")
    w = window_for("eth", "5m", now)
    assert w.slugs == (f"eth-updown-5m-{int(ts('2026-09-29T16:05:00'))}",)
    w = window_for("sol", "15m", now)
    assert w.start == int(ts("2026-09-29T16:00:00")) and w.end - w.start == 900


def test_hourly_slug_et_and_dst():
    w = window_for("btc", "1h", ts("2026-09-29T16:30:00"))      # 12:30 EDT
    assert w.slugs[0] == "bitcoin-up-or-down-september-29-12pm-et"
    assert w.slugs[1] == "bitcoin-up-or-down-september-29-2026-12pm-et"
    assert w.start == int(ts("2026-09-29T16:00:00"))
    w = window_for("doge", "1h", ts("2026-11-02T17:30:00"))     # 12:30 EST (после перехода)
    assert w.slugs[0] == "dogecoin-up-or-down-november-2-12pm-et"
    w = window_for("eth", "1h", ts("2026-09-29T04:10:00"))      # 00:10 EDT
    assert w.slugs[0] == "ethereum-up-or-down-september-29-12am-et"


def test_daily_noon_to_noon():
    w = window_for("btc", "1d", ts("2026-09-29T15:00:00"))      # 11:00 EDT -> окно до сегодня 12:00
    assert w.slugs[0] == "bitcoin-up-or-down-on-september-29"
    assert w.end == int(ts("2026-09-29T16:00:00")) and w.end - w.start == 86400
    w = window_for("btc", "1d", ts("2026-09-29T16:30:00"))      # после полудня -> до завтра
    assert w.slugs[0] == "bitcoin-up-or-down-on-september-30"


def test_4h_candidates_cover_now():
    now = ts("2026-09-29T17:20:00")
    w = window_for("btc", "4h", now)
    assert all(int(s.rsplit("-", 1)[1]) <= now < int(s.rsplit("-", 1)[1]) + 14400 for s in w.slugs)
    assert len(w.slugs) == 4


def _mkt(outcomes=("Up", "Down"), end_iso=None, src=None, meta=None):
    m = {"outcomes": json.dumps(list(outcomes)), "clobTokenIds": json.dumps(["A", "B"]),
         "conditionId": "0x1", "resolutionSource": src}
    if end_iso:
        m["endDate"] = end_iso
    ev = {"eventMetadata": json.dumps(meta)} if meta else {}
    return ev, m


def test_parse_updown_validates_end_and_kind():
    w = window_for("btc", "1h", ts("2026-09-29T16:30:00"))
    ev, m = _mkt(("Down", "Up"), "2026-09-29T17:00:00Z", "https://www.binance.com/en/trade/BTC_USDT",
                 {"priceToBeat": 84000.5})
    mm = parse_updown(w, w.slugs[0], ev, m)
    assert mm.up_token == "B" and mm.ref_kind == "binance" and mm.price_to_beat == 84000.5
    ev, m = _mkt(end_iso="2026-09-29T19:00:00Z")          # не то окно
    assert parse_updown(w, w.slugs[0], ev, m) is None
    assert ref_kind_of("https://data.chain.link/streams/eth-usd-twap-60s-streams") == "twap60"


def test_feed_multi_symbol():
    f = MultiFeed(["btc", "eth"])
    f.handle_rtds(json.dumps({"topic": "crypto_prices_twap_sixty",
                              "payload": {"symbol": "eth/usd", "value": 3000.5, "timestamp": 1000}}))
    f.handle_rtds(json.dumps({"topic": "crypto_prices_chainlink",
                              "payload": {"symbol": "sol/usd", "value": 150, "timestamp": 1000}}))
    f.handle_binance(json.dumps({"stream": "btcusdt@bookTicker",
                                 "data": {"s": "BTCUSDT", "b": "84000", "a": "84002"}}))
    assert f.at_exact("eth", "twap60", 1000) == 3000.5
    assert f.at_exact("sol", "spot", 1000) is None           # монета не подписана
    assert f.latest("btc", "binance") == 84001


def test_ws_trades_and_api_normalize():
    msg = json.dumps([{"event_type": "book"},
                      {"event_type": "last_trade_price", "asset_id": "A", "price": "0.62",
                       "size": "15", "side": "BUY", "timestamp": "1790000000123"}])
    t = parse_clob_ws_trades(msg)
    assert t == [{"asset_id": "A", "price": "0.62", "size": "15", "side": "BUY",
                  "exchange_ts": "1790000000123"}]
    assert parse_clob_ws_trades("PONG") == []
    n = normalize_api_trade({"proxyWallet": "0xw", "side": "SELL", "asset": "A", "price": 0.3,
                             "size": 10, "timestamp": 1790000000, "outcome": "Down",
                             "transactionHash": "0xt"})
    assert n["wallet"] == "0xw" and n["outcome"] == "Down" and n["price"] == 0.3


def test_rotating_gz_and_truncated_read():
    d = Path(tempfile.mkdtemp())
    w = RotatingGzCsv(d, "x", ["a", "b"], flush_every_s=0)
    t0 = ts("2026-09-29T10:59:59")
    w.write({"a": 1, "b": 2}, ts=t0)
    w.write({"a": 3, "b": 4}, ts=t0 + 2)     # следующий час -> новый файл
    w.close()
    w2 = RotatingGzCsv(d, "x", ["a", "b"], flush_every_s=0)   # перезапуск в том же часу
    w2.write({"a": 5, "b": 6}, ts=t0 + 3)
    w2.close()
    df = load_gz_csvs(d, "x")
    assert sorted(df.a.tolist()) == [1, 3, 5]
    # оборванный файл: пишем, сбрасываем, обрезаем хвост
    p = d / "y_20260929_10.csv.gz"
    fh = gzip.open(p, "wt", newline="")
    fh.write("a,b\n" + "".join(f"{i},{i}\n" for i in range(1000)))
    fh.flush()
    raw = p.read_bytes()
    p.write_bytes(raw[:-3] if len(raw) > 3 else raw)
    txt = read_gz_csv_tolerant(p)
    assert txt.startswith("a,b\n") and txt.count("\n") >= 2


def test_multi_report_runs_on_synthetic_data():
    random.seed(3)
    d = Path(tempfile.mkdtemp())
    from logger.multi_logger import TICK_FIELDS, API_TRADE_FIELDS, WINDOW_FIELDS
    tw = RotatingGzCsv(d, "ticks", TICK_FIELDS, flush_every_s=0)
    aw = RotatingGzCsv(d, "trades_api", API_TRADE_FIELDS, flush_every_s=0)
    import csv
    with open(d / "windows.csv", "w", newline="") as fh:
        ww = csv.DictWriter(fh, WINDOW_FIELDS)
        ww.writeheader()
        s0 = int(ts("2026-09-29T10:00:00"))
        for k in range(40):
            s = s0 + 300 * k
            bp = 84000.0
            ep = 3000.0
            path = []
            for i in range(0, 300, 1):
                bp += random.gauss(0, 3)
                ep = ep * (1 + (random.gauss(0, 3) / 84000)) + random.gauss(0, 0.05)
                path.append((i, bp, ep))
            for coin, idx, p0 in (("btc", 1, 84000.0), ("eth", 2, 3000.0)):
                last = path[-1][idx]
                outcome = "UP" if last >= p0 else "DOWN"
                for i, b, e in path:
                    px = b if coin == "btc" else e
                    scale = 40 if coin == "btc" else 1.5
                    mid = min(max(0.5 + (px - p0) / scale, 0.02), 0.98)
                    tw.write({"ts_unix": s + i, "key": f"{coin}-5m", "coin": coin, "interval": "5m",
                              "slug": f"{coin}-{s}", "window_start": s, "window_end": s + 300,
                              "seconds_to_close": 300 - i, "binance_mid": px,
                              "up_bid": round(mid - .01, 2), "up_ask": round(mid + .01, 2),
                              "down_bid": round(.99 - mid, 2), "down_ask": round(1.01 - mid, 2)}, ts=s + i)
                    if i % 20 == 0:
                        aw.write({"key": f"{coin}-5m", "slug": f"{coin}-{s}", "window_start": s,
                                  "window_end": s + 300, "winner": outcome, "outcome": "Up",
                                  "side": "BUY", "price": round(mid, 2), "size": 100,
                                  "wallet": f"0x{random.randint(1, 5)}"}, ts=s + 300)
                ww.writerow({"key": f"{coin}-5m", "coin": coin, "interval": "5m", "slug": f"{coin}-{s}",
                             "window_start": s, "window_end": s + 300, "ref_kind": "twap60",
                             "resolution_source": "chainlink", "outcome": outcome,
                             "outcome_source": "data_api", "api_trades": 15})
    tw.close()
    aw.close()
    r = subprocess.run([sys.executable, str(ROOT / "analysis/multi_report.py"), "--dir", str(d)],
                       capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stderr[-3000:]
    for part in ("1. Покрытие", "2. Калибровка", "3. Сделки", "4. Опережение", "BTC-рынок -> ETH-рынок"):
        assert part in r.stdout, part


def test_kline_outcome_and_combined_outcomes():
    import core.multi as cm
    s0 = int(ts("2026-09-29T19:00:00"))

    class R:
        def __init__(self, js): self._js = js
        def raise_for_status(self): pass
        def json(self): return self._js
    orig = cm._session.get
    try:
        cm._session.get = lambda url, params=None, timeout=None: R(
            [[s0 * 1000, "100.0", "0", "0", "99.5", "0", (s0 + 3600) * 1000 - 1]])
        assert cm.binance_kline_outcome("btc", s0, s0 + 3600) == ("DOWN", 100.0, 99.5)
        cm._session.get = lambda url, params=None, timeout=None: R(
            [[(s0 + 60) * 1000, "1", "0", "0", "2", "0", 0]])
        assert cm.binance_kline_outcome("btc", s0, s0 + 3600)[0] is None   # не та свеча
        # 1d: Close минутной свечи 12:00 ET старта против Close свечи 12:00 ET конца
        d0 = s0 - 3 * 3600
        closes = {d0 * 1000: "100.0", (d0 + 86400) * 1000: "101.0"}
        cm._session.get = lambda url, params=None, timeout=None: R(
            [[params["startTime"], "0", "0", "0", closes[params["startTime"]], "0", params["startTime"] + 59999]])
        assert cm.binance_kline_outcome("btc", d0, d0 + 86400) == ("UP", 100.0, 101.0)
        closes[(d0 + 86400) * 1000] = "100.0"
        assert cm.binance_kline_outcome("btc", d0, d0 + 86400)[0] == "SPLIT"
    finally:
        cm._session.get = orig
    # windows.csv (старый формат с исходом) + outcomes.csv (официальный и по свече)
    import csv
    from logger.multi_logger import WINDOW_FIELDS, OUTCOME_FIELDS
    sys.path.insert(0, str(ROOT / "analysis"))
    from multi_report import load_windows
    d = Path(tempfile.mkdtemp())
    with open(d / "windows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, WINDOW_FIELDS); w.writeheader()
        w.writerow({"key": "btc-5m", "slug": "a", "outcome": "UP", "outcome_source": "data_api"})
        w.writerow({"key": "btc-1h", "slug": "b", "outcome_source": "pending", "ref_kind": "binance"})
        w.writerow({"key": "btc-1h", "slug": "c", "outcome_source": "pending", "ref_kind": "binance"})
    with open(d / "outcomes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, OUTCOME_FIELDS); w.writeheader()
        w.writerow({"slug": "b", "outcome_kline": "DOWN", "outcome_source": "kline_only"})
        w.writerow({"slug": "b", "outcome": "DOWN", "outcome_source": "gamma"})
        w.writerow({"slug": "c", "outcome_kline": "UP", "outcome_source": "kline_only"})
    W = load_windows(d).set_index("slug")
    assert W.loc["a", "outcome_final"] == "UP" and W.loc["a", "final_src"] == "data_api"
    assert W.loc["b", "outcome_final"] == "DOWN" and W.loc["b", "final_src"] == "gamma"
    assert W.loc["c", "outcome_final"] == "UP" and W.loc["c", "final_src"] == "kline"


def test_maker_fill_logic():
    import numpy as np
    sys.path.insert(0, str(ROOT / "analysis"))
    from maker_sim import simulate_fill, size_at, tick_for
    assert size_at("0.48:120|0.47:300", 0.47) == 300 and size_at("", 0.5) == 0
    assert tick_for(0.5) == 0.01 and tick_for(0.97) == 0.001
    ts = np.array([10, 11, 12, 20])
    px = np.array([0.48, 0.48, 0.48, 0.48])
    sz = np.array([10.0, 10.0, 10.0, 10.0])
    # очередь перед нами 25: до t=15 съели 20 (сделка в t=10 — в секунду постановки, не считается)
    assert simulate_fill(ts, px, sz, 10, 15, 0.48, 25, 20) == 0
    assert simulate_fill(ts, px, sz, 10, 25, 0.48, 25, 20) == 5      # 30 - 25
    assert simulate_fill(ts, px, sz, 9, 25, 0.48, 0, 20) == 20
    # сделка ниже нашей цены = уровень пробит, исполнены целиком
    assert simulate_fill(np.array([12]), np.array([0.47]), np.array([1.0]), 10, 20, 0.48, 1000, 20) == 20
    # сделка выше нашей цены (по аску) нас не касается
    assert simulate_fill(np.array([12]), np.array([0.49]), np.array([500.0]), 10, 20, 0.48, 0, 20) == 0


def test_gamma_market_by_slug_falls_back_to_events():
    import core.pm as pm

    class R:
        def __init__(self, js): self._js, self.status_code = js, 200
        def json(self): return self._js
    orig = pm._session.get
    try:
        pm._session.get = lambda url, params=None, timeout=None: R(
            [] if url.endswith("/markets") else [{"markets": [{"slug": "inner", "closed": True}]}])
        assert pm.gamma_market_by_slug("bitcoin-up-or-down-x")["slug"] == "inner"
    finally:
        pm._session.get = orig


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"OK   {name}")
            except Exception as ex:
                import traceback
                fails += 1
                print(f"FAIL {name}: {ex!r}")
                traceback.print_exc()
    sys.exit(1 if fails else 0)
