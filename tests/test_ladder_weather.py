"""
Офлайн-тесты лесенок и погоды. Запуск:  py tests/test_ladder_weather.py
"""
import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from core.events import parse_event, parse_strike, parse_bucket, temp_unit, Event  # noqa: E402
from core.external import parse_deribit_name, station_from_source, _daily_long  # noqa: E402
from core.multi import RotatingGzCsv  # noqa: E402


def _ev(title, items, src="", desc=""):
    return {"slug": "s", "title": title, "endDate": "2026-10-20T16:00:00Z", "resolutionSource": src,
            "description": desc, "markets": [
                {"slug": f"m{i}", "conditionId": f"0x{i}", "groupItemTitle": t, "question": t,
                 "outcomes": json.dumps(["Yes", "No"]), "clobTokenIds": json.dumps([f"Y{i}", f"N{i}"])}
                for i, t in enumerate(items)]}


def test_parse_event_and_strikes():
    ev = parse_event(_ev("Bitcoin above ___ on October 20?", ["80,000", "82,000", "84,000"]))
    assert len(ev.markets) == 3 and ev.markets[0].yes_token == "Y0" and ev.end
    assert parse_strike("84,000", ev) == (84000.0, math.inf)
    assert parse_strike("82,000-84,000") == (82000.0, 84000.0)
    assert parse_strike("<80,000") == (-math.inf, 80000.0)
    assert parse_strike(">90,000") == (90000.0, math.inf)
    rng = Event("r", "Bitcoin price on October 20?", None, None, "")
    assert parse_strike("84,000", rng) == (84000.0, 84000.0)


def test_parse_buckets_and_units():
    assert parse_bucket("15°C") == (15.0, 15.0)
    assert parse_bucket("14°C or below") == (-math.inf, 14.0)
    assert parse_bucket("24°C or higher") == (24.0, math.inf)
    assert parse_bucket("62-63°F") == (62.0, 63.0)
    assert parse_bucket("62 to 63") == (62.0, 63.0)
    assert parse_bucket("61 or below") == (-math.inf, 61.0)
    assert parse_bucket("-3°C") == (-3.0, -3.0)
    ev = parse_event(_ev("Highest temperature in NYC on October 20?", ["61°F or below", "62-63°F"]))
    assert temp_unit(ev) == "F"
    ev = parse_event(_ev("Highest temperature in Amsterdam on October 20?", ["14°C or below", "15°C"]))
    assert temp_unit(ev) == "C"


def test_external_parsers():
    exp, k, t = parse_deribit_name("BTC-13OCT26-84000-C")
    assert k == 84000 and t == "C" and exp == 1791878400   # 2026-10-13 08:00 UTC
    assert parse_deribit_name("BTC-PERPETUAL") is None
    assert station_from_source("https://www.wunderground.com/history/daily/nl/schiphol/EHAM") == "EHAM"
    assert station_from_source("no link") is None
    rows = _daily_long({"utc_offset_seconds": 7200, "daily": {"time": ["2026-10-20"],
                        "temperature_2m_max_member01": [17.2]}}, "ens")
    assert rows == [{"kind": "ens", "date": "2026-10-20", "key": "temperature_2m_max_member01",
                     "value": 17.2, "utc_offset": 7200}]


def test_arbitrage_detection():
    import pandas as pd
    from multiout import bucket_sum_arb
    B = pd.DataFrame({"event_slug": ["e"] * 3, "ts": [1] * 3, "yes_ask": [0.30, 0.30, 0.30],
                      "yes_ask_size": [50, 20, 40], "no_ask": [0.75, 0.75, 0.75], "no_ask_size": [9, 9, 9]})
    A = bucket_sum_arb(B)
    yes = A[A.type == "все Yes"].iloc[0]
    assert yes.edge > 0.05 and yes.max_sets == 20           # 1 - 0.9 - комиссии
    assert (A.type == "все No").sum() == 0                   # 2.25 > n-1 = 2 -> нет арбитража


def _write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _run(script, d):
    r = subprocess.run([sys.executable, str(ROOT / "analysis" / script), "--dir", str(d)],
                       capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def test_ladder_report_end_to_end():
    from logger.ladder_logger import BOOK_FIELDS, DERIBIT_FIELDS, OUTCOME_FIELDS
    random.seed(5)
    d = Path(tempfile.mkdtemp())
    bw = RotatingGzCsv(d, "books", BOOK_FIELDS, flush_every_s=0)
    dw = RotatingGzCsv(d, "deribit", DERIBIT_FIELDS, flush_every_s=0)
    outs = []
    t0 = 1791000000
    for e in range(6):
        end = t0 + 86400 * (e + 1)
        spot0 = 84000 + random.gauss(0, 1500)
        final = spot0 * math.exp(random.gauss(0, 0.02))
        strikes = [80000 + 1000 * i for i in range(9)]
        for ts in range(end - 30 * 3600, end, 900):
            spot = spot0 + (final - spot0) * (1 - (end - ts) / (30 * 3600))
            tte = end - ts
            for k in strikes:
                sd = 0.5 * math.sqrt(tte / 31557600)
                p = 0.5 * (1 + math.erf(math.log(spot / k) / sd / math.sqrt(2)))
                p = min(max(p, 0.01), 0.99)
                ya = round(min(p + 0.01, 0.999), 3)
                na = round(min(1 - p + 0.01, 0.999), 3)
                if k == 83000 and ts == end - 20 * 3600:   # подложим нарушение монотонности
                    ya = 0.05
                bw.write({"ts": ts, "event_slug": f"bitcoin-above-{e}", "kind": "above", "coin": "bitcoin",
                          "end": end, "tte_s": tte, "market_slug": f"m{e}-{k}", "title": f"{k:,}", "lo": k,
                          "hi": math.inf, "yes_bid": round(p - 0.01, 3), "yes_ask": ya, "yes_ask_size": 100,
                          "no_bid": round(1 - p - 0.01, 3), "no_ask": na, "no_ask_size": 100, "spot": spot}, ts=ts)
                if ts % 3600 == 0:
                    for typ in "CP":
                        dw.write({"ts": ts, "instrument": f"BTC-X-{k}-{typ}", "currency": "BTC", "expiry": end + 3600,
                                  "strike": k, "type": typ, "mark_iv": 50.0, "underlying": spot}, ts=ts)
        for k in strikes:
            outs.append({"event_slug": f"bitcoin-above-{e}", "kind": "above", "coin": "bitcoin", "end": end,
                         "market_slug": f"m{e}-{k}", "title": f"{k:,}", "lo": k, "hi": math.inf,
                         "winner": "Yes" if final > k else "No", "source": "data_api",
                         "binance_close_at_end": final})
    bw.close()
    dw.close()
    _write(d / "outcomes.csv", OUTCOME_FIELDS, outs)
    out = _run("ladder_report.py", d)
    for part in ("=== 1.", "=== 2.", "=== 4.", "=== 5.", "> K' совпало в 100.00%"):
        assert part in out, part
    assert "Снимков с плюсом после комиссий" in out      # нашёл подложенное нарушение


def test_weather_report_end_to_end():
    from logger.weather_logger import BOOK_FIELDS, FC_FIELDS, METAR_FIELDS, EVENT_FIELDS, OUTCOME_FIELDS
    random.seed(7)
    d = Path(tempfile.mkdtemp())
    bw = RotatingGzCsv(d, "books", BOOK_FIELDS, flush_every_s=0)
    fw = RotatingGzCsv(d, "forecasts", FC_FIELDS, flush_every_s=0)
    mw = RotatingGzCsv(d, "metar", METAR_FIELDS, flush_every_s=0)
    evs, outs = [], []
    buckets = [(-math.inf, 14)] + [(t, t) for t in range(15, 24)] + [(24, math.inf)]
    for day in range(1, 9):
        date = f"2026-10-{day:02d}"
        day0 = 1790812800 + (day - 1) * 86400 - 7200          # полночь по местному времени (UTC+2)
        end = day0 + 86400 + 7200
        actual = round(19 + random.gauss(0, 2))
        slug = f"highest-temperature-in-amsterdam-on-october-{day}-2026"
        evs.append({"slug": slug, "city": "Amsterdam", "station": "EHAM", "unit": "C", "date": date,
                    "title": f"Highest temperature in Amsterdam on October {day}?", "end": end,
                    "buckets": "ok"})
        for ts in range(day0 - 86400, day0 + 86400, 3600):
            fw.write({"ts": ts, "station": "EHAM", "city": "Amsterdam", "unit": "C", "kind": "ens",
                      "date": date, "key": "m", "value": 0, "utc_offset": 7200}, ts=ts)
            for mbr in range(10):
                fw.write({"ts": ts, "station": "EHAM", "city": "Amsterdam", "unit": "C", "kind": "ens",
                          "date": date, "key": f"temperature_2m_max_member{mbr:02d}",
                          "value": actual + random.gauss(0, 1.5), "utc_offset": 7200}, ts=ts)
            for i, (lo, hi) in enumerate(buckets):
                inb = lo <= actual <= hi
                p = 0.6 if inb else 0.04
                bw.write({"ts": ts, "event_slug": slug, "city": "Amsterdam", "station": "EHAM", "unit": "C",
                          "date": date, "end": end, "tte_s": end - ts, "market_slug": f"{slug}-{i}",
                          "title": "x", "lo": lo, "hi": hi, "yes_bid": p - 0.01, "yes_ask": p + 0.01,
                          "yes_ask_size": 30, "no_bid": 0.98 - p, "no_ask": 0.995 - p if not inb else 0.41,
                          "no_ask_size": 30}, ts=ts)
        for h in range(0, 24):
            t = day0 + h * 3600
            mw.write({"ts": t, "station": "EHAM", "obs_time": t,
                      "temp_c": actual - abs(h - 14) * 0.5}, ts=t)
        for i, (lo, hi) in enumerate(buckets):
            outs.append({"event_slug": slug, "city": "Amsterdam", "station": "EHAM", "unit": "C", "date": date,
                         "market_slug": f"{slug}-{i}", "title": "x", "lo": lo, "hi": hi,
                         "winner": "Yes" if lo <= actual <= hi else "No", "source": "data_api"})
    for w in (bw, fw, mw):
        w.close()
    _write(d / "events.csv", EVENT_FIELDS, evs)
    _write(d / "outcomes.csv", OUTCOME_FIELDS, outs)
    out = _run("weather_report.py", d)
    for part in ("=== 1.", "=== 2.", "=== 3.", "=== 4.", "=== 5.", "Brier"):
        assert part in out, part
    assert "Корзин, которые уже были мёртвыми" in out


def test_units_stations_and_old_events():
    import time as _t
    import core.events as ce
    # правила упоминают обе единицы, но корзины в °C -> C
    ev = parse_event(_ev("Highest temperature in Amsterdam on October 20?", ["14°C or below", "15°C"],
                         desc="Wunderground may show °F; resolves in degrees Celsius"))
    assert temp_unit(ev) == "C"
    # корзины без единиц: решают слова в правилах
    ev = parse_event(_ev("Highest temperature in Paris on October 20?", ["14 or below", "15"],
                         desc="highest temperature in degrees Celsius. Celsius."))
    assert temp_unit(ev) == "C"
    ev = parse_event(_ev("Highest temperature in NYC on October 20?", ["61 or below", "62-63"],
                         desc="in degrees Fahrenheit"))
    assert temp_unit(ev) == "F"
    assert station_from_source("https://www.wunderground.com/history/daily/gb/london/EGLC/date/2026-10-20") == "EGLC"
    assert station_from_source("see https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA.") == "KLGA"
    assert station_from_source("https://www.weather.gov/wrh/timeseries?site=KMIA") == "KMIA"
    assert station_from_source("https://www.weather.gov/wrh/timeseries?site=eham") == "EHAM"
    # прошлогоднее событие (slug без года) отбрасывается
    old = parse_event(_ev("Bitcoin above ___ on October 12?", ["80,000"]))
    old.end = int(_t.time()) - 365 * 86400
    orig = ce.fetch_event
    try:
        ce.fetch_event = lambda slug: old
        assert ce.EventCache().ensure("bitcoin-above-on-october-12") is None
    finally:
        ce.fetch_event = orig


def test_fetch_winner_sources():
    import core.events as ce
    sm = parse_event(_ev("Bitcoin above ___ on X?", ["84,000"])).markets[0]

    class R:
        def __init__(self, js, code=200): self._js, self.status_code, self.text = js, code, str(js)
        def json(self): return self._js
    orig = ce._session.get
    try:
        ce._session.get = lambda url, params=None, timeout=None: (
            R({"closed": True, "tokens": [{"outcome": "Yes", "winner": False, "price": 0},
                                          {"outcome": "No", "winner": True, "price": 1}]})
            if "/markets/" in url else R({}, 404))
        assert ce.fetch_winner(sm) == ("No", "clob")

        def gamma_only(url, params=None, timeout=None):
            if "clob" in url or "data-api" in url:
                return R({}, 404)
            if params and params.get("closed") == "true":
                return R([{"closed": True, "outcomePrices": '["1", "0"]'}])
            return R([])
        ce._session.get = gamma_only
        assert ce.fetch_winner(sm) == ("Yes", "gamma_cid")
        ce._session.get = lambda url, params=None, timeout=None: R([], 200) if "gamma" in url else R({}, 404)
        assert ce.fetch_winner(sm) == (None, None)
    finally:
        ce._session.get = orig


def test_weather_strategies_finds_planted_bias():
    """Подкладываем перекос: фаворит по 0.90 выигрывает в 97% случаев."""
    from logger.weather_logger import BOOK_FIELDS, OUTCOME_FIELDS
    random.seed(11)
    d = Path(tempfile.mkdtemp())
    bw = RotatingGzCsv(d, "books", BOOK_FIELDS, flush_every_s=0)
    outs = []
    t0 = 1791000000
    for e in range(160):
        end = t0 + e * 7200
        fav_wins = random.random() < 0.97
        slug = f"ev{e}"
        for i, (name, yes_ask) in enumerate([("fav", 0.90), ("dog", 0.08), ("x", 0.04)]):
            won = (i == 0 and fav_wins) or (i == 1 and not fav_wins)
            for h in (30, 20, 10, 4):
                ts = end - h * 3600
                bw.write({"ts": ts, "event_slug": slug, "city": f"c{e % 7}", "end": end, "tte_s": end - ts,
                          "market_slug": f"{slug}-{name}", "yes_ask": yes_ask, "yes_ask_size": 100,
                          "no_ask": round(1.01 - yes_ask, 3), "no_ask_size": 100}, ts=ts)
            outs.append({"event_slug": slug, "market_slug": f"{slug}-{name}", "winner": "Yes" if won else "No"})
    bw.close()
    _write(d / "outcomes.csv", OUTCOME_FIELDS, outs)
    r = subprocess.run([sys.executable, str(ROOT / "analysis" / "weather_strategies.py"), "--dir", str(d)],
                       capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stderr[-3000:]
    passed = [l for l in r.stdout.splitlines() if l.startswith("Прошли фильтры")]
    assert passed and "Yes 0.85–0.95" in passed[0], r.stdout[-2500:]


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
