"""
Внешние источники данных для лесенок и погоды (все бесплатные, без ключей):
  * Deribit  — опционы BTC/ETH (по ним считаем "профессиональную" вероятность страйка)
  * Binance  — текущая цена (лесенки резолвятся по Binance BTC/USDT)
  * Open-Meteo — прогнозы погоды: несколько моделей + ансамбли, геокодинг городов
  * aviationweather.gov — METAR: фактические наблюдения на аэропортовых станциях
    (Polymarket резолвит погоду по Wunderground, а он показывает те же станции)
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone

from core.pm import _session, log

DERIBIT = "https://www.deribit.com/api/v2/public"
BINANCE = "https://api.binance.com/api/v3"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ENS = "https://ensemble-api.open-meteo.com/v1/ensemble"
OPEN_METEO_GEO = "https://geocoding-api.open-meteo.com/v1/search"
AVWX = "https://aviationweather.gov/api/data/metar"

MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
                                      "SEP", "OCT", "NOV", "DEC"], 1)}


# --------------------------------------------------------------------- #
# Deribit
# --------------------------------------------------------------------- #
def parse_deribit_name(name: str):
    """'BTC-13OCT26-84000-C' -> (expiry_unix (08:00 UTC), strike, 'C')."""
    m = re.match(r"^([A-Z]+)-(\d{1,2})([A-Z]{3})(\d{2})-(\d+(?:d\d+)?)-([CP])$", name)
    if not m:
        return None
    day, mon, yy = int(m.group(2)), MONTHS.get(m.group(3)), int(m.group(4))
    if not mon:
        return None
    exp = int(datetime(2000 + yy, mon, day, 8, tzinfo=timezone.utc).timestamp())
    return exp, float(m.group(5).replace("d", ".")), m.group(6)


def deribit_options(currency: str, max_days: float = 10.0) -> list[dict]:
    r = _session.get(f"{DERIBIT}/get_book_summary_by_currency",
                     params={"currency": currency, "kind": "option"}, timeout=10)
    r.raise_for_status()
    now = time.time()
    out = []
    for x in r.json().get("result") or []:
        p = parse_deribit_name(x.get("instrument_name", ""))
        if not p or p[0] - now > max_days * 86400 or p[0] < now:
            continue
        out.append({"instrument": x["instrument_name"], "currency": currency, "expiry": p[0],
                    "strike": p[1], "type": p[2], "mark_iv": x.get("mark_iv"),
                    "mark_price": x.get("mark_price"), "bid": x.get("bid_price"), "ask": x.get("ask_price"),
                    "underlying": x.get("underlying_price"), "index": x.get("estimated_delivery_price"),
                    "oi": x.get("open_interest")})
    return out


# --------------------------------------------------------------------- #
# Binance
# --------------------------------------------------------------------- #
def binance_mids(symbols: list[str]) -> dict:
    import json
    r = _session.get(f"{BINANCE}/ticker/bookTicker",
                     params={"symbols": json.dumps(symbols, separators=(",", ":"))}, timeout=6)
    r.raise_for_status()
    return {x["symbol"]: (float(x["bidPrice"]) + float(x["askPrice"])) / 2 for x in r.json()}


def binance_close_at(symbol: str, ts: int) -> float | None:
    """Close минутной свечи Binance, которая НАЧИНАЕТСЯ в ts (для проверки резолва)."""
    try:
        r = _session.get(f"{BINANCE}/klines", params={"symbol": symbol, "interval": "1m",
                                                      "startTime": ts * 1000, "limit": 1}, timeout=6)
        k = r.json()
        if k and int(k[0][0]) == ts * 1000 and int(k[0][6]) < time.time() * 1000:
            return float(k[0][4])
    except Exception as e:
        log(f"[binance] close {symbol} {ts}: {e}")
    return None


# --------------------------------------------------------------------- #
# Станции и геокодинг
# --------------------------------------------------------------------- #
# Координаты распространённых станций (ICAO), по которым Polymarket резолвит погоду.
# Если станции нет в таблице — координаты города берутся из геокодера Open-Meteo.
STATIONS = {
    "EHAM": (52.3086, 4.7639), "EGLC": (51.5053, 0.0553), "EGLL": (51.4700, -0.4543),
    "LFPG": (49.0097, 2.5479), "LFPO": (48.7233, 2.3794), "EDDM": (48.3538, 11.7861),
    "LEMD": (40.4719, -3.5626), "EPWA": (52.1657, 20.9671), "EFHK": (60.3172, 24.9633),
    "UUEE": (55.9726, 37.4146), "OEJN": (21.6796, 39.1565), "LTAC": (40.1281, 32.9951),
    "CYYZ": (43.6777, -79.6248), "KLGA": (40.7769, -73.8740), "KJFK": (40.6413, -73.7781),
    "KORD": (41.9742, -87.9073), "KMDW": (41.7868, -87.7522), "KMIA": (25.7959, -80.2870),
    "KLAX": (33.9416, -118.4085), "KSFO": (37.6213, -122.3790), "KDAL": (32.8471, -96.8518),
    "KDFW": (32.8998, -97.0403), "KHOU": (29.6454, -95.2789), "KIAH": (29.9902, -95.3368),
    "KAUS": (30.1975, -97.6664), "KDEN": (39.8561, -104.6737), "KSEA": (47.4502, -122.3088),
    "KATL": (33.6407, -84.4277), "KSJC": (37.3639, -121.9289), "KPHX": (33.4352, -112.0101),
    "KBOS": (42.3656, -71.0096), "RKSI": (37.4602, 126.4407), "RJTT": (35.5494, 139.7798),
    "VHHH": (22.3080, 113.9185), "RCTP": (25.0797, 121.2342), "WSSS": (1.3644, 103.9915),
    "ZBAA": (40.0799, 116.6031), "ZSPD": (31.1443, 121.8083), "ZGSZ": (22.6393, 113.8107),
    "ZGGG": (23.3924, 113.2988), "ZHHH": (30.7838, 114.2081), "ZUCK": (29.7192, 106.6417),
    "ZUUU": (30.5785, 103.9471), "NZWN": (-41.3272, 174.8053), "SAEZ": (-34.8222, -58.5358),
    "SBGR": (-23.4356, -46.4731), "LIRF": (41.8003, 12.2389), "LEBL": (41.2974, 2.0833),
    "EDDB": (52.3667, 13.5033), "LSZH": (47.4647, 8.5492), "LOWW": (48.1103, 16.5697),
}


# Станции, подтверждённые правилами рынков (используются, если ссылку разобрать не удалось)
CITY_STATION = {"amsterdam": "EHAM", "london": "EGLC", "nyc": "KLGA", "new york": "KLGA", "taipei": "RCSS"}


def station_from_source(*texts) -> str | None:
    """ICAO-код станции из ссылок в правилах: Wunderground (…/history/daily/nl/schiphol/EHAM,
    в том числе с /date/… после кода) или параметры вида ?site=KLGA / ?ids=EHAM."""
    pats = [r"wunderground\.com/[^\s\"'<>)]*?/([A-Z]{4})(?=[/?#\s\"'<>).,]|$)",
            r"[?&](?:site|station|stationId|stationid|ids|icao|id)=([A-Z]{4})\b"]
    for t in texts:
        if not t:
            continue
        for p in pats:
            m = re.search(p, str(t), re.I)
            if m:
                return m.group(1).upper()
    return None


_geo_cache: dict = {}


def geocode(name: str):
    if name in _geo_cache:
        return _geo_cache[name]
    res = None
    try:
        r = _session.get(OPEN_METEO_GEO, params={"name": name, "count": 1}, timeout=8)
        js = r.json().get("results") or []
        if js:
            res = (js[0]["latitude"], js[0]["longitude"])
    except Exception as e:
        log(f"[geo] {name}: {e}")
    _geo_cache[name] = res
    return res


# --------------------------------------------------------------------- #
# Прогнозы Open-Meteo
# --------------------------------------------------------------------- #
DET_MODELS = "ecmwf_ifs025,gfs_seamless,icon_seamless,ukmo_seamless,meteofrance_seamless,gem_seamless"
ENS_MODELS = "ecmwf_ifs025,gfs025,icon_seamless"


def _daily_long(js: dict, kind: str) -> list[dict]:
    """Ответ Open-Meteo -> длинные строки (date, key, value) по всем полям daily."""
    d = js.get("daily") or {}
    dates = d.get("time") or []
    off = js.get("utc_offset_seconds")
    rows = []
    for k, vals in d.items():
        if k == "time" or not isinstance(vals, list):
            continue
        for date, v in zip(dates, vals):
            if v is not None:
                rows.append({"kind": kind, "date": date, "key": k, "value": v, "utc_offset": off})
    return rows


_last_err: dict = {}


def _log_once(key: str, msg: str, every_s: int = 3600):
    if time.time() - _last_err.get(key, 0) > every_s:
        _last_err[key] = time.time()
        log(msg + "  (повторы этой ошибки скрыты на час)")


def forecast_daily_max(lat, lon, unit: str, days: int = 4) -> list[dict]:
    rows = []
    common = {"latitude": lat, "longitude": lon, "daily": "temperature_2m_max", "timezone": "auto",
              "forecast_days": days, "temperature_unit": "fahrenheit" if unit == "F" else "celsius"}
    try:
        r = _session.get(OPEN_METEO, params={**common, "models": DET_MODELS}, timeout=(5, 20))
        if r.status_code == 200:
            rows += _daily_long(r.json(), "det")
        else:
            _log_once("om_det", f"[open-meteo] HTTP {r.status_code}: {r.text[:150]}")
    except Exception as e:
        _log_once("om_det", f"[open-meteo] api.open-meteo.com недоступен: {type(e).__name__}. "
                            f"Ансамбли берутся с другого сервера и продолжают писаться.")
    try:
        r = _session.get(OPEN_METEO_ENS, params={**common, "models": ENS_MODELS}, timeout=(5, 30))
        if r.status_code == 200:
            rows += _daily_long(r.json(), "ens")
        else:
            _log_once("om_ens", f"[open-meteo ens] HTTP {r.status_code}: {r.text[:150]}")
    except Exception as e:
        _log_once("om_ens", f"[open-meteo ens] {type(e).__name__}: {e}")
    return rows


# --------------------------------------------------------------------- #
# METAR (факт)
# --------------------------------------------------------------------- #
def metar(stations: list[str], hours: int = 3) -> list[dict]:
    out = []
    for i in range(0, len(stations), 40):
        ids = ",".join(stations[i:i + 40])
        try:
            r = _session.get(AVWX, params={"ids": ids, "format": "json", "hours": hours}, timeout=15)
            if r.status_code != 200:
                continue
            for x in r.json() or []:
                t = x.get("obsTime")
                if isinstance(t, str):
                    try:
                        t = int(datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp())
                    except ValueError:
                        t = None
                out.append({"station": x.get("icaoId"), "obs_time": t, "temp_c": x.get("temp"),
                            "max_c_6h": x.get("maxT"), "raw": x.get("rawOb")})
        except Exception as e:
            log(f"[metar] {ids[:30]}…: {e}")
    return out
