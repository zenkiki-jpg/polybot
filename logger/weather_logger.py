"""
Логгер погодных рынков Polymarket: "Highest temperature in <город> on <дата>?"
+ прогнозы Open-Meteo (6 моделей и 3 ансамбля) + фактические наблюдения METAR.

Пишет в data/weather/:
  books_YYYYMMDD_HH.csv.gz    — снимки стакана каждой корзины (Yes/No), раз в 5 минут,
                                а в последние 12 часов до конца — раз в 2 минуты.
  forecasts_YYYYMMDD_HH.csv.gz — прогнозы максимума на дни событий, раз в час на станцию:
                                детерминированные модели (ECMWF, GFS, ICON, UKMO, Météo-France,
                                GEM) и члены ансамблей. Формат длинный: date, key, value.
  metar_YYYYMMDD_HH.csv.gz    — наблюдения станций раз в 10 минут (температура, макс. за 6ч).
  events.csv                  — события: город, станция, единицы, конец, правила.
  outcomes.csv                — победившая корзина каждого события.
  trades_YYYYMMDD_HH.csv.gz   — все сделки события после резолва.

Запуск:
    py logger/weather_logger.py
    py logger/weather_logger.py --cities amsterdam,london,paris,nyc --days 2
"""
import argparse
import csv
import queue
import re
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pm import log  # noqa: E402
from core.multi import MONTHS, RotatingGzCsv, fetch_books_batch, fetch_all_trades, normalize_api_trade  # noqa: E402
from core.events import EventCache, EventResolver, discover_by_tag, parse_bucket, temp_unit  # noqa: E402
from core.external import (STATIONS, CITY_STATION, station_from_source, geocode,  # noqa: E402
                           forecast_daily_max, metar)

DEFAULT_CITIES = ("amsterdam,london,paris,munich,madrid,warsaw,helsinki,moscow,toronto,nyc,chicago,"
                  "miami,los-angeles,san-francisco,dallas,houston,austin,denver,seattle,atlanta,"
                  "seoul,tokyo,hong-kong,taipei,singapore,beijing,shanghai,shenzhen,guangzhou,wuhan,"
                  "chongqing,chengdu,wellington,buenos-aires,sao-paulo,jeddah,ankara,berlin,rome")
BOOK_FIELDS = ["ts", "event_slug", "city", "station", "unit", "date", "end", "tte_s", "market_slug",
               "title", "lo", "hi", "yes_bid", "yes_bid_size", "yes_ask", "yes_ask_size", "no_bid",
               "no_bid_size", "no_ask", "no_ask_size", "yes_bids", "yes_asks"]
FC_FIELDS = ["ts", "station", "city", "unit", "lat", "lon", "kind", "date", "key", "value", "utc_offset"]
METAR_FIELDS = ["ts", "station", "obs_time", "temp_c", "max_c_6h", "raw"]
EVENT_FIELDS = ["first_seen", "slug", "city", "station", "lat", "lon", "unit", "date", "title", "end",
                "resolution_source", "n_markets", "buckets", "description"]
OUTCOME_FIELDS = ["event_slug", "city", "station", "unit", "date", "market_slug", "title", "lo", "hi",
                  "winner", "source", "logged_at"]
TRADE_FIELDS = ["event_slug", "market_slug", "title", "exchange_ts", "asset_id", "outcome", "side",
                "price", "size", "wallet", "tx"]


def candidates(cities, days, now):
    t = datetime.fromtimestamp(now, timezone.utc)
    out = []
    for c in cities:
        for k in range(-1, days + 1):
            d = (t + timedelta(days=k)).date()
            m, dd, y = MONTHS[d.month - 1], d.day, d.year
            out += [f"highest-temperature-in-{c}-on-{m}-{dd}-{y}", f"highest-temperature-in-{c}-on-{m}-{dd}"]
    return out


def event_date(slug: str):
    m = re.search(r"-on-([a-z]+)-(\d{1,2})(?:-(\d{4}))?$", slug)
    if not m or m.group(1) not in MONTHS:
        return None
    y = int(m.group(3)) if m.group(3) else datetime.now(timezone.utc).year
    return f"{y:04d}-{MONTHS.index(m.group(1)) + 1:02d}-{int(m.group(2)):02d}"


def city_of(title: str, slug: str) -> str:
    m = re.search(r"highest temperature in (.+?) on ", title or "", re.I)
    if m:
        return re.sub(r"\s*\(.*?\)", "", m.group(1)).strip()
    m = re.search(r"highest-temperature-in-(.+?)-on-", slug)
    return m.group(1).replace("-", " ") if m else ""


def run(a):
    cities = [c.strip() for c in a.cities.split(",") if c.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    books_w = RotatingGzCsv(out, "books", BOOK_FIELDS)
    fc_w = RotatingGzCsv(out, "forecasts", FC_FIELDS)
    metar_w = RotatingGzCsv(out, "metar", METAR_FIELDS)
    trades_w = RotatingGzCsv(out, "trades", TRADE_FIELDS)

    def plain(name, fields):
        p = out / name
        new = not p.exists() or p.stat().st_size == 0
        fh = open(p, "a", newline="", encoding="utf-8")
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
            fh.flush()
        return fh, w
    ev_fh, ev_w = plain("events.csv", EVENT_FIELDS)
    oc_fh, oc_w = plain("outcomes.csv", OUTCOME_FIELDS)
    lock = threading.Lock()

    cache = EventCache(refresh_s=3600, miss_retry_s=3600)
    info: dict[str, dict] = {}       # event slug -> city/station/coords/unit/date
    seen: set = set()
    last_sample: dict[str, float] = {}
    jobs: "queue.Queue" = queue.Queue()
    counters = {"rows": 0, "fc": 0, "metar": 0, "resolved": 0, "trades": 0}

    def on_done(ev, res):
        i = info.get(ev.slug, {})
        with lock:
            for sm in ev.markets:
                w, src = res.get(sm.slug, (None, "timeout"))
                oc_w.writerow({"event_slug": ev.slug, "city": i.get("city"), "station": i.get("station"),
                               "unit": i.get("unit"), "date": i.get("date"), "market_slug": sm.slug,
                               "title": sm.title, "lo": sm.lo, "hi": sm.hi, "winner": w, "source": src,
                               "logged_at": datetime.now(timezone.utc).isoformat()})
            oc_fh.flush()
        counters["resolved"] += 1
        if not a.no_trades:
            jobs.put(ev)

    resolver = EventResolver(on_done)
    resolver.start()

    def trades_worker():
        while True:
            ev = jobs.get()
            for sm in ev.markets:
                if sm.condition_id:
                    rows, _ = fetch_all_trades(sm.condition_id)
                    for r in rows:
                        trades_w.write({"event_slug": ev.slug, "market_slug": sm.slug, "title": sm.title,
                                        **normalize_api_trade(r)}, ts=ev.end or time.time())
                    counters["trades"] += len(rows)
    threading.Thread(target=trades_worker, daemon=True).start()

    def side_loop():
        """Прогнозы раз в час, METAR раз в 10 минут."""
        last_fc: dict[tuple, float] = {}
        time.sleep(60)  # даём основному циклу найти события
        while True:
            now = time.time()
            act = [info[e.slug] for e in cache.active(now) if e.slug in info]
            stations = sorted({i["station"] for i in act if i.get("station")})
            if stations:
                ts = time.time()
                for r in metar(stations, hours=2):
                    metar_w.write({"ts": round(ts), **r}, ts=ts)
                    counters["metar"] += 1
            done = set()
            for i in act:
                key = (i.get("station") or i["city"], i["unit"])
                if key in done or i.get("lat") is None or now - last_fc.get(key, 0) < a.forecast_every:
                    continue
                done.add(key)
                last_fc[key] = now
                ts = time.time()
                for r in forecast_daily_max(i["lat"], i["lon"], i["unit"]):
                    fc_w.write({"ts": round(ts), "station": i.get("station"), "city": i["city"], "unit": i["unit"],
                                "lat": i["lat"], "lon": i["lon"], **r}, ts=ts)
                    counters["fc"] += 1
            time.sleep(600)
    threading.Thread(target=side_loop, daemon=True).start()

    log(f"Погодный логгер: {len(cities)} городов (+ поиск по тегу), папка {out}")
    last_disc = last_status = 0.0
    shown_src = 0
    try:
        while True:
            now = time.time()
            if now - last_disc > 900:
                last_disc = now
                cands = candidates(cities, a.days, now)
                for tag in ("weather", "daily-temperature", "temperature"):
                    cands += discover_by_tag(tag, r"^Highest temperature in ")
                for slug in dict.fromkeys(cands):
                    ev = cache.ensure(slug, parse_bucket)
                    if ev is None or slug in seen:
                        continue
                    seen.add(slug)
                    st = station_from_source(ev.resolution_source, ev.description)
                    city = city_of(ev.title, slug)
                    if st is None:
                        st = CITY_STATION.get(city.lower())
                        if shown_src < 3:
                            shown_src += 1
                            log(f"    станция не найдена в правилах {slug}. Источник: {ev.resolution_source!r}; "
                                f"правила: {(ev.description or '')[:300]!r}")
                    unit = temp_unit(ev)
                    coords = STATIONS.get(st) if st else None
                    if coords is None and city:
                        coords = geocode(city)
                    info[slug] = {"city": city, "station": st, "unit": unit, "date": event_date(slug),
                                  "lat": coords[0] if coords else None, "lon": coords[1] if coords else None}
                    resolver.add(ev)
                    with lock:
                        ev_w.writerow({"first_seen": datetime.now(timezone.utc).isoformat(), "slug": slug,
                                       **info[slug], "title": ev.title, "end": ev.end,
                                       "resolution_source": ev.resolution_source, "n_markets": len(ev.markets),
                                       "buckets": "|".join(f"{m.title}={m.lo},{m.hi}" for m in ev.markets),
                                       "description": (ev.description or "")[:1500]})
                        ev_fh.flush()
                    bad = [m.title for m in ev.markets if m.lo is None]
                    log(f"[+] {slug}: {city}, станция {st or '?'}, °{unit}, корзин {len(ev.markets)}"
                        + (f" | НЕ разобраны: {bad}" if bad else ""))

            due = []
            for ev in cache.active(now):
                tte = (ev.end - now) if ev.end else 10 ** 9
                every = 120 if tte < 12 * 3600 else 300
                if now - last_sample.get(ev.slug, 0) >= every:
                    last_sample[ev.slug] = now
                    due.append(ev)
            if due:
                toks = [t for ev in due for sm in ev.markets for t in (sm.yes_token, sm.no_token)]
                books = fetch_books_batch(toks)
                for ev in due:
                    i = info.get(ev.slug, {})
                    for sm in ev.markets:
                        y, n = books.get(sm.yes_token), books.get(sm.no_token)
                        row = {"ts": round(now), "event_slug": ev.slug, "city": i.get("city"),
                               "station": i.get("station"), "unit": i.get("unit"), "date": i.get("date"),
                               "end": ev.end, "tte_s": round(ev.end - now) if ev.end else None,
                               "market_slug": sm.slug, "title": sm.title, "lo": sm.lo, "hi": sm.hi}
                        if y:
                            row.update(yes_bid=y.best_bid, yes_bid_size=y.best_bid_size, yes_ask=y.best_ask,
                                       yes_ask_size=y.best_ask_size, yes_bids=y.depth_str("bids", 3),
                                       yes_asks=y.depth_str("asks", 3))
                        if n:
                            row.update(no_bid=n.best_bid, no_bid_size=n.best_bid_size, no_ask=n.best_ask,
                                       no_ask_size=n.best_ask_size)
                        books_w.write(row, ts=now)
                        counters["rows"] += 1

            if now - last_status > 60:
                last_status = now
                act = cache.active(now)
                cities_now = sorted({info[e.slug]["city"] for e in act if e.slug in info})
                print(f"[{datetime.now():%H:%M:%S}] событий {len(act)} в {len(cities_now)} городах | "
                      f"строк стакана {counters['rows']} | прогнозов {counters['fc']} | METAR {counters['metar']} | "
                      f"ждут резолва {resolver.count()}, зарезолвлено {counters['resolved']} | "
                      f"сделок {counters['trades']}", flush=True)
                if not act:
                    print("           ни одного события не найдено — пришли этот вывод, проверю формат slug",
                          flush=True)
            time.sleep(max(0.5, 10 - (time.time() - now)))
    except KeyboardInterrupt:
        log("Остановлено.")
    finally:
        for w in (books_w, fc_w, metar_w, trades_w):
            w.close()
        ev_fh.close()
        oc_fh.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Логгер погодных рынков Polymarket")
    ap.add_argument("--cities", default=DEFAULT_CITIES, help="slug-и городов через запятую")
    ap.add_argument("--days", type=int, default=2, help="на сколько дней вперёд искать события")
    ap.add_argument("--forecast-every", type=int, default=3600, help="как часто брать прогноз, с")
    ap.add_argument("--no-trades", action="store_true")
    ap.add_argument("--out", default="data/weather")
    run(ap.parse_args())
