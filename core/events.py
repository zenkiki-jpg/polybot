"""
Общее для рынков "с несколькими исходами" (лесенки страйков, корзины температур).

На Polymarket такое событие (event) — это набор отдельных бинарных рынков Yes/No,
по одному на страйк/корзину. У каждого свои токены, свой стакан, свой резолв.
Подпись исхода лежит в поле groupItemTitle ("84,000", "15°C", "62-63°F" и т.п.).
"""
from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

from core.pm import _session, _maybe_json, GAMMA, DATA_API, CLOB, log


# --------------------------------------------------------------------- #
# Модель события
# --------------------------------------------------------------------- #
@dataclass
class SubMarket:
    event_slug: str
    slug: str
    condition_id: str | None
    title: str               # groupItemTitle: "84,000", "15°C", "61°F or below"
    question: str
    yes_token: str
    no_token: str
    outcomes: list           # ["Yes","No"] в порядке Gamma
    lo: float | None = None  # разобранный диапазон (страйк или корзина)
    hi: float | None = None


@dataclass
class Event:
    slug: str
    title: str
    end: int | None
    resolution_source: str | None
    description: str
    markets: list = field(default_factory=list)
    fetched_at: float = field(default_factory=time.time)


def _iso(s):
    if not s:
        return None
    try:
        return int(datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def parse_event(ev: dict) -> Event | None:
    mkts = ev.get("markets") or []
    if not mkts:
        return None
    out = Event(slug=ev.get("slug"), title=ev.get("title") or "",
                end=_iso(ev.get("endDate")) or _iso(mkts[0].get("endDate")),
                resolution_source=ev.get("resolutionSource") or mkts[0].get("resolutionSource"),
                description=ev.get("description") or mkts[0].get("description") or "")
    for m in mkts:
        outcomes = _maybe_json(m.get("outcomes")) or []
        tokens = _maybe_json(m.get("clobTokenIds")) or []
        if len(outcomes) != 2 or len(tokens) != 2:
            continue
        lab = [str(o).strip().lower() for o in outcomes]
        if "yes" not in lab or "no" not in lab:
            continue
        out.markets.append(SubMarket(
            event_slug=out.slug, slug=m.get("slug") or "", condition_id=m.get("conditionId"),
            title=str(m.get("groupItemTitle") or m.get("question") or ""),
            question=str(m.get("question") or ""),
            yes_token=tokens[lab.index("yes")], no_token=tokens[lab.index("no")],
            outcomes=list(outcomes)))
    return out if out.markets else None


def fetch_event(slug: str) -> Event | None:
    try:
        r = _session.get(f"{GAMMA}/events", params={"slug": slug}, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, list) and data:
            return parse_event(data[0])
    except Exception as e:
        log(f"[gamma] {slug}: {e}")
    return None


def discover_by_tag(tag_slug: str, title_regex: str, limit: int = 500) -> list[str]:
    """Slug-и открытых событий с тегом, у которых заголовок подходит под регэксп.
    Если Gamma не поддержит фильтр — вернётся пусто, и сработают кандидаты slug."""
    rx = re.compile(title_regex, re.I)
    out = []
    try:
        r = _session.get(f"{GAMMA}/events", params={"tag_slug": tag_slug, "closed": "false",
                                                    "active": "true", "limit": limit}, timeout=10)
        if r.status_code == 200 and isinstance(r.json(), list):
            out = [e.get("slug") for e in r.json() if rx.search(e.get("title") or "")]
    except Exception as e:
        log(f"[gamma] discover {tag_slug}: {e}")
    return [s for s in out if s]


class EventCache:
    """Кэш событий: находим один раз, потом только перечитываем раз в refresh_s
    (страйки иногда добавляют) и помним промахи, чтобы не долбить API."""

    def __init__(self, refresh_s=1800, miss_retry_s=1800, max_past_s=2 * 86400):
        self.events: dict[str, Event] = {}
        self.miss: dict[str, float] = {}
        self.refresh_s, self.miss_retry_s, self.max_past_s = refresh_s, miss_retry_s, max_past_s
        self._lock = threading.Lock()

    def ensure(self, slug: str, parse_fn=None) -> Event | None:
        now = time.time()
        with self._lock:
            ev = self.events.get(slug)
            if ev and now - ev.fetched_at < self.refresh_s:
                return ev
            if not ev and now - self.miss.get(slug, 0) < self.miss_retry_s:
                return None
        fresh = fetch_event(slug)
        # slug без года находит прошлогоднее событие с той же датой — такие отбрасываем
        if fresh is not None and fresh.end is not None and fresh.end < now - self.max_past_s:
            fresh = None
        with self._lock:
            if fresh is None:
                if ev:
                    ev.fetched_at = now
                    return ev
                self.miss[slug] = now
                return None
            if parse_fn:
                for m in fresh.markets:
                    m.lo, m.hi = parse_fn(m.title, fresh)
            self.events[slug] = fresh
            return fresh

    def active(self, now: float, grace_s: int = 0) -> list[Event]:
        with self._lock:
            return [e for e in self.events.values() if e.end is None or e.end + grace_s > now]

    def drop(self, slug: str):
        with self._lock:
            self.events.pop(slug, None)


# --------------------------------------------------------------------- #
# Разбор подписей
# --------------------------------------------------------------------- #
_num = r"-?\d[\d,]*(?:\.\d+)?"


def _f(s: str) -> float:
    return float(s.replace(",", ""))


def parse_strike(title: str, ev: Event | None = None) -> tuple[float | None, float | None]:
    """Лесенка "above": "84,000" -> (84000, inf). Диапазон: "82,000-84,000" -> (82000, 84000).
    Хвосты: "<80,000" / "80,000 or below" -> (-inf, 80000); ">90,000" -> (90000, inf)."""
    t = title.replace("$", "").replace("−", "-").strip()
    is_above = ev is not None and "above" in (ev.title or "").lower()
    m = re.match(rf"^({_num})\s*(?:-|–|to)\s*({_num})$", t)
    if m:
        return _f(m.group(1)), _f(m.group(2))
    m = re.match(rf"^(?:<|below|under)\s*({_num})$", t, re.I) or re.match(rf"^({_num})\s*(?:or below|or less|or lower)$", t, re.I)
    if m:
        return -math.inf, _f(m.group(1))
    m = re.match(rf"^(?:>|above|over)\s*({_num})$", t, re.I) or re.match(rf"^({_num})\s*(?:or above|or more|or higher|\+)$", t, re.I)
    if m:
        return _f(m.group(1)), math.inf
    m = re.match(rf"^[↑↓]?\s*({_num})$", t)
    if m:
        v = _f(m.group(1))
        return (v, math.inf) if is_above or ev is None else (v, v)
    return None, None


def temp_unit(ev: Event) -> str:
    """°C или °F. Сначала смотрим на подписи корзин (самое надёжное), потом на слова
    в правилах. В правилах могут встречаться обе единицы, поэтому считаем упоминания."""
    titles = " ".join(m.title for m in ev.markets)
    has_c, has_f = "°C" in titles or "ºC" in titles, "°F" in titles or "ºF" in titles
    if has_c != has_f:
        return "C" if has_c else "F"
    txt = f"{ev.title} {ev.description}".lower()
    c = txt.count("celsius") + txt.count("°c")
    f = txt.count("fahrenheit") + txt.count("°f")
    if c != f:
        return "C" if c > f else "F"
    return "C"


def parse_bucket(title: str, ev: Event | None = None) -> tuple[float | None, float | None]:
    """Корзины температуры в целых градусах (как их отдаёт станция):
    "15°C" -> (15,15); "14°C or below" -> (-inf,14); "24°C or higher" -> (24,inf);
    "62-63°F" / "62 to 63" -> (62,63)."""
    t = title.replace("°", "").replace("º", "").replace("−", "-")
    t = re.sub(r"\s*[CF]\b", "", t).strip()
    m = re.match(r"^(-?\d+)\s*(?:-|–|to)\s*(-?\d+)$", t, re.I)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"^(-?\d+)\s*(?:or below|or lower|or less|and below|or colder)$", t, re.I) or \
        re.match(r"^(?:<=?|below)\s*(-?\d+)$", t, re.I)
    if m:
        return -math.inf, float(m.group(1))
    m = re.match(r"^(-?\d+)\s*(?:or higher|or above|or more|and above|or warmer|\+)$", t, re.I) or \
        re.match(r"^(?:>=?|above)\s*(-?\d+)$", t, re.I)
    if m:
        return float(m.group(1)), math.inf
    m = re.match(r"^(-?\d+)$", t)
    if m:
        v = float(m.group(1))
        return v, v
    return None, None


# --------------------------------------------------------------------- #
# Резолв бинарного Yes/No рынка (общий, не только Up/Down)
# --------------------------------------------------------------------- #
def _label(sm: SubMarket, idx: int) -> str:
    return str(sm.outcomes[idx]).strip().capitalize()


def _winner_from_prices(sm: SubMarket, m: dict, src: str):
    """Закрытый рынок Gamma: outcomePrices ["1","0"] / ["0","1"] / ["0.5","0.5"]."""
    prices = _maybe_json(m.get("outcomePrices"))
    closed = m.get("closed") in (True, "true", "True")
    if not closed or not prices or len(prices) != 2:
        return None, None
    try:
        p = [float(x) for x in prices]
    except (TypeError, ValueError):
        return None, None
    if max(p) >= 0.99 and min(p) <= 0.01:
        return _label(sm, 0 if p[0] > p[1] else 1), src
    if abs(p[0] - 0.5) < 0.01 and abs(p[1] - 0.5) < 0.01:
        return "SPLIT", src
    return None, None


def fetch_winner(sm: SubMarket, debug: bool = False) -> tuple[str | None, str | None]:
    """'Yes' | 'No' | 'SPLIT' | None и источник. Пробует по очереди:
      1) CLOB /markets/<condition_id> — у токенов есть флаг winner;
      2) Data API /v2/resolutions — выплаты по исходам;
      3) Gamma /markets по condition_ids и по slug, с closed=true (по умолчанию Gamma
         может не отдавать закрытые рынки — из-за этого раньше ничего не находилось)."""
    def dbg(name, r):
        if debug:
            body = r.text[:600] if hasattr(r, "text") else str(r)[:600]
            log(f"  [debug] {name}: HTTP {getattr(r, 'status_code', '?')} {body}")

    cid = sm.condition_id
    if cid:
        try:
            r = _session.get(f"{CLOB}/markets/{cid}", timeout=8)
            dbg("clob /markets", r)
            if r.status_code == 200:
                js = r.json()
                toks = js.get("tokens") or []
                win = [t for t in toks if t.get("winner") in (True, "true")]
                if len(win) == 1:
                    return str(win[0].get("outcome", "")).strip().capitalize(), "clob"
                if js.get("closed") and len(toks) == 2:
                    pr = [float(t.get("price") or 0) for t in toks]
                    if abs(pr[0] - 0.5) < 0.01 and abs(pr[1] - 0.5) < 0.01:
                        return "SPLIT", "clob"
        except Exception as e:
            log(f"[clob] winner {sm.slug}: {e}")
        try:
            r = _session.get(f"{DATA_API}/v2/resolutions", params={"condition": cid}, timeout=8)
            dbg("data-api /v2/resolutions", r)
            if r.status_code == 200:
                for row in r.json().get("data") or []:
                    if str(row.get("status", "")).lower() != "resolved":
                        continue
                    p = [float(x) for x in (row.get("payouts") or [])]
                    if len(p) == 2:
                        if p[0] == p[1]:
                            return "SPLIT", "data_api"
                        return _label(sm, 0 if p[0] > p[1] else 1), "data_api"
        except Exception as e:
            log(f"[data-api] {sm.slug}: {e}")
    for params, name in (({"condition_ids": cid, "closed": "true"}, "gamma_cid"),
                         ({"slug": sm.slug, "closed": "true"}, "gamma_slug"),
                         ({"slug": sm.slug}, "gamma")):
        if not all(params.values()):
            continue
        try:
            r = _session.get(f"{GAMMA}/markets", params=params, timeout=8)
            dbg(f"gamma /markets {params}", r)
            if r.status_code == 200 and isinstance(r.json(), list) and r.json():
                w, src = _winner_from_prices(sm, r.json()[0], name)
                if w:
                    return w, src
        except Exception as e:
            log(f"[gamma] winner {sm.slug}: {e}")
    return None, None


class EventResolver:
    """После окончания события опрашивает исход каждого его рынка (с затуханием)
    и вызывает on_done(event, {market_slug: (winner, source)}), когда все известны
    или вышло время."""

    def __init__(self, on_done, give_up_s=48 * 3600):
        self.on_done, self.give_up_s = on_done, give_up_s
        self.pending: dict[str, Event] = {}
        self.results: dict[str, dict] = {}
        self.next_try: dict[str, float] = {}
        self.tries: dict[str, int] = {}
        self._lock = threading.Lock()

    def add(self, ev: Event):
        with self._lock:
            self.pending.setdefault(ev.slug, ev)

    def count(self):
        with self._lock:
            return len(self.pending)

    def _loop(self):
        while True:
            time.sleep(15)
            now = time.time()
            with self._lock:
                evs = list(self.pending.values())
            for ev in evs:
                if ev.end is None or now < ev.end + 60 or now < self.next_try.get(ev.slug, 0):
                    continue
                res = self.results.setdefault(ev.slug, {})
                for sm in ev.markets:
                    if sm.slug not in res:
                        w, src = fetch_winner(sm)
                        if w:
                            res[sm.slug] = (w, src)
                done = len(res) == len(ev.markets)
                if done or now - ev.end > self.give_up_s:
                    with self._lock:
                        self.pending.pop(ev.slug, None)
                    self.results.pop(ev.slug, None)
                    try:
                        self.on_done(ev, res)
                    except Exception as e:
                        log(f"[resolver] {ev.slug}: {e}")
                else:
                    k = self.tries.get(ev.slug, 0) + 1
                    self.tries[ev.slug] = k
                    self.next_try[ev.slug] = now + min(60 * 2 ** min(k, 5), 1800)

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
