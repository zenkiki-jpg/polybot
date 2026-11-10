"""
Общее ядро для логгера, paper-трейдера и анализа (v2).

Что здесь исправлено по сравнению с v1:
  * Стакан берётся из /book и лучшие цены считаются как max(bids) / min(asks).
    В v1 бралось bids[0]/asks[0] — а Polymarket отдаёт уровни от худшего к
    лучшему, отсюда "призрачные" 0.01/0.99. Потом v1 перешёл на /price, где
    перепутал стороны (side=BUY возвращает лучший БИД, а не аск).
  * Токены Up/Down сопоставляются по названию исхода, а не по порядку.
  * Референсная цена = тот поток Chainlink, который указан в resolutionSource
    рынка (сейчас это BTC/USD TWAP 60s), а не самодельное среднее спота.
  * Price to beat: сначала eventMetadata.priceToBeat из Gamma, иначе
    наблюдение RTDS с timestamp ровно на старте окна. Если нет ни того, ни
    другого — окно пропускается, а не подставляется "примерная" цена.
  * Исход берётся из официального резолва (Data API /v2/resolutions, запасной
    вариант — outcomePrices закрытого рынка в Gamma), а не из своей цены.
  * Комиссия тейкера: shares * rate * p * (1 - p), считается на каждый филл.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal

import requests

# Windows-консоль: не падать на Δ/кириллице при перенаправлении вывода, печатать сразу
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
RTDS_WS = "wss://ws-live-data.polymarket.com"
BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"

WINDOW = 300
DEFAULT_FEE_RATE = 0.07  # крипто-категория, см. docs.polymarket.com/trading/fees

_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (polybot research logger)",
    "Accept": "application/json",
})
_pool = ThreadPoolExecutor(max_workers=8)


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------- #
# Окна
# --------------------------------------------------------------------- #
def window_start(ts: float) -> int:
    t = int(ts)
    return t - (t % WINDOW)


def slug_for(start: int) -> str:
    return f"btc-updown-5m-{start}"


# --------------------------------------------------------------------- #
# Рынок (Gamma)
# --------------------------------------------------------------------- #
def _maybe_json(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return v
    return v


def reference_kind(resolution_source: str | None) -> str:
    """По ссылке resolutionSource определяем, какой поток Chainlink решает исход."""
    s = (resolution_source or "").lower()
    if "twap-60" in s or "twap60" in s or "twap-60s" in s:
        return "twap60"
    if "twap-30" in s or "twap30" in s:
        return "twap30"
    return "spot"


@dataclass
class Market:
    slug: str
    start: int
    end: int
    condition_id: str | None
    event_id: str | None
    outcomes: list            # ["Up", "Down"] в том порядке, что отдал Gamma
    token_ids: list           # в том же порядке, что outcomes
    up_token: str
    down_token: str
    resolution_source: str | None
    ref_kind: str
    price_to_beat: float | None
    fee_rate: float
    min_order_size: float | None
    fetched_at: float = field(default_factory=time.time)

    def token_for(self, side: str) -> str:
        return self.up_token if side == "UP" else self.down_token


def parse_market(slug: str, event: dict | None, mkt: dict) -> Market | None:
    outcomes = _maybe_json(mkt.get("outcomes")) or []
    tokens = _maybe_json(mkt.get("clobTokenIds")) or []
    if len(outcomes) != 2 or len(tokens) != 2:
        return None
    labels = [str(o).strip().lower() for o in outcomes]
    if "up" not in labels or "down" not in labels:
        return None
    up_token = tokens[labels.index("up")]
    down_token = tokens[labels.index("down")]

    start = int(slug.rsplit("-", 1)[-1])

    # priceToBeat может лежать в eventMetadata события или рынка, иногда строкой
    ptb = None
    for holder in (event or {}, mkt):
        meta = _maybe_json(holder.get("eventMetadata")) if holder else None
        if isinstance(meta, dict) and meta.get("priceToBeat") not in (None, "", 0, "0"):
            try:
                ptb = float(meta["priceToBeat"])
                break
            except (TypeError, ValueError):
                pass

    res_src = mkt.get("resolutionSource") or (event or {}).get("resolutionSource")

    fee_rate = DEFAULT_FEE_RATE
    if mkt.get("feesEnabled") is False:
        fee_rate = 0.0

    mos = mkt.get("orderMinSize")
    try:
        mos = float(mos) if mos is not None else None
    except (TypeError, ValueError):
        mos = None

    return Market(
        slug=slug, start=start, end=start + WINDOW,
        condition_id=mkt.get("conditionId"),
        event_id=str((event or {}).get("id")) if event and event.get("id") is not None else None,
        outcomes=list(outcomes), token_ids=list(tokens),
        up_token=up_token, down_token=down_token,
        resolution_source=res_src, ref_kind=reference_kind(res_src),
        price_to_beat=ptb, fee_rate=fee_rate, min_order_size=mos,
    )


def fetch_market(slug: str) -> Market | None:
    try:
        r = _session.get(f"{GAMMA}/events", params={"slug": slug}, timeout=6)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            ev = data[0]
            mkts = ev.get("markets") or []
            if mkts:
                return parse_market(slug, ev, mkts[0])
        # запасной путь: /markets
        r = _session.get(f"{GAMMA}/markets", params={"slug": slug}, timeout=6)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return parse_market(slug, None, data[0])
    except (requests.RequestException, ValueError) as e:
        log(f"[gamma] {slug}: {e}")
    return None


# --------------------------------------------------------------------- #
# Стакан (CLOB)
# --------------------------------------------------------------------- #
@dataclass
class Book:
    token_id: str
    bids: list      # [(price, size)] от ЛУЧШЕГО к худшему (по убыванию цены)
    asks: list      # [(price, size)] от ЛУЧШЕГО к худшему (по возрастанию цены)
    fetched_at: float
    min_order_size: float | None = None

    @property
    def best_bid(self):
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self):
        return self.asks[0][0] if self.asks else None

    @property
    def best_bid_size(self):
        return self.bids[0][1] if self.bids else None

    @property
    def best_ask_size(self):
        return self.asks[0][1] if self.asks else None

    def depth_str(self, side: str = "asks", levels: int = 5) -> str:
        lv = self.asks if side == "asks" else self.bids
        return "|".join(f"{p:g}:{s:g}" for p, s in lv[:levels])


def parse_book(token_id: str, raw: dict, fetched_at: float | None = None) -> Book:
    def lv(key):
        out = []
        for x in raw.get(key) or []:
            try:
                p, s = float(x["price"]), float(x["size"])
            except (KeyError, TypeError, ValueError):
                continue
            if s > 0:
                out.append((p, s))
        return out

    bids = sorted(lv("bids"), key=lambda t: -t[0])   # порядок в ответе не важен
    asks = sorted(lv("asks"), key=lambda t: t[0])
    mos = raw.get("min_order_size")
    try:
        mos = float(mos) if mos is not None else None
    except (TypeError, ValueError):
        mos = None
    return Book(token_id, bids, asks, fetched_at or time.time(), mos)


def fetch_book(token_id: str) -> Book | None:
    t0 = time.time()
    try:
        r = _session.get(f"{CLOB}/book", params={"token_id": token_id}, timeout=4)
        if r.status_code == 404:
            return None  # книга закрыта / рынок уже не торгуется
        r.raise_for_status()
        return parse_book(token_id, r.json(), t0)
    except (requests.RequestException, ValueError) as e:
        log(f"[clob] book {token_id[:10]}…: {e}")
        return None


def fetch_books(market: Market) -> tuple[Book | None, Book | None]:
    fu = _pool.submit(fetch_book, market.up_token)
    fd = _pool.submit(fetch_book, market.down_token)
    return fu.result(), fd.result()


# --------------------------------------------------------------------- #
# Комиссия и симуляция исполнения
# --------------------------------------------------------------------- #
def taker_fee(shares: float, price: float, rate: float) -> float:
    return shares * rate * price * (1.0 - price)


@dataclass
class Fill:
    status: str            # filled | partial | no_liquidity | too_small
    shares: float = 0.0
    cost: float = 0.0      # сколько $ ушло на сами шары (без комиссии)
    fee: float = 0.0
    avg_price: float | None = None
    worst_price: float | None = None
    best_ask: float | None = None


def simulate_buy(book: Book | None, usd: float, max_price: float, fee_rate: float,
                 min_shares: float | None = None) -> Fill:
    """Покупка на `usd` долларов по аскам, не дороже max_price.
    Идём по уровням стакана — никакой "эвристики проскальзывания"."""
    if book is None or not book.asks:
        return Fill("no_liquidity")
    best = book.best_ask
    shares = cost = fee = 0.0
    worst = None
    left = usd
    for p, s in book.asks:
        if p > max_price or p >= 1.0 or left <= 1e-9:
            break
        take = min(s, left / p)
        shares += take
        cost += take * p
        fee += taker_fee(take, p, fee_rate)
        left -= take * p
        worst = p
    if shares <= 0:
        return Fill("no_liquidity", best_ask=best)
    mos = min_shares if min_shares is not None else book.min_order_size
    if mos and shares < mos:
        return Fill("too_small", best_ask=best)
    status = "filled" if left <= usd * 0.01 else "partial"
    return Fill(status, shares, cost, fee, cost / shares, worst, best)


def settle_pnl(fill: Fill, won: bool) -> float:
    if fill.shares <= 0:
        return 0.0
    payout = fill.shares if won else 0.0
    return payout - fill.cost - fill.fee


# --------------------------------------------------------------------- #
# Официальный исход
# --------------------------------------------------------------------- #
def parse_resolution_payouts(market: Market, rows: list) -> str | None:
    for row in rows or []:
        if str(row.get("status", "")).lower() != "resolved":
            continue
        payouts = row.get("payouts")
        if not payouts or len(payouts) != 2:
            continue
        if payouts[0] == payouts[1]:
            return "SPLIT"
        winner_idx = 0 if payouts[0] > payouts[1] else 1
        label = str(market.outcomes[winner_idx]).strip().lower()
        return "UP" if label == "up" else "DOWN"
    return None


def parse_gamma_outcome(market: Market, mkt: dict) -> str | None:
    prices = _maybe_json(mkt.get("outcomePrices"))
    if not mkt.get("closed") or not prices or len(prices) != 2:
        return None
    try:
        p = [float(x) for x in prices]
    except (TypeError, ValueError):
        return None
    if sorted(p) != [0.0, 1.0]:
        return None
    label = str(market.outcomes[p.index(1.0)]).strip().lower()
    return "UP" if label == "up" else "DOWN"


def fetch_outcome(market: Market) -> tuple[str | None, str | None]:
    """Возвращает (UP|DOWN|SPLIT|None, источник)."""
    if market.condition_id:
        try:
            r = _session.get(f"{DATA_API}/v2/resolutions",
                             params={"condition": market.condition_id}, timeout=6)
            if r.status_code == 200:
                out = parse_resolution_payouts(market, r.json().get("data"))
                if out:
                    return out, "data_api"
        except (requests.RequestException, ValueError) as e:
            log(f"[data-api] {market.slug}: {e}")
    try:
        mk = gamma_market_by_slug(market.slug)
        if mk:
            out = parse_gamma_outcome(market, mk)
            if out:
                return out, "gamma"
    except (requests.RequestException, ValueError) as e:
        log(f"[gamma] outcome {market.slug}: {e}")
    return None, None


def gamma_market_by_slug(slug: str) -> dict | None:
    """Рынок из Gamma по slug. У часовых/дневных slug события и рынка могут
    отличаться, поэтому если /markets ничего не нашёл — берём рынок из /events."""
    r = _session.get(f"{GAMMA}/markets", params={"slug": slug}, timeout=6)
    if r.status_code == 200:
        data = r.json()
        if isinstance(data, list) and data:
            return data[0]
    r = _session.get(f"{GAMMA}/events", params={"slug": slug}, timeout=6)
    if r.status_code == 200:
        data = r.json()
        if isinstance(data, list) and data and data[0].get("markets"):
            return data[0]["markets"][0]
    return None


# --------------------------------------------------------------------- #
# Потоки цен: Chainlink через RTDS (spot, TWAP30, TWAP60) + Binance
# --------------------------------------------------------------------- #
RTDS_TOPICS = {
    "crypto_prices_chainlink": "spot",
    "crypto_prices_twap_thirty": "twap30",
    "crypto_prices_twap_sixty": "twap60",
}


def _parse_value(payload: dict) -> float | None:
    fav = payload.get("full_accuracy_value")
    if fav not in (None, ""):
        try:
            return float(Decimal(str(fav)) / Decimal(10) ** 18)
        except Exception:
            pass
    v = payload.get("value")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


class PriceFeed:
    """Держит историю наблюдений по каждому потоку: (obs_ts_ms, value, recv_time)."""

    STALE_S = 15.0

    def __init__(self, keep_seconds: int = 1200, use_binance: bool = True):
        self._lock = threading.Lock()
        self._hist = {k: deque() for k in ("spot", "twap30", "twap60", "binance")}
        self._keep_ms = keep_seconds * 1000
        self._use_binance = use_binance
        self._ws = {}

    # --- приём данных (вызывается и из тестов напрямую) ---
    def add(self, kind: str, obs_ts_ms: int, value: float, recv: float | None = None):
        if value is None or obs_ts_ms is None:
            return
        recv = recv if recv is not None else time.time()
        with self._lock:
            h = self._hist[kind]
            if h and obs_ts_ms <= h[-1][0]:
                if obs_ts_ms == h[-1][0]:
                    return  # дубликат
                # запоздавшее наблюдение: вставляем по порядку
                items = list(h)
                items.append((obs_ts_ms, value, recv))
                items.sort(key=lambda x: x[0])
                h.clear()
                h.extend(items)
            else:
                h.append((obs_ts_ms, value, recv))
            cutoff = h[-1][0] - self._keep_ms
            while h and h[0][0] < cutoff:
                h.popleft()

    def handle_rtds_message(self, message: str):
        if not message or not message.lstrip().startswith("{"):
            return
        try:
            data = json.loads(message)
        except ValueError:
            return
        kind = RTDS_TOPICS.get(data.get("topic"))
        if not kind:
            return
        payload = data.get("payload") or {}
        if str(payload.get("symbol", "btc/usd")).lower() != "btc/usd":
            return
        batch = payload.get("data")
        if isinstance(batch, list):  # первичный снапшот, если прислали
            for item in batch:
                v = _parse_value(item)
                ts = item.get("timestamp")
                if v is not None and ts is not None:
                    self.add(kind, int(ts), v)
            return
        v = _parse_value(payload)
        ts = payload.get("timestamp")
        if v is not None and ts is not None:
            self.add(kind, int(ts), v)

    def handle_binance_message(self, message: str):
        try:
            d = json.loads(message)
            b, a = float(d["b"]), float(d["a"])
        except (ValueError, KeyError, TypeError):
            return
        now = time.time()
        self.add("binance", int(now * 1000), (a + b) / 2, now)

    # --- чтение ---
    def latest(self, kind: str, max_age: float | None = None):
        """(value, obs_ts_ms) последнего наблюдения или (None, None), если устарело."""
        max_age = self.STALE_S if max_age is None else max_age
        with self._lock:
            h = self._hist[kind]
            if not h:
                return None, None
            ts, v, recv = h[-1]
        if time.time() - recv > max_age:
            return None, None
        return v, ts

    def at_exact(self, kind: str, ts_ms: int, tol_ms: int = 0):
        """Наблюдение с timestamp ровно ts_ms (± tol_ms). Нужен для price to beat."""
        with self._lock:
            best = None
            for t, v, _ in reversed(self._hist[kind]):
                if abs(t - ts_ms) <= tol_ms:
                    if best is None or abs(t - ts_ms) < abs(best[0] - ts_ms):
                        best = (t, v)
                if t < ts_ms - tol_ms - 5000:
                    break
        return best[1] if best else None

    def has_coverage(self, kind: str, ts_ms: int) -> bool:
        """Видели ли мы поток и до, и после ts_ms (т.е. точно не пропустили момент)."""
        with self._lock:
            h = self._hist[kind]
            if not h:
                return False
            return h[0][0] <= ts_ms <= h[-1][0]

    # --- сетевые потоки ---
    def start(self):
        import websocket  # websocket-client

        def rtds_loop():
            while True:
                try:
                    def on_open(ws):
                        sub = {"action": "subscribe", "subscriptions": [
                            {"topic": t, "type": "update", "filters": '{"symbol":"btc/usd"}'}
                            for t in RTDS_TOPICS
                        ]}
                        ws.send(json.dumps(sub))

                        def pinger():
                            while True:
                                time.sleep(5)
                                try:
                                    ws.send("PING")
                                except Exception:
                                    return
                        threading.Thread(target=pinger, daemon=True).start()

                    ws = websocket.WebSocketApp(
                        RTDS_WS, on_open=on_open,
                        on_message=lambda ws, m: self.handle_rtds_message(m))
                    self._ws["rtds"] = ws
                    ws.run_forever()
                except Exception as e:
                    log(f"[rtds] {e}")
                log("[rtds] переподключение через 3с")
                time.sleep(3)

        def binance_loop():
            while True:
                try:
                    ws = websocket.WebSocketApp(
                        BINANCE_WS, on_message=lambda ws, m: self.handle_binance_message(m))
                    self._ws["binance"] = ws
                    ws.run_forever(ping_interval=20, ping_timeout=10)
                except Exception as e:
                    log(f"[binance] {e}")
                time.sleep(3)

        def watchdog():
            # "тихо" подвисшее соединение: тиков нет, а сокет формально жив
            while True:
                time.sleep(10)
                with self._lock:
                    h = self._hist["twap60"] or self._hist["spot"]
                    last = h[-1][2] if h else None
                if last and time.time() - last > 30 and self._ws.get("rtds"):
                    log("[rtds] нет тиков 30с — рву соединение")
                    try:
                        self._ws["rtds"].close()
                    except Exception:
                        pass

        threading.Thread(target=rtds_loop, daemon=True).start()
        threading.Thread(target=watchdog, daemon=True).start()
        if self._use_binance:
            threading.Thread(target=binance_loop, daemon=True).start()


# --------------------------------------------------------------------- #
# Price to beat
# --------------------------------------------------------------------- #
def price_to_beat(market: Market, feed: PriceFeed) -> tuple[float | None, str | None]:
    if market.price_to_beat is not None:
        return market.price_to_beat, "gamma"
    v = feed.at_exact(market.ref_kind, market.start * 1000)
    if v is not None:
        return v, "rtds_exact"
    return None, None


# --------------------------------------------------------------------- #
# Кэш рынков + фоновый резолвер
# --------------------------------------------------------------------- #
class MarketCache:
    """Кэширует рынок по slug; пока нет priceToBeat — перезапрашивает раз в N секунд."""

    def __init__(self, refresh_s: float = 10.0):
        self._m: dict[str, Market] = {}
        self._last_try: dict[str, float] = {}
        self.refresh_s = refresh_s

    def get(self, slug: str) -> Market | None:
        m = self._m.get(slug)
        now = time.time()
        need = m is None or (m.price_to_beat is None and now - m.fetched_at > self.refresh_s)
        if need and now - self._last_try.get(slug, 0) > (2.0 if m is None else self.refresh_s):
            self._last_try[slug] = now
            fresh = fetch_market(slug)
            if fresh is not None:
                if m is not None and fresh.price_to_beat is None:
                    fresh.price_to_beat = m.price_to_beat
                self._m[slug] = fresh
                m = fresh
        return m

    def prune(self, keep_after: int):
        for s in [s for s, m in self._m.items() if m.end < keep_after]:
            self._m.pop(s, None)
            self._last_try.pop(s, None)


class Resolver:
    """Фоновый поток: для закрытых окон узнаёт официальный исход.
    Колбэк on_resolved(market, outcome, source) вызывается ровно один раз на окно."""

    def __init__(self, on_resolved, give_up_after_s: int = 1800, poll_s: float = 10.0):
        self._pending: dict[str, Market] = {}
        self._lock = threading.Lock()
        self._on = on_resolved
        self.give_up = give_up_after_s
        self.poll_s = poll_s

    def add(self, market: Market):
        with self._lock:
            self._pending.setdefault(market.slug, market)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def _loop(self):
        next_try: dict[str, float] = {}
        tries: dict[str, int] = {}
        while True:
            time.sleep(self.poll_s)
            now = time.time()
            with self._lock:
                items = list(self._pending.values())
            for m in items:
                if now < m.end + 5 or now < next_try.get(m.slug, 0):
                    continue
                out, src = fetch_outcome(m)
                give_up = getattr(m, "give_up_s", None) or self.give_up
                if out is None and now - m.end < give_up:
                    # чем дольше нет резолва, тем реже спрашиваем (10с … 10 мин)
                    k = tries.get(m.slug, 0) + 1
                    tries[m.slug] = k
                    next_try[m.slug] = now + min(self.poll_s * 2 ** min(k, 6), 600)
                    continue
                with self._lock:
                    self._pending.pop(m.slug, None)
                next_try.pop(m.slug, None)
                tries.pop(m.slug, None)
                try:
                    self._on(m, out, src or "timeout")
                except Exception as e:
                    log(f"[resolver] колбэк упал на {m.slug}: {e}")

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
