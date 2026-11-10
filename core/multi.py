"""
Ядро расширенного логгера: все монеты × все интервалы Up/Down рынков Polymarket.

Форматы slug (проверено по открытым источникам, но Polymarket их иногда меняет,
поэтому для часовых/дневных пробуем варианты и проверяем по endDate):
  5m / 15m / 4h : {coin}-updown-{5m|15m|4h}-{start_unix}
  1h            : {name}-up-or-down-{month}-{day}-{h}{am|pm}-et         (время ET)
                  {name}-up-or-down-{month}-{day}-{year}-{h}{am|pm}-et
  1d            : {name}-up-or-down-on-{month}-{day}[-{year}]
                  окно = полдень ET вчера -> полдень ET в день из slug

Источник резолва у разных рынков разный (Chainlink spot / TWAP, свечи Binance),
поэтому логгер пишет resolutionSource каждого окна и ВСЕ потоки цен, а истиной
считается только официальный исход из API.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception as _e:  # на Windows нужен пакет tzdata
    raise SystemExit("Нужен пакет tzdata: py -m pip install tzdata") from _e

from core.pm import (_session, _pool, _maybe_json, _parse_value, reference_kind,
                     parse_book, fetch_book, DEFAULT_FEE_RATE, GAMMA, CLOB, DATA_API, log)

COINS = {
    #  код    имя в часовых/дневных slug   RTDS-символ   Binance
    "btc":  ("bitcoin",  "btc/usd",  "btcusdt"),
    "eth":  ("ethereum", "eth/usd",  "ethusdt"),
    "sol":  ("solana",   "sol/usd",  "solusdt"),
    "xrp":  ("xrp",      "xrp/usd",  "xrpusdt"),
    "doge": ("dogecoin", "doge/usd", "dogeusdt"),
}
INTERVALS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
# как часто писать тик, секунд (длинные окна — реже, иначе файлы раздуются)
DEFAULT_SAMPLE = {"5m": 1, "15m": 2, "1h": 5, "4h": 15, "1d": 30}

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]


# --------------------------------------------------------------------- #
# Окна и slug
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class WindowSpec:
    coin: str
    interval: str
    start: int
    end: int
    slugs: tuple  # кандидаты, пробуются по порядку

    @property
    def key(self) -> str:
        return f"{self.coin}-{self.interval}"


def window_for(coin: str, interval: str, now: float) -> WindowSpec | None:
    name = COINS[coin][0]
    n = INTERVALS[interval]
    if interval in ("5m", "15m"):
        s = int(now) - int(now) % n
        return WindowSpec(coin, interval, s, s + n, (f"{coin}-updown-{interval}-{s}",))
    if interval == "4h":
        # выравнивание 4ч окон неизвестно наверняка -> пробуем все сдвиги на целый час
        base = int(now) - int(now) % 3600
        cands = []
        for k in range(4):
            s = base - k * 3600
            if s <= now < s + n:
                cands.append(s)
        # предпочитаем кратное 4ч по UTC, потом по ET
        cands.sort(key=lambda s: (s % n != 0, (s + 4 * 3600) % n != 0))
        slugs = tuple(f"{coin}-updown-4h-{s}" for s in cands)
        return WindowSpec(coin, interval, cands[0], cands[0] + n, slugs)
    if interval == "1h":
        t = datetime.fromtimestamp(now, ET).replace(minute=0, second=0, microsecond=0)
        s = int(t.timestamp())
        h12 = t.hour % 12 or 12
        ap = "am" if t.hour < 12 else "pm"
        m, d, y = MONTHS[t.month - 1], t.day, t.year
        return WindowSpec(coin, interval, s, s + 3600, (
            f"{name}-up-or-down-{m}-{d}-{h12}{ap}-et",
            f"{name}-up-or-down-{m}-{d}-{y}-{h12}{ap}-et",
        ))
    if interval == "1d":
        t = datetime.fromtimestamp(now, ET)
        end_date = t.date() if t.hour < 12 else t.date() + timedelta(days=1)
        end_dt = datetime(end_date.year, end_date.month, end_date.day, 12, tzinfo=ET)
        start_date = end_date - timedelta(days=1)
        start_dt = datetime(start_date.year, start_date.month, start_date.day, 12, tzinfo=ET)
        m, d, y = MONTHS[end_date.month - 1], end_date.day, end_date.year
        return WindowSpec(coin, interval, int(start_dt.timestamp()), int(end_dt.timestamp()), (
            f"{name}-up-or-down-on-{m}-{d}",
            f"{name}-up-or-down-on-{m}-{d}-{y}",
        ))
    return None


# --------------------------------------------------------------------- #
# Рынок
# --------------------------------------------------------------------- #
@dataclass
class MMarket:
    coin: str
    interval: str
    slug: str
    start: int
    end: int
    condition_id: str | None
    outcomes: list
    token_ids: list
    up_token: str
    down_token: str
    resolution_source: str | None
    ref_kind: str
    price_to_beat: float | None
    fee_rate: float
    fetched_at: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        return f"{self.coin}-{self.interval}"


def _iso_to_unix(s):
    if not s:
        return None
    try:
        return int(datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def ref_kind_of(src: str | None) -> str:
    s = (src or "").lower()
    if "binance" in s:
        return "binance"
    return reference_kind(src)


def parse_updown(spec: WindowSpec, slug: str, event: dict | None, mkt: dict) -> MMarket | None:
    outcomes = _maybe_json(mkt.get("outcomes")) or []
    tokens = _maybe_json(mkt.get("clobTokenIds")) or []
    if len(outcomes) != 2 or len(tokens) != 2:
        return None
    labels = [str(o).strip().lower() for o in outcomes]
    if "up" not in labels or "down" not in labels:
        return None
    # проверка, что это именно нужное окно
    end = _iso_to_unix(mkt.get("endDate") or (event or {}).get("endDate"))
    if end is not None and abs(end - spec.end) > 180:
        return None
    ptb = None
    for holder in (event or {}, mkt):
        meta = _maybe_json(holder.get("eventMetadata")) if holder else None
        if isinstance(meta, dict) and meta.get("priceToBeat") not in (None, "", 0, "0"):
            try:
                ptb = float(meta["priceToBeat"])
                break
            except (TypeError, ValueError):
                pass
    src = mkt.get("resolutionSource") or (event or {}).get("resolutionSource")
    return MMarket(
        coin=spec.coin, interval=spec.interval, slug=slug, start=spec.start, end=spec.end,
        condition_id=mkt.get("conditionId"), outcomes=list(outcomes), token_ids=list(tokens),
        up_token=tokens[labels.index("up")], down_token=tokens[labels.index("down")],
        resolution_source=src, ref_kind=ref_kind_of(src), price_to_beat=ptb,
        fee_rate=0.0 if mkt.get("feesEnabled") is False else DEFAULT_FEE_RATE,
    )


def fetch_updown(spec: WindowSpec) -> MMarket | None:
    for slug in spec.slugs:
        sp = spec
        tail = slug.rsplit("-", 1)[-1]
        if spec.interval in ("5m", "15m", "4h") and tail.isdigit():
            s0 = int(tail)  # у 4h кандидаты с разным стартом
            sp = WindowSpec(spec.coin, spec.interval, s0, s0 + INTERVALS[spec.interval], spec.slugs)
        try:
            r = _session.get(f"{GAMMA}/events", params={"slug": slug}, timeout=6)
            if r.status_code != 200:
                continue
            data = r.json()
            if isinstance(data, list) and data and data[0].get("markets"):
                ev = data[0]
                m = parse_updown(sp, slug, ev, ev["markets"][0])
                if m:
                    return m
        except (Exception,) as e:  # сеть/JSON — просто пробуем следующий вариант
            log(f"[gamma] {slug}: {e}")
    return None


class MultiMarketCache:
    """(coin, interval) -> рынок текущего окна. Не долбит API, если рынка нет."""

    def __init__(self, ptb_refresh_s=15.0, miss_retry_s=60.0):
        self._by_slug: dict[str, MMarket] = {}
        self._miss: dict[tuple, float] = {}
        self._lock = threading.Lock()
        self.ptb_refresh_s = ptb_refresh_s
        self.miss_retry_s = miss_retry_s
        self.missing_reported: set = set()

    def get(self, coin: str, interval: str, now: float) -> MMarket | None:
        spec = window_for(coin, interval, now)
        if spec is None:
            return None
        with self._lock:
            for s in spec.slugs:
                m = self._by_slug.get(s)
                if m is not None:
                    break
            else:
                m = None
            miss_t = self._miss.get(spec.slugs)
        if m is not None:
            if m.price_to_beat is None and time.time() - m.fetched_at > self.ptb_refresh_s:
                m.fetched_at = time.time()
                fresh = fetch_updown(spec)
                if fresh is not None and fresh.price_to_beat is not None:
                    m.price_to_beat = fresh.price_to_beat
            return m
        if miss_t is not None and time.time() - miss_t < self.miss_retry_s:
            return None
        fresh = fetch_updown(spec)
        with self._lock:
            if fresh is None:
                self._miss[spec.slugs] = time.time()
                if spec.key not in self.missing_reported:
                    self.missing_reported.add(spec.key)
                    log(f"[warn] не нашёл рынок {spec.key}: пробовал {', '.join(spec.slugs)}")
                return None
            self._by_slug[fresh.slug] = fresh
            return fresh

    def prune(self, now: float):
        with self._lock:
            for s in [s for s, m in self._by_slug.items() if m.end < now - 3600]:
                self._by_slug.pop(s, None)
            for k in [k for k, t in self._miss.items() if now - t > 3600]:
                self._miss.pop(k, None)


# --------------------------------------------------------------------- #
# Стаканы пачкой
# --------------------------------------------------------------------- #
def fetch_books_batch(tokens: list[str]) -> dict:
    """token_id -> Book. Сначала POST /books пачками, при ошибке — по одному параллельно."""
    out = {}
    t0 = time.time()
    try:
        for i in range(0, len(tokens), 100):
            chunk = tokens[i:i + 100]
            r = _session.post(f"{CLOB}/books", json=[{"token_id": t} for t in chunk], timeout=6)
            r.raise_for_status()
            for raw in r.json() or []:
                tid = raw.get("asset_id") or raw.get("token_id")
                if tid:
                    out[tid] = parse_book(tid, raw, t0)
        return out
    except Exception as e:
        log(f"[clob] /books пачкой не вышло ({e}) — беру по одному")
    futs = {t: _pool.submit(fetch_book, t) for t in tokens}
    for t, f in futs.items():
        b = f.result()
        if b is not None:
            out[t] = b
    return out


# --------------------------------------------------------------------- #
# Потоки цен по всем монетам
# --------------------------------------------------------------------- #
RTDS_KIND = {"crypto_prices_chainlink": "spot", "crypto_prices_twap_thirty": "twap30",
             "crypto_prices_twap_sixty": "twap60"}
SYM_TO_COIN = {v[1]: k for k, v in COINS.items()}
BIN_TO_COIN = {v[2].upper(): k for k, v in COINS.items()}


class MultiFeed:
    STALE_S = 15.0

    def __init__(self, coins, keep_s=900):
        self.coins = list(coins)
        self._h: dict[tuple, deque] = {}
        self._lock = threading.Lock()
        self._keep_ms = keep_s * 1000
        self._ws = {}

    def add(self, coin, kind, ts_ms, value, recv=None):
        if value is None or ts_ms is None:
            return
        recv = recv if recv is not None else time.time()
        with self._lock:
            h = self._h.setdefault((coin, kind), deque())
            if h and ts_ms <= h[-1][0]:
                if ts_ms == h[-1][0]:
                    return
                items = sorted(list(h) + [(ts_ms, value, recv)], key=lambda x: x[0])
                h.clear()
                h.extend(items)
            else:
                h.append((ts_ms, value, recv))
            cut = h[-1][0] - self._keep_ms
            while h and h[0][0] < cut:
                h.popleft()

    def handle_rtds(self, message: str):
        if not message or not message.lstrip().startswith("{"):
            return
        try:
            d = json.loads(message)
        except ValueError:
            return
        kind = RTDS_KIND.get(d.get("topic"))
        p = d.get("payload") or {}
        coin = SYM_TO_COIN.get(str(p.get("symbol", "")).lower())
        if not kind or coin not in self.coins:
            return
        if isinstance(p.get("data"), list):
            for it in p["data"]:
                self.add(coin, kind, it.get("timestamp"), _parse_value(it))
            return
        ts = p.get("timestamp")
        self.add(coin, kind, int(ts) if ts is not None else None, _parse_value(p))

    def handle_binance(self, message: str):
        try:
            d = json.loads(message)
            d = d.get("data", d)
            coin = BIN_TO_COIN.get(str(d["s"]).upper())
            mid = (float(d["b"]) + float(d["a"])) / 2
        except (ValueError, KeyError, TypeError, AttributeError):
            return
        if coin in self.coins:
            now = time.time()
            self.add(coin, "binance", int(now * 1000), mid, now)

    def latest(self, coin, kind, max_age=None):
        max_age = self.STALE_S if max_age is None else max_age
        with self._lock:
            h = self._h.get((coin, kind))
            if not h:
                return None
            ts, v, recv = h[-1]
        return v if time.time() - recv <= max_age else None

    def at_exact(self, coin, kind, ts_ms, tol_ms=0):
        with self._lock:
            h = self._h.get((coin, kind)) or ()
            best = None
            for t, v, _ in reversed(h):
                if abs(t - ts_ms) <= tol_ms and (best is None or abs(t - ts_ms) < abs(best[0] - ts_ms)):
                    best = (t, v)
                if t < ts_ms - tol_ms - 5000:
                    break
        return best[1] if best else None

    def last_recv(self, coin, kind):
        with self._lock:
            h = self._h.get((coin, kind))
            return h[-1][2] if h else None

    def start(self, use_binance=True):
        """По отдельному соединению RTDS на каждую монету, с фильтром по символу —
        так же, как в BTC-логгере, где это работало стабильно. Сторож рвёт
        соединение, если по монете 30 с нет ни одного тика (тихое зависание)."""
        import websocket

        def rtds(coin):
            sym = COINS[coin][1]
            while True:
                try:
                    def on_open(ws):
                        ws.send(json.dumps({"action": "subscribe", "subscriptions": [
                            {"topic": t, "type": "update", "filters": json.dumps({"symbol": sym}, separators=(",", ":"))}
                            for t in RTDS_KIND]}))

                        def ping():
                            while True:
                                time.sleep(5)
                                try:
                                    ws.send("PING")
                                except Exception:
                                    return
                        threading.Thread(target=ping, daemon=True).start()
                    ws = websocket.WebSocketApp("wss://ws-live-data.polymarket.com", on_open=on_open,
                                                on_message=lambda w, m: self.handle_rtds(m))
                    self._ws[f"rtds_{coin}"] = ws
                    self._ws_started[coin] = time.time()
                    ws.run_forever()
                except Exception as e:
                    log(f"[rtds {coin}] {e}")
                self.reconnects[coin] = self.reconnects.get(coin, 0) + 1
                time.sleep(3)

        def watchdog():
            while True:
                time.sleep(10)
                now = time.time()
                for coin in self.coins:
                    last = max(filter(None, [self.last_recv(coin, "spot"), self.last_recv(coin, "twap60")]),
                               default=None)
                    started = self._ws_started.get(coin, now)
                    ref = last if last and last > started else started
                    if now - ref > 30 and self._ws.get(f"rtds_{coin}"):
                        log(f"[rtds {coin}] 30с без тиков — переподключаюсь")
                        try:
                            self._ws[f"rtds_{coin}"].close()
                        except Exception:
                            pass
                        self._ws_started[coin] = now

        def binance():
            streams = "/".join(f"{COINS[c][2]}@bookTicker" for c in self.coins)
            url = f"wss://stream.binance.com:9443/stream?streams={streams}"
            while True:
                try:
                    ws = websocket.WebSocketApp(url, on_message=lambda w, m: self.handle_binance(m))
                    self._ws["binance"] = ws
                    ws.run_forever(ping_interval=20, ping_timeout=10)
                except Exception as e:
                    log(f"[binance] {e}")
                time.sleep(3)

        self._ws_started = {}
        self.reconnects = {}
        for c in self.coins:
            threading.Thread(target=rtds, args=(c,), daemon=True).start()
        threading.Thread(target=watchdog, daemon=True).start()
        if use_binance:
            threading.Thread(target=binance, daemon=True).start()


# --------------------------------------------------------------------- #
# Сделки: живой поток CLOB (market channel)
# --------------------------------------------------------------------- #
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def parse_clob_ws_trades(message: str) -> list[dict]:
    """Достаёт события last_trade_price из сообщения market-канала."""
    if not message or message in ("PONG", "PING"):
        return []
    try:
        d = json.loads(message)
    except ValueError:
        return []
    items = d if isinstance(d, list) else [d]
    out = []
    for it in items:
        if isinstance(it, dict) and it.get("event_type") == "last_trade_price":
            out.append({
                "asset_id": it.get("asset_id"), "price": it.get("price"), "size": it.get("size"),
                "side": it.get("side"), "exchange_ts": it.get("timestamp"),
            })
    return out


class TradeStream:
    """Подписка на сделки по текущему набору токенов. При смене набора — переподключение."""

    def __init__(self, on_trade):
        self.on_trade = on_trade
        self._tokens: frozenset = frozenset()
        self._ws = None
        self._lock = threading.Lock()
        self.last_msg = 0.0

    def set_tokens(self, tokens):
        tokens = frozenset(t for t in tokens if t)
        with self._lock:
            if tokens == self._tokens:
                return
            self._tokens = tokens
            ws = self._ws
        if ws is not None:
            try:
                ws.close()  # цикл переподключится с новым набором
            except Exception:
                pass

    def start(self):
        import websocket

        def loop():
            while True:
                with self._lock:
                    toks = list(self._tokens)
                if not toks:
                    time.sleep(1)
                    continue
                try:
                    def on_open(ws):
                        ws.send(json.dumps({"assets_ids": toks, "type": "market"}))

                        def ping():
                            while True:
                                time.sleep(10)
                                try:
                                    ws.send("PING")
                                except Exception:
                                    return
                        threading.Thread(target=ping, daemon=True).start()

                    def on_msg(ws, m):
                        self.last_msg = time.time()
                        for t in parse_clob_ws_trades(m):
                            self.on_trade(t)
                    ws = websocket.WebSocketApp(CLOB_WS, on_open=on_open, on_message=on_msg)
                    with self._lock:
                        self._ws = ws
                    ws.run_forever()
                except Exception as e:
                    log(f"[clob-ws] {e}")
                time.sleep(0.5)

        threading.Thread(target=loop, daemon=True).start()


# --------------------------------------------------------------------- #
# Сделки: полная история окна из Data API (после закрытия)
# --------------------------------------------------------------------- #
def fetch_all_trades(condition_id: str, max_rows: int = 20000) -> tuple[list[dict], str | None]:
    """Все сделки рынка. Сначала /v2/trades (курсор), затем старый /trades (offset)."""
    rows, cursor = [], None
    try:
        while len(rows) < max_rows:
            params = {"condition": condition_id, "limit": 500}
            if cursor:
                params["cursor"] = cursor
            r = _session.get(f"{DATA_API}/v2/trades", params=params, timeout=10)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            js = r.json()
            rows.extend(js.get("data") or [])
            cursor = (js.get("pagination") or {}).get("next_cursor")
            if not cursor:
                return rows, "v2"
    except Exception as e:
        if rows:
            return rows, "v2_partial"
        log(f"[data-api] v2/trades {condition_id[:10]}…: {e} — пробую /trades")
    rows = []
    try:
        off = 0
        while len(rows) < max_rows:
            r = _session.get(f"{DATA_API}/trades",
                             params={"market": condition_id, "limit": 500, "offset": off}, timeout=10)
            if r.status_code != 200:
                break
            chunk = r.json() or []
            if isinstance(chunk, dict):
                chunk = chunk.get("data") or []
            rows.extend(chunk)
            if len(chunk) < 500:
                break
            off += 500
        return rows, "v1" if rows else None
    except Exception as e:
        log(f"[data-api] /trades {condition_id[:10]}…: {e}")
        return rows, None


def normalize_api_trade(t: dict) -> dict:
    """Приводит строку сделки (v1 или v2) к общему виду."""
    g = lambda *ks: next((t.get(k) for k in ks if t.get(k) not in (None, "")), None)
    return {
        "exchange_ts": g("timestamp", "match_time", "matchTime", "created_at"),
        "asset_id": g("asset", "asset_id", "token_id", "tokenId"),
        "outcome": g("outcome"),
        "side": g("side", "taker_side"),
        "price": g("price"),
        "size": g("size", "shares"),
        "usdc": g("usdc_size", "usdcSize", "size_usdc", "notional_usdc"),
        "wallet": g("proxyWallet", "proxy_wallet", "user", "taker"),
        "tx": g("transactionHash", "transaction_hash", "tx_hash"),
    }


# --------------------------------------------------------------------- #
# Запись: CSV.gz с ротацией по часам (UTC)
# --------------------------------------------------------------------- #
class RotatingGzCsv:
    """Пишет data/<dir>/<prefix>_YYYYMMDD_HH.csv.gz. Каждый час — новый файл,
    так при сбое теряется максимум текущий час. Потокобезопасно."""

    def __init__(self, directory: Path, prefix: str, fields: list[str], flush_every_s: float = 5.0):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix, self.fields = prefix, fields
        self._lock = threading.Lock()
        self._hour = None
        self._fh = self._w = None
        self._last_flush = 0.0
        self.flush_every_s = flush_every_s
        self.rows = 0

    def _open(self, hour_key: str):
        if self._fh:
            self._fh.close()
        path = self.dir / f"{self.prefix}_{hour_key}.csv.gz"
        new = not path.exists()
        self._fh = gzip.open(path, "at", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=self.fields, extrasaction="ignore")
        if new:
            self._w.writeheader()
        self._hour = hour_key

    def write(self, row: dict, ts: float | None = None):
        ts = ts if ts is not None else time.time()
        hk = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y%m%d_%H")
        with self._lock:
            if hk != self._hour:
                self._open(hk)
            self._w.writerow(row)
            self.rows += 1
            if time.time() - self._last_flush > self.flush_every_s:
                self._fh.flush()
                self._last_flush = time.time()

    def close(self):
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None


def read_gz_csv_tolerant(path) -> str:
    """Читает .csv.gz, даже если последний кусок оборван (процесс убили посреди записи)."""
    raw = Path(path).read_bytes()
    out = bytearray()
    import zlib
    while raw:
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            out += d.decompress(raw)
        except zlib.error:
            break
        if not d.eof:
            break  # оборванный хвост — берём то, что успели
        raw = d.unused_data
    text = out.decode("utf-8", errors="replace")
    # отрезаем последнюю неполную строку
    cut = text.rfind("\n")
    return text[:cut + 1] if cut >= 0 else ""


def load_gz_csvs(directory, prefix, usecols=None):
    import pandas as pd
    files = sorted(Path(directory).glob(f"{prefix}_*.csv.gz"))
    frames = []
    for f in files:
        txt = read_gz_csv_tolerant(f)
        if txt.count("\n") > 1:
            frames.append(pd.read_csv(io.StringIO(txt), low_memory=False,
                                      usecols=(lambda c: c in usecols) if usecols else None))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------- #
# Исход по свече Binance (для рынков, у которых resolutionSource = Binance)
# --------------------------------------------------------------------- #
BINANCE_REST = "https://api.binance.com/api/v3/klines"
_KLINE_INTERVAL = {3600: "1h", 14400: "4h", 86400: "1d", 900: "15m", 300: "5m"}


def _kline(symbol, interval, start_ms):
    r = _session.get(BINANCE_REST, params={"symbol": symbol, "interval": interval,
                                           "startTime": start_ms, "limit": 1}, timeout=6)
    r.raise_for_status()
    k = r.json()
    if not k or int(k[0][0]) != start_ms:
        return None
    return k[0]


def binance_kline_outcome(coin: str, start: int, end: int) -> tuple[str | None, float | None, float | None]:
    """Исход по правилам рынков на Binance.
      1h: свеча 1h, начинающаяся в start: UP, если close >= open.
      1d: две минутные свечи — "Close" свечи 12:00 ET в день старта против "Close"
          свечи 12:00 ET в день окончания; больше -> UP, меньше -> DOWN, равно -> SPLIT.
    Возвращает (исход, цена_старта, цена_конца) или (None, ...), если свечи ещё нет."""
    sym = COINS[coin][2].upper()
    try:
        if end - start == 86400:
            a = _kline(sym, "1m", start * 1000)
            b = _kline(sym, "1m", end * 1000)
            if a is None or b is None or int(b[6]) > time.time() * 1000:
                return None, float(a[4]) if a else None, None
            o, c = float(a[4]), float(b[4])
            return ("UP" if c > o else "DOWN" if c < o else "SPLIT"), o, c
        iv = _KLINE_INTERVAL.get(end - start)
        if iv is None:
            return None, None, None
        k = _kline(sym, iv, start * 1000)
        if k is None:
            return None, None, None
        o, c = float(k[1]), float(k[4])
        if int(k[6]) > time.time() * 1000:   # свеча ещё не закрылась
            return None, o, None
        return ("UP" if c >= o else "DOWN"), o, c
    except Exception as e:
        log(f"[binance] kline {coin} {start}: {e}")
        return None, None, None


@dataclass
class OutcomeRef:
    """Минимум, который нужен core.pm.fetch_outcome."""
    slug: str
    condition_id: str | None
    outcomes: list
    end: int = 0


def outcome_ref_by_slug(slug: str) -> OutcomeRef | None:
    from core.pm import gamma_market_by_slug
    try:
        m = gamma_market_by_slug(slug)
        if m:
            return OutcomeRef(slug, m.get("conditionId"), _maybe_json(m.get("outcomes")) or [])
    except Exception as e:
        log(f"[gamma] {slug}: {e}")
    return None
