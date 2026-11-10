"""
Загрузка исторических свечей BTCUSDT с Binance.
Публичный REST API, авторизация НЕ требуется, бесплатно, без ключей.

Документация: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
"""
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def _to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    start: datetime = None,
    end: datetime = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Скачивает свечи за период [start, end], постранично (Binance отдаёт
    максимум 1000 свечей за запрос), с уважением к rate-limit.
    """
    if start is None or end is None:
        raise ValueError("Укажи start и end (datetime, UTC)")

    all_rows = []
    cur_start = _to_ms(start)
    end_ms = _to_ms(end)

    while cur_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur_start,
            "endTime": end_ms,
            "limit": limit,
        }
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break

        all_rows.extend(rows)
        last_open_time = rows[-1][0]
        cur_start = last_open_time + 1  # следующая свеча

        # Binance rate limit — не спамим
        time.sleep(0.25)

        if len(rows) < limit:
            break

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    if df.empty:
        return df

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df[["open_time", "close_time", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates(subset="open_time").reset_index(drop=True)
    return df


def fetch_klines_cached(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    start: datetime = None,
    end: datetime = None,
) -> pd.DataFrame:
    """
    Та же загрузка, но с кэшем в CSV — чтобы не дёргать Binance повторно
    при каждом прогоне бэктеста.
    """
    key = f"{symbol}_{interval}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    cache_path = CACHE_DIR / key

    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["open_time", "close_time"])
        return df

    df = fetch_klines(symbol, interval, start, end)
    if not df.empty:
        df.to_csv(cache_path, index=False)
    return df


if __name__ == "__main__":
    # Быстрая проверка при запуске напрямую
    from datetime import timedelta

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=1)
    df = fetch_klines_cached(start=start, end=end)
    print(f"Загружено свечей: {len(df)}")
    print(df.head())
