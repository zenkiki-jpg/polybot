"""
Реальная торговля на Polymarket через официальный py-clob-client.

⚠️ ЭТО РАБОТАЕТ С НАСТОЯЩИМИ ДЕНЬГАМИ, КОГДА TRADING_DRY_RUN=false. ⚠️

Устроено так же, как live/paper_trader.py (та же логика сигнала, тот же
TWAP, та же непрерывная проверка до закрытия окна) — но при срабатывании
сигнала реально отправляет ордер на биржу вместо виртуальной записи.

ПРЕДОХРАНИТЕЛИ (не отключаются флагами командной строки, только вручную
в .env, чтобы нельзя было случайно обойти по невнимательности):
  - TRADING_DRY_RUN=true по умолчанию — ничего не отправляется, только лог.
  - TRADING_MAX_STAKE_PER_TRADE — жёсткий потолок ставки в $, ниже которого
    урезается ЛЮБОЙ расчёт (Kelly или fixed), что бы они ни насчитали.
  - TRADING_MAX_DAILY_LOSS_FRACTION — при превышении дневного лимита
    убытка бот перестаёт открывать новые позиции до следующего дня.

Перед первым использованием:
    1. Прочитай live/WALLET_SETUP.md полностью.
    2. Скопируй .env.example -> .env, впиши приватный ключ.
    3. python live/real_trader.py --setup   (получить API-креды один раз)
    4. python live/real_trader.py --check-balance   (проверить, что всё видит)
    5. Запускай ТОЛЬКО с dry-run=true много дней, сверяя с paper trading.
    6. Только потом, осознанно, меняй TRADING_DRY_RUN на false в .env.
"""
import sys as _sys
_sys.stderr.write(
    "\n[real_trader] ОТКЛЮЧЁН в v2.\n"
    "Этот файл построен на логике v1 (перепутанные bid/ask, самодельный TWAP,\n"
    "исход по своей цене, market-ордер без FOK и без проверки ответа, нет funder\n"
    "для email-кошельков, нет redeem). Пока paper v2 не покажет устойчивый плюс\n"
    "на новых данных, реальной торговли нет смысла. Когда покажет — напишем\n"
    "исполнение заново на core/pm.py.\n\n")
_sys.exit(2)


import sys
import os
import csv
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone, date
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
import builtins
import functools
print = functools.partial(builtins.print, flush=True)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from logger.polymarket_logger import (
    current_window_slug, fetch_market_by_slug, parse_clob_token_ids,
    fetch_price, fetch_btc_price_binance, start_chainlink_stream,
    get_chainlink_btc_price, WINDOW_SECONDS,
)
from backtest.strategy import StrategyConfig, estimate_win_prob, kelly_stake, apply_win_fee
from live.paper_trader import TwapBuffer, compute_signal, WindowState


# --------------------------------------------------------------------- #
# Конфигурация из .env — читается один раз при старте
# --------------------------------------------------------------------- #
def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val else default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


PRIVATE_KEY = os.environ.get("POLYMARKET_PRIVATE_KEY", "").strip()
SIGNATURE_TYPE = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "0"))
API_KEY = os.environ.get("POLYMARKET_API_KEY", "").strip()
API_SECRET = os.environ.get("POLYMARKET_API_SECRET", "").strip()
API_PASSPHRASE = os.environ.get("POLYMARKET_API_PASSPHRASE", "").strip()

MAX_STAKE_PER_TRADE = _env_float("TRADING_MAX_STAKE_PER_TRADE", 2.0)
MAX_DAILY_LOSS_FRACTION = _env_float("TRADING_MAX_DAILY_LOSS_FRACTION", 0.10)
DRY_RUN = _env_bool("TRADING_DRY_RUN", True)

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet


def _mask_key(k: str) -> str:
    if not k:
        return "(пусто)"
    return k[:6] + "..." + k[-4:]


def get_client():
    """Создаёт ClobClient. Приватный ключ используется ТОЛЬКО локально для
    криптографической подписи — никуда не отправляется как есть."""
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds

    if not PRIVATE_KEY:
        print("[ошибка] POLYMARKET_PRIVATE_KEY не задан в .env. "
              "Прочитай live/WALLET_SETUP.md")
        sys.exit(1)

    creds = None
    if API_KEY and API_SECRET and API_PASSPHRASE:
        creds = ApiCreds(api_key=API_KEY, api_secret=API_SECRET, api_passphrase=API_PASSPHRASE)

    client = ClobClient(
        host=CLOB_HOST,
        chain_id=CHAIN_ID,
        key=PRIVATE_KEY,
        creds=creds,
        signature_type=SIGNATURE_TYPE,
    )
    return client


def cmd_setup():
    """Получить (или пересоздать) API-креды CLOB и вывести, что вписать в .env."""
    client = get_client()
    print(f"Кошелёк: {client.get_address()}")
    print("Получаю/создаю API-креды...")
    creds = client.create_or_derive_api_creds()
    print("\nГотово! Впиши это в .env (или я сейчас предложу дописать сам):\n")
    print(f"POLYMARKET_API_KEY={creds.api_key}")
    print(f"POLYMARKET_API_SECRET={creds.api_secret}")
    print(f"POLYMARKET_API_PASSPHRASE={creds.api_passphrase}")

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        answer = input("\nАвтоматически дописать эти строки в .env? (y/n): ").strip().lower()
        if answer == "y":
            content = env_path.read_text()
            for key, val in [("POLYMARKET_API_KEY", creds.api_key),
                              ("POLYMARKET_API_SECRET", creds.api_secret),
                              ("POLYMARKET_API_PASSPHRASE", creds.api_passphrase)]:
                if f"{key}=" in content:
                    lines = content.splitlines()
                    lines = [f"{key}={val}" if l.startswith(f"{key}=") else l for l in lines]
                    content = "\n".join(lines) + "\n"
                else:
                    content += f"\n{key}={val}\n"
            env_path.write_text(content)
            print(".env обновлён.")


def cmd_check_balance():
    """Проверка, что бот вообще видит кошелёк и баланс — БЕЗ ордеров."""
    client = get_client()
    print(f"Кошелёк: {client.get_address()}")
    try:
        balance = client.get_balance_allowance()
        print(f"Баланс/allowance: {balance}")
    except Exception as e:
        print(f"[ошибка] Не удалось получить баланс: {e}")
        print("Возможные причины: не пройден --setup, нет approve на USDC, "
              "неверный signature_type.")


# --------------------------------------------------------------------- #
# Основной торговый цикл — почти идентичен paper_trader.py, но с реальным
# размещением ордера вместо виртуальной записи, плюс жёсткие предохранители.
# --------------------------------------------------------------------- #
class DailyLossGuard:
    """Kill switch: считает реализованный PnL за календарный день, при
    превышении лимита убытка блокирует НОВЫЕ сделки до следующего дня."""
    def __init__(self, bankroll_reference: float, max_loss_fraction: float):
        self.bankroll_reference = bankroll_reference
        self.max_loss_fraction = max_loss_fraction
        self.current_day = date.today()
        self.realized_pnl_today = 0.0

    def register_pnl(self, pnl: float):
        today = date.today()
        if today != self.current_day:
            self.current_day = today
            self.realized_pnl_today = 0.0
        self.realized_pnl_today += pnl

    def trading_allowed(self) -> bool:
        today = date.today()
        if today != self.current_day:
            self.current_day = today
            self.realized_pnl_today = 0.0
        loss_limit = -abs(self.bankroll_reference * self.max_loss_fraction)
        return self.realized_pnl_today > loss_limit


def place_real_order(client, token_id: str, price: float, size_usd: float, dry_run: bool):
    """Размещает FOK market-ордер на покупку. Возвращает (успех, детали)."""
    from py_clob_client.clob_types import MarketOrderArgs
    from py_clob_client.order_builder.constants import BUY

    if dry_run:
        print(f"  [DRY-RUN] Ордер НЕ отправлен. Был бы: BUY token={token_id[:12]}... "
              f"price=${price:.3f} size=${size_usd:.2f}")
        return True, {"dry_run": True}

    try:
        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=size_usd,
            side=BUY,
            price=price,
        )
        signed_order = client.create_market_order(order_args)
        resp = client.post_order(signed_order)
        print(f"  [РЕАЛЬНЫЙ ОРДЕР] Отправлен: {resp}")
        return True, resp
    except Exception as e:
        print(f"  [ОШИБКА ОРДЕРА] Не удалось разместить: {e}")
        return False, {"error": str(e)}


def run_real_trader(cfg: StrategyConfig, interval: int, out_path: Path, initial_bankroll: float):
    print("=" * 70)
    print(f"РЕЖИМ: {'DRY-RUN (безопасно, ордера не отправляются)' if DRY_RUN else '⚠️  РЕАЛЬНАЯ ТОРГОВЛЯ ⚠️'}")
    print(f"Максимальная ставка на сделку: ${MAX_STAKE_PER_TRADE}")
    print(f"Дневной лимит убытка: {MAX_DAILY_LOSS_FRACTION*100:.0f}% от банкролла")
    print("=" * 70)

    if not DRY_RUN:
        print("\n⚠️  ВНИМАНИЕ: TRADING_DRY_RUN=false — БОТ БУДЕТ ТОРГОВАТЬ НА РЕАЛЬНЫЕ ДЕНЬГИ.")
        confirm = input("Введи ТОЧНО фразу 'я понимаю риски' чтобы продолжить: ")
        if confirm.strip() != "я понимаю риски":
            print("Подтверждение не получено. Останавливаюсь.")
            return

    client = get_client()
    print(f"Кошелёк: {client.get_address()}")

    guard = DailyLossGuard(initial_bankroll, MAX_DAILY_LOSS_FRACTION)
    bankroll = initial_bankroll
    state = WindowState()
    twap = TwapBuffer(window_seconds=30.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_path.exists() and out_path.stat().st_size > 0
    CSV_FIELDS = ["event", "logged_at_utc", "window_slug", "direction", "signal_value",
                  "entry_price_real", "stake", "won", "pnl", "bankroll_after", "dry_run"]

    print("Подключаюсь к Chainlink через RTDS WebSocket Polymarket...")
    start_chainlink_stream()
    time.sleep(2)
    print(f"Запущено. Банкролл (для расчёта размера ставки): ${bankroll:.2f}\n")

    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()

        cached_market = None

        try:
            while True:
                now = datetime.now(timezone.utc)
                now_unix = int(now.timestamp())
                window_ts, slug = current_window_slug(now_unix)
                seconds_to_close = WINDOW_SECONDS - (now_unix - window_ts)

                price_tick = get_chainlink_btc_price() or fetch_btc_price_binance()
                if price_tick is not None:
                    twap.add(now_unix, price_tick)
                price = twap.value()

                if slug != state.slug:
                    if state.pending is not None and state.last_price is not None:
                        trade = state.pending
                        actual_direction = "UP" if state.last_price > state.open_price else "DOWN"
                        won = trade["direction"] == actual_direction
                        entry_price = trade["entry_price_real"]
                        stake = trade["stake"]
                        fill_price = min(entry_price + stake * cfg.slippage_per_dollar_stake, 0.999)
                        if won:
                            payout = stake / fill_price
                            pnl = apply_win_fee(payout, cfg) - stake
                        else:
                            pnl = -stake

                        bankroll += pnl
                        guard.register_pnl(pnl)

                        writer.writerow({
                            "event": "SETTLED", "logged_at_utc": now.isoformat(),
                            "window_slug": state.slug, "direction": trade["direction"],
                            "signal_value": trade["signal_value"], "entry_price_real": entry_price,
                            "stake": round(stake, 4), "won": won, "pnl": round(pnl, 4),
                            "bankroll_after": round(bankroll, 4), "dry_run": DRY_RUN,
                        })
                        f.flush()
                        print(f"\n{'✅ ВЫИГРЫШ' if won else '❌ ПРОИГРЫШ'} {state.slug}: "
                              f"{trade['direction']}, entry=${entry_price:.3f}, "
                              f"stake=${stake:.2f}, pnl={pnl:+.2f}, банкролл=${bankroll:.2f}\n")

                    elif state.slug is not None:
                        print(f"[{now:%H:%M:%S}] Окно {state.slug} закрыто — сделки не было")

                    state = WindowState()
                    state.slug = slug
                    state.open_price = price
                    cached_market = fetch_market_by_slug(slug)

                if price is not None:
                    state.last_price = price

                if (not state.traded and cached_market is not None
                        and state.open_price is not None and price is not None
                        and seconds_to_close <= cfg.entry_offset_seconds
                        and guard.trading_allowed()):

                    delta = price - state.open_price
                    signal_value = compute_signal(delta, state.open_price, cfg)

                    if signal_value >= cfg.delta_threshold:
                        direction = "UP" if delta > 0 else "DOWN"
                        token_ids = parse_clob_token_ids(cached_market)

                        if len(token_ids) >= 2:
                            token_id = token_ids[0] if direction == "UP" else token_ids[1]
                            entry_price = fetch_price(token_id, "BUY")

                            if entry_price is not None and 0.0 < entry_price < 1.0:
                                if cfg.stake_mode == "kelly":
                                    win_prob = estimate_win_prob(signal_value, cfg,
                                                                  seconds_remaining=seconds_to_close)
                                    stake = kelly_stake(entry_price, win_prob, bankroll, cfg)
                                else:
                                    stake = cfg.stake_fixed

                                # ЖЁСТКИЙ предохранитель — независимо от расчёта стратегии
                                stake = min(stake, MAX_STAKE_PER_TRADE)

                                if stake > 0:
                                    print(f"[{now:%H:%M:%S}] Сигнал: {direction}, "
                                          f"signal={signal_value:.3f}, entry=${entry_price:.3f}, "
                                          f"stake=${stake:.2f}")
                                    success, details = place_real_order(
                                        client, token_id, entry_price, stake, DRY_RUN)

                                    if success:
                                        state.pending = {
                                            "direction": direction, "signal_value": signal_value,
                                            "entry_price_real": entry_price, "stake": stake,
                                        }
                                        writer.writerow({
                                            "event": "OPEN", "logged_at_utc": now.isoformat(),
                                            "window_slug": slug, "direction": direction,
                                            "signal_value": round(signal_value, 4),
                                            "entry_price_real": entry_price,
                                            "stake": round(stake, 4), "won": "", "pnl": "",
                                            "bankroll_after": "", "dry_run": DRY_RUN,
                                        })
                                        f.flush()
                                        # Сделка реально открыта -> в этом окне больше
                                        # не пробуем. Если сигнал ЕЩЁ НЕ дотянул до
                                        # порога — traded НЕ ставим, проверяем дальше.
                                        state.traded = True

                if not guard.trading_allowed():
                    print(f"[{now:%H:%M:%S}] 🛑 Дневной лимит убытка достигнут — "
                          f"торговля приостановлена до завтра.")

                time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\nОстановлено. Финальный банкролл: ${bankroll:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Реальная торговля Polymarket BTC 5-min (с dry-run)")
    parser.add_argument("--setup", action="store_true", help="Получить API-креды один раз")
    parser.add_argument("--check-balance", action="store_true", help="Проверить баланс без ордеров")
    parser.add_argument("--interval", type=int, default=3)
    parser.add_argument("--out", type=str, default="data/real_trades.csv")
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--offset", type=int, default=60)
    parser.add_argument("--kelly-fraction", type=float, default=0.5)
    parser.add_argument("--kelly-prob-cap", type=float, default=0.94)
    parser.add_argument("--stake-mode", type=str, default="fixed", choices=["fixed", "kelly"])
    parser.add_argument("--stake-fixed", type=float, default=1.0)
    parser.add_argument("--bankroll", type=float, default=20.0,
                         help="Условный банкролл ДЛЯ РАСЧЁТА размера ставки (не обязательно = реальный баланс)")
    parser.add_argument("--fee-on-win", type=float, default=0.02,
                         help="Комиссия с выигрышного payout (0-1). ПРОВЕРЬ актуальную цифру на Polymarket")
    parser.add_argument("--slippage-per-dollar", type=float, default=0.0008,
                         help="Проскальзывание на $1 ставки (грубая эвристика)")
    args = parser.parse_args()

    if args.setup:
        cmd_setup()
        sys.exit(0)
    if args.check_balance:
        cmd_check_balance()
        sys.exit(0)

    cfg = StrategyConfig(
        delta_threshold=args.threshold,
        entry_offset_seconds=args.offset,
        stake_mode=args.stake_mode,
        stake_fixed=args.stake_fixed,
        kelly_fraction=args.kelly_fraction,
        kelly_prob_cap=args.kelly_prob_cap,
        initial_bankroll=args.bankroll,
        fee_on_win=args.fee_on_win,
        slippage_per_dollar_stake=args.slippage_per_dollar,
    )
    run_real_trader(cfg, args.interval, Path(args.out), args.bankroll)
