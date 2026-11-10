"""
Бэктест-движок: разбивает 1-минутные свечи BTC на 5-минутные окна,
эмулирует "5-min up/down" рынок Polymarket и прогоняет стратегию.
"""
from dataclasses import dataclass, field
import pandas as pd

from .strategy import (StrategyConfig, implied_entry_price, estimate_win_prob,
                        kelly_stake, apply_win_fee)


@dataclass
class Trade:
    window_start: pd.Timestamp
    direction: str          # "UP" или "DOWN"
    delta: float
    entry_price: float
    stake: float
    won: bool
    pnl: float
    bankroll_after: float
    believed_win_prob: float  # во что верила модель (для сверки вера vs факт)
    fee_paid: float = 0.0     # сколько комиссии съедено (0, если проигрыш)


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)  # [(timestamp, bankroll)]

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.won)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0][1]
        max_dd = 0.0
        for _, bal in self.equity_curve:
            peak = max(peak, bal)
            dd = (peak - bal) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def total_fees_paid(self) -> float:
        return sum(t.fee_paid for t in self.trades)

    @property
    def believed_vs_actual(self) -> dict | None:
        """Сверка 'во что верила модель' vs 'что было по факту'. Большое
        расхождение (как мы видели: 88.6% веры против 73.56% факта) —
        предупреждающий знак, что калибровка не соответствует именно
        этому периоду истории."""
        if not self.trades:
            return None
        avg_believed = sum(t.believed_win_prob for t in self.trades) / len(self.trades)
        actual = self.win_rate
        return {
            "believed_win_rate_pct": round(avg_believed * 100, 2),
            "actual_win_rate_pct": round(actual * 100, 2),
            "gap_pct_points": round((avg_believed - actual) * 100, 2),
        }

    def summary(self) -> dict:
        s = {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.total_trades - self.wins,
            "win_rate_pct": round(self.win_rate * 100, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_fees_paid": round(self.total_fees_paid, 2),
            "final_bankroll": round(self.equity_curve[-1][1], 2) if self.equity_curve else None,
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
        }
        bva = self.believed_vs_actual
        if bva:
            s.update(bva)
        return s


def run_backtest(df_1m: pd.DataFrame, cfg: StrategyConfig) -> BacktestResult:
    """
    df_1m: DataFrame с колонками open_time, close_time, open, high, low, close
           (1-минутные свечи, отсортированные по времени, без пропусков в идеале)
    cfg:   параметры стратегии
    """
    if df_1m.empty:
        raise ValueError("Пустой набор данных — нечего бэктестить")

    df = df_1m.reset_index(drop=True)
    bankroll = cfg.initial_bankroll
    result = BacktestResult()
    result.equity_curve.append((df.loc[0, "open_time"], bankroll))

    # Сколько минутных свечей проходит до момента входа внутри 5-мин окна
    window_len = 5
    entry_after_candles = window_len - max(1, round(cfg.entry_offset_seconds / 60))
    entry_after_candles = max(1, min(window_len - 1, entry_after_candles))

    n_windows = len(df) // window_len
    for w in range(n_windows):
        chunk = df.iloc[w * window_len: (w + 1) * window_len]
        if len(chunk) < window_len:
            break

        window_open = chunk.iloc[0]["open"]
        window_start_time = chunk.iloc[0]["open_time"]
        entry_price_ref = chunk.iloc[entry_after_candles - 1]["close"]
        window_close = chunk.iloc[-1]["close"]

        delta = entry_price_ref - window_open
        signal_value = abs(delta) / window_open * 100 if cfg.delta_unit == "percent" else abs(delta)

        if signal_value < cfg.delta_threshold:
            continue  # сигнала недостаточно, пропускаем окно

        direction = "UP" if delta > 0 else "DOWN"
        actual_direction = "UP" if window_close > window_open else "DOWN"
        won = direction == actual_direction

        # Шаг 1: базовая цена (без проскальзывания) — нужна только чтобы
        # прикинуть размер ставки, ещё не зная её саму (курица и яйцо).
        base_entry_price = implied_entry_price(signal_value, cfg, seconds_remaining=cfg.entry_offset_seconds)
        believed_win_prob = estimate_win_prob(signal_value, cfg, seconds_remaining=cfg.entry_offset_seconds)

        if cfg.stake_mode == "kelly":
            stake = kelly_stake(base_entry_price, believed_win_prob, bankroll, cfg)
        else:
            stake = cfg.stake_fixed

        if stake > bankroll:
            break  # банкролла не хватает на эту ставку — останавливаем прогон

        if stake <= 0:
            continue  # Kelly говорит "эджа нет для этой сделки" — просто пропускаем, НЕ обрываем весь бэктест

        # Шаг 2: теперь, зная реальный размер ставки, пересчитываем цену
        # УЖЕ С проскальзыванием — крупная Kelly-ставка получает более
        # реалистичную (худшую) цену, чем мелкая fixed-ставка.
        entry_price = implied_entry_price(signal_value, cfg, seconds_remaining=cfg.entry_offset_seconds,
                                           stake=stake)

        fee_paid = 0.0
        if won:
            payout = stake / entry_price  # $1 за акцию при выигрыше
            payout_after_fee = apply_win_fee(payout, cfg)
            fee_paid = payout - payout_after_fee
            pnl = payout_after_fee - stake
        else:
            pnl = -stake

        bankroll += pnl

        result.trades.append(Trade(
            window_start=window_start_time,
            direction=direction,
            delta=delta,
            entry_price=entry_price,
            stake=stake,
            won=won,
            pnl=pnl,
            bankroll_after=bankroll,
            believed_win_prob=believed_win_prob,
            fee_paid=fee_paid,
        ))
        result.equity_curve.append((window_start_time, bankroll))

        if bankroll <= 0:
            break

    return result
