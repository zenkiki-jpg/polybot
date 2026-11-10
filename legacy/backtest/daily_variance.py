"""
Проверка "а что если бы я торговал каждый ОТДЕЛЬНЫЙ день за последние N дней"
— с текущей (уже откалиброванной) моделью цены.

В отличие от grid_search.py (который ищет параметры), этот скрипт
ФИКСИРУЕТ параметры и смотрит на РАЗБРОС результата по дням — именно
то, что нужно, чтобы ответить на вопрос "может ли один хороший день быть
просто везением, а не системным свойством".

Запуск:
    python backtest/daily_variance.py --days 30
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data.binance_fetch import fetch_klines_cached
from backtest.strategy import StrategyConfig
from backtest.engine import run_backtest


def run_daily_variance(days: int, threshold: float, offset: int,
                        stake_mode: str, kelly_fraction: float,
                        initial_bankroll: float):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = fetch_klines_cached(symbol="BTCUSDT", interval="1m", start=start, end=end)
    if df.empty:
        print("Не удалось загрузить данные с Binance.")
        return

    df["_day"] = df["open_time"].dt.date
    day_list = sorted(df["_day"].unique())
    print(f"Тестирую {len(day_list)} отдельных дней ({day_list[0]} — {day_list[-1]})")
    print(f"Параметры: threshold=${threshold}, offset={offset}с, "
          f"stake_mode={stake_mode}, банкролл каждый день заново с ${initial_bankroll}\n")

    rows = []
    for day in day_list:
        day_df = df[df["_day"] == day].drop(columns="_day").reset_index(drop=True)
        if len(day_df) < 50:
            continue  # неполный день (например, первый/последний в выборке)

        cfg = StrategyConfig(
            delta_threshold=threshold,
            entry_offset_seconds=offset,
            stake_mode=stake_mode,
            stake_fixed=10.0,
            kelly_fraction=kelly_fraction,
            initial_bankroll=initial_bankroll,
        )
        result = run_backtest(day_df, cfg)
        s = result.summary()
        rows.append({
            "день": day,
            "сделок": s["total_trades"],
            "winrate_%": s["win_rate_pct"],
            "pnl_$": s["total_pnl"],
            "доходность_%": (s["total_pnl"] / initial_bankroll * 100) if s["total_trades"] else 0,
            "drawdown_%": s["max_drawdown_pct"],
        })

    rdf = pd.DataFrame(rows)
    if rdf.empty:
        print("Недостаточно данных ни на один полный день.")
        return

    print(rdf.to_string(index=False))

    profitable_days = (rdf["pnl_$"] > 0).sum()
    total_days = len(rdf)
    print("\n" + "=" * 60)
    print("ИТОГО ПО ДНЯМ")
    print("=" * 60)
    print(f"Прибыльных дней: {profitable_days} из {total_days} ({profitable_days/total_days*100:.0f}%)")
    print(f"Средняя дневная доходность: {rdf['доходность_%'].mean():+.2f}%")
    print(f"Медианная дневная доходность: {rdf['доходность_%'].median():+.2f}%")
    print(f"Стандартное отклонение доходности: {rdf['доходность_%'].std():.2f}%")
    print(f"Худший день: {rdf['доходность_%'].min():+.2f}%")
    print(f"Лучший день:  {rdf['доходность_%'].max():+.2f}%")
    print(f"Средняя дневная просадка: {rdf['drawdown_%'].mean():.2f}%")

    print("\nКак это читать:")
    print("- Если 'Прибыльных дней' сильно больше 50% и среднее заметно > 0,")
    print("  а стандартное отклонение не поглощает среднее с запасом — это уже")
    print("  выглядит как системное свойство, а не везение одного дня.")
    print("- Если разброс (std) сравним или больше среднего — один хороший день")
    print("  почти ничего не доказывает, нужно больше независимых дней и/или")
    print("  меньший риск на сделку.")

    return rdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=15.0)
    parser.add_argument("--offset", type=int, default=10)
    parser.add_argument("--stake-mode", type=str, default="kelly", choices=["fixed", "kelly"])
    parser.add_argument("--kelly-fraction", type=float, default=0.5)
    parser.add_argument("--bankroll", type=float, default=100.0)
    args = parser.parse_args()
    run_daily_variance(args.days, args.threshold, args.offset,
                        args.stake_mode, args.kelly_fraction, args.bankroll)
