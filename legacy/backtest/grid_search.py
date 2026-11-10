"""
Grid search по параметрам стратегии — С train/test разделением,
чтобы не подбирать параметры, которые просто "подошли" под конкретный
кусок истории (overfitting).

Идея:
1. Скачиваем N дней истории BTC.
2. Делим ХРОНОЛОГИЧЕСКИ на train (первые ~70%) и test (последние ~30%).
3. Перебираем сетку параметров, считаем метрику ТОЛЬКО на train.
4. Берём top-N комбинаций по train и проверяем их на test — которые
   держатся на test, а не разваливаются, те и заслуживают доверия.
5. Используем ТОЛЬКО fixed stake для перебора (Kelly сильно "раздувает"
   и искажает сравнение параметров — оценивай сначала сигнал, потом
   уже думай про сайзинг).

Метрика: не просто total_pnl, а risk-adjusted —
    score = total_pnl / max(max_drawdown_pct, 1)
Так мы не выбираем параметры, которые дают много прибыли ценой диких
просадок (как в примере с 97% drawdown).

Запуск:
    python backtest/grid_search.py --days 30
"""
import sys
import argparse
import itertools
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data.binance_fetch import fetch_klines_cached
from backtest.strategy import StrategyConfig
from backtest.engine import run_backtest


def score_result(result) -> float:
    s = result.summary()
    if s["total_trades"] < 20:
        return float("-inf")  # слишком мало сделок — не доверяем результату
    dd = max(s["max_drawdown_pct"], 1.0)
    return s["total_pnl"] / dd


def run_grid_search(days: int, train_frac: float = 0.7, block_shuffle: bool = False):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = fetch_klines_cached(symbol="BTCUSDT", interval="1m", start=start, end=end)
    if df.empty:
        print("Не удалось загрузить данные с Binance.")
        return

    if block_shuffle:
        # Делим на дневные блоки и перемешиваем — так "удачный день" не может
        # целиком осесть в одной части. Честнее оценивает средний эффект,
        # хуже подходит, если хочешь смотреть именно на устойчивость во времени.
        import numpy as np
        df["_day"] = df["open_time"].dt.date
        days_list = sorted(df["_day"].unique())
        rng = np.random.default_rng(42)
        rng.shuffle(days_list)
        split_day_idx = int(len(days_list) * train_frac)
        train_days = set(days_list[:split_day_idx])
        df_train = df[df["_day"].isin(train_days)].drop(columns="_day").reset_index(drop=True)
        df_test = df[~df["_day"].isin(train_days)].drop(columns="_day").reset_index(drop=True)
        print("Режим: block_shuffle — дни перемешаны случайно между train/test")
    else:
        split_idx = int(len(df) * train_frac)
        df_train = df.iloc[:split_idx].reset_index(drop=True)
        df_test = df.iloc[split_idx:].reset_index(drop=True)

    print(f"Всего свечей: {len(df)} | train: {len(df_train)} | test: {len(df_test)}")
    print(f"Train период: {df_train.iloc[0]['open_time']} -> {df_train.iloc[-1]['open_time']}")
    print(f"Test период:  {df_test.iloc[0]['open_time']} -> {df_test.iloc[-1]['open_time']}")

    # Сетка параметров для перебора
    thresholds = [5, 10, 15, 20, 25, 30]
    offsets = [60, 90, 120, 150, 180]

    train_results = []
    for threshold, offset in itertools.product(thresholds, offsets):
        cfg = StrategyConfig(
            delta_threshold=threshold,
            entry_offset_seconds=offset,
            stake_mode="fixed",
            stake_fixed=10.0,
            initial_bankroll=300.0,
        )
        try:
            result = run_backtest(df_train, cfg)
            train_results.append({
                "threshold": threshold,
                "offset": offset,
                "score": score_result(result),
                **result.summary(),
            })
        except Exception as e:
            print(f"  ошибка на threshold={threshold}, offset={offset}: {e}")

    train_df = pd.DataFrame(train_results).sort_values("score", ascending=False)
    print("\n=== ТОП-10 комбинаций на TRAIN (не доверяем им напрямую!) ===")
    print(train_df.head(10)[["threshold", "offset", "score", "total_trades",
                              "win_rate_pct", "total_pnl", "max_drawdown_pct"]].to_string(index=False))

    # Проверяем топ-10 по train на test — вот это уже честная проверка
    top10 = train_df.head(10)
    test_rows = []
    for _, row in top10.iterrows():
        cfg = StrategyConfig(
            delta_threshold=row["threshold"],
            entry_offset_seconds=row["offset"],
            stake_mode="fixed",
            stake_fixed=10.0,
            initial_bankroll=300.0,
        )
        result = run_backtest(df_test, cfg)
        test_rows.append({
            "threshold": row["threshold"],
            "offset": row["offset"],
            "train_score": row["score"],
            "test_score": score_result(result),
            **{f"test_{k}": v for k, v in result.summary().items()},
        })

    test_df = pd.DataFrame(test_rows).sort_values("test_score", ascending=False)
    print("\n=== Те же комбинации, проверенные на TEST (независимые данные) ===")
    print(test_df[["threshold", "offset", "train_score", "test_score",
                    "test_total_trades", "test_win_rate_pct", "test_total_pnl",
                    "test_max_drawdown_pct"]].to_string(index=False))

    # --- Понятный вердикт по каждой комбинации ---
    print("\n=== ВЕРДИКТ (человеческим языком) ===")
    n_train_positive = (train_df["score"] > 0).sum()
    n_total = len(train_df)
    print(f"Комбинаций с положительным score на TRAIN: {n_train_positive} из {n_total}")

    if n_train_positive == 0:
        print("\n⚠️  НИ ОДНА комбинация не была прибыльной на train-периоде "
              f"({df_train.iloc[0]['open_time']:%d.%m} — {df_train.iloc[-1]['open_time']:%d.%m}).")
        print("Это значит: текущая ценовая модель (price_model_k/power) на ")
        print("представительном куске истории НЕ показывает надёжного эджа —")
        print("даже если на test что-то выглядит красиво, это, скорее всего,")
        print("особенности конкретных нескольких дней, а не устойчивая закономерность.")
        print("\nЧто можно сделать дальше:")
        print("  1. Собрать РЕАЛЬНЫЕ логи Polymarket за более длинный и разнообразный")
        print("     период (не 15 часов, а недели), и переоткалибровать price_model.")
        print("  2. Проверить, не переобучена ли текущая модель под один трендовый день.")
    else:
        stable = test_df[(test_df["test_score"] > 0) &
                          (test_df["train_score"] > 0)]
        if len(stable) > 0:
            print(f"\n✅ Комбинаций, прибыльных И на train, И на test: {len(stable)}")
            print("Это и есть кандидаты, которым можно чуть больше доверять:")
            print(stable[["threshold", "offset", "train_score", "test_score"]].to_string(index=False))
        else:
            print("\n⚠️  Прибыльные на train комбинации не подтвердились на test.")
            print("Похоже на подгонку параметров под шум train-периода.")

    print("\nОбщее правило: доверяй только тем комбинациям, которые держатся на ОБОИХ")
    print("периодах — резкий разрыв train vs test (в любую сторону) означает, что")
    print("результат определяется особенностями конкретных дат, а не сигналом.")

    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="Сколько дней истории использовать")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Доля данных под train")
    parser.add_argument("--block-shuffle", action="store_true",
                         help="Перемешать дни случайно между train/test (устраняет эффект 'удачного периода')")
    args = parser.parse_args()
    run_grid_search(args.days, args.train_frac, args.block_shuffle)
