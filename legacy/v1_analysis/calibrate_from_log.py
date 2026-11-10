"""
Калибровка price_model_k / price_model_power / win_prob_table напрямую
из твоего live_log.csv (лога Polymarket) — без ручного пересчёта в чате.

Делает то же самое, что мы считали вручную ранее:
  1. Для каждого закрытого 5-минутного окна берёт дельту BTC (open -> точка
     входа) и реальную цену покупки токена в этот момент — точка на offset
     используется для калибровки price_model_k/power (кривая цены).
  2. Считает win_prob_table_by_time — ДВУМЕРНУЮ таблицу "дельта + оставшееся
     время -> реальный winrate", используя ВСЕ замеры внутри каждого окна
     (не только одну точку) — так сигнал за 8 секунд до закрытия и тот же
     сигнал за 120 секунд до закрытия оцениваются по-разному, как и должно
     быть (см. обсуждение: одна и та же дельта ближе к концу окна значит
     куда больше, чем та же дельта в начале).
  3. Сохраняет всё в backtest/calibrated_params.json — GUI, grid_search и
     paper_trader подхватят это автоматически при следующем запуске.

Запуск:
    python backtest/calibrate_from_log.py --csv data/live_log.csv
    python backtest/calibrate_from_log.py --csv data/live_log.csv --offset 60
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from backtest.strategy import TIME_BUCKETS, time_bucket_label

OUT_PATH = Path(__file__).parent / "calibrated_params.json"


def price_model(delta_abs, k, power):
    raw = 0.5 + k * (delta_abs ** power)
    return np.clip(raw, 0.5, 0.999)


def _resolve_price_col(df: pd.DataFrame) -> str:
    if "btc_price_chainlink" in df.columns and df["btc_price_chainlink"].notna().sum() > len(df) * 0.5:
        return "btc_price_chainlink"
    return "btc_price_binance"


def extract_single_point_per_window(df: pd.DataFrame, entry_offset_seconds: int) -> pd.DataFrame:
    """Одна точка на окно (ближайшая к entry_offset_seconds) — используется
    ТОЛЬКО для калибровки price_model_k/power (кривая цены), как раньше."""
    price_col = _resolve_price_col(df)
    records = []
    for slug, g in df.groupby("window_slug"):
        g = g.sort_values("seconds_to_close", ascending=False)
        if g["seconds_to_close"].min() > 5:
            continue

        window_open_price = g.iloc[0][price_col]
        window_close_price = g.iloc[-1][price_col]
        if pd.isna(window_open_price) or pd.isna(window_close_price):
            continue
        actual_direction = 1 if window_close_price > window_open_price else -1

        g = g.copy()
        g["dist_to_target"] = (g["seconds_to_close"] - entry_offset_seconds).abs()
        entry_row = g.loc[g["dist_to_target"].idxmin()]

        delta = entry_row[price_col] - window_open_price
        predicted_direction = 1 if delta > 0 else (-1 if delta < 0 else 0)
        if predicted_direction == 0:
            continue

        entry_price = entry_row["yes_best_ask"] if predicted_direction == 1 else entry_row["no_best_ask"]
        if pd.isna(entry_price):
            continue

        records.append({
            "window_slug": slug,
            "abs_delta": abs(delta),
            "won": predicted_direction == actual_direction,
            "entry_price_real": entry_price,
        })

    return pd.DataFrame(records)


def extract_all_points_per_window(df: pd.DataFrame) -> pd.DataFrame:
    """ВСЕ замеры внутри каждого окна — используется и для 2D win_prob_table,
    и для 2D price_model (дельта + оставшееся время). Каждое окно даёт ~60
    точек вместо одной."""
    price_col = _resolve_price_col(df)
    records = []
    for slug, g in df.groupby("window_slug"):
        g = g.sort_values("seconds_to_close", ascending=False)
        if g["seconds_to_close"].min() > 5:
            continue  # окно не досмотрено до конца

        window_open_price = g.iloc[0][price_col]
        window_close_price = g.iloc[-1][price_col]
        if pd.isna(window_open_price) or pd.isna(window_close_price):
            continue
        actual_direction = 1 if window_close_price > window_open_price else -1

        for row in g.itertuples():
            row_price = getattr(row, price_col)
            if pd.isna(row_price):
                continue
            delta = row_price - window_open_price
            predicted_direction = 1 if delta > 0 else (-1 if delta < 0 else 0)
            if predicted_direction == 0:
                continue

            abs_delta = abs(delta)
            pct_delta = abs_delta / window_open_price * 100 if window_open_price else 0.0
            entry_price_real = (row.yes_best_ask if predicted_direction == 1
                                 else row.no_best_ask)

            records.append({
                "window_slug": slug,
                "seconds_to_close": row.seconds_to_close,
                "abs_delta": abs_delta,
                "pct_delta": pct_delta,
                "won": predicted_direction == actual_direction,
                "entry_price_real": entry_price_real,
            })

    return pd.DataFrame(records)


def build_win_prob_table(points: pd.DataFrame, delta_col: str, thresholds: list,
                          prob_cap: float = 0.95, min_n: int = 5) -> list:
    table = []
    for t in thresholds:
        sub = points[points[delta_col] >= t]
        if len(sub) < min_n:
            break
        wr = min(sub["won"].mean(), prob_cap)
        table.append([t, round(wr, 4), len(sub)])
    return table


def build_win_prob_table_by_time(points: pd.DataFrame, delta_col: str, thresholds: list,
                                  prob_cap: float = 0.95, min_n: int = 5) -> dict:
    """Строит СВОЮ таблицу winrate-по-threshold ДЛЯ КАЖДОГО временного бакета."""
    points = points.copy()
    points["time_bucket"] = points["seconds_to_close"].apply(time_bucket_label)

    result = {}
    for bucket_label, bucket_points in points.groupby("time_bucket"):
        table = build_win_prob_table(bucket_points, delta_col, thresholds, prob_cap, min_n)
        if table:
            result[bucket_label] = table
    return result


def build_price_model_by_time(points: pd.DataFrame, delta_col: str, min_n: int = 30) -> dict:
    """Фитит СВОЮ кривую price_model (k, power) ДЛЯ КАЖДОГО временного
    бакета — вместо одной кривой на все времена. Использует РЕАЛЬНЫЕ цены
    (yes_best_ask/no_best_ask) из каждого замера внутри окна, а не только
    точку на фиксированном offset."""
    points = points.copy()
    points["time_bucket"] = points["seconds_to_close"].apply(time_bucket_label)
    points = points.dropna(subset=["entry_price_real"])

    result = {}
    for bucket_label, bucket_points in points.groupby("time_bucket"):
        if len(bucket_points) < min_n:
            continue
        x = bucket_points[delta_col].values
        y = bucket_points["entry_price_real"].values
        try:
            popt, _ = curve_fit(price_model, x, y, p0=[0.05, 0.4], maxfev=5000)
            k_fit, power_fit = popt
            y_pred = price_model(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            result[bucket_label] = {
                "k": round(float(k_fit), 5),
                "power": round(float(power_fit), 4),
                "r_squared": round(float(r2), 3),
                "n": len(bucket_points),
            }
        except Exception:
            continue  # не удалось подобрать кривую для этого бакета — пропускаем
    return result


def load_log_csv(csv_path: str) -> pd.DataFrame:
    """Устойчивое чтение live_log.csv, даже если файл "склеен" из старого
    формата логгера (11 колонок, без Chainlink) и нового (12 колонок,
    с Chainlink) — такое бывает, если дописывать новой версией логгера
    в файл, начатый старой версией."""
    OLD_FIELDS = [
        "logged_at_utc", "window_start_unix", "window_slug", "seconds_to_close",
        "btc_price_binance", "yes_best_bid", "yes_best_ask", "yes_mid",
        "no_best_bid", "no_best_ask", "no_mid",
    ]
    NEW_FIELDS = [
        "logged_at_utc", "window_start_unix", "window_slug", "seconds_to_close",
        "btc_price_binance", "btc_price_chainlink", "yes_best_bid", "yes_best_ask",
        "yes_mid", "no_best_bid", "no_best_ask", "no_mid",
    ]

    try:
        return pd.read_csv(csv_path, parse_dates=["logged_at_utc"])
    except Exception:
        pass  # файл смешанного формата — разбираем построчно ниже

    print("⚠️  Файл смешанного формата (менялось число колонок посреди лога) — "
          "разбираю построчно, это нормально после апгрейда логгера.")

    import csv as csv_module
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv_module.reader(f)
        header = next(reader)
        for line_num, raw_row in enumerate(reader, start=2):
            if len(raw_row) == len(NEW_FIELDS):
                rows.append(dict(zip(NEW_FIELDS, raw_row)))
            elif len(raw_row) == len(OLD_FIELDS):
                row = dict(zip(OLD_FIELDS, raw_row))
                row["btc_price_chainlink"] = None
                rows.append(row)
            # иначе — совсем битая строка, тихо пропускаем

    df = pd.DataFrame(rows)
    df["logged_at_utc"] = pd.to_datetime(df["logged_at_utc"], errors="coerce")
    for col in ["window_start_unix", "seconds_to_close", "btc_price_binance",
                "btc_price_chainlink", "yes_best_bid", "yes_best_ask", "yes_mid",
                "no_best_bid", "no_best_ask", "no_mid"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Восстановлено строк: {len(df)}")
    return df


def run_calibration(csv_path: str, entry_offset_seconds: int):
    df = load_log_csv(csv_path)
    print(f"Загружено строк: {len(df)}")

    # Убираем старый баг /book (0.01/0.99 плейсхолдер), если он есть в логе
    if "yes_best_bid" in df.columns and "yes_best_ask" in df.columns:
        placeholder = ((df["yes_best_bid"] == 0.01) & (df["yes_best_ask"] == 0.99))
        df = df[~placeholder].copy()

    price_col = _resolve_price_col(df)
    print(f"Использую как эталон цены BTC: {price_col}")

    # --- 1. Одна точка на окно -> калибровка price_model_k/power (как раньше) ---
    points = extract_single_point_per_window(df, entry_offset_seconds)
    points = points.dropna(subset=["entry_price_real"])
    print(f"Закрытых окон с валидной точкой калибровки: {len(points)}")

    if len(points) < 20:
        print("\n⚠️  Слишком мало данных (<20 окон) для надёжной калибровки.")
        print("Собери логгер подольше и повтори — но всё равно посчитаю на том, что есть.")

    if len(points) < 5:
        print("Данных совсем недостаточно — прерываю.")
        return

    x = points["abs_delta"].values
    y = points["entry_price_real"].values

    try:
        popt, _ = curve_fit(price_model, x, y, p0=[0.01, 0.5], maxfev=5000)
        k_fit, power_fit = popt
        y_pred = price_model(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    except Exception as e:
        print(f"Ошибка калибровки кривой: {e}")
        return

    flat_win_prob_table = build_win_prob_table(points, "abs_delta", [0, 10, 20, 30, 50, 75, 100])

    # --- 2. ВСЕ точки внутри каждого окна -> 2D таблица (дельта + время) ---
    all_points = extract_all_points_per_window(df)
    print(f"\nВсего точек для 2D-калибровки (все замеры внутри окон): {len(all_points)}")

    dollar_thresholds = [0, 10, 20, 30, 50, 75, 100]
    percent_thresholds = [0, 0.01, 0.02, 0.05, 0.1, 0.15]

    win_prob_table_by_time_dollar = build_win_prob_table_by_time(
        all_points, "abs_delta", dollar_thresholds)
    win_prob_table_by_time_percent = build_win_prob_table_by_time(
        all_points, "pct_delta", percent_thresholds)

    # --- 3. price_model ТОЖЕ по временным бакетам, на тех же реальных ценах ---
    price_model_by_time_dollar = build_price_model_by_time(all_points, "abs_delta")
    price_model_by_time_percent = build_price_model_by_time(all_points, "pct_delta")

    result = {
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(csv_path),
        "entry_offset_seconds": entry_offset_seconds,
        "n_windows": len(points),
        "n_points_2d": len(all_points),
        "price_model_k": round(float(k_fit), 5),
        "price_model_power": round(float(power_fit), 4),
        "r_squared": round(float(r2), 3),
        "win_prob_table": flat_win_prob_table,
        "win_prob_table_by_time": {
            "dollar": win_prob_table_by_time_dollar,
            "percent": win_prob_table_by_time_percent,
        },
        "price_model_by_time": {
            "dollar": price_model_by_time_dollar,
            "percent": price_model_by_time_percent,
        },
        "time_buckets": TIME_BUCKETS,
        "overall_win_rate": round(float(points["won"].mean()), 4),
    }

    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ КАЛИБРОВКИ")
    print("=" * 60)
    print(f"Окон использовано:        {result['n_windows']}")
    print(f"price_model_k:            {result['price_model_k']}")
    print(f"price_model_power:        {result['price_model_power']}")
    print(f"R^2 фита:                 {result['r_squared']}  "
          f"({'неплохо' if r2 > 0.6 else 'слабо — модель объясняет цену лишь частично'})")
    print(f"Общий winrate по выборке: {result['overall_win_rate']*100:.1f}%")
    print("\nПлоская таблица winrate по threshold (для обратной совместимости):")
    print(f"{'threshold':>10} | {'winrate':>8} | {'n':>5}")
    for t, wr, n in flat_win_prob_table:
        print(f"{t:>10} | {wr*100:>7.1f}% | {n:>5}")

    print("\n2D-таблица winrate по (threshold $, время до закрытия):")
    for bucket, table in win_prob_table_by_time_dollar.items():
        print(f"\n  Бакет времени '{bucket}с до закрытия':")
        print(f"  {'threshold':>10} | {'winrate':>8} | {'n':>5}")
        for t, wr, n in table:
            print(f"  {t:>10} | {wr*100:>7.1f}% | {n:>5}")

    print("\n2D price_model (k/power) по временным бакетам:")
    print(f"  {'бакет':>10} | {'k':>8} | {'power':>7} | {'R^2':>6} | {'n':>6}")
    for bucket, p in price_model_by_time_dollar.items():
        print(f"  {bucket:>10} | {p['k']:>8} | {p['power']:>7} | {p['r_squared']:>6} | {p['n']:>6}")

    print(f"\nСохранено в: {OUT_PATH}")
    print("GUI, grid_search и paper_trader подхватят это автоматически при следующем запуске.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Путь к live_log.csv")
    parser.add_argument("--offset", type=int, default=120,
                         help="За сколько секунд до закрытия считать точку входа")
    args = parser.parse_args()
    run_calibration(args.csv, args.offset)
