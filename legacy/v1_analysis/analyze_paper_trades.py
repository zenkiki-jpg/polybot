"""
Анализ накопленного лога paper trading (data/paper_trades.csv) — считает
статистику по обоим портфелям (fixed и kelly), не останавливая живой процесс
(файл можно читать, пока paper_trader.py продолжает в него дописывать).

Использование:
    python live/analyze_paper_trades.py --csv data/paper_trades.csv
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


CSV_FIELDS = [
    "event", "logged_at_utc", "window_slug", "window_open_price",
    "direction", "signal_value", "seconds_to_close_at_entry", "entry_price_real",
    "believed_win_prob",
    "won",
    "fixed_stake", "fixed_pnl", "fixed_bankroll_after", "fixed_skipped",
    "kelly_stake", "kelly_pnl", "kelly_bankroll_after", "kelly_skipped",
]

# Более старая версия схемы (до того, как добавили seconds_to_close_at_entry
# и believed_win_prob) — на случай файлов, начатых старой версией скрипта.
OLD_CSV_FIELDS = [
    "event", "logged_at_utc", "window_slug", "window_open_price",
    "direction", "signal_value", "entry_price_real",
    "won",
    "fixed_stake", "fixed_pnl", "fixed_bankroll_after", "fixed_skipped",
    "kelly_stake", "kelly_pnl", "kelly_bankroll_after", "kelly_skipped",
]


def load_paper_trades_csv(csv_path: str) -> pd.DataFrame | None:
    """Устойчивое чтение — на случай пустого/битого файла (например, если
    файл случайно создан, но в него ничего не записалось, или заголовок
    повредился из-за одновременной записи двух процессов в один файл)."""
    p = Path(csv_path)

    if not p.exists():
        print(f"[ошибка] Файл не найден: {csv_path}")
        return None

    if p.stat().st_size == 0:
        print(f"[ошибка] Файл ПУСТОЙ (0 байт): {csv_path}")
        print("Похоже, процесс либо не успел ничего записать, либо упал сразу после старта.")
        return None

    with open(csv_path, encoding="utf-8", errors="replace") as f:
        first_line = f.readline().strip()

    # Файл БЕЗ заголовка вообще (данные начинаются прямо с первой строки) —
    # это баг старых версий paper_trader.py, если целевой файл существовал
    # пустым при старте. Определяем по числу полей в первой строке.
    if not first_line.startswith("event,"):
        n_fields = len(first_line.split(","))
        if n_fields == len(CSV_FIELDS):
            names = CSV_FIELDS
        elif n_fields == len(OLD_CSV_FIELDS):
            names = OLD_CSV_FIELDS
        else:
            print(f"[ошибка] Файл без заголовка, и число полей ({n_fields}) не "
                  f"совпадает ни с одной известной схемой ({len(CSV_FIELDS)} или "
                  f"{len(OLD_CSV_FIELDS)}). Файл повреждён сильнее обычного, "
                  f"нужно смотреть руками.")
            return None

        print(f"[warn] В файле нет строки-заголовка (старый баг paper_trader.py, "
              f"уже исправлен). Читаю по угаданной схеме ({n_fields} полей).")
        try:
            df = pd.read_csv(csv_path, names=names, header=None)
            df["logged_at_utc"] = pd.to_datetime(df["logged_at_utc"], errors="coerce")
            # Если это старая (короткая) схема — недостающих колонок не будет,
            # добавляем их как NaN, чтобы остальной код не падал на KeyError
            for col in CSV_FIELDS:
                if col not in df.columns:
                    df[col] = pd.NA
            return df
        except Exception as e:
            print(f"[ошибка] Не удалось прочитать даже с угаданной схемой: {e}")
            return None

    # Сначала пробуем нормально
    try:
        return pd.read_csv(csv_path, parse_dates=["logged_at_utc"])
    except Exception as e:
        print(f"[warn] Обычное чтение не удалось: {e}")

    # Смотрим на первые строки "как есть" для диагностики
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        first_lines = [next(f, "") for _ in range(3)]
    print("\nПервые строки файла (для диагностики):")
    for i, line in enumerate(first_lines):
        print(f"  [{i}] {line.rstrip()[:200]}")

    header = first_lines[0].strip()
    if "logged_at_utc" not in header:
        print("\n⚠️  В заголовке файла НЕТ колонки 'logged_at_utc' — заголовок битый")
        print("или отсутствует вовсе (например, из-за одновременной записи двух")
        print("процессов в один и тот же файл, или файл был создан вручную/пустым).")
        print("\nЧто делать: удали/переименуй этот файл и перезапусти сессию заново")
        print("с явным --out на новое имя файла — не давай двум процессам писать")
        print("в один и тот же CSV одновременно.")
        return None

    # Заголовок вроде на месте, но что-то ещё пошло не так (например, где-то
    # посреди файла строка с другим числом полей) — пробуем восстановить
    # построчно, пропуская битые строки, лучше частичные данные чем ничего.
    print("\nПробую восстановить построчно, пропуская повреждённые строки...")
    try:
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="warn")
        df["logged_at_utc"] = pd.to_datetime(df["logged_at_utc"], errors="coerce")
        print(f"Восстановлено строк: {len(df)}")
        return df
    except Exception as e2:
        print(f"[ошибка] Не удалось восстановить файл даже частично: {e2}")
        return None


def compute_stats(df: pd.DataFrame, prefix: str) -> dict:
    """prefix: 'fixed' или 'kelly'"""
    settled = df[df["event"] == "SETTLED"].copy()
    skipped_col = f"{prefix}_skipped"
    pnl_col = f"{prefix}_pnl"
    bank_col = f"{prefix}_bankroll_after"

    traded = settled[settled[skipped_col] == False].copy()  # noqa: E712
    n_skipped = (settled[skipped_col] == True).sum()  # noqa: E712

    if traded.empty:
        return {
            "сделок": 0, "пропущено": int(n_skipped), "winrate_%": 0.0,
            "итог_$": 0.0, "финальный_банкролл": None, "max_drawdown_%": 0.0,
        }

    wins = (traded["won"] == True).sum()  # noqa: E712
    n = len(traded)
    total_pnl = traded[pnl_col].sum()

    # Просадка по цепочке банкролла именно этого портфеля
    equity = traded[bank_col].tolist()
    peak = equity[0]
    max_dd = 0.0
    for b in equity:
        peak = max(peak, b)
        if peak > 0:
            dd = (peak - b) / peak * 100
            max_dd = max(max_dd, dd)

    return {
        "сделок": n,
        "пропущено": int(n_skipped),
        "winrate_%": round(wins / n * 100, 2),
        "итог_$": round(total_pnl, 2),
        "финальный_банкролл": round(equity[-1], 2),
        "max_drawdown_%": round(max_dd, 2),
    }


def compute_belief_gap(df: pd.DataFrame) -> dict | None:
    """Сверка 'во что верила модель (believed_win_prob)' vs 'что было по
    факту (won)' — по ВСЕМ сработавшим сигналам, не разделяя fixed/kelly
    (сигнал общий для обеих стратегий). Большой разрыв — предупреждающий
    знак, что калибровка не соответствует текущему периоду (см. пример
    с обвалом бэктеста: верили в 88.6%, по факту вышло 73.56%)."""
    settled = df[df["event"] == "SETTLED"].copy()
    if "believed_win_prob" not in settled.columns or settled.empty:
        return None
    settled = settled.dropna(subset=["believed_win_prob"])
    if settled.empty:
        return None

    avg_believed = settled["believed_win_prob"].mean() * 100
    actual = (settled["won"] == True).mean() * 100  # noqa: E712
    return {
        "n": len(settled),
        "believed_win_rate_pct": round(avg_believed, 2),
        "actual_win_rate_pct": round(actual, 2),
        "gap_pct_points": round(avg_believed - actual, 2),
    }


def run_analysis(csv_path: str):
    df = load_paper_trades_csv(csv_path)
    if df is None:
        return
    print(f"Загружено строк: {len(df)}")

    opened = (df["event"] == "OPEN").sum()
    settled = (df["event"] == "SETTLED").sum()
    print(f"Открыто сделок: {opened} | Закрыто (settled): {settled}")

    if settled == 0:
        print("\nПока нет ни одной завершённой сделки — рано подводить итоги.")
        return

    first_time = df["logged_at_utc"].min()
    last_time = df["logged_at_utc"].max()
    duration = last_time - first_time
    print(f"Период: {first_time} — {last_time} ({duration})")

    print("\n" + "=" * 60)
    for prefix, label in [("fixed", "FIXED"), ("kelly", "KELLY")]:
        s = compute_stats(df, prefix)
        print(f"\n{label}:")
        for k, v in s.items():
            print(f"  {k}: {v}")
    print("=" * 60)

    gap = compute_belief_gap(df)
    if gap:
        print(f"\nСВЕРКА 'ВЕРА vs ФАКТ' (по {gap['n']} сработавшим сигналам):")
        print(f"  Модель верила в winrate:  {gap['believed_win_rate_pct']}%")
        print(f"  Реальный winrate:         {gap['actual_win_rate_pct']}%")
        print(f"  Расхождение:              {gap['gap_pct_points']:+.2f} п.п.")
        if abs(gap["gap_pct_points"]) > 10:
            print("  ⚠️  Расхождение больше 10 п.п. — калибровка заметно разошлась")
            print("      с текущим периодом. Возможно, стоит собрать больше данных")
            print("      этого периода и переоткалиброваться, либо снизить kelly_fraction.")
    else:
        print("\n(Сверка 'вера vs факт' недоступна — это старый CSV без колонки "
              "believed_win_prob, обнови paper_trader.py и начни новый файл)")

    # Погодовая/подневная разбивка — полезно смотреть, нет ли резких провалов
    settled_df = df[df["event"] == "SETTLED"].copy()
    settled_df["day"] = settled_df["logged_at_utc"].dt.date
    print("\nПо дням (fixed_pnl):")
    daily = settled_df.groupby("day")["fixed_pnl"].agg(["count", "sum"])
    daily.columns = ["сделок", "pnl_$"]
    print(daily.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/paper_trades.csv")
    args = parser.parse_args()
    run_analysis(args.csv)
