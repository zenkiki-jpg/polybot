"""
Детальный анализ Kelly-портфеля из paper_trades.csv — не просто общая
статистика (это уже есть в analyze_paper_trades.py), а разбор ПОЧЕМУ
получился именно такой результат: какие дни/сделки внесли наибольший
вклад, как вело себя проскальзывание, сколько съедено на комиссиях.

Использование:
    python live/analyze_kelly.py --csv data/paper_trades.csv
    python live/analyze_kelly.py --csv data/paper_trades.csv --portfolio fixed
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from live.analyze_paper_trades import load_paper_trades_csv


def run_kelly_detail(csv_path: str, portfolio: str, top_n: int):
    df = load_paper_trades_csv(csv_path)
    if df is None:
        return

    settled = df[df["event"] == "SETTLED"].copy()
    skipped_col = f"{portfolio}_skipped"
    stake_col = f"{portfolio}_stake"
    pnl_col = f"{portfolio}_pnl"
    fill_col = f"{portfolio}_fill_price"
    slip_col = f"{portfolio}_slippage_cost"
    fee_col = f"{portfolio}_fee_paid"
    bank_col = f"{portfolio}_bankroll_after"

    missing = [c for c in [skipped_col, stake_col, pnl_col] if c not in settled.columns]
    if missing:
        print(f"[ошибка] В файле нет нужных колонок: {missing}")
        print("Похоже, файл записан старой версией paper_trader.py (до добавления "
              "детального трекинга проскальзывания/комиссии) — начни новую сессию "
              "новой версией скрипта, чтобы получить эти данные.")
        return

    traded = settled[settled[skipped_col] == False].copy()  # noqa: E712
    n_skipped = int((settled[skipped_col] == True).sum())  # noqa: E712

    if traded.empty:
        print(f"Портфель '{portfolio}': ни одной совершённой сделки (пропущено {n_skipped}).")
        return

    traded["day"] = pd.to_datetime(traded["logged_at_utc"]).dt.date
    has_detail = fill_col in traded.columns and traded[fill_col].notna().any()

    print("=" * 70)
    print(f"ДЕТАЛЬНЫЙ АНАЛИЗ ПОРТФЕЛЯ: {portfolio.upper()}")
    print("=" * 70)
    print(f"Всего сделок: {len(traded)} | Пропущено: {n_skipped} "
          f"({n_skipped/(len(traded)+n_skipped)*100:.1f}% от всех сигналов)")

    # --- 1. Общая статистика ---
    wins = (traded["won"] == True).sum()  # noqa: E712
    total_pnl = traded[pnl_col].sum()
    print(f"Win rate: {wins/len(traded)*100:.2f}%")
    print(f"Total PnL: {total_pnl:+.2f}$")

    # --- 2. Распределение размера ставок ---
    print("\n" + "-" * 70)
    print("РАЗМЕР СТАВОК")
    print("-" * 70)
    stakes = traded[stake_col]
    print(f"  Мин:     ${stakes.min():.2f}")
    print(f"  Медиана: ${stakes.median():.2f}")
    print(f"  Среднее: ${stakes.mean():.2f}")
    print(f"  90-й перцентиль: ${stakes.quantile(0.9):.2f}")
    print(f"  Макс:    ${stakes.max():.2f}")
    if portfolio == "kelly" and stakes.max() > stakes.median() * 5:
        print("  ⚠️  Максимальная ставка сильно (>5x) превышает медианную —")
        print("      признак того, что Kelly сильно нарастил ставки после удачной")
        print("      серии. Стоит проверить, не совпадает ли пик с 'хорошим днём'")
        print("      из подневной разбивки ниже — велика вероятность, что это")
        print("      разовый выброс, а не типичное поведение.")

    # --- 3. По дням ---
    print("\n" + "-" * 70)
    print("ПО ДНЯМ")
    print("-" * 70)
    daily = traded.groupby("day").agg(
        сделок=(pnl_col, "count"),
        winrate=("won", lambda s: (s == True).mean() * 100),  # noqa: E712
        pnl=(pnl_col, "sum"),
        средняя_ставка=(stake_col, "mean"),
        макс_ставка=(stake_col, "max"),
    ).round(2)
    print(daily.to_string())

    # --- 4. Лучшие / худшие дни ---
    print("\n" + "-" * 70)
    print(f"ТОП-{top_n} ЛУЧШИХ ДНЕЙ")
    print("-" * 70)
    print(daily.sort_values("pnl", ascending=False).head(top_n).to_string())

    print(f"\nТОП-{top_n} ХУДШИХ ДНЕЙ")
    print("-" * 70)
    print(daily.sort_values("pnl", ascending=True).head(top_n).to_string())

    best_day_pnl = daily["pnl"].max()
    total_from_best_day = best_day_pnl
    pct_from_best_day = (total_from_best_day / total_pnl * 100) if total_pnl != 0 else 0
    if abs(pct_from_best_day) > 40:
        print(f"\n⚠️  Один лучший день даёт {pct_from_best_day:.0f}% всего PnL —")
        print("    итоговый результат сильно зависит от одного удачного периода,")
        print("    а не от стабильного эджа на каждый день. Осторожнее с выводами.")

    # --- 5. Лучшие / худшие ОТДЕЛЬНЫЕ сделки ---
    print("\n" + "-" * 70)
    print(f"ТОП-{top_n} ЛУЧШИХ СДЕЛОК")
    print("-" * 70)
    cols_to_show = ["logged_at_utc", "window_slug", "direction", "signal_value",
                     "entry_price_real", stake_col, pnl_col]
    if has_detail:
        cols_to_show.insert(5, fill_col)
    best_trades = traded.sort_values(pnl_col, ascending=False).head(top_n)
    print(best_trades[cols_to_show].to_string(index=False))

    print(f"\nТОП-{top_n} ХУДШИХ СДЕЛОК")
    print("-" * 70)
    worst_trades = traded.sort_values(pnl_col, ascending=True).head(top_n)
    print(worst_trades[cols_to_show].to_string(index=False))

    # --- 6. Проскальзывание ---
    if has_detail:
        print("\n" + "-" * 70)
        print("ПРОСКАЛЬЗЫВАНИЕ")
        print("-" * 70)
        slip = traded[slip_col].dropna()
        if len(slip) > 0:
            total_slip = slip.sum()
            print(f"  Суммарно потеряно на проскальзывании: ${total_slip:.2f}")
            print(f"  В среднем на сделку: ${slip.mean():.4f}")
            print(f"  Максимум за одну сделку: ${slip.max():.4f}")
            print(f"  Как доля от Total PnL: {abs(total_slip/total_pnl*100):.1f}%"
                  if total_pnl != 0 else "")
        else:
            print("  Нет данных о проскальзывании в этом файле.")

        # --- 7. Комиссии ---
        print("\n" + "-" * 70)
        print("КОМИССИИ")
        print("-" * 70)
        fees = traded[fee_col].dropna()
        if len(fees) > 0:
            total_fee = fees.sum()
            print(f"  Суммарно уплачено комиссий: ${total_fee:.2f}")
            print(f"  В среднем на выигрышную сделку: ${fees[fees > 0].mean():.4f}"
                  if (fees > 0).any() else "  Комиссий не было")
            print(f"  Как доля от Total PnL: {abs(total_fee/total_pnl*100):.1f}%"
                  if total_pnl != 0 else "")
        else:
            print("  Нет данных о комиссиях в этом файле.")

        combined_cost = (slip.sum() if len(slip) else 0) + (fees.sum() if len(fees) else 0)
        pnl_without_costs = total_pnl + combined_cost
        print(f"\n  PnL БЕЗ учёта комиссии/проскальзывания был бы: {pnl_without_costs:+.2f}$")
        print(f"  PnL С учётом (реальный): {total_pnl:+.2f}$")
        if pnl_without_costs != 0:
            print(f"  Издержки съели {(1 - total_pnl/pnl_without_costs)*100:.1f}% "
                  f"от 'идеальной' прибыли")
    else:
        print("\n(Детали по проскальзыванию/комиссии недоступны — старый файл, "
              "начни новую сессию новой версией paper_trader.py)")

    # --- 8. Разбивка по времени срабатывания (если есть seconds_to_close_at_entry) ---
    if "seconds_to_close_at_entry" in traded.columns and traded["seconds_to_close_at_entry"].notna().any():
        print("\n" + "-" * 70)
        print("ПО ВРЕМЕНИ СРАБАТЫВАНИЯ СИГНАЛА (секунд до закрытия окна)")
        print("-" * 70)
        buckets = [15, 30, 60, 120, 300]

        def bucket_label(s):
            prev = 0
            for edge in buckets:
                if s <= edge:
                    return f"{prev}-{edge}"
                prev = edge
            return f"{buckets[-1]}+"

        traded["time_bucket"] = traded["seconds_to_close_at_entry"].apply(bucket_label)
        by_time = traded.groupby("time_bucket").agg(
            сделок=(pnl_col, "count"),
            winrate=("won", lambda s: (s == True).mean() * 100),  # noqa: E712
            pnl=(pnl_col, "sum"),
        ).round(2)
        print(by_time.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/paper_trades.csv")
    parser.add_argument("--portfolio", type=str, default="kelly", choices=["kelly", "fixed"])
    parser.add_argument("--top", type=int, default=5, help="Сколько лучших/худших показывать")
    args = parser.parse_args()
    run_kelly_detail(args.csv, args.portfolio, args.top)
