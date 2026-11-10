"""
Оконное приложение (Tkinter) для бэктеста стратегии "5-min BTC up/down".

Запуск:
    python gui/app.py
"""
import sys
import threading
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from data.binance_fetch import fetch_klines_cached
from backtest.strategy import StrategyConfig, _CALIBRATED
from backtest.engine import run_backtest


class BacktestApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Polymarket BTC 5-min Bot — Backtester (Free / Local)")
        self.geometry("1100x720")

        self._build_controls()
        self._build_chart()
        self._build_stats()

        if _CALIBRATED:
            n = _CALIBRATED.get("n_windows", "?")
            date = _CALIBRATED.get("calibrated_at", "")[:10]
            self.status_var.set(f"Загружена калибровка из calibrated_params.json "
                                 f"({n} окон, от {date})")

    # ---------------------------------------------------------------- UI --
    def _build_controls(self):
        frame = ttk.LabelFrame(self, text="Параметры")
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        def add_field(label, default, row, col):
            ttk.Label(frame, text=label).grid(row=row, column=col * 2, sticky="w", padx=4, pady=3)
            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(frame, textvariable=var, width=12)
            entry.grid(row=row, column=col * 2 + 1, sticky="w", padx=4, pady=3)
            return var

        # Строка 0
        self.days_var = add_field("Дней истории:", 7, 0, 0)
        self.threshold_var = add_field("Delta threshold ($):", 50, 0, 1)
        self.offset_var = add_field("Вход за N сек до закрытия:", 120, 0, 2)

        # Строка 1
        self.stake_var = add_field("Ставка за сделку ($) [если fixed]:", 10, 1, 0)
        self.bankroll_var = add_field("Начальный банкролл ($):", 300, 1, 1)
        self.price_k_var = add_field("price_model_k:", StrategyConfig().price_model_k, 1, 2)

        # Строка 2
        self.price_power_var = add_field("price_model_power:", StrategyConfig().price_model_power, 2, 0)
        self.kelly_fraction_var = add_field("Kelly fraction (0.5=half):", 0.5, 2, 1)

        # Строка 3 — комиссия и проскальзывание (раньше нигде не учитывались!)
        self.fee_on_win_var = add_field("Комиссия с выигрыша (0-1):", StrategyConfig().fee_on_win, 3, 0)
        self.slippage_var = add_field("Проскальзывание на $1 ставки:",
                                       StrategyConfig().slippage_per_dollar_stake, 3, 1)

        ttk.Label(frame, text="Режим ставки:").grid(row=2, column=4, sticky="w", padx=4, pady=3)
        self.stake_mode_var = tk.StringVar(value="fixed")
        stake_mode_combo = ttk.Combobox(
            frame, textvariable=self.stake_mode_var, state="readonly", width=12,
            values=["fixed", "kelly"],
        )
        stake_mode_combo.grid(row=2, column=5, sticky="w", padx=4, pady=3)

        # Строка 4 — кнопка и статус
        self.run_btn = ttk.Button(frame, text="Запустить бэктест", command=self._run_clicked)
        self.run_btn.grid(row=4, column=0, columnspan=2, pady=8, sticky="w", padx=4)

        self.status_var = tk.StringVar(value="Готов.")
        ttk.Label(frame, textvariable=self.status_var, foreground="#555").grid(
            row=4, column=2, columnspan=4, sticky="w"
        )

    def _build_chart(self):
        self.figure = Figure(figsize=(9, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Кривая эквити (банкролл во времени)")
        self.ax.set_xlabel("Время")
        self.ax.set_ylabel("Банкролл, $")

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

    def _build_stats(self):
        frame = ttk.LabelFrame(self, text="Статистика")
        frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)

        self.stats_labels = {}
        keys = [
            ("total_trades", "Сделок"),
            ("win_rate_pct", "Win rate, %"),
            ("total_pnl", "Total PnL, $"),
            ("total_fees_paid", "Комиссий уплачено, $"),
            ("final_bankroll", "Финальный банкролл, $"),
            ("max_drawdown_pct", "Max drawdown, %"),
        ]
        for i, (key, label) in enumerate(keys):
            ttk.Label(frame, text=label + ":").grid(row=0, column=i * 2, padx=6, pady=4, sticky="e")
            var = tk.StringVar(value="—")
            ttk.Label(frame, textvariable=var, font=("TkDefaultFont", 10, "bold")).grid(
                row=0, column=i * 2 + 1, padx=6, pady=4, sticky="w"
            )
            self.stats_labels[key] = var

        # Вторая строка — сверка "вера модели vs факт" (диагностика Kelly)
        diag_keys = [
            ("believed_win_rate_pct", "Верила в winrate, %"),
            ("actual_win_rate_pct", "Факт winrate, %"),
            ("gap_pct_points", "Расхождение, п.п."),
        ]
        for i, (key, label) in enumerate(diag_keys):
            ttk.Label(frame, text=label + ":", foreground="#666").grid(
                row=1, column=i * 2, padx=6, pady=2, sticky="e")
            var = tk.StringVar(value="—")
            ttk.Label(frame, textvariable=var, foreground="#666").grid(
                row=1, column=i * 2 + 1, padx=6, pady=2, sticky="w")
            self.stats_labels[key] = var

    # ----------------------------------------------------------- actions --
    def _run_clicked(self):
        self.run_btn.config(state="disabled")
        self.status_var.set("Загружаю данные и считаю...")
        threading.Thread(target=self._run_backtest_thread, daemon=True).start()

    def _run_backtest_thread(self):
        try:
            days = int(float(self.days_var.get()))
            stake_mode = "percent_bankroll" if self.stake_mode_var.get() == "kelly" else "fixed"
            cfg = StrategyConfig(
                delta_threshold=float(self.threshold_var.get()),
                entry_offset_seconds=int(float(self.offset_var.get())),
                stake_fixed=float(self.stake_var.get()),
                initial_bankroll=float(self.bankroll_var.get()),
                price_model_k=float(self.price_k_var.get()),
                price_model_power=float(self.price_power_var.get()),
                stake_mode="kelly" if self.stake_mode_var.get() == "kelly" else "fixed",
                kelly_fraction=float(self.kelly_fraction_var.get()),
                fee_on_win=float(self.fee_on_win_var.get()),
                slippage_per_dollar_stake=float(self.slippage_var.get()),
            )

            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)
            df = fetch_klines_cached(symbol="BTCUSDT", interval="1m", start=start, end=end)

            if df.empty:
                self.after(0, lambda: self._on_error("Не удалось загрузить данные с Binance"))
                return

            result = run_backtest(df, cfg)
            self.after(0, lambda: self._on_success(result))

        except Exception as e:
            self.after(0, lambda: self._on_error(str(e)))

    def _on_success(self, result):
        summary = result.summary()
        for key, var in self.stats_labels.items():
            var.set(str(summary.get(key, "—")))

        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)

        if result.equity_curve:
            xs = [t for t, _ in result.equity_curve]
            ys = [b for _, b in result.equity_curve]
            start_bankroll = ys[0]

            ax1.plot(xs, ys, color="#2563eb", linewidth=1.6, zorder=3)
            ax1.axhline(y=start_bankroll, color="#999", linestyle="--", linewidth=0.9,
                        label=f"Старт (${start_bankroll:.0f})")

            # Закрашиваем зону выше/ниже стартового банкролла — сразу видно,
            # где стратегия была в плюсе, а где в просадке
            ys_arr = ys
            ax1.fill_between(xs, ys_arr, start_bankroll,
                              where=[y >= start_bankroll for y in ys_arr],
                              color="#22c55e", alpha=0.12, interpolate=True)
            ax1.fill_between(xs, ys_arr, start_bankroll,
                              where=[y < start_bankroll for y in ys_arr],
                              color="#ef4444", alpha=0.12, interpolate=True)

            # Точки выигрышных / проигрышных сделок поверх кривой
            win_x = [t.window_start for t in result.trades if t.won]
            win_y = [t.bankroll_after for t in result.trades if t.won]
            lose_x = [t.window_start for t in result.trades if not t.won]
            lose_y = [t.bankroll_after for t in result.trades if not t.won]
            ax1.scatter(win_x, win_y, color="#16a34a", s=14, zorder=4, label="Выигрыш")
            ax1.scatter(lose_x, lose_y, color="#dc2626", s=14, zorder=4, label="Проигрыш")

            ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
            ax1.grid(True, linestyle=":", alpha=0.4)
            ax1.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")

            # Гистограмма PnL по сделкам — показывает разброс, не только тренд
            pnls = [t.pnl for t in result.trades]
            colors = ["#16a34a" if p >= 0 else "#dc2626" for p in pnls]
            ax2.bar(range(len(pnls)), pnls, color=colors, width=0.8)
            ax2.axhline(y=0, color="#999", linewidth=0.8)
            ax2.set_ylabel("PnL сделки, $")
            ax2.set_xlabel("№ сделки по порядку")
            ax2.grid(True, linestyle=":", alpha=0.4)

        ax1.set_title("Кривая эквити (банкролл во времени)")
        ax1.set_ylabel("Банкролл, $")
        self.figure.autofmt_xdate()
        self.figure.tight_layout()
        self.canvas.draw()

        self.status_var.set(f"Готово. Сделок: {summary['total_trades']}")
        self.run_btn.config(state="normal")

    def _on_error(self, msg):
        self.status_var.set("Ошибка.")
        self.run_btn.config(state="normal")
        messagebox.showerror("Ошибка бэктеста", msg)


if __name__ == "__main__":
    app = BacktestApp()
    app.mainloop()
