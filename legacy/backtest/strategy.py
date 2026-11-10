"""
Параметры стратегии + модель "цены входа".

ВАЖНО (честно): у нас нет бесплатного источника исторических цен токенов
Polymarket (ордербук по каждому 5-минутному рынку за прошлые месяцы просто
не архивируется публично). Поэтому entry_price мы МОДЕЛИРУЕМ по величине
дельты — чем сильнее цена уже сдвинулась к моменту входа, тем ближе цена
токена к $1 (рынок больше уверен в исходе).

Это единственное разумное приближение без реальных данных. Результат
бэктеста на этой модели — оценка ПОРЯДКА ВЕЛИЧИНЫ, не точный прогноз.
Когда дойдём до paper trading на живых данных Polymarket — заменим модель
реальными котировками.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

_CALIBRATED_PATH = Path(__file__).parent / "calibrated_params.json"

# Временные бакеты "сколько секунд остаётся до закрытия окна" — единое место
# истины, используется и здесь, и в calibrate_from_log.py, чтобы границы
# совпадали между калибровкой и использованием.
TIME_BUCKETS = [15, 30, 60, 120, 300]


def time_bucket_label(seconds_remaining: float) -> str:
    prev = 0
    for edge in TIME_BUCKETS:
        if seconds_remaining <= edge:
            return f"{prev}-{edge}"
        prev = edge
    return f"{TIME_BUCKETS[-1]}+"


def _load_calibrated() -> dict | None:
    """Если рядом лежит calibrated_params.json (создан calibrate_from_log.py),
    подхватываем откалиброванные значения автоматически. Иначе — дефолты."""
    if _CALIBRATED_PATH.exists():
        try:
            return json.loads(_CALIBRATED_PATH.read_text())
        except Exception:
            return None
    return None


_CALIBRATED = _load_calibrated()


def _default_k(unit: str = "dollar") -> float:
    if _CALIBRATED and _CALIBRATED.get(unit) and _CALIBRATED[unit].get("price_model_k") is not None:
        return _CALIBRATED[unit]["price_model_k"]
    if _CALIBRATED and unit == "dollar" and "price_model_k" in _CALIBRATED:
        return _CALIBRATED["price_model_k"]  # старый формат JSON (только доллары)
    return 0.08387 if unit == "dollar" else 0.194  # процентный дефолт — грубая оценка


def _default_power(unit: str = "dollar") -> float:
    if _CALIBRATED and _CALIBRATED.get(unit) and _CALIBRATED[unit].get("price_model_power") is not None:
        return _CALIBRATED[unit]["price_model_power"]
    if _CALIBRATED and unit == "dollar" and "price_model_power" in _CALIBRATED:
        return _CALIBRATED["price_model_power"]
    return 0.4425 if unit == "dollar" else 0.42


def _default_win_prob_table(unit: str = "dollar") -> tuple:
    """Возвращает кортеж (threshold, winrate, n) — n нужен, чтобы Kelly знал,
    насколько доверять конкретной ячейке (см. estimate_win_prob)."""
    if _CALIBRATED and _CALIBRATED.get(unit) and _CALIBRATED[unit].get("win_prob_table"):
        return tuple(tuple(row) for row in _CALIBRATED[unit]["win_prob_table"])
    if _CALIBRATED and unit == "dollar" and _CALIBRATED.get("win_prob_table"):
        return tuple(tuple(row) for row in _CALIBRATED["win_prob_table"])
    # Дефолты без калибровки — считаем условно "средней" надёжности (n=200)
    if unit == "dollar":
        return (
            (0.0, 0.823, 200), (10.0, 0.895, 200), (20.0, 0.918, 200),
            (30.0, 0.921, 200), (50.0, 0.921, 200),
        )
    return (
        (0.0, 0.75, 200), (0.02, 0.85, 200), (0.05, 0.90, 200), (0.1, 0.92, 200),
    )


def _default_win_prob_table_by_time(unit: str = "dollar") -> dict:
    """2D-таблица: {временной_бакет: ((threshold, prob, n), ...)}. Пустой
    словарь, если calibrate_from_log.py ещё не считал 2D-версию (старый
    JSON) — тогда estimate_win_prob() откатится на плоскую таблицу."""
    raw = None
    if _CALIBRATED and _CALIBRATED.get("win_prob_table_by_time"):
        raw = _CALIBRATED["win_prob_table_by_time"].get(unit)
    if not raw:
        return {}
    return {
        bucket: tuple(tuple(row) for row in table)
        for bucket, table in raw.items()
    }


def _default_price_model_by_time(unit: str = "dollar") -> dict:
    """2D price_model: {временной_бакет: (k, power)}. Пустой словарь, если
    calibrate_from_log.py ещё не считал 2D-версию — implied_entry_price()
    откатится на плоскую (одну на все времена) кривую."""
    raw = None
    if _CALIBRATED and _CALIBRATED.get("price_model_by_time"):
        raw = _CALIBRATED["price_model_by_time"].get(unit)
    if not raw:
        return {}
    return {
        bucket: (info["k"], info["power"])
        for bucket, info in raw.items()
    }


@dataclass
class StrategyConfig:
    # За сколько секунд до закрытия окна принимаем решение
    entry_offset_seconds: int = 120

    # В каких единицах измеряется сигнал/threshold:
    # "dollar"  — абсолютный сдвиг цены BTC в $ (как раньше)
    # "percent" — сдвиг в % от цены на момент открытия окна (устойчивее
    #             к изменению уровня цены BTC со временем — см. обсуждение
    #             гиста Archetapp, там сигнал именно в %)
    delta_unit: str = "dollar"

    # Минимальный сдвиг, чтобы вообще входить в сделку.
    # В единицах delta_unit: доллары (напр. 15) или проценты (напр. 0.05 = 0.05%)
    delta_threshold: float = 50.0

    # Модель цены токена: entry_price = clamp(0.5 + k * x ** power, min, max)
    # где x — дельта в единицах delta_unit. Дефолты берутся из
    # calibrated_params.json (если создан calibrate_from_log.py — там теперь
    # сразу считаются ОБА варианта, для $ и для %).
    price_model_k: float = field(default_factory=lambda: _default_k("dollar"))
    price_model_power: float = field(default_factory=lambda: _default_power("dollar"))
    price_model_k_pct: float = field(default_factory=lambda: _default_k("percent"))
    price_model_power_pct: float = field(default_factory=lambda: _default_power("percent"))
    price_model_min: float = 0.5
    price_model_max: float = 0.999

    # Размер ставки
    stake_mode: str = "fixed"          # "fixed" или "kelly"
    stake_fixed: float = 10.0          # $ за сделку, если fixed

    # --- Kelly criterion ---
    kelly_fraction: float = 0.5

    win_prob_table: tuple = field(default_factory=lambda: _default_win_prob_table("dollar"))
    win_prob_table_pct: tuple = field(default_factory=lambda: _default_win_prob_table("percent"))
    win_prob_table_by_time: dict = field(default_factory=lambda: _default_win_prob_table_by_time("dollar"))
    win_prob_table_by_time_pct: dict = field(default_factory=lambda: _default_win_prob_table_by_time("percent"))
    price_model_by_time: dict = field(default_factory=lambda: _default_price_model_by_time("dollar"))
    price_model_by_time_pct: dict = field(default_factory=lambda: _default_price_model_by_time("percent"))
    kelly_prob_cap: float = 0.94
    kelly_max_stake_fraction: float = 0.05
    # z для нижней границы Уилсона: 1.96 = 95% доверие (стандартный выбор).
    # Больше z -> Kelly консервативнее относится к маленьким выборкам.
    # z=0 полностью отключает поправку (использует сырую точечную оценку,
    # как было раньше).
    kelly_confidence_z: float = 1.96

    initial_bankroll: float = 300.0

    # --- Комиссия и проскальзывание — раньше НЕ учитывались нигде! ---
    #
    # Комиссия: по нескольким независимым источникам за 2026 год у Polymarket
    # ТЕПЕРЬ есть комиссия (раньше её не было) — цифры расходятся в деталях
    # (то ли 2% с выигрышного payout, то ли taker fee по категориям, для
    # Crypto около 1.8%), поэтому берём консервативную оценку 2% с выигрыша
    # как дефолт. ПРОВЕРЬ актуальную цифру на самом Polymarket перед тем как
    # доверять итоговым цифрам буквально — сюда легко может закрасться
    # неточность из-за противоречивых источников.
    fee_on_win: float = 0.02

    # Проскальзывание: не фиксированное число, а РАСТЁТ вместе с размером
    # ставки относительно банкролла — грубая имитация того, что крупный
    # ордер сам "проедает" стакан и получает среднюю цену хуже котируемой
    # (см. пример с рассчётом в чате: покупка $40 при глубине $10 на лучшей
    # цене даёт среднюю \$0.787 вместо \$0.75). Это ГРУБАЯ эвристика, не
    # измерение реальной глубины стакана (у нас нет надёжного /book) —
    # только чтобы не притворяться, что крупные Kelly-ставки исполняются
    # по идеальной котируемой цене.
    slippage_per_dollar_stake: float = 0.0008  # +0.08 центов цены на каждый $1 ставки
    slippage: float = 0.0  # оставлено для обратной совместимости/ручной надбавки

    def active_price_model_params(self) -> tuple[float, float]:
        """Возвращает (k, power) для ТЕКУЩЕГО delta_unit."""
        if self.delta_unit == "percent":
            return self.price_model_k_pct, self.price_model_power_pct
        return self.price_model_k, self.price_model_power

    def active_price_model_by_time(self) -> dict:
        if self.delta_unit == "percent":
            return self.price_model_by_time_pct
        return self.price_model_by_time

    def active_win_prob_table(self) -> tuple:
        if self.delta_unit == "percent":
            return self.win_prob_table_pct
        return self.win_prob_table

    def active_win_prob_table_by_time(self) -> dict:
        if self.delta_unit == "percent":
            return self.win_prob_table_by_time_pct
        return self.win_prob_table_by_time


def implied_entry_price(x: float, cfg: StrategyConfig, seconds_remaining: float | None = None,
                         stake: float | None = None) -> float:
    """x — дельта в единицах cfg.delta_unit -> смоделированная цена токена (0..1).

    Если передан seconds_remaining — используем 2D-кривую (своя пара k/power
    для этого временного бакета), если она есть в калибровке. Иначе (или
    если для бакета нет данных) — откатываемся на плоскую (одну на все
    времена) кривую, как раньше.

    Если передан stake — добавляем проскальзывание, растущее вместе с
    размером ставки (см. StrategyConfig.slippage_per_dollar_stake).
    """
    k, power = None, None
    if seconds_remaining is not None:
        by_time = cfg.active_price_model_by_time()
        if by_time:
            bucket = time_bucket_label(seconds_remaining)
            pair = by_time.get(bucket)
            if pair:
                k, power = pair

    if k is None:
        k, power = cfg.active_price_model_params()

    raw = 0.5 + k * (x ** power)
    price = max(cfg.price_model_min, min(cfg.price_model_max, raw))

    dynamic_slippage = (stake * cfg.slippage_per_dollar_stake) if stake else 0.0
    price = price + cfg.slippage + dynamic_slippage
    return min(price, 0.999)  # цена токена физически не может быть >= $1


def apply_win_fee(payout: float, cfg: StrategyConfig) -> float:
    """Применяет комиссию Polymarket к выигрышному payout. НЕ применяется
    к проигрышам (комиссия берётся только с выигрыша, по всем найденным
    источникам)."""
    return payout * (1.0 - cfg.fee_on_win)


def wilson_lower_bound(p_hat: float, n: int, z: float = 1.96) -> float:
    """Нижняя граница доверительного интервала Уилсона для доли p_hat,
    посчитанной на выборке размера n.

    Смысл простыми словами: "насколько маленькое значение winrate я бы
    всё ещё считал правдоподобным, учитывая размер выборки". При большом n
    граница почти совпадает с p_hat (мы уверены в оценке). При маленьком n
    граница сильно съезжает к 50% (мы НЕ уверены, поэтому подстраховываемся
    и считаем эдж более скромным, чем сырая цифра говорит).

    Это и есть ответ на "чтобы Kelly различал эджи по степени доверия",
    а не слепо верил любой цифре из таблицы, даже посчитанной на 9 сделках.
    """
    if n <= 0:
        return 0.5
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    margin = (z * ((p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) ** 0.5)) / denom
    return max(0.0, center - margin)


def estimate_win_prob(x: float, cfg: StrategyConfig, seconds_remaining: float | None = None) -> float:
    """x — дельта в единицах cfg.delta_unit. Оценка истинной вероятности
    выигрыша для Kelly — НЕ сырая цифра из таблицы, а её консервативная
    нижняя граница (Уилсон), учитывающая размер выборки за этой ячейкой.

    Если передан seconds_remaining — используем 2D-таблицу (дельта + время
    до закрытия), если она есть в калибровке. Иначе (или если для этого
    временного бакета нет данных) — откатываемся на плоскую таблицу.
    """
    table = None
    if seconds_remaining is not None:
        by_time = cfg.active_win_prob_table_by_time()
        if by_time:
            bucket = time_bucket_label(seconds_remaining)
            table = by_time.get(bucket)

    if not table:
        table = cfg.active_win_prob_table()

    p_hat, n = table[0][1], table[0][2] if len(table[0]) > 2 else 200
    for row in table:
        threshold = row[0]
        if x >= threshold:
            p_hat = row[1]
            n = row[2] if len(row) > 2 else 200

    p_conservative = wilson_lower_bound(p_hat, n, z=cfg.kelly_confidence_z)
    return min(p_conservative, cfg.kelly_prob_cap)


def kelly_stake(entry_price: float, win_prob: float, bankroll: float, cfg: StrategyConfig) -> float:
    """Half/full Kelly размер ставки в $.

    f* = (b*p - q) / b,  b = payout за $1 при выигрыше = 1/entry_price - 1
    """
    if entry_price <= 0 or entry_price >= 1:
        return 0.0
    b = 1.0 / entry_price - 1.0
    q = 1.0 - win_prob
    f_full = (b * win_prob - q) / b if b > 0 else 0.0
    f_full = max(0.0, f_full)  # никогда не ставим отрицательную долю
    f = f_full * cfg.kelly_fraction
    f = min(f, cfg.kelly_max_stake_fraction)  # страховочный потолок
    return bankroll * f
