# Шпаргалка команд

Все команды — из корня проекта. На Windows вместо `python` можно писать `py`.
В `cmd` для перехода на другой диск: `cd /d D:\путь\к\polybot`.

## Тесты (без сети)

```bash
python tests/test_core.py
python tests/test_multi.py
python tests/test_ladder_weather.py
```

## BTC 5m (первые эксперименты)

```bash
python logger/polymarket_logger.py                 # -> data/ticks.csv, data/windows.csv
python live/paper_trader.py                        # -> data/paper_v2.csv
python analysis/paper_report.py --csv data/paper_v2.csv
python analysis/edge_report.py --ticks data/ticks.csv --windows data/windows.csv
```

## 25 рынков Up/Down

```bash
python logger/multi_logger.py                      # -> data/multi/
python analysis/backfill_outcomes.py               # докачать исходы
python analysis/multi_report.py
python analysis/maker_sim.py
python analysis/maker_sim.py --since 2026-10-12    # проверка на новых днях
```

## Крипто-лесенки

```bash
python logger/ladder_logger.py                     # -> data/ladder/
python analysis/backfill_event_outcomes.py --dir data/ladder
python analysis/ladder_report.py
```

## Погода

```bash
python logger/weather_logger.py                    # -> data/weather/
python analysis/backfill_event_outcomes.py --dir data/weather
python analysis/weather_report.py
python analysis/weather_strategies.py
python analysis/weather_strategies.py --since 2026-10-21
```

## Сохранить отчёт в файл

```bash
python analysis/weather_report.py > weather_report.txt
```

В PowerShell `>` пишет в UTF-16. Если нужен UTF-8:
`python analysis/weather_report.py | Out-File -Encoding utf8 weather_report.txt`
