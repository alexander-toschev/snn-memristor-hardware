# Свежая карта стабильных ячеек

Один конечный процесс: A измеряет все256 адресов и повторяемость R/M/H;
выбор фиксируется до B; B проверяет обе ориентации знаковых пар; C обучается
на отдельных fit/selection масках, замораживается до heldout. Все физические
операции — public MemriBoard Connector, штатные converters и INI.
Калибровка выполняется на Raspberry; readout и accuracy — позже на ПК.

Из корня репозитория, по одной команде:

```bash
python -B experiments/stable_cell_calibration_20261020/experiment.py --dry-run --preflight-source hardware
```

```bash
python -B experiments/stable_cell_calibration_20261020/experiment.py --run-simulator
```

После программного сравнения оператор запускает плату:

```bash
../../.venv/bin/python -B experiments/stable_cell_calibration_20261020/experiment.py --run-hardware
```

Последняя команда рассчитана на текущий Pi checkout внутри
`Documents/snn-memristor-hardware/.tmp/survey-20261019`; в другом checkout
используется его Python с зависимостями MemriBoard. Драйвер задаёт
`configs/raspberry_board.ini`. Никаких фоновых запусков или detach.

Максимум411536 impacts,20480 pulses,600с явных ожиданий. Soft deadline3600с
останавливает новые команды, но не прерывает уже заблокированный драйвер.
Current/ADC/nativeR guards сохраняются; skipped измерение не становится нулём.
Прерванный процесс сохраняет partial raw/map; продолжение с середины
не предусмотрено. SET/RESET меняют физические состояния.

Выход: `results/stable_cell_calibration_20261020/<UUID>/` с `map.json`,
`observations.json`, `C_*.json`, полным chunked gzip raw, исходным INI,
provenance и итоговым `manifest.json`. SHA публичных файлов записываются,
ручного source approval нет. Приватные драйверы отдельно не проверяются:
`private_runtime_not_independently_reverified`. Перед Git обязателен secret scan.

Карта различает quality-qualified и `diagnostic_best_available`: заранее
фиксированный fallback выбирает одну пару и до8 лучших общих BL, если
строгий quality rectangle отсутствует. Mandatory valid/span/sign failure
останавливает следующий этап; soft noise/drift quality failures не вызывают
нового подбора, дополнительных импульсов или скрытого расширения порогов.
Калибровка C относится к состоянию checkerboard при0,05В/до4 активных BL;
после записи CIFAR-весов нужны свежие basis/C/heldout. Это ещё не accuracy.
