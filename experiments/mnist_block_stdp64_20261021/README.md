# Совместимый старый путь EXP_FMNIST_STDP

Фактическая реализация перенесена в
[`experiments/EXP_FMNIST_STDP/experiment.py`](../EXP_FMNIST_STDP/experiment.py),
block-specific операции — в соседний [core.py](../EXP_FMNIST_STDP/core.py).
Общие модули остаются в корневом `experiment_common/`.

Здешние `experiment.py` и `core.py` — только compatibility reexports.
Старый CLI продолжает работать с теми же аргументами и результатами.
Каноническая точка входа не импортирует реализацию из этого каталога.

[Описание протокола, ограничения и команды](../EXP_FMNIST_STDP/README.md).
Исторические данные, manifest и source hashes не переписываются.
