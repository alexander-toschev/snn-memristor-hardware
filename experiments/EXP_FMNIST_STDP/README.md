# EXP_FMNIST_STDP

Пользовательское имя эксперимента **MNIST64 block-STDP через native MemriBoard
Simulator**. Датасет остаётся **MNIST**, не Fashion-MNIST: переименована точка
входа, а не данные, модель или протокол.

`experiment.py` — строго типизированная тонкая обёртка над
[существующей реализацией](../mnist_block_stdp64_20261021/experiment.py).
Она передаёт все CLI-аргументы без изменений. Старый путь продолжает работать.
Подробности операций, границы цифрового/мемристорного вычисления, guards и
журналирования: [протокол](../mnist_block_stdp64_20261021/README.md).

Общие Collector, cache, MNIST-входы, LIF/readout и writer выделены в корневой
[`experiment_common/`](../../experiment_common/README.md), а не дублируются
рядом с этим entrypoint. `experiment.py` остаётся на своём месте.

Из корня репозитория:

```powershell
.venv-mnist-pilot/Scripts/python.exe -X utf8 -B experiments/EXP_FMNIST_STDP/experiment.py --help
```

Полный SIM-прогон с выбранным по времени блоком256:

```powershell
.venv-mnist-pilot/Scripts/python.exe -X utf8 -B experiments/EXP_FMNIST_STDP/experiment.py --run-simulator --block-size 256 --wall-hours 7.75
```

Последний полный прогон завершён:66,36%,10000test,22988s. Новый запуск
рефакторингом не выполнялся; исторические результаты сохранены отдельно.
`-X utf8` позволяет вывести русскую справку со стрелками в Windows-консоли.
Переименование не перезапускает его и не меняет исходники процесса.
Результаты по-прежнему сохраняются в
`results/mnist_block_stdp64_20261021/<UUID>/`; исторические manifest/INI/raw
не перемещаются и не переписываются. Обёртка не меняет `BASE` или schema.
Hardware-режима у этого опыта нет.
