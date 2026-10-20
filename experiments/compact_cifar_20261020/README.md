# Новый компактный CIFAR: только CPU training/readout

RGB48→8 LIF→10, отдельный checkpoint; это не прежняя64-filter сеть.
Один seed,4500 train/500 validation,50 epochs Adam, selected epoch по
validation CE. Pilot100 фиксируется до обучения, test forward — только
после сохранения выбранного checkpoint. Все файлы внутри репозитория.

Из корня:

```bash
python -B experiments/compact_cifar_20261020/experiment.py --run
```

Используются `.deps` и `.deps-classifier`, CPU float32,2threads.
Результат — `results/compact_cifar_20261020/<UUID>/`: checkpoint, W/A/b,
все epoch/permutation/RNG данные, input freeze, отдельные acquisition
fixture без labels и evaluation labels, `CPU_PILOT100.json`.

CPU accuracy явно программная. На100 test приводятся confusion, silence,
ties и informative coverage; тихий bias-only ответ не доказывает полезный
MVM. Физический gain и mapping устанавливаются по фактической stable map
перед аппаратной записью; здесь они не выдумываются. MemriBoard этот код
не импортирует и не вызывает.

Физический executor — отдельный `physical_experiment.py`. После завершения
карты, из корня репозитория, команды выполняются по одной:

```bash
python -B experiments/compact_cifar_20261020/physical_experiment.py --prepare-plan --cpu-session results/compact_cifar_20261020/1aa6dcc7-b729-450e-8f95-beb6344bc075 --map-session results/stable_cell_calibration_20261020/MAP_UUID --output results/compact_cifar_20261020/physical_plan.json
```

На Pi оператор запускает опубликованный план:

```bash
../../.venv/bin/python -B experiments/compact_cifar_20261020/physical_experiment.py --run-hardware --plan results/compact_cifar_20261020/physical_plan.json
```

После переноса полного измеренного сеанса на ПК:

```bash
python -B experiments/compact_cifar_20261020/physical_experiment.py --classify --session results/compact_cifar_physical_20261020/PHYSICAL_UUID --output results/compact_cifar_20261020/PHYSICAL_REPORT.json
```

`MAP_UUID`/`PHYSICAL_UUID` заменяются фактическими UUID. Prepare не вызывает
плату. Plan фиксирует все384 коэффициента, gain из измеренных обеих знаковых
ориентаций карты, точный inference budget и максимум9216 programming pulses.
В каждом tile заново измеряется basis/C/heldout; каждые20 изображений —
retention basis. Raw/C currents суммируются по всем48 входам до единственного
LIF update. Missing не заменяется CPU; частичный сеанс не классифицируется.
NativeSIM запускается тем же entry с `--run-simulator` и планом source=simulator;
план нельзя выдавать за SIM-qualified, если соответствующая SIM-карта не получена.
