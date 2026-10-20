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
