# MVM при постоянной положительной маске

Read-only диагностика: 4290 вызовов, 0 импульсов SET/RESET, максимум 600 секунд.
Два контекста: BL14+BL16–31 и все 32 BL; WL0/7. Напряжение фона 0,025 В,
на BL14/16/20/30/31 по очереди 0,05 или 0,075 В, три повтора.
В каждом bracket сначала свежий mode7 R всех активных ячеек, затем
baseline → perturbation → baseline с одинаковой положительной маской.
Проверяется полный ток max(requested, native DAC-decoded V)/R и headroom.
Все измерения — публичный MemriBoard Connector, конвертеры и INI.

Heldout-комбинации [16,20], [20,30,31], [14,16,20,30,31] при 0,05 В
проверяют сумму измеренных singleton-приращений. Это проверяемая гипотеза,
а не цифровая подмена MVM. Два уровня сравниваются по фактическому отношению
DAC increments, не по идеальному коэффициенту 2. Relative residual имеет
знаменатель max(abs(predicted), 1 ADC); абсолютная ошибка сохранена отдельно.

Заранее фиксированный инженерный screening: mean singleton > max(5 ADC,5 SD),
ошибка scaled delta и mean heldout residual ≤ max(5 ADC,10% abs(predicted)),
абсолютный drift baseline ≤5 ADC. Это не гарантия MNIST и не аттестация 256 ячеек;
негативный target не запрещает отдельно исследовать другие адреса. Пороги не
подбираются после данных. Частный драйвер не изменяется.

Из корня репозитория:

```powershell
.venv-mnist-pilot/Scripts/python.exe -B experiments/mvm_fixed_mask_diagnostic_20261022/experiment.py --run-simulator
```

На Raspberry Pi (оператор):

```bash
../../.venv/bin/python -B experiments/mvm_fixed_mask_diagnostic_20261022/experiment.py --run-hardware --publish
```

`--dry-run` проверяет конфигурацию без открытия Connector. Результаты находятся в
`results/mvm_fixed_mask_diagnostic_20261022/<UUID>`. Публикация после закрытия порта
сохраняет точный архив, JSON/HTML summary, выполняет secret scan и штатные hooks/LFS.
