# MNIST128 с диагональной калибровкой перед LIF

Новый вариант прежнего pooled32→8 conductance pilot. Старые core/SPEC/raw
и исход `uninformative_calibration` аппаратного e5dd5913… сохранены.
Причина изменения: общего gain для четырёх WL не было, индивидуальные
допустимые gains были у всех. Числа из старого fit не переносятся в новый run.

На32 свежих fit-изображениях для каждого WL0–3 независимо выбирается gain
из прежнего grid[1,2,4,8,10,20,40,80,100,200,400,800,1600]. Допуск прежний:
8–64postspikes,≥4активных изображений,≤8posts/image,0posts на пустом input.
Среди допустимых — min(abs(total_posts−32),gain). Контрольный WL4+i получает
тот же gain, что обучаемый WLi. Диагональ фиксируется до preprobe/train;
нет подстройки на postprobe, retention или labels. Нет допустимого gain
у хотя бы одного обучаемого WL — `uninformative_calibration`, без train.

Все реальные команды/guards/preparation/checkpoints наследуются неизменными
из прежнего Collector: public MemriBoard Connector/converters/INI;
MVM.05V,≤4BL; RESET1.4V/SET1.7V; ограничения тока и ADC прежние.
Максимум172288impacts/2560pulses/2GiBraw, начальный диск6GiB/резерв2GiB.
Только128train,32fit,16probe×3=208presentations;128learning/128controlcells.
Это механизм обучения проводимостей, не accuracy и не full10k.
Диагональ масштабирует измеренный ток перед цифровым LIF, не заменяет
физическую операцию CPU-весами. Шум тоже масштабируется.

Локальный nativeSIM (один новый run):

```sh
python -B experiments/mnist_diagonal_20261020/experiment.py --run-simulator
```

На Pi после публикации, команды по одной:

```sh
../../.venv/bin/python -B experiments/mnist_diagonal_20261020/experiment.py --dry-run
```

```sh
../../.venv/bin/python -B experiments/mnist_diagonal_20261020/experiment.py --run-hardware
```

Результаты `results/mnist_diagonal_20261020/UUID/`. HW lock и close/finally
сохранены. SIM initialarray сохраняется отдельно; hardware pulse submitted0,
физическая длительность неизвестна. SIMpulse.0001s/100modelsteps не заявляется
эквивалентом реального времени платы. Private runtime не переаттестуется.

После terminal outcome локальный независимый replay:

```sh
python -B tools/audit_mnist_diagonal.py --session results/mnist_diagonal_20261020/UUID --output reports/2026-10-20/mnist-diagonal/UUID_AUDIT.json
```

Replay наследует прежние проверки native запросов/ответов, guards, ownership,
controls, ADC conversion и LIF/STDP math. Отдельно пересчитывает индивидуальный
выбор из freshfit raw и проверяет применение одной frozen диагонали ко всем
последующим изображениям. Ни старый sharedgain verdict, ни старые результаты
не переписываются. Новый main/reference counterfactual не выполняется;
существующий scalarSIMreference — историческое сравнение, не точный контроль
новой диагональной модели. Парные аппаратные no-update WL остаются в пилоте.
