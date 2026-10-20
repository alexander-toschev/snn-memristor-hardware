# CIFAR100 subset10: четыре clean checkpoint воспроизведены

Повторный **чистый PyTorch eval** на 1000 official test примерах для каждого
seed завершён. Все восемь исторических FS/TWD чисел совпали точно по числу
правильных ответов. Обучения, платы и native MemriBoard simulator не было.

| Seed | Архивный / повторный FS | Архивный / повторный TWD | Strict FS | Strict TWD | Silent |
|---|---:|---:|---:|---:|---:|
| 0 | 72.5% / 72.5% | 76.4% / 76.4% | 71.9% | 76.0% | 88 |
| 1 | 73.9% / 73.9% | 77.0% / 77.0% | 73.2% | 76.4% | 67 |
| 2 | 74.1% / 74.1% | 76.7% / 76.7% | 73.8% | 76.5% | 100 |
| 42 | 75.4% / 75.4% | 78.2% / 78.2% | 74.9% | 77.7% | 88 |

Исторические метрики используют conventional argmax и приведены для точного
сопоставления. В strict варианте silent output или tie даёт abstain/ошибку,
знаменатель остаётся 1000. Поэтому архивные значения нельзя интерпретировать
как accuracy только уверенных спайковых решений. Silent входит в tied, эти
счётчики нельзя складывать. Полные tie counts, predictions, output spikes и
TWD scores сохранены по каждому seed.

Модель: RGB Conv64/128/256→FC512→10, T12, LIF tau0.5/threshold1, BNTT,
use_twd=true и training twd_weight0.1, memristor_sigma0. `TWD=1` означает
включение decoder, не коэффициент его loss равный1. Checkpoint каждого seed
— исторический best по validation FS, новый выбор/подгонка не выполнялись.
Классы fine-label `[0,1,2,3,5,8,13,14,17,19]`, original test order.

Суммарное время четырёх eval — 124.30 с (CPU, 2 threads, batch16);
загрузка Python/Torch и dataset добавляет время. Результаты:
`results/cifar_confirmatory_20261020/clean-four-seeds-20261020/`.
Веса и runtime source копии лежат внутри репозитория. SHA checkpoint
совпали с историческими summary; все необходимые test bytes также проверены.

```console
python -B experiments/cifar_confirmatory_20261020/evaluate.py --seeds 0 1 2 42 --output-dir results/cifar_confirmatory_20261020/new-clean-reproduction
```

Эти результаты **не подтверждают** literalMVM через native simulator или
аппаратную accuracy. Отдельный nativeSIM programming/readback hybrid тоже
ещё не запускался и не должен называться полноценным аналоговым MVM.
