# MNIST: accuracy на замороженной после STDP плате

Отдельный read-only протокол: заново измеряем official train0..127 для
локального readout и official test0..99 для оценки. Не используем изменявшиеся
во время STDP train-отклики или прежний probe16 в качестве test accuracy.
Источники — завершённые diagonal128: HW `a4327210-2cc4-4c19-bb7b-b8820d2e84e4`,
SIM `ee0db984-cd00-4052-95e6-069e71c720eb`.

Ровно 56 336 native impacts, 0 программирующих импульсов, 228 предъявлений.
Start/end checkpoints читают все 256 ячеек. SET/RESET, новая подготовка,
подбор gain и STDP запрещены. Старые gains сохраняются вместе с парными
контролями WL4..7. MVM 0.05 В, до 4 BL одновременно, исходные current/ADC
guards; только public MemriBoard Connector, штатные converters и INI.
SIM копирует **final** `array.cb` родителя в новую сессию; родитель не меняется.
Аппаратное состояние программно не восстанавливается: сравниваем parent→start
и start→end G, не обещаем отсутствия drift/read-disturbance.

Локально: независимый raw replay, WL0..3 raster32 (128 признаков), отдельный
контроль WL4..7. Train-only mean/population std, ridge λ=1, one-hot10,
нештрафуемый bias. Оба readout сохраняются **до** открытия test labels.
Основная strict accuracy: silent raster или top-score tie (abs ≤1e−12)
даёт abstain/ошибку, знаменатель всегда 100. Conventional argmax, coverage,
conditional accuracy, confusion и все predictions сохраняются отдельно.
Wilson95 описывает этот малый фиксированный prefix; это не точность full10k
и не доказательство улучшения относительно обучения без STDP.

Команды из корня репозитория, по одной:

```console
python -B experiments/mnist_accuracy_20261020/experiment.py --dry-run --preflight-source hardware
python -B experiments/mnist_accuracy_20261020/experiment.py --run-hardware
```

На PC (данные MNIST IDX нужны только локальному readout):

```console
python -B experiments/mnist_accuracy_20261020/experiment.py --run-simulator
python -B experiments/mnist_accuracy_20261020/analyze.py --session results/mnist_accuracy_20261020/UUID --output-dir reports/2026-10-20/mnist-accuracy-SOURCE
```

Labels отсутствуют в переносимом fixture; они читаются только `analyze.py`.
Результат acquisition: `results/mnist_accuracy_20261020/UUID`.
Метрики: `ACCURACY.json`, raw replay: `AUDIT.json`, обученные readout и
точные measured features сохраняются отдельно. Оценка времени по предыдущему
аппаратному run — несколько минут; запись raw/публикация добавляют время.
Новых приватных SHA/receipt/grant барьеров нет:
`private_runtime_not_independently_reverified`.
