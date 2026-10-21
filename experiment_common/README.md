# Общие компоненты экспериментов

`experiment.py` и научный порядок этапов остаются в `experiments/<name>/`.
Этот корневой пакет содержит повторно используемые функции, не новый сценарий.

| Модуль | Ответственность |
|---|---|
| `native_memriboard.py` | Публичный Connector: SI-команда/ответ, валидация, R/G, native25-bit MVM и guards |
| `native_output.py` | Сериализация native-ответов и захват вывода драйвера |
| `stdp_collector.py` | Общий конкретный pulse-quantized MNIST/STDP протокол:200ног, initialization/feedback, fixed gain, finite caps |
| `exact_cache.py` | Кеш только точных повторов native ADC/V, поколения и invalidation |
| `mnist_stdp.py` | Fresh initialization и цифровой BindsNET LIF/traces/WeightDependentPostPre; без W@input |
| `mnist_data.py`, `mnist_inputs.py`, `poisson.py` | Официальные IDX и архивный content-seeded Poisson/patch masks |
| `lif.py` | Цифровой LIF100ticks и features из **измеренного** LUT |
| `readout.py`, `schema.py` | Train-only scaler/SGD25, итоговые метрики и типы |
| `mnist_checkpoint.py` | Конкретный сохранённый архивный64-filter checkpoint; не вызывается fresh-STDP опытом |
| `durable.py` | Типизированные atomic JSON, SHA и ограниченный gzip-writer |
| `runtime.py` | Repo-relative импорты, **SIM-only** MNIST INI preflight и измеренная шкала проводимости |

`stdp_collector.py` и MNIST-модули названы по конкретному протоколу, а не
объявлены подходящими для любой сети/платы. Например, native mask содержит25
входов, напряжениеMVM0,05V, readback3повтора. Новые scientific параметры нельзя
незаметно менять при переиспользовании. Публичный `manager.board.Connector`
остаётся единственной границей I/O; обращений к внутреннему драйверу нет.

Block- и timestep-эксперименты используют один shared STDP collector и digital
step. Архивный `native.py` наследует общие чтения/guards, но сохраняет свои
программирование готовых весов и singleton-alpha. Его `inputs/cache/lif/schema`
содержат reexports. Старый `readout.py` имеет короткие forwarding-функции,
чтобы сохранить тестовый/API patch `readout.labels` через явный `label_reader`.
Здесь нет подмены `sys.modules` или временного изменения общих globals.
У CIFAR-resumable старый durable-путь экспортирует shared writer, а его
специфический `simulated_process_loss` остаётся в CIFAR-каталоге.

Не копируем датасет в каждый опыт: существующие IDX лежат в репозиторном
`experiments/mnist_baseline_20261019/data/MNIST/raw/`. Characterization JSON
сохраняется как прежний неизменный артефакт в
`experiments/mnist_archive64_20261020/calibration/`. Это данные, не runtime
импорты чужого эксперимента. Historical reference encoder используется только
как независимый тестовый эталон.

Новые manifests записывают SHA реальных shared исходников через runtime или
центральный calibration preflight. Старые manifests неизменны и относятся
к исходникам соответствующего Git commit; новые алиасы не доказывают равенство
старых source hashes. Старые независимые научные сценарии массово не мигрируем.

Проверка всех shared исходников и трёх затронутых MNIST-маршрутов:

```powershell
python -m mypy --config-file typing/mnist_block_stdp64/mypy.ini
.venv-mnist-pilot/Scripts/python.exe -X utf8 -B -m unittest tests.test_mnist_block_stdp64 tests.test_mnist_online_stdp64 tests.test_mnist_archive64 tests.test_experiment_common
```

Точные stubs описывают только внешние untyped API BindsNET/MemriBoard/root
legacy. Реальные новые `experiment_common/*.py` входят в strict-проверку,
не заменены stub-файлами и не скрыты ignore-правилами.
