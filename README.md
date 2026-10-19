# MemriBoard: CIFAR Conv1 и проверка мемристорного тракта

Аппаратные операции выполняются только через публичный MemriBoard Connector и штатные converters. BN и LIF в текущем тракте цифровые. Новые опыты имеют отдельный `experiment.py` в своей папке; корневой `experiment.py` сохраняет исторические сценарии.

**Текущая работа, 19 октября:** [план и состояние выполнения](docs/plans/EXPERIMENTS_20261019.md). Обследование всех 256 ячеек и отдельный MVM/LIF-прогон завершены на плате и native SIM. Готовится полный проход одного изображения CIFAR с фактическими погрешностями; нового аппаратного классификационного результата пока нет.

| Эксперимент | Точка входа / результат |
|---|---|
| Все 256 ячеек: RESET/SET и карта | [experiments/survey256_20261019/experiment.py](experiments/survey256_20261019/experiment.py) · [отчёт и карта](reports/2026-10-19/survey256-hardware-review/REPORT_RU.md) |
| Совместные входы 1–32, матрица C, цифровой LIF | [experiments/parallel_mvm_20261019/experiment.py](experiments/parallel_mvm_20261019/experiment.py) · [HW/SIM и ограничения коррекции](reports/2026-10-19/parallel-mvm-review/HW_REPORT_RU.md) |
| Полное изображение CIFAR, все 64 Conv1-фильтра | [experiments/cifar_full_case0_20261019/experiment.py](experiments/cifar_full_case0_20261019/experiment.py) · [протокол, команды и бюджет](experiments/cifar_full_case0_20261019/README.md); новый маршрут проходит native SIM |
| Архив MNIST: проверка готовности | [experiments/mnist_recovery_20261019/experiment.py](experiments/mnist_recovery_20261019/experiment.py) · [что нужно восстановить](experiments/mnist_recovery_20261019/README.md) |

[Отчёт разработчику: операции HW и SIM, измерения и ограничения](docs/developer-reports/2026-10-17/REPORT_RU.md) · [Команды воспроизведения](docs/developer-reports/2026-10-17/REPRODUCTION.md)

## Предыдущий CIFAR-результат, 16 октября

Ограниченный no-BL8 опыт завершён на HW и native SIM: четыре Conv1-фильтра, 300 коэффициентов, 108 участвовавших физических адресов, 10 настоящих патчей CIFAR-100, T=12 и 5 повторов. HW прошёл прежние проверки качества в **22 из 44 микротайлов**, ошибка весов post-all NRMSE — **16,8571%**. Полнота сбора не означает точность всех ячеек, одновременное размещение всех весов или accuracy полного классификатора.

- [Карты и сравнение до/после исключения BL8](reports/2026-10-16/hardware-stability-comparison/REPORT.md).
- [Причины оставшихся отказов](reports/2026-10-16/hardware-stability-comparison/DIAGNOSTIC_REVIEW.md).
- [Прямое сравнение измеренных HW и SIM выходов](reports/2026-10-16/direct-hardware-simulator/REPORT.md).
- [Границы готовности полного CIFAR](docs/plans/FULL_CIFAR_READINESS_20261016.md).

## Где что находится

| Путь | Назначение |
|---|---|
| `experiment.py` | Исторические сценарии и режимы; без флагов — справка |
| `experiments/*/experiment.py` | Отдельные новые опыты с собственными протоколами и README |
| `configs/` | Зафиксированные протоколы, INI и параметры платы |
| `weights/` | Checkpoint, исходные веса, fixture, карты и их происхождение |
| `results/` | Полные журналы измерений, manifest, независимые аудиты; не переписывать |
| `reports/` | Компактные опубликованные карты и сводки со ссылками/SHA на исходные данные |
| [docs/](docs/README.md) | Инструкции, протоколы, планы и исторические review |
| [tools/](tools/README.md) | Удобный вход в офлайн-анализ и запуск тестов |
| [tests/](tests/README.md) | Unit-тесты, FakeConnector и проверки форматов без платы |
| `experiments/`, `full_cifar_reference/` | Исторический эксперимент и CPU-эталон сети |
| `vendor/MemriBoard/` | Зафиксированный публичный MemriBoard; private-драйверы только локально |
| `.tmp/` | Игнорируемая локальная временная работа, не публикационный каталог |

Некоторые `.py` и три старых теста намеренно остаются в корне: их пути и байты входят в source pins исторических протоколов. [Почему их не переносили](docs/REPOSITORY_LAYOUT.md). Все новые самостоятельные документы и утилиты размещаются в соответствующих папках.

## Получить данные Git LFS

После clone или обновления ветки загрузите настоящие файлы весов и результатов:

```sh
git lfs install --local
git lfs pull
```

Git LFS должен быть установлен в окружении. До запуска проверок убедитесь, что
веса и журналы загружены, а не остались текстовыми LFS-pointer файлами.
`git lfs ls-files` показывает объекты LFS; штатные SHA-проверки протоколов
проверяют содержимое загруженных файлов. После следующего `git pull` повторите
`git lfs pull`, если автоматическая загрузка отключена или была прервана.

## Начать без подключения к плате

Команды выполняются из корня репозитория:

```sh
python -B experiment.py --help
python -B tools/offline.py --help
python -B tools/offline.py compare-traces --help
```

Зависимости и перенос GitHub → Raspberry описаны в [REPRODUCE](docs/guides/REPRODUCE.md). Полный офлайн-набор тестов:

```sh
python -B tools/run_tests.py
```

Для выбранных тестов:

```sh
python -B tools/run_tests.py tests.test_compare_fresh_traces tests.test_export_fresh_trace -v
```

Тесты используют сохранённые данные, CPU и FakeConnector. Native simulator и hardware здесь не запускаются. Для CPU-классификатора нужны дополнительные зависимости из `requirements-classifier.txt`.

## Запуски и воспроизводимость

Сценарии выбираются только явным флагом `experiment.py`; preflight — отдельный `--dry-run`, штатный simulator — `--run-simulator`, плата — `--run-hardware`. Пример и бюджет каждого сценария находятся в [протоколах](docs/README.md#протоколы-ограниченных-опытов). Наличие команды не отменяет проверку source pins, исходных данных, разрешения оператора и срока конкретного протокола. Замороженные сценарии 16 октября имеют исторический cutoff и не являются готовой командой нового опыта в понедельник.

В VS Code первый профиль F5 показывает справку без I/O. Остальные профили сохранены для соответствующих сценариев; никакого автоматического аппаратного запуска нет. `.venv` задаётся относительно workspace для Windows/Linux.

Полные raw и manifest сохраняются без изменения. Компактный отчёт не заменяет полный raw-аудит. Воспроизведение исторического состояния требует записанного в manifest Git commit; перенос документации не меняет acquisition source. Native-длительность аппаратного импульса неизвестна: 125 модельных шагов simulator нельзя называть измеренными 125 мкс платы.

[Правила работы](AGENTS.md) · [Секреты и публикация](SECURITY.md) · [Сторонний код](THIRD_PARTY_NOTICE.md)
