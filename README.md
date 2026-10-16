# CIFAR Conv1 — MemriBoard

Выполнен ограниченный аппаратный эксперимент: четыре Conv1-фильтра, 300 обученных коэффициентов, 10 фиксированных настоящих CIFAR-патчей, 12 шагов и 5 повторов. Сумма всех 75 коэффициентов фильтра поступает в обученный BN и цифровой LIF. Это **первый слой, не accuracy полного CIFAR-классификатора**.

[Итог HW/SIM](results/cifar_feedback_reviews/remaining_hardware_vs_simulator_20261016/REPORT.md) · [Карта участвовавших адресов](results/cifar_feedback_reviews/remaining_hardware_vs_simulator_20261016/hardware_contributing_map.svg) · [Независимый аппаратный review](results/cifar_feedback_reviews/remaining_hardware_20261016/INDEPENDENT_REVIEW.md).

Аппаратный аудит: **669378 проверок, 0 ошибок**. Полнота сбора подтверждена, точность записи весов остаётся неудовлетворительной. Из десяти заранее выбранных резервных адресов девять измерены в CIFAR; WL3/BL31 не прошёл ограниченный RESET-screening. Карта показывает объединение участвовавших адресов во времени, а не конечные проводимости или одновременно размещённые 300 весов.

## Единая точка входа

Все команды выполняются из корня репозитория. Без флагов `experiment.py` показывает справку.

| Сценарий | Флаг | Новые коэффициенты | Максимум вызовов / записей |
|---|---|---:|---:|
| [Резерв A](CIFAR_FEEDBACK_RESERVE.md) | `--cifar-feedback-reserve` | 13 | 24323 / 3232 |
| [Резервы B/C](CIFAR_REMAINING_RESERVES.md) | `--cifar-feedback-remaining-reserves` | 20 | 40378 / 5260 |

Preflight без подключения:

```sh
python experiment.py --help
python experiment.py --cifar-feedback-reserve --dry-run
python experiment.py --cifar-feedback-remaining-reserves --dry-run
```

Dry-run проверяет исходники, INI, веса и привязанные результаты родителей без подключения к плате. Частные файлы драйвера должны находиться локально; отсутствие или несовпадение SHA вызывает явную ошибку. Это не проверка проводки.

Новый прогон штатного simulator:

```sh
python experiment.py --cifar-feedback-reserve --run-simulator
python experiment.py --cifar-feedback-remaining-reserves --run-simulator
```

B/C использует фиксированную опубликованную A-сессию своего backend; только что запущенный A не подставляется автоматически. Итог содержит 20 новых B/C + 13 исторических A + 267 исходных коэффициентов. Старые результаты не перезаписываются.

Аппаратный запуск выбирается отдельным `--run-hardware`, изменяет состояния платы и требует намеренного разрешения оператора. [Воспроизведение и окружение GitHub → Raspberry](REPRODUCE.md).

## VS Code и файлы

Первые четыре профиля F5: A preflight, B/C preflight, A simulator, B/C simulator. Автоматического аппаратного запуска нет. Python берётся из `.venv` внутри репозитория; Windows/Linux пути заданы относительно workspace.

- Настройки: `configs/cifar_feedback_reserve.json`, `configs/cifar_feedback_remaining_reserves.json`, `configs/raspberry_board.ini`.
- Веса и патчи: `weights/conv1_weights.npy`, `weights/prepared_cifar100_conv1.json`.
- Результаты: `results/cifar_feedback_reserve/`, `results/cifar_feedback_remaining_reserves/`.
- Аудиторы: `audit_cifar_feedback_reserve.py`, `audit_cifar_feedback_remaining_reserves.py`.

Аппаратные операции выполняются только через публичный MemriBoard Connector и штатные converters. Native-длительность записи неизвестна; 125 модельных шагов simulator не являются измеренными 125 мкс платы. Исходные ADC, ошибки, команды, хеши и родительские результаты сохраняются.

Исторические сценарии: [адаптивное размещение](CIFAR_FEEDBACK_ADAPTIVE.md), [четыре фазы](HARDWARE_PLAN.md), [подготовка MVM](CALIBRATED_MVM_OPERATIONS.md). Их старые карты и пороги не заменяют результат последнего A/B/C.
