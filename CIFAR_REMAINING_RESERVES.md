# B/C: конечный пакет оставшихся резервов

Сценарий `experiment.py --cifar-feedback-remaining-reserves` дополняет
проверенный опыт A двумя заранее фиксированными блоками B/C. Запуск всегда
ручной. Режим без флагов выводит справку; preflight не открывает Connector.
Перед физическим запуском нужны review, чистый commit, native SIM и независимый
аудит его результата. Исходные runtime A и исторические данные не изменены.

| Блок | Новые коэффициенты | RESET-кандидаты | Максимум вызовов | Максимум записывающих импульсов |
|---|---|---:|---:|---:|
| B | filter2 parent0, tap0–10 | P14 / N15 | 21421 | 2794 |
| C | filter1 parent0, tap0–8 | P12 / N15 | 18957 | 2466 |
| Итого | 20 | 56 | 40378 | 5260 |

B добавляет WL5/BL15. При свежем тихом отклике WL4/BL7 и WL5/BL6 занимают tap0,
WL5/BL15 — отрицательную ногу tap1. C добавляет WL3/BL12, BL13, BL31; при свежем
тихом отклике WL2/BL3 занимает положительную ногу tap0, отрицательные BL12/13/31
— tap0/1/2. Все подходящие резервные адреса занимают назначенные слоты до
заполнения остальных исходным порядком. Нетихий резерв сохраняет свой результат;
вместо него допустим другой свежий тихий адрес того же фиксированного пула.

Тихий отклик означает прежние ограничения actual mode10 mean/SD, а не нулевую
проводимость или доказанную программируемость. Пороги, токовые ограничения,
проверки native R/ADC, 55 RESET- и 109 SET-уровней, число повторов и фиксированный
gain HW=1 / SIM=0.03 не изменены. Один проход, без дополнительных адресов и retry.

Недостаток полной ёмкости B останавливает B до SET и сохраняет raw. Единственное
специальное исключение `RouteCapacityError` позволяет выполнить C один раз,
поскольку это другой заранее предусмотренный блок. Если один блок не завершён,
результат не объявляется полным: старый B/C не подставляется. Транспортная ошибка,
некорректный native R/ADC, safety или budget error немедленно останавливают оба.

## Закреплённые родители

- HW: [A e17464e9](results/cifar_feedback_reserve/e17464e9-a601-4b7e-adf5-8a1d56de41d2/manifest.json),
  [независимый аудит](results/cifar_feedback_reviews/reserve_hardware_20261016/audit.json).
- SIM: [A 55134577](results/cifar_feedback_reserve/55134577-a813-4d1b-a3de-67c81ab285ca/manifest.json),
  [независимый аудит](results/cifar_feedback_reviews/reserve_simulator_20261016/audit.json).

Точные SHA обоих manifest/audit заданы в `reserve_remaining_plan.py` и config.
Проверяются identity/backend, все result-файлы, ancestry42 и исторические
счётчики. Выбор parent фиксирован; пользовательский resume или автоматическая
замена другим parent не предусмотрены. Из A исключаются ровно B11+C9, остальные
280 импортируются неизменными, включая новый блок A13. Fresh SIM начинает новый
штатный array; импорт280 является историческим измерением, не восстановлением
непрерывного состояния массива. Это ограничение сохраняется в manifest.

## Измерения и результат

После всех записей каждого блока выполняются свежие readback, anchors и
независимая калибровка C2. Матрица и единицы фиксируются до 600 новых CIFAR
partials. Decoder использует только измеренные строки и фиксированный gain;
подгонка отдельного коэффициента `w/measured` запрещена. После измерений — новые
anchors/readback. Плохая fidelity сохраняется, не скрывает диагностические данные.

Оба завершённых блока дают 1200 новых partials и вместе с parent — ровно 300
обученных taps. Сумма всех 75 taps предшествует обученному BN[t] и одному
цифровому LIF шагу: четыре фильтра, 10 patches, T12, пять повторов, 2400 выходов.
Это последовательные one-hot измерения; не полный classifier accuracy и не
утверждение точной записи всех весов. Реальные origins/imports сохраняются.

Новые назначения могут вытеснить ранее уникально использованный адрес.
Охват пересчитывается по фактически contributing адресам; нельзя прибавлять
число резервов к прошлому охвату. Пакет не доказывает максимум всех 256 ячеек.

## Ручные команды

Зависимости и приватные файлы описаны в [REPRODUCE.md](REPRODUCE.md).
Из корня checkout, команды по одной, foreground:

```sh
git pull --ff-only origin codex/full-cifar-diagnostic
```

```sh
.venv/bin/python -B experiment.py --cifar-feedback-remaining-reserves --dry-run
```

```sh
.venv/bin/python -B experiment.py --cifar-feedback-remaining-reserves --run-simulator
```

После проверки SIM и разрешения основного агента/оператора:

```sh
.venv/bin/python -B experiment.py --cifar-feedback-remaining-reserves --run-hardware
```

На Windows используйте `.venv/Scripts/python.exe`. Dry-run проверяет реальные
parent/source pins и требует приватные hardware-файлы, хотя I/O не выполняет.
Результат: `results/cifar_feedback_remaining_reserves/<uuid>`, отдельные raw
chunks для B/C, frozen route/C/calibration, partials, disposition и manifest.
`screening_complete`, `quiet`, `route_assigned`, `cifar_complete` раздельны:
незавершённое измерение не превращается ни в успех, ни в нетихий адрес.

Offline-проверки без backend:

```sh
python -B -m unittest test_reserve_remaining_plan test_remaining_collector -v
```
