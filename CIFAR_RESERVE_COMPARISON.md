# Offline-сравнение резервного маршрута A

`compare_cifar_reserve.py` читает сохранённые результаты. Он не импортирует
сборщик, драйвер или аудитор и не вызывает плату либо Git. Чистые функции
`compare_cifar_adaptive.py` загружаются только после проверки закреплённого SHA.
Полный raw replay остаётся отдельным независимым аудитом.

Компаратор принимает только reviewed A protocol и audit SHA
`372f48a05abc652cda0f2c9d5891660a9c23429f53d18ea84749c587f5e10459`.
Проверяются связи audit→manifest→использованные результаты, parent287,
вложенная история42, fixed gain и счётчики без двойного учёта предков.
Маршруты B/C требуют отдельной reviewed схемы; переименование аргументов
не делает их совместимыми с A.

Из корня репозитория после получения обоих независимых аудитов:

```text
python compare_cifar_reserve.py --sim-session results/cifar_feedback_reserve/55134577-a813-4d1b-a3de-67c81ab285ca --sim-audit results/cifar_feedback_reviews/reserve_simulator_20261016/audit.json --hardware-session results/cifar_feedback_reserve/HARDWARE_SESSION_ID --hardware-audit results/cifar_feedback_reviews/HARDWARE_AUDIT_FOLDER/audit.json --output results/cifar_feedback_reviews/reserve_hardware_vs_simulator
```

`HARDWARE_SESSION_ID` и `HARDWARE_AUDIT_FOLDER` — заполнители, не полученные
здесь аппаратные результаты. Выходная папка должна быть новой. Все пути
относительны репозиторию; абсолютные пути и выход за его пределы отклоняются.

Результат — `comparison.json` и `REPORT.md`:

- При полном маршруте: 2400 aligned HW/SIM/trained/own-W_eff выходов с
  per-filter Conv/BN/LIF, различиями спайков и изменчивостью пяти повторов.
- Новые 13 весов и 600 вкладов до BN/LIF отдельно от 287 исторических весов.
  Полный результат содержит 300 весов, но не 300 новых измерений.
- Фактически использованный union адресов после замещения, добавленные и
  вытесненные адреса, fresh quiet/disposition и реальные SET-счётчики резервов.
- При проверенной неудаче route gate: partial-отчёт без полного2400 сравнения.
  Старые исключённые13 не возвращаются в результат. Audit failure отклоняется.

Отсутствующий `acquisition_complete` допустим только в исходной форме
аппаратного failed-manifest при связанном audit `acquisition_complete=false`
и отсутствии summary/completed_microtiles. Успех требует явного `true`.

Тесты запускаются только offline:

```text
python test_compare_cifar_reserve.py
```

15 тестов включают реальные артефакты native SIM A, нарушения hash/gain/
ancestry/coverage и исходную failed-форму manifest без completion flag.
Проверенный SIM-only отчёт:
`results/cifar_feedback_reviews/A_SIM_FINDINGS_20261016/FINDINGS.md`.

Assigned не означает точно записанный вес; audit PASS не является fidelity
PASS. CIFAR100 classification accuracy и абсолютный максимум всех состояний
массива здесь не оцениваются. HW/SIM состояния, история и масштабы различны;
пять повторов не считаются независимыми. Калибровка не подгоняется по CIFAR.
