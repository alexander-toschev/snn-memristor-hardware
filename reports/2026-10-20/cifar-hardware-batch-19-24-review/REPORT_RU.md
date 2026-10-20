# Аппаратные CIFAR stages19–24: локальный аудит

Все шесть завершённых сессий независимо воспроизведены из raw новым
зарегистрированным provenance-v2 auditor без private-driver receipts.
Нативные команды, ADC/R, порядок, guards, programming и numerical replay
проверены; оборудование/модель не запускались. Суммарное время шести
последовательных audit subprocesses146,687с; команды/exit/timing находятся
в `AUDIT_EXECUTION.json`, stdout/stderr в `stage-N.log`.

| Stage | Numerical replay | Допуск сразу после записи | Heldout raw RMSE, мВ | Heldout C RMSE, мВ |
|---:|---|---:|---:|---:|
|19|strict PASS|12/128|36,740|27,339|
|20|FAIL_condition_only, portable policy|10/44|34,378|25,921|
|21|strict PASS|11/128|26,235|23,723|
|22|strict PASS|23/128|19,180|17,623|
|23|FAIL_condition_only, portable policy|6/44|33,021|20,742|
|24|strict PASS|15/128|37,184|30,998|

`FAIL_condition_only` не переименован в strict PASS. В соответствующих
`PREFIX_AUDIT.json` сохранены original/local condition diagnostics и
проверка неизменности остальных решений по reviewed excluded-rank-condition
policy. Raw и payload не подменялись.

Для новых шести stages проверено1769316 impacts,8650 programming pulses
и600 коэффициентов. Прямой допуск:
`abs((effective-trained)*.125)<=max(.00061V,.05*abs(trained*.125))`.
Сразу после программирования его выдержали77/600; после записи всех ячеек
63/600 raw и61/600 C; в финальном basis66/600 raw и67/600 C.
Финальный raw имеет366/600 точно нулевых effective weights и32 неправильных
знака ненулевого target; финальный C —346 нулевых и48 неправильных знаков.
Следовательно, numerical-audit acceptance не означает точную запись весов.

Средний по шести heldout RMSE снизился31,123→24,391мВ, улучшение6/6 stages.
Это ошибка аддитивности относительно измеренного singleton reference,
не ошибка относительно желаемых trained weights и не accuracy классификации.
Полные RMSE/relative-L2/допуски/retention и source SHA функции расчёта
содержатся в `SUMMARY.json`.

Generic review принят в `generic-review/BATCH_REVIEW.json`: теперь допустим
непрерывный сохранённый prefix0..24 (25 из48 stages). Его подтверждённые
операции:7441738 impacts и35657 pulses. Это нижняя граница физического итога:
неизвестный потерянный хвост первоначальной оборванной сессии не обнулён.
Полный classifier не выполнен, dataset accuracy не измерена.
`NEXT_PLAN.json` не создавался; продолжение старой all-256 раскладки не запущено.

Public/INI/source/raw проверки сохранены. Для приватного runtime честный
статус `private_runtime_not_independently_reverified`; Git не считается
доказательством байтов untracked драйверов. Исторические receipt-backed
отчёты prefix0..18 остаются неизменными, новые reports используют v2.
Публикацию/commit/push делает основной оператор отдельно от этого аудита.
