# Локальный review любой завершённой партии CIFAR

Уточнение20октября: текущий локальный маршрут использует
[RUNTIME_PROVENANCE_V2_20261020.md](RUNTIME_PROVENANCE_V2_20261020.md).
Новые private receipts не обязательны; v1 source freeze ниже остаётся
историческим. Numerical replay, source/INI/payload/geometry checks сохранены.

`tools/review_cifar_batch.py` заменяет ручную настройку номера партии в новом
review. Исторический `review_cifar_batch_20261020.py` и frozen acquisition/auditor
не меняются. Новый CLI не запускает плату, модель, numerical raw replay или Git.

Сначала root переносит **завершённые** сессии и исходные raw (старые receipts можно сохранить как evidence) через Git
и отдельно выполняет reviewed `audit_cifar_interrupted_prefix.py --stage-session`.
Каждый новый `PREFIX_AUDIT.json` должен уже иметь completed literal stage,
успешный terminal и оба measured payload. Если нужен переносимый численный
режим, он запрашивается явно у auditor с неизменным numerical replay; новый review сохраняет
`FAIL_condition_only` и проверяет утверждённые условия его исключения.
Это не превращается в строгий numerical PASS.

Для партии19…24 после появления шести реальных audit reports:

```text
python -B tools/review_cifar_batch.py --plan reports/2026-10-20/cifar-hardware-batch-13-18-review/HW_BATCH_19_25_PLAN.json --stage-report "$STAGE_19_REPORT" --stage-report "$STAGE_20_REPORT" --stage-report "$STAGE_21_REPORT" --stage-report "$STAGE_22_REPORT" --stage-report "$STAGE_23_REPORT" --stage-report "$STAGE_24_REPORT" --output-dir reports/2026-10-20/cifar-hardware-batch-19-24-review/generic-review
```

Переменные — точные repo-relative пути готовых `PREFIX_AUDIT.json`, заданные
оператором; сам CLI не ищет сессии и не угадывает UUID. На момент подготовки
этой команды партия19…24 ещё выполняется; её реальные PASS-отчёты этим изменением
не создавались и live-файлы не читались.

Проверяются старый prefix из exact plan, все и только новые stages диапазона,
отсутствие повторных session/stage/coefficient, source/grant/receipt bindings,
полный completed result inventory, SHA каждого сжатого raw файла, последний
terminal footer и measured payloads. Старый прерванный original prefix остаётся
отдельным случаем с неизвестной стоимостью отброшенного хвоста; новый completed
stage не может использовать этот путь допуска.

По умолчанию создаются только новые `BATCH_REVIEW.json` и `PREFIX_INDEX.json`.
Выход разрешён только внутри `reports/` или `.tmp/`, чтобы не менять inventory
сохранённых acquisition-сессий. Чтение terminal gzip ограничено8MiB expanded
по размеру chunk прежнего Writer и проверяет gzip CRC/EOF.
Index содержит ссылки на исходные stage certificates; он не выдаёт себя за
результат raw replay или full tensor. Выходная папка обязана отсутствовать.
Ошибка проверки не создаёт принятый prefix или план.

Только явный `--next-stop-stage-exclusive N` дополнительно создаёт `NEXT_PLAN.json`
через прежний reviewed planner/contract и рассчитывает прежние per-stage
impact/pulse bounds. `N` должен быть больше конца принятой партии и не больше48.
План не публикуется и не запускается автоматически; fresh Pi preflight и решение
root обязательны. После принятия всех48 stages новый план не нужен; полная
классификация/сравнение остаются отдельным шагом.

Commit/push status этим review не проверяется. Полное покрытие не означает
правильный CIFAR-ответ и тем более dataset accuracy. Tests используют synthetic
saved evidence и ограниченные mocks источников; новых аппаратных результатов нет.
