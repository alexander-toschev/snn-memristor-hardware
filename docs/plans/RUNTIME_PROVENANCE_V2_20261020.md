# Локальный аудит: provenance v2, без повторных private receipts

Изменение по прямому указанию пользователя20октября: Git переносит сохранённые
сессии и tracked исходники; повторные ручные SHA двух private MemriCORE файлов
и operator receipts больше не обязательны для локального аудита.

`tools/runtime_provenance.py` разделяет доказательства:

- Все public source/INI, protocol, fixtures и weights проверяются как раньше.
- Исторические изменённые metadata/offline source проверяются по **известному
  локальному Git commit** `b7232c4367096d1c29eeb4d4c0c7e40645a99142`, exact path
  и записанному SHA. Используется read-only `git show`; ни fetch, ни checkout,
  ни hooks/filters не запускаются. Нет нужного Git object — аудит останавливается.
- Для двух явно перечисленных private файлов сохраняются зарегистрированные
  acquisition SHA, но локальный аудит их не читает и не требует remote receipt.
  Статус: `private_runtime_not_independently_reverified`; они **не Git-verified**.
  Это ограничение provenance, не доказательство неизменности private runtime.
- Существующий receipt можно передать явно; он проверяется и сохраняется как
  прежнее наблюдение оператора. Нового наблюдения после каждой партии не требуется.

Изменяются существующие offline entrypoints, общий numerical replay не копируется.
Семь original CIFAR source, resumable Collector/entry/durable, физический протокол,
напряжения, штатный Connector и INI selection не меняются. STDP local-transfer
использует ту же provenance policy; его numerical tail остаётся точной копией
ранее проверенного frozen replay. Pi по-прежнему выполняет штатный acquisition
preflight; эта политика относится к локальному анализу перенесённых результатов.

`tools/runtime_provenance_versions.json` явно регистрирует старые/новые producer
SHA. Старые отчёты и receipts не переписываются. Новые отчёты имеют собственный
actual auditor SHA и `provenance_generation=v2`, никогда не представляются как
результат выполнения прежнего SHA. Старый stage registry/grants сохраняются;
новый registry `tools/cifar_stage_source_registry_v2.json` и grants в
`reports/2026-10-20/runtime-provenance-v2/` создаются отдельно после review.
Текущая аппаратная партия не меняет checkout во время сбора.

Позднее20октября пользователь переключил следующий опыт на stable-cell subset.
Новые v2 registry/HW grants для прежней all256 раскладки **не активируются**.
Этот patch открывает только локальный audit собранных25 stages; они используют
свои неизменные исторические registry/grants. Optional next-plan закрыт без
отдельного нового допуска.25stages не являются полным CIFAR/accuracy.

После переноса **завершённой** сессии (переменные задаются точными repo-relative
путями оператором):

```text
python -B tools/audit_cifar_interrupted_prefix.py --session "$SESSION" --stage-session --portable-condition --output-dir "$NEW_AUDIT_DIR"
python -B tools/review_cifar_batch.py --plan "$PLAN" --stage-report "$REPORT_1" --stage-report "$REPORT_2" --output-dir "$NEW_REVIEW_DIR"
python -B tools/classify_cifar_continuation.py --session "$ORIGINAL_PREFIX_SESSION" --session "$STAGE_SESSION_13" --session "$OTHER_COMPLETED_STAGE_SESSIONS" --portable-condition --output-dir "$NEW_COMPOSITE_DIR"
python -B tools/audit_conductance_stdp_local_transfer.py --session "$COMPLETED_STDP_SESSION" --output "$NEW_STDP_REPORT"
```

В classifier `--session` повторяется отдельно для **каждой** конкретной сессии;
`OTHER_COMPLETED_STAGE_SESSIONS` выше обозначает одну переменную, не список,
разделённый пробелами. Полный classifier требует ровно48 stages; при меньшем
числе данных ошибочно дополненных нулями/CPU нет. Практическую полную команду
оператор составляет после получения всех stage UUID, не заранее выдумывает их.
`FAIL_condition_only`, rank-margin16, heldout, raw/payload/inventory/terminal,
finite budgets, silent/ties и отсутствие dataset accuracy остаются прежними.

Generic review по умолчанию не требует допуска **будущего** hardware запуска;
он проверяет historical grants фактически собранных сессий. Опция next-plan
отдельно требует нового reviewed execution grant. Ни одна команда выше не
запускает плату. Сохранённые v1_SOURCE_FREEZE остаются историей; v2 имеет свой review.
