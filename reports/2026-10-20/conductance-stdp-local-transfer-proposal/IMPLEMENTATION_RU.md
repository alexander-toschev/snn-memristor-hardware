# STDP: реализация локального переноса, ожидает независимый review

По отдельному поручению root после предложения создан `tools/audit_conductance_stdp_local_transfer.py`. Старые источники acquisition и оба старых аудитора не изменены. Новый файл453+ строк, из них281 — статическая читаемая копия numerical/raw tail. Runtime проверяет SHA старого файла, точное совпадение tail после CRLF→LF и фиксированный SHA этого фрагмента. Никакого динамического выполнения извлечённого текста, monkeypatch или изменения старых проверок нет.

Пять offline-тестов прошли за1,481с: подмены session/start/final/source-map/INI/launcher/qualification bindings, SHA/размера/пути private-файла, третьего private-файла, устаревшего времени наблюдения, evidence и SIM receipt отклоняются. Сохранённый native SIM d27… разобран полностью; все численные JSON-поля точно равны прежнему независимому отчёту, кроме нового SHA самого аудитора. Ошибка первой тестовой попытки касалась лишь сравнения Python integer keys со строковыми JSON keys; после JSON-нормализации сравнение проходит, replay не менялся.

Для hardware новый route сначала вызывает старую неизменённую `launcher_attestation()`. Только provenance допускает свежий root-observed receipt для строго двух разрешённых private-путей. Все прочие local source bytes/INI и полный result inventory проверяются. Receipt должен быть наблюдён после `finished_utc` и связывать точные session/start/final/INI/protocol/source-map/launcher/qualifying-SIM/source-freeze SHA. Его наблюдение не является криптографической удалённой аттестацией и не доказывает независимо историю содержимого файлов во время опыта. Итог не называется старым strict-local-full PASS.

Новый CLI (только после review и загрузки exact raw):

```bash
python -B tools/audit_conductance_stdp_local_transfer.py --session results/conductance_stdp_hardware_20261019/SESSION --private-source-receipt reports/2026-10-20/STDP_SESSION_PRIVATE_RECEIPT.json --output reports/2026-10-20/STDP_SESSION_LOCAL_AUDIT.json
```

`SESSION` и receipt — placeholders, сейчас аппаратной STDP-сессии нет. Сначала старый строгий source/INI preflight на Pi, затем разрешённый root опыт. После terminal outcome root снимает маленькое чтение SHA/stat двух private-файлов и идентичности host/UTC/HEAD; никакого драйвера оно не импортирует. Полный raw/provenance публикуется через LFS; численный replay выполняется на ПК.

Receipt schema: `stdp-private-source-operator-receipt-v1`, `operator=root`, `verification=remote_operator_verified`, `local_private_bytes_available=false`, `cryptographic_remote_attestation=false`; `binding` должен точно совпасть с результатом read-only `receipt_binding(session, manifest)`. `records` — ровно два словаря path/observed_sha256/expected_sha256/size_bytes/resolved_path. Отдельный сохранённый JSON наблюдения содержит `evidence_type=operator_transcription_of_visible_terminal_stdout`, `identity={observed_utc,host,git_head}`, тот же binding и records; его repo-relative путь и SHA указываются в receipt. Root должен действительно наблюдать и сохранить эти факты, а не скопировать их из манифеста. CIFAR receipt не принимается как STDP evidence.

Научные ограничения прежние: это software/logical STDP через measured MVM и physical SET/RESET, не доказательство intrinsic Δt-STDP или улучшения распознавания. No-eligible calibration — допустимый финальный исход. Длительность HW неизвестна. Никаких новых аппаратных обращений в этой подготовке не было.
