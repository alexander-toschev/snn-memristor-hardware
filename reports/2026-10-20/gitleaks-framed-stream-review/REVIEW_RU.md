# Исправление полного сканирования binary stdin, 20 октября 2026

Gitleaks8.30.1 распознаёт gzip в начале stdin как `application/gzip`, пишет
`skipping binary file` и возвращает0 без проверки содержимого. Явный Python
writer после этого получает BrokenPipe; `subprocess.run(input=...)` может
скрыть раннее закрытие. Источник: официальный
[sources/file.go v8.30.1](https://raw.githubusercontent.com/gitleaks/gitleaks/v8.30.1/sources/file.go),
`fileFragments`, первая MIME-проверка при `totalLines == 0`.

Root воспроизвёл это на Pi для raw713598байт; decoded поток проходил сканер.
Локально воспроизведено на другом неизменённом gzip375879байт: binary skip,
exit0, BrokenPipe после32768 успешно flushedбайт. Это ошибка полноты
сканирования, не обнаруженная утечка. Прежний raw-scanned/exit0 сам по себе
не является доказательством сканирования бинарных payload. Исторические
отчёты не переписываются; текущая публикация проходит новый полный scan.

## Изменение

Общий `tools/security/scan_staged_lfs.py:scan_chunks` добавляет фиксированный
ASCII/LF frame длиной не менее1024байт только в scanner stdin, затем передаёт
каждый исходный байт без изменений. Это предотвращает MIME binary skip,
сохраняя стандартные правила Gitleaks, запрет allow-комментариев, отсутствие
size exclusions и изоляцию config. Framing не записывается в измеренные
файлы, LFS объекты или Git blobs и не меняет их SHA.

Staged-LFS hook и `tools/run_cifar_batch.py` используют этот общий streaming
helper. Wrapper отдельно сканирует исходный gzip и все decoded bytes; CRC,
обработка нескольких gzip members и bounded decompression прежние. Метаданные
содержат payload_bytes/framing_bytes/scanner_input_bytes. Любой BrokenPipe,
short write, read error, ненулевой exit либо timeout блокирует публикацию;
никакого игнорирования ошибок или принятия exit0 после write failure нет.
HOOKS pin общего helper обновлён явно. Другие hooks, source/grant/plan checks,
baseHEAD binding, allowlist и методы acquisition не изменены.

## Проверка

39/39 тестов PASS13.163s с фактическим Gitleaks8.30.1. Синтетические
нерабочие canaries в начале/середине/конце raw и decoded данных обнаружены;
проверены binary magic, NUL, большой хвост, точная передача всех байтов,
BrokenPipe при exit0, short write, ошибка источника и timeout. Старый тест
12MiB binary-prefix тоже PASS. IsolatedGit/LFS tests подтверждают прежние
commit/retry/guard ограничения без изменений рабочего репозитория и сети.
Первый sandbox-run не смог запустить Git Bash/LFS; сохранён отдельно.
Повтор тех же тестов с разрешённым внешним shell прошёл полностью.
SHA и точные логи: `AUTHOR_REVIEW.json`, `TESTS_UNSANDBOXED.log`.
Независимый reviewer даёт отдельный вердикт; этот текст — авторская проверка.

## Возобновление публикации без аппаратного повтора

Root сообщил: batch `c93d6fde-a709-4ec1-935d-2612c1ec8696` завершён
2026-10-20T11:42:30.179528+00:00, childexit0, attribution true, hardware lock
отсутствует; все stages19–24 completed,1769316 impacts/8650 pulses.
`PUBLICATION.json` не создан. Это операторские сведения, не semantic audit.

Порядок: root публикует только reviewed scanner fix перед provenance changes;
Pi после завершения child обновляется fast-forward. Затем существующий
`tools/run_cifar_batch.py preserve --source hardware --plan` с прежним
`reports/2026-10-20/cifar-hardware-batch-13-18-review/HW_BATCH_19_25_PLAN.json`,
шестью точными `--session` из старого `acquisition_exit.json` и `--push`
создаёт новый evidence batch, проверяет final inventories и публикует данные.
`preserve` не запускает acquisition. Новый baseHEAD относится к новому batch;
старый baseHEAD check не ослабляется. Старые четыре файла batch сохраняются
неизменёнными и публикуются root отдельным exact-allowlist commit после нового
полного scan. Старый batch не переписывается, raw не перепаковывается.
Дальше локальный semantic audit каждой сессии; успешная публикация не PASS
эксперимента. Новый publisher, обход hooks и физический повтор не требуются.
