# Конечная партия CIFAR и сохранение результатов

Новая обёртка `tools/run_cifar_batch.py` не меняет сбор, физический протокол
или frozen источники. До независимого review это подготовленный код, а не
разрешение аппаратного запуска. Плату запускает основной оператор на Pi.

После review и публикации кода команда из корня checkout:

```text
python3 -B tools/run_cifar_batch.py run --source hardware --plan reports/2026-10-20/cifar-hardware-batch-13-18-review/HW_BATCH_19_25_PLAN.json --push
```

План задаёт конечные stages19…24. Обёртка вызывает опубликованный
`experiment.py` foreground, без detach и без повторов. Точный поток stdout/stderr
сохраняется в `results/cifar_batch_preservation/<UUID>/console.log` через LFS;
в консоль идут короткие статусы примерно раз в5мин. Native per-impact журнал
по-прежнему создаёт сам frozen collector в каждой сессии.

После выхода дочернего процесса проверяются attribution всех новых UUID,
start/final inventory и отсутствие аппаратного lock. Отсутствующий final
допускается только для подтверждённо завершённого дочернего процесса этой
обёртки; это partial, а не completed. Неясная принадлежность, чужие изменения,
оставшийся lock или неоднозначный индекс оставляют данные локально для разбора.
Lock чужого аппаратного процесса обёртка не удаляет.

Сканирование секретов включает исходные байты каждого файла и все декодируемые
gzip members. Truncated gzip сохраняется как truncated; повреждение, которое
мешает проверить доступный поток целиком, блокирует публикацию. Никакой
исправленный или выдуманный footer не добавляется. Gitleaks работает без
filename/size/inline-comment исключений; вывод сканера и Git не печатается.

Git получает только точный список файлов этих сессий и собственного batch.
Проверяются LFS pointers, локальные байты, штатные `.githooks` и результат hooks.
Публикуется обычный commit, затем необязательный push строго в
`alexander-toschev/snn-memristor-hardware`,
`refs/heads/codex/full-cifar-diagnostic`. Auth использует только уже настроенный
credential cache; интерактивный ввод и чтение секретных файлов отсутствуют.
Каждая Git-команда явно включает штатный `core.hooksPath=.githooks` и
`gc.auto=0`; network-команды используют только `cache --timeout=43200`.
Это включает reviewed hooks даже на Pi без локальной настройки hooksPath.
При неуспешном push commit остаётся локально. Успешный pre-push/LFS upload
и remote ref проверяются; отдельного скачивания LFS объектов для readback нет.

Повтор публикации: скопировать поле `batch` из статуса в переменную `BATCH`,
затем выполнить только:

```text
python3 -B tools/run_cifar_batch.py finalize --batch "$BATCH" --push
```

`finalize` не вызывает acquisition. Повтор после собственного commit отправляет
тот же commit; другой HEAD или изменение evidence блокируют операцию.
Для уже завершённых сессий вне wrapper есть `preserve --source hardware --plan …`
с отдельным `--session` для каждого точного UUID. Этот режим требует реальные
финальные manifests и также не запускает эксперимент.

Статусы `acquisition`, `preservation`, `push` раздельны. Даже exit0,
committed и pushed **не означают semantic audit PASS**. Строгий raw replay,
составление48 stages, классификация и сравнение сSIM выполняются отдельно локально.
При разрыве всей терминальной сессии автосохранение не обещается:
повторная связь нужна для явного разбора/finalize; физический stage автоматически
не повторяется. Unit-тесты используют synthetic files/mocked Git. Отдельный
integration-тест создаёт временный Git-репозиторий и проверяет настоящий
commit/LFS pointer, отказ synthetic hook и успешный retry; scanner и hooks
в этом тесте — заглушки. Production Gitleaks, remote upload, nativeSIM и плата
этими тестами не квалифицируются.
