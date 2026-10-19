# Завершение текущих прогонов 19 октября

**Активный маршрут — resumable whole-stage CIFAR.** Исходный полный процесс
прерван, его неизменные raw опубликованы коммитом `11290a9`. Команды штатного
completed-run ниже сохранены как исторические; они не подходят к interrupted
сессии без final manifest. Новый маршрут:
[cifar_resumable_20261019](../../experiments/cifar_resumable_20261019/README.md).
Только независимо проверенные целые тайлы могут использоваться повторно;
первый незавершённый тайл получает полную новую запись, R и C. Checkpoints
каждые16позиций сохраняют свидетельства, не разрешая смешивать состояния
внутри тайла. До аппаратного продолжения нужны source review и nativeSIM
квалификация abrupt-process-loss с полным повтором прерванного тайла.
Все команды foreground, без detach/автоматических повторов импульсов.
Pi — только MemriBoard acquisition; тяжёлый raw audit/classifier — локально.

**Инцидент16:30MSK:** аппаратная CIFAR-сессия `f21e608c…` прервана:
процесс отсутствует, final manifest отсутствует, lock остался, последний gzip
оборван. Ниже сохранён порядок для штатного завершения, но сейчас нельзя
выполнять его как completed-процедуру или переходить к STDP. Сначала сохранить
неизменные bytes и отдельный inventory; lock не удалять, запись не повторять.
[Факты и проект продолжения](../../reports/2026-10-19/cifar-full-case0-review/HARDWARE_INTERRUPTION_20261019_RU.md).

Команды выполняются **по одной, foreground**, с проверкой результата перед
следующей. Никаких detach, фонового автопродолжения, токенов в командах или
изменения выполняющихся исходников. Реальное I/O — только MemriBoard.
Уточнение пользователя: Raspberry максимально оставляем для физических
мемристорных операций и записи сырых данных через public MemriBoard.
Тяжёлые raw replay/classifier, обработка признаков, обучение head и сравнение
результатов выполняются локально. Все аппаратные запуски координирует root.

## Raspberry: закончить CIFAR, сохранить неизменные raw и provenance

Локальная ветка Pi `codex/survey-20261019`, стартовый HEAD `c67526b`, PID18068.
Единственная remote-ветка публикации — `codex/full-cifar-diagnostic`.
Имя локальной ветки Pi или локального publication worktree не задаёт remote.

```bash
cd ~/Documents/snn-memristor-hardware/.tmp/survey-20261019
```

```bash
ps -p 18068 -o pid,stat,etime,args
```

Пока показан процесс CIFAR — ждать его завершения; не обновлять checkout и
не запускать другой аппаратный опыт. После возврата приглашения shell:

```bash
../../.venv/bin/python -B -c 'import json; from pathlib import Path; p=Path("results/cifar_full_case0_20261019/f21e608c-059f-4dce-afe0-6edf5c26133c/manifest.json"); m=json.loads(p.read_text()); print({k:m.get(k) for k in ("status","source","scope","impacts","pulses")}); assert m["status"]=="completed" and m["source"]=="hardware" and m["scope"]=="full"; actual={f.relative_to(p.parent).as_posix() for f in p.parent.rglob("*") if f.is_file()}; assert actual==set(m["result_sha256"])|{"manifest.json"}; print("inventory files",len(actual)); assert not Path("results/read_all/.hardware.lock").exists()'
```

Ошибка/отсутствующий manifest/оставшийся lock — остановка процедуры, а не
разрешение удалить lock или повторить запись. Сначала разобрать сохранённые
данные. Ключи финального manifest именно `impacts` и `pulses`: их возвращает
`Collector.run()`, затем entrypoint переносит через `manifest.update(result)`.
Не путать с `native_impacts`/`programming_pulses` внутри raw-аудита.
Проверка выше читает только manifest и имена файлов; это лёгкая проверка
завершения/inventory, **не strict raw audit**. Manifest, start_manifest,
protocol, INI, SHA исходников и сырой журнал остаются неизменными. На Pi
не запускать тяжёлый classifier/replay даже при наличии `.deps-classifier`.

## После сбора: обновить Git без массового LFS, опубликовать именно результат

Remote уже впереди стартового Pi HEAD. Сначала дождаться финальных raw/provenance,
затем fast-forward; коммит результатов **после FF**, чтобы не создать
расхождение веток. При любой ошибке FF остановиться, не делать reset/stash
или принудительное слияние автоматически.

```bash
git status --short
```

```bash
git -c gc.auto=0 -c credential.helper='cache --timeout=43200' fetch origin codex/full-cifar-diagnostic
```

```bash
GIT_LFS_SKIP_SMUDGE=1 git -c gc.auto=0 merge --ff-only FETCH_HEAD
```

```bash
git diff --cached --name-only
```

До добавления новых файлов index должен быть пуст. Добавить только текущую
сессию; будущий локальный отчёт пока не существует. Не `git add results/`
и не `git add .`:

```bash
git add -- results/cifar_full_case0_20261019/f21e608c-059f-4dce-afe0-6edf5c26133c
```

```bash
git diff --cached --stat
```

```bash
git -c core.hooksPath=.githooks commit -m "Preserve full CIFAR hardware raw and provenance for local audit"
```

Этот commit запускает штатный secret-check staged Git и проверку содержимого
staged LFS objects через `tools/security/scan_staged_lfs.py`. Отсутствующий
Gitleaks, секрет или отсутствующий LFS object блокирует коммит; обходить hook
нельзя. Raw gzip/array.cb попадут в LFS по существующим `.gitattributes`.
Не включать credential-файлы, PAT, приватный MemriCORE или консольную историю.

```bash
git -c gc.auto=0 -c credential.helper='cache --timeout=43200' -c core.hooksPath=.githooks push origin HEAD:codex/full-cifar-diagnostic
```

Дождаться успешной загрузки Git и LFS. При запросе авторизации использовать
настроенный credential helper/скрытый ввод; PAT не помещать в команду или URL.
Если remote успел измениться, не использовать force push.

Публикация неизменных raw/provenance предшествует тяжёлому локальному аудиту
и сама по себе не означает PASS или точность классификации.

## Локально: сначала квалифицировать перенос аудита

После публикации получить в локальный checkout именно аппаратную сессию через
Git/LFS, проверить все content SHA и происхождение. Затем подготовить strict
raw audit/classifier и [сравнение SIM/HW](CIFAR_FULL_REPORT_COMPARISON_20261019.md)
в локальной согласованной среде. Старые отчёты и семь frozen CIFAR sources
не переписывать. Сам classifier ищет `.deps-classifier` и `.deps` относительно
checkout; наличие папок ещё не квалифицирует платформу.

**Локальный перенос пока не прошёл квалификацию.** Нужно отдельно решить
наличие/проверку записанных SHA приватного MemriCORE без его публикации и
проверить платформенные численные различия, особенно rank7 и выбор матрицы C.
Нельзя пропускать private-SHA проверки, подменять матрицу, менять допуски или
объявлять Windows replay эквивалентным Pi без review. До этой квалификации
нет утверждённой команды локального аппаратного classifier. Это ограничение
не является разрешением перенести тяжёлую обработку обратно на Pi.
После квалификации и успешного аудита отчёт создаётся новым файлом локально,
проверяется независимо и публикуется отдельным коммитом. Проверять silence/ties
отдельно от целостности сбора.

## После освобождения платы: STDP v2

Обновлённая ветка должна содержать reviewed launcher
`experiments/conductance_stdp_hardware_20261019/experiment.py` и дополнительный
auditor. Квалифицирующей SIM-сессии нужны только **два LFS payload**; обычные
JSON/source уже получены Git. Не скачивать весь архив:

```bash
git lfs pull origin --include="results/conductance_stdp_calibrated_20261019/d27da87e-9928-4303-971e-215afbc63158/array.cb,results/conductance_stdp_calibrated_20261019/d27da87e-9928-4303-971e-215afbc63158/raw/raw.jsonl.gz" --exclude=""
```

```bash
../../.venv/bin/python -B experiments/conductance_stdp_hardware_20261019/experiment.py --dry-run
```

Требуются `registered=true`, `hardware_lock_present=false`,
`connector_opened=false`, `board_access=false`. Preflight обязан пройти с
реальными приватными MemriCORE-файлами и их закреплёнными SHA, INI и всеми
квалифицирующими SIM-артефактами. Недостающие данные/источники не обходить.
После подтверждённого освобождения платы и принятия preflight основным агентом:

```bash
../../.venv/bin/python -B experiments/conductance_stdp_hardware_20261019/experiment.py --run-hardware
```

Это новый UUID, не повтор CIFAR. Предел6712 impact/128 импульсов; fresh fit
на плате, SIM gain800 не переносится. `uninformative_calibration`/exit3 —
допустимая остановка без обучения после подготовки; не повторять с другим gain.
Для completed или этого терминального статуса взять **точный UUID из вывода**:

```bash
STDP_SESSION=results/conductance_stdp_hardware_20261019/REPLACE_WITH_ACTUAL_UUID
```

Сохранить final status, raw, observations и provenance; публиковать их с Pi
для последующего **локального** аудита. Нужен supplemental hardware auditor
`tools/audit_conductance_stdp_hardware.py`: он проверяет новый launcher и
квалификацию, затем вызывает frozen native raw replay. Один старый SIM auditor
не проверяет все поля нового аппаратного запуска. Ошибка запуска/аудита
сохраняется как результат; следующий эксперимент автоматически не стартует.

Его локальная среда и private source-SHA проверки также требуют квалификации;
на Pi не запускать raw replay/postprocessing. Для публикации исходных данных
использовать только точную сессию, без ещё не существующего отчёта:

```bash
git add -- "$STDP_SESSION"
```

```bash
git diff --cached --stat
```

```bash
git -c core.hooksPath=.githooks commit -m "Preserve calibrated conductance STDP hardware raw for local audit"
```

```bash
git -c gc.auto=0 -c credential.helper='cache --timeout=43200' -c core.hooksPath=.githooks push origin HEAD:codex/full-cifar-diagnostic
```

Перед add index снова должен быть пуст, а UUID заменён фактическим. Для
failed-сессии, которую supplemental auditor отклоняет, публикацию исходных
данных и отдельного разбора согласовать по фактическому содержимому;
несуществующий успешный отчёт не создавать.

## Локальный MNIST

Сессия `7a536bf8-0269-4edd-b454-a8ba2c33c219` **остановлена после head epoch5**,
это partial, а не completed full-run; финальной test accuracy нет. Не запускать
на ней completed-only финальный аудит и не менять её исходники/артефакты.
`python_refactor` готовит отдельный локальный recovery, validator проверяет
его контракт; восстановление не считать уже выполненным.
Требования completed-only проверки и интерпретация exit3/`review_required` в
[MNIST_FULL_FINAL_AUDIT_20261019.md](MNIST_FULL_FINAL_AUDIT_20261019.md)
относятся к будущему завершённому квалифицированному результату; команда со
старым UUID не превращает partial в полный результат. Это CPU-аудит в
локальной `.venv-mnist-pilot`, без повторного
обучения, GPU или платы. Старый smoke128 auditor для full результата не подходит.
