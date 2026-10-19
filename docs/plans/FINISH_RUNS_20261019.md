# Завершение текущих прогонов 19 октября

Команды выполняются **по одной, foreground**, с проверкой результата перед
следующей. Никаких detach, фонового автопродолжения, токенов в командах или
изменения выполняющихся исходников. Реальное I/O — только MemriBoard.

## Raspberry: сначала закончить CIFAR и сохранить строгий аудит

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
../../.venv/bin/python -B -c 'import json; from pathlib import Path; p=Path("results/cifar_full_case0_20261019/f21e608c-059f-4dce-afe0-6edf5c26133c/manifest.json"); m=json.loads(p.read_text()); print({k:m.get(k) for k in ("status","source","scope","impacts","pulses")}); assert m["status"]=="completed" and m["source"]=="hardware" and m["scope"]=="full"; assert not Path("results/read_all/.hardware.lock").exists()'
```

Ошибка/отсутствующий manifest/оставшийся lock — остановка процедуры, а не
разрешение удалить lock или повторить запись. Сначала разобрать сохранённые
данные. Ключи финального manifest именно `impacts` и `pulses`: их возвращает
`Collector.run()`, затем entrypoint переносит через `manifest.update(result)`.
Не путать с `native_impacts`/`programming_pulses` внутри raw-аудита.
Для completed выполнить **до fetch/merge**:

```bash
../../.venv/bin/python -B experiments/cifar_full_case0_20261019/classifier.py --session results/cifar_full_case0_20261019/f21e608c-059f-4dce-afe0-6edf5c26133c --output reports/2026-10-19/cifar-full-case0-review/NATIVE_HW_FULL_CLASSIFIER.json
```

Classifier проверяет источник/manifest/raw, выполняет strict replay и затем
CPU/raw/C downstream. Он не открывает плату. На Pi проверено наличие
**checkout-local `.deps-classifier/torch`** и `../../.venv/bin/python`.
Сам classifier добавляет `.deps-classifier` и `.deps` этого checkout в импорт;
`PYTHONPATH` и родительский `.deps-classifier` не нужны. Существующий отчёт
не перезаписывать. Exit0 и созданный отчёт означают завершение аудита,
но не автоматически хорошую точность; проверить silence/ties отдельно.

## После аудита: обновить Git без массового LFS, опубликовать именно результат

Remote уже впереди стартового Pi HEAD. Сначала сохранить аудит на диске,
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
сессию и её отчёт; не `git add results/` и не `git add .`:

```bash
git add -- results/cifar_full_case0_20261019/f21e608c-059f-4dce-afe0-6edf5c26133c reports/2026-10-19/cifar-full-case0-review/NATIVE_HW_FULL_CLASSIFIER.json
```

```bash
git diff --cached --stat
```

```bash
git -c core.hooksPath=.githooks commit -m "Record full CIFAR MemriBoard hardware case0 and strict audit"
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

После независимого review полного аппаратного отчёта выполнить на Pi
[сравнение завершённых SIM/HW отчётов](CIFAR_FULL_REPORT_COMPARISON_20261019.md).
Оно использует опубликованный отдельный инструмент и обе точные сессии,
проверяет исходники, декодеры/тишину и метрики; raw SIM повторно не скачивается.
Сохранённый `FULL_SIM_HARDWARE_COMPARISON.json` также проверить и опубликовать
отдельным коммитом с теми же hooks в `codex/full-cifar-diagnostic`.
Это сравнение ещё не выполнено; результат нынешнего аппаратного сбора ожидается.

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

```bash
../../.venv/bin/python -B tools/audit_conductance_stdp_hardware.py --session "$STDP_SESSION" --output reports/2026-10-19/conductance-stdp-calibrated-review/NATIVE_HW_SUPPLEMENTAL_AUDIT.json
```

Нужен именно supplemental hardware auditor: он проверяет новый launcher и
квалификацию, затем вызывает frozen native raw replay. Один старый SIM auditor
не проверяет все поля нового аппаратного запуска. Ошибка запуска/аудита
сохраняется как результат; следующий эксперимент автоматически не стартует.

Для публикации этого результата также использовать только точную сессию:

```bash
git add -- "$STDP_SESSION" reports/2026-10-19/conductance-stdp-calibrated-review/NATIVE_HW_SUPPLEMENTAL_AUDIT.json
```

```bash
git diff --cached --stat
```

```bash
git -c core.hooksPath=.githooks commit -m "Record calibrated conductance STDP hardware and supplemental audit"
```

```bash
git -c gc.auto=0 -c credential.helper='cache --timeout=43200' -c core.hooksPath=.githooks push origin HEAD:codex/full-cifar-diagnostic
```

Перед add index снова должен быть пуст, а UUID заменён фактическим. Для
failed-сессии, которую supplemental auditor отклоняет, публикацию исходных
данных и отдельного разбора согласовать по фактическому содержимому;
несуществующий успешный отчёт не создавать.

## Локальный MNIST

Текущий full STDP UUID `7a536bf8-0269-4edd-b454-a8ba2c33c219` проверять только
после `manifest.json: completed`. Точная foreground команда и интерпретация
exit3/`review_required` опубликованы в
[MNIST_FULL_FINAL_AUDIT_20261019.md](MNIST_FULL_FINAL_AUDIT_20261019.md).
Это отдельный CPU-аудит сохранённых данных в `.venv-mnist-pilot`, без повторного
обучения, GPU или платы. Старый smoke128 auditor для full результата не подходит.
