# CIFAR: безопасное продолжение по целым тайлам

Статус: новый маршрут на review; gates SIM/HW закрыты. Это не результат
аппаратного продолжения и не разрешение запускать плату сейчас.

`experiment.py` — foreground acquisition через public MemriBoard Connector,
штатные converters и INI. Все методы физической записи, fresh-R/current guards,
калибровки и MVM наследуются без изменения из закреплённого collector. Прямых
driver/GPIO/SPI/serial/interface вызовов и динамической загрузки драйвера нет.

Каждый из 48 тайлов получает отдельные UUID, Connector, ограниченную запись,
свежие basis/fit/selection/heldout/C, все 784 позиции и final basis. Физический
протокол прежний: 0.05V, не более 4 активных BL, 8 WL, T12, 8 уровней SET/RESET
до 3V, неизвестная аппаратная длительность. Максимумы всех 48 новых тайлов:
14,457,648 impacts и 115,200 импульсов. План с принятым префиксом вычисляет
точный остаточный верхний бюджет. Это НЕ общий фактический расход всех попыток:
непопавший в raw хвост старой сессии неизвестен; её прежний protocol ceiling
15,000,000 — только консервативная верхняя граница.

Raw сохраняется gzip level1 без потерь. Каждые 16 коэффициентов и 16 позиций,
а также на границах этапов, текущий gzip закрывается, файл fsync, индекс/progress
заменяются атомарно; на POSIX синхронизируется и каталог. Chunk≤8MiB expanded,
stage≤2GiB expanded, вся новая попытка≤32GiB expanded. Перед стартом резервируются
две копии бюджета для working tree/Git LFS плюс 8GiB свободного места. При ошибке
первая попытка прекращается, сохраняет failed manifest, закрывает Connector;
автоматического повторения импульсов нет. Native stdout сохраняется gzip,
консоль показывает прогресс каждые16позиций.

**Частые checkpoints сохраняют свидетельства, но не позволяют продолжать внутри
тайла.** Повторно использовать можно только целый тайл после независимого
семантического raw replay. У прерванного тайла сохраняется вся исходная raw,
затем он полностью программируется и измеряется заново с новой C. Нельзя
смешивать позиции, состояния или C между попытками. Lock нельзя автоматически
снимать: основной агент сначала подтверждает отсутствие старого процесса.

Native SIM квалификация: stage0 полностью; stage1 аварийно завершает процесс
через SIM-only `os._exit(86)` после17позиций, до закрытия нового gzip. Индекс
по-прежнему подтверждает16позиций;17-я существует только в software marker и
неполном хвосте. Не создаются final manifest/footer/утверждение о close.
Новый процесс полностью повторяет
stage1 и stage2. Это 4 фильтра/75 taps/784 позиции/T12, а не классификация полной
сети. При каждом новом Connector SIM исходный массив создаётся заново из
фиксированного seed; этот source context change записан явно. Каждый тайл всё
равно требует собственной новой программы/калибровки.

После source review/регистрации основной агент создаёт первый план:

```sh
python tools/plan_cifar_continuation.py --source simulator --purpose qualification --output reports/2026-10-19/cifar-resume-review/SIM_INITIAL_PLAN.json
python experiments/cifar_resumable_20261019/experiment.py --dry-run --preflight-source simulator --plan reports/2026-10-19/cifar-resume-review/SIM_INITIAL_PLAN.json
python experiments/cifar_resumable_20261019/experiment.py --run-simulator --qualify-process-loss --plan reports/2026-10-19/cifar-resume-review/SIM_INITIAL_PLAN.json
```

Далее локальный `tools/audit_cifar_interrupted_prefix.py --stage-session`
принимает завершённую stage0-сессию. `tools/plan_cifar_continuation.py` с её
`--prefix-report` создаёт новый план, начинающийся со stage1. Второй запуск
использует этот план без `--qualify-process-loss`. Точные UUID/пути команд
формируются из фактических результатов, а не подставляются заранее.

Для платы план `--purpose full --source hardware` строится из независимых
сертификатов старого непрерывного префикса и завершённых новых stage sessions.
`--stop-stage-exclusive` задаёт конечную малую партию (например,13..19 означает
шесть тайлов13–18). Target остаётся48, но завершение партии не объявляется
завершением полного CIFAR. Следующая команда требует нового плана с audited
префиксом. Пользователь выбрал закончить текущий тайл до22:00 и продолжить
завтра; основной агент выбирает партию по актуальному времени с запасом,
ночного автоматического продолжения и автоповтора нет.
GitHub→Pi checkout→`experiment.py --dry-run`, затем явный `--run-hardware`.
Pi выполняет только физический сбор и лёгкую provenance/preflight проверку.

Тяжёлый replay и классификация выполняются **локально**:
`tools/classify_cifar_continuation.py --session ...` для каждого явно указанного
источника. Он повторно вызывает независимый raw auditor, проверяет все48stage
ровно один раз и собирает raw/C tensors 12×64×28×28×3. Три вклада складываются
до одного обученного BN/LIF шага. Missing/duplicate/nonfinite вклад вызывает
ошибку, без CPU/zero fill. `--qualification-only` допускает ровно3SIMstage и
никогда не запускает полный classifier. Обычный режим сравнивает CPU, measured
raw и measured C; оставшиеся слои/LIF цифровые. Один case0 — не test accuracy;
silent/tied decoder argmax не означает распознавания. GainSIM.015/HW.125 —
явная разница источников, а не доказанная физическая эквивалентность.

Private Pi source evidence переносится только по отдельно reviewed receipt
policy, без публикации приватных исходников и без generic missing-source
fallback. Старые full-auditor/source gates остаются неизменными.

Prefix replay sources фиксируются до регистрации новой acquisition-схемы.
Отдельный `tools/cifar_stage_source_registry.json` содержит reviewed source SHA
нового маршрута; это устраняет циклическое взаимное хеширование. Реестра
с регистрацией ещё нет; его создаёт основной агент после review. Раздельные
immutable execution grants `SIMULATOR_EXECUTION_GRANT.json` и
`HARDWARE_EXECUTION_GRANT.json` находятся в `reports/2026-10-19/cifar-resume-review/`.
Каждый grant закрепляет4новыхpy,4offline tools и stage registry. Сессия сохраняет
точные bytes/SHA grant; HW grant дополнительно ссылается на SHA фактической
nativeSIM qualification. SIM grant и исходники послеSIM не редактируются;
позднее разрешение HW добавляет новый внешний файл, не меняет код. В финальном
qualification вызове обязателен `--interrupted-session` с фактической SIM
сессией без manifest: CRC/index/request-result пары до16 проверяются отдельно,
неполный17-й хвост сохраняется и не участвует в tensor.
