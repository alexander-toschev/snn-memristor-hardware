# Discovery SET/RESET и проводимостей всех 256 ячеек

Пользователь запросил калибровку карты, включая SET/RESET и подбор проводимости.
Сценарий отдельный: никакой записи CIFAR-весов и автоматического следующего
эксперимента. Подготовленный код сам ничего не запускает.

Текущая версия — `cell-conductance-discovery-v2-gui-write-guard`.
Историческая аппаратная сессия `f32eab10-9abd-4d8c-a4c5-3ed49a3cb8fb`
использовала v1 без per-pulse current guard; её протокол и результаты не меняются.

На Raspberry используется Git checkout этого репозитория, а не копия команд
в сторонней установленной GUI-папке. Обновите проверенную ветку через
`git pull --ff-only`, зафиксируйте `git rev-parse HEAD`, активируйте локальную
`.venv` и выполняйте команды ниже из корня репозитория. Подготовка окружения
и private driver files описана в [REPRODUCE.md](REPRODUCE.md). Эти файлы
остаются локальными; source fingerprints проверяются до подключения.

```sh
python experiment.py --calibrate-cells --dry-run
python experiment.py --calibrate-cells --run-simulator
python experiment.py --calibrate-cells --run-hardware
```

Dry-run проверяет аппаратные source fingerprints, INI и точный bounded protocol
без импорта драйвера/подключения. Hardware требует локальные operator-provided
файлы из прежней инструкции. Simulator создаёт один новый штатный массив 32×8
в каталоге сессии, проверяет совпадение simulator INI и Connector config; доступ
к interface не используется. Все физические I/O — только публичный Connector.

## Что выполняется

План — `configs/cell_calibration.json`, все WL0..7 и BL0..31, один цикл:

1. Три mode7 vol=0 чтения каждой из 256 ячеек перед первым программированием.
2. Перед RESET очередной ячейки — local_pre_reset checkpoint. Это уже может
   быть после записи предыдущих ячеек; не путать с начальной общей картой.
3. RESET .3..3 В, шаг .01, максимум 271 решение о ненулевом pulse, до native q≤.005.
   Post-RESET checkpoint подтверждает mean и last read q≤.005 перед допуском SET.
   Immediate crossing сохраняется отдельно. Если RESET не достигнут либо
   подтверждение не прошло, SET данной ячейки пропускается без retry; сбор
   terminal checkpoint сохраняется, затем рассматривается следующая ячейка.
   Подтверждающие mode7 чтения выполняются до checkpoint MVM, не после него.
4. SET такая же лестница, не более 271 решения; первые пересечения q=.01/.02/.03/
   .04/.05 вызывают checkpoint. Перескок нескольких целей одним pulse — ОДНО
   измеренное состояние со списком crossed_targets. .05 — верх тестируемого
   окна, не максимум возможностей ячейки. Post-read ниже цели не вызывает
   повторного допульсования и не удаляется из данных.
5. Терминальное фактическое состояние измеряется и при недостижении .05.
   Если после последнего checkpoint не было нового write, terminal ссылается
   на тот же checkpoint; это не ещё одно независимое состояние. Максимум семь
   разных checkpoint/ячейку. Без нового write состояние не считается доказанно
   неизменным: read-disturbance не исключён.
6. После всех ячеек — три mode7 чтения каждой для сравнения с terminal и
   initial G. Различие описательное, причинный write-disturb не установлен;
   сохранность ранней MVM-кривой поздним измерением G не доказана.

Checkpoint: пять mode7 vol=0 при nominal read=.3 В; затем пять повторов
`zero-before, .025, .05, .075, .1, zero-after` с единственным активным BL
на соответствующей WL. Это 35 impact. Model guard проверяет конечный G,
ток при .1 В≤5 мА и прогноз≤80% ADC reference; наблюдаемый output также≤80%.
Это software bounds, не физический compliance и не аппаратная калибровка.

Перед каждым ненулевым mode7 v2 повторяет GUI-проверку для валидного R:
`current_predict = requested_voltage / last_valid_mode7_native_R`.
При sign0 допускается ≤0.04 А; при sign1 ≤`board.soft_cc` (0.002 А).
Равенство разрешено. Последний R берётся из `result[0]` успешного mode7 данной
ячейки, включая checkpoint-чтения; это не среднее G, не целевой q и не native R
ответа mode10. После каждого успешного mode7 он обновляется. При отсутствии,
нулевом, отрицательном или нечисловом/бесконечном R сценарий останавливается:
GUI-подстановку epsilon вместо нуля мы не применяем.

Превышение тока сохраняется как `event=skipped` с planned_command_si, previous R,
resistance_source_task_id, predicted current, limit, step и programming_decision_id.
Connector не вызывается; нет task_id/ADC/q/ответа. Шаг лестницы продвигается,
а q, last_write и crossed_targets не меняются. Это ограниченный по току поиск,
не неисправность ячейки и не transport failure. Длительность и реальные токи
импульсов этой проверкой не измеряются; это не аппаратный compliance.

Верхние пределы: **138752 решения о записи**, не более **138752 реальных попыток
ненулевого pulse**, **203008 impact** = 256×(542+7×35)
+ 768 initial + 768 final. Достижение endpoint раньше уменьшает фактический объём.
Ни RESET miss, ни SET miss сами по себе не останавливают исследование остальных
ячеек. Ошибка транспорта/ID, ADC saturation/invalid, invalid model R/G,
нарушение budgets/model guards/наблюдаемого headroom — остановка всего run.
Терминальные измерения не выполняются в обход такой остановки.
`programming_decisions = programming_pulse_attempts + programming_skipped_current`;
per-cell reset/set_decisions, reset/set_pulses и reset/set_skipped_current
разделены. Попытка с потерянным ответом считается выполненной попыткой вызова,
пропуск — нет. Нулевые skips полного simulator run не доказывают ветку guard:
граничные условия и skip-переходы проверяются отдельно в offline fixtures.

## Что означают напряжения и данные

На физическом драйвере mode7 не использует pulse-width: native длительность
неизвестна, передаётся 0 как метаданные вызова. Simulator получает 125 мкс,
которые его модель использует как 125 шагов; это не эквивалент платы.
Hardware mode7 возвращает среднее десяти ADC, simulator — один simulated ADC.
Пять повторов hardware mode7 не объявляются 50 независимыми измерениями.

Native converters неизменны. .025/.05/.075/.1 В превращаются в DAC20/41/61/82;
native d2v с тремя знаками даёт .024/.05/.074/.1 В. Сохраняются requested SI,
DAC и native decoded values. Последние используются симулятором, но не доказывают
измеренное напряжение физической платы. Mode10 использует a2v(gain=1), ADC0
допустим; readback R — a2r при nominal .3 В. G и MVM относятся к разным
трассам/напряжениям: их несоответствие не доказывает ошибку аппаратного gain.

## Результаты и отдельный offline отчёт

Сессия `results/cell_calibrations/<uuid>/`: start/final manifest, protocol/INI
snapshots, Git/source hashes, полный raw request/result, компактные
checkpoints.jsonl, cell_results.jsonl, initial/final_array.jsonl и summary.
Request flush предшествует impact; result сохраняется даже при исключении.
Счётчик pulse учитывает попытки, доставка при потере ответа неизвестна.
Shared hardware lock общий с предыдущими сценариями; сторонние приложения
им не блокируются. close_port не гарантирует аппаратный reset после исключения.

После завершения, без аппаратуры:

```sh
python render_cell_calibration.py --session results/cell_calibrations/SESSION_UUID
```

Отдельный новый `report/` содержит HTML8×32, cells.csv, states.csv и JSON с
hashes. Renderer не импортируется runner и не меняет исходные данные.
В v2 CSV/JSON/HTML показывают причины и число пропусков по току отдельно от
исхода измерений; current_limited не превращает завершённый сбор в failed.
При чтении v1 эти счётчики отсутствуют, а отчёт явно сообщает, что guard не был включён.
Он различает best_observed_state (исторический) и terminal_state, затем
сравнивает terminal G5 с final G3. Никакая готовая маска не экспортируется.

Per-state screening использует прежние предварительные SNR≥5, CVhigh≤5%,
двухуровневую линейность .05/.1 относительно пяти zero-before. Zero-after
показывается отдельно. G после MVM внутри checkpoint не измеряется, поэтому
per-state G stability неизвестна. Поздний G: |ΔG|/Gterminal≤max(5%,3SEfraction),
SE использует SDterminal²/5 и SDfinal²/3. Это не qualification repeatability.

Дополнительный affine fit: raw ADC против DAC-кода на hardware или против
native decoded V на simulator. Первые четыре повтора дают fit, пятый — временной
holdout. .025/.075 — дополнительные диагностические точки, не новый pass gate.
Fit не корректирует raw и не делает физический нелинейный MVM линейным.
Здесь MVM — отклик одного активного BL на одной WL в каждом checkpoint.
Совместное умножение нескольких выбранных BL/фильтров — отдельный следующий
эксперимент; ни успешный simulator run, ни этот отчёт его не заменяют.

Offline тесты:

```sh
python -m unittest test_cell_calibration -v
```
