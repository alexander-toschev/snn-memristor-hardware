# Выборочная проверка карты MemriBoard — 22 октября

Сценарий `experiment.py` действительно измеряет плату через публичный
`manager.board.Connector`, штатные converters и INI. Общий read-only Collector
находится в `experiment_common/selective_metrology.py`. SET/RESET отсутствуют;
попытка передать записывающую команду отвергается до `impact`.

## Исходная карта и фиксированная выборка

Последняя аппаратная калибровка: `4c8c70ca-f819-4c82-83c8-8fed7f9fe3b0`,
20 октября, completed, 293294 вызова/8960 импульсов.
SHA исходного `map.json`:
`e452e89b3923873de5138142d9511b689d2d10d5245d2c1222f73d46d446b085`.
Публичный архив и inventory:
`results/stable_cell_calibration_20261020/archives/4c8c70ca-f819-4c82-83c8-8fed7f9fe3b0.{tar.gz,json}`.
Полный прежний raw для этого запуска скачивать не нужно: `sample.json` содержит
неизменяемую до измерений выборку, исходные оценки и identity карты.

Карта **уже исходно diagnostic_best_available, qualified=false**. Выбранные
WL[0,7] × BL[2,3,6,10,11,12,14,15] прошли обязательную проверку обеих полярностей
signed response, но не строгие quality noise/drift критерии. Это16ячеек,
8дифференциальных коэффициентов, не подтверждение200ячеек для4×25filters.

Проверяем все16 выбранных ячеек и по4 на остальных6WL: лучший score,
ближайший к прежнему порогу span/SNR, нижняя/верхняя историческая G;
повторы адресов пропускаются детерминированно. Всего40/256.
Точные адреса и причины записаны в `sample.json` ДО первого обращения.

После карты выполнялись CIFAR и MNIST записи (UUID перечислены в sample).
Историческая G не является текущим target, её изменение не объявляем retention
ошибкой. Этот список известных сессий не является полным доказательством
отсутствия других воздействий.

## Чтение и критерии

Четыре обхода: initial, delay0, delay10, delay60. Задержки отсчитываются от
конца первого обхода; фактическое время каждого checkpoint сохраняется.
На каждую ячейку10 независимых `zero → fresh mode7 R → singleton .05V → zero`.
Ровно6400 вызовов при полном выполнении:1600mode7+4800mode10, **0импульсов**.
Wall-budget600s, explicit wait≤60s. Ориентир1–3мин, не обещание срока.
В SIM создаётся отдельный массив; это проверка сценария, не валидация HWкарты.

Сохраняются прежние формулы `stable_cell_calibration_20261020/metrics.py`:
SD≤max(2ADC,.05×исторический span); drift от нового initial≤max(3ADC,.05×span);
zero shift≤max(2ADC,3×SD(zero_after−zero_before)/sqrt(10)).
Negative centered ADC не обрезается. Невалидный R/ADC, ток>.005A,
предсказанный выход≥3.3V, измеренный>4V останавливают сбор без подмены данных.
Ток вычисляется по fresh R и max(.05V, штатный DAC→V).

Даже PASS означает только повторяемость нынешнего чтения выбранных40cells.
Он не переводит старую карту в qualified и не проверяет programmable range,
signed pair orientations или все256cells. При отклонениях сначала расширять
соответствующие участки; полной перекалибровки по умолчанию нет.

Чтобы повторно подтвердить программируемый диапазон после intervening writes,
нужны исходные два цикла R→M→H, где RESET8levels(.3,.6,.9,1.2,1.5,2,2.5,3V),
M — первые4SETlevels, H — ещё4SETlevels, те же10repeats/delays.
Только16selectedcells:≤17664calls/512pulses,≤360s wait.
Прежняя B-проверка обеих signed orientations, M/M residual и checkerboard
на8slots:≤6656calls/768pulses,≤240s wait. Итого≤24320calls/1280pulses.
Это следующий зависимый сценарий после оценки read-only данных, сейчас код
его автоматически не исполняет. Исходные guards .04A RESET/.002A SET сохраняются;
HWduration0 означает неизвестную фактическую длительность, не отсутствие импульса.

## Запуск

```powershell
.venv-mnist-pilot/Scripts/python.exe -B experiments/selective_map_validation_20261022/experiment.py --dry-run
.venv-mnist-pilot/Scripts/python.exe -B experiments/selective_map_validation_20261022/experiment.py --run-simulator
```

На Pi из корня репозитория:

```bash
.venv/bin/python -B experiments/selective_map_validation_20261022/experiment.py --run-hardware
```

`sample/protocol/start_manifest` сохраняются прежде Connector. Hardware lock,
close_port в finally, raw requests/results/guards/zero brackets, completed/failed
manifest и hashes доступны в `results/selective_map_validation_20261022/<UUID>`.
Нет private-driver receipt/check, статус честно
`private_runtime_not_independently_reverified`.

Аппаратный MNIST не должен использовать exact SIM LUT: физическое чтение
шумит, а эта карта подтверждала лишь16 выбранных cells. Нужны явное размещение,
свежие аппаратные MVM и суммирование всех измеренных вкладов до LIF;
чтение весов с последующим идеальным CPU W@x не считается аппаратным forward.
