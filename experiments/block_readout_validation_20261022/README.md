# Проверка 15 блоков для последовательного MNIST readout

Точная карта `BLOCK_SELECTION.json`: 15 групп, 33 коэффициента, 66 уникальных
ячеек. Не исключаем плохие входы внутри блока и не меняем их перестановку.
Это только проверка нынешнего readout, без программирования, STDP, LIF и accuracy.

Один foreground-прогон: **74 518 вызовов, 0 SET/RESET, до600с**.
Basis: 3 повтора singleton-входов, 8658вызовов. Затем фиксируется SHA basis.
Validation: 5 раз проверяются все 137 непустых бинарных масок, 65 860 вызовов.
Каждая singleton-маска повторяет вход basis во времени; её нельзя называть
новой комбинацией. Маски с несколькими входами не используются при fit.
Порядок блоков, масок и двух WL-ног чередуется на повторах. Сохраняется полный
raw до 256 МиБ; после закрытия порта он архивируется без изменения байтов.

Три заранее заданных бинарных профиля оцениваются раздельно:

1. `ordinary_delta05`: фон0, вход0,05В.
2. `biased_delta025`: фон0,025В на всех BL блока, вход0,05В.
3. `biased_delta05`: тот же фон, вход0,075В.

Каждая нога: свежий mode7 R всех активных → baseline×4 → signal×4 → baseline×4.
Все четыре mode10 идут через публичный Connector/converters/INI и сохраняются.
Проверяется sum(max(requested, DAC-decoded V)/R)≤5мА,330I<3,3В и ADC≤4В.
Signed-ответ — измеренная plus-нога минус minus-нога, не расчёт из G.

Primary readout всегда усредняет4выборки. `first_sample_diagnostic` описывает
те же данные по первой выборке, но не может быть выбран по результату.
Оператор будущего readout обязан сохранить четыре реальные выборки на уровень.

До данных заданы критерии: leg mean>max(5ADC,5SD), max baseline drift≤5ADC;
ошибка validation mean≤max(5ADC,10%abs(predicted)). Signed SD≤5ADC, signed
drift≤5ADC, signed error≤max(5ADC,10%(abs(predPlus)+abs(predMinus))). Signed
zero/cancellation помечается `zero_or_unresolved` и κ, но не исключает блок
сам по себе. Prediction строится только из singleton basis того же профиля.
Линейность между двумя амплитудами biased — отдельная диагностика по фактическим
DAC increments; это не обязательный критерий бинарного профиля.

Статус `stable_current_state_readout` требует все входы и все проверки профиля.
Полные, но шумные/дрейфующие данные дают `candidate` с причинами; неполные или
неположительный leg basis — `fail`. Среди стабильных профилей фиксированный
приоритет ordinary05 → biased025 → biased05. Это не programming/MNIST GO;
аппаратный scheduler последовательного MNIST ещё не реализован.

Из корня репозитория (SIM):
```powershell
.venv-mnist-pilot/Scripts/python.exe -B experiments/block_readout_validation_20261022/experiment.py --run-simulator
```
Оператор на Pi:
```bash
../../.venv/bin/python -B experiments/block_readout_validation_20261022/experiment.py --run-hardware --publish
```
`--dry-run` не открывает Connector. После закрытия порта публикация сохраняет
архив/summary/HTML, проверяет секреты, использует штатные Git hooks/LFS.
Результаты: `results/block_readout_validation_20261022/<UUID>`.
