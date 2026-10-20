# Архивная MNIST64: native MemriBoard SIM → CPU LIF/readout

Оцениваются **сохранённые** STDP-веса `weights/mnist_baseline_20261019/model.pt`.
Нового STDP-обучения нет (`new_stdp_training=false`), физическая плата не открывается.
64×5×5 фильтра, stride1,24×24 позиции, Poisson100 и LIF сохранены из архива.

## Подготовка и команды

Из корня репозитория, для нового окружения Python3.11:

```powershell
py -3.11 -m venv .venv-mnist-pilot
.venv-mnist-pilot/Scripts/python.exe -m pip install --require-hashes --no-deps -r experiments/mnist_feature_pilot_20261019/bootstrap.lock
.venv-mnist-pilot/Scripts/python.exe -m pip install --require-hashes --no-deps --no-build-isolation -r experiments/mnist_feature_pilot_20261019/requirements.lock
.venv-mnist-pilot/Scripts/python.exe -m pip install --target .deps -r requirements.txt
```

В уже подготовленной рабочей папке повторно устанавливать пакеты не нужно.
Checkpoint/официальные IDX должны быть извлечены Git LFS в их repo-relative пути.

```powershell
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_archive64_20261020/experiment.py --pilot
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_archive64_20261020/experiment.py --run-simulator --pilot
```

Первая команда — preflight без Connector/нейросети. Вторая — отдельный пилот
128train/100test. По последнему указанию пользователя основной запуск использует
первые **10000 официальных train** и все **10000 официальных test**, без `--pilot`:

```powershell
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_archive64_20261020/experiment.py --run-simulator --batch-size 8
```

Это обучение нового цифрового readout на 10000 изображениях, а не повтор
архивного STDP на 30000: свёрточные STDP-веса остаются сохранёнными архивными.
Протокол помечен `mnist-archive64-native-fullmask-lut-v2-train10000`;
ранний пилот и расчёт `FULL_INVENTORY.json` для 30000/10000 сохраняют свой
исходный смысл и не являются измерениями нового объёма.
Пилот не выдаётся за архивную полную accuracy. Результаты в
`results/mnist_archive64_20261020/<UUID>/`. Hardware CLI отсутствует.

## Где выполняется каждая операция

| Этап | Исполнитель | Данные/смысл |
|---|---|---|
| Изображение/Poisson | CPU, архивный encoder | uint8/255, rate.0015, seed123, content-deterministic,100ticks |
| Patch | CPU |25битный mask5×5, без умножения на W |
| Target/feedback | CPU | Разность проводимостей, следующий импульс по nativeR, без labels |
| SET/RESET/readR | Public Connector → native SIM | Настоящие команды штатному симулятору, не физической плате |
| MVM | Public Connector mode10 → native SIM | Одна полная сумма всех активных BL, до25;7 остальных BL нулевые |
| Guards | CPU до команды | V/R и предел тока/ADC; это защита, не замена MVM |
| ADC→V | Штатный converter, CPU | Те же параметры/округление, что GUI |
| Differential/α | CPU | Вычитание двух измеренных WL, преобразование масштаба |
| LUT hit | CPU, без нового Connector call | Exact повтор уже измеренного mask; нет интерполяции |
| LIF | BindsNET CPU | Leak/threshold/reset/refractory; нет нового STDP/WTA |
| Counts/scaler/head | CPU | log1p, train-only zscore,25epochsSGD |
| Accuracy | CPU | Test labels только после сохранения head/scaler |

SIM выбирается `cb_type='simulator'`, INI задаёт board_type/электрические параметры.
Только публичные `Connector.open_port/impact/close_port` и штатные converters.
**W@input в runtime нет. Разбиения на microvectors нет**: цифровое сложение
частичных MVM отсутствует. Цифровыми остаются разность двух ног, α и LIF/readout.

## Программирование и калибровка

W положительны: `Gminus=L`, `Gplus=L+W*span/max(Wfilter)`, G в Siemens.
L≈25.972µS, span≈128.018µS взяты из заранее измеренной label-free native SIM
characterization `f237904b-0560-4a9a-8c33-f22c69345650`. Точные копии её
manifest/observations находятся в `calibration/`; полный исходный raw-архив
сохранён в `results/cifar_confirmatory_native_20261020/archives/`.

Каждая нога:8RESET .3/.6/.9/1.2/1.5/2/2.5/3V по125µs, затем≤96 feedback
SET1.7V/RESET1.4V по10µs. Это125/10 шагов модели SIM, не аттестованная HW длительность.
Stop при |G-target|≤.02span. SET≤2mA, RESET≤40mA, MVM≤5mA/ожидаемый выход<3.3V.
Промахи target сохраняются; идеальные веса не подставляются.

NativeR использует штатное чтение≈.3V с делителем, MVM — .05V.
Масштаб определён из реальных singleton MVM:
`α=Σ(W·ΔV)/Σ(ΔV²)` по фильтру, без меток/intercept. Из каждой разности вычитается
измеренная разность нулевого вектора. Это не утверждение одинаковой проводимости
при .3/.05V. Singleton RMSE и R-readback RMSE — разные диагностические метрики.

## Exact cache и LIF

После записей тайла — readback/conditioning, три полных прохода mask×WL:
исходный, повтор, заключительный обратный. MVM ADC и converterV должны совпасть
точно; иначе остановка без усреднения/fallback. Все freshR guards сохраняются
и проходят токовую проверку, но не обязаны совпадать: last_resistance меняется
при чтениях. В native исходниках .05V не изменяет state после conditioning,
mode10 не использует last_resistance. На физическую плату это допущение не переносится.
Каждый новый stage — новая cache generation, missing mask вызывает ошибку.

64 фильтра —16 последовательных программирований по4, а не одновременно64
на32×8. Их измеренные LUT объединяются на CPU. Независимые LIF позволяют выполнить
все64 канала за один проход encoder, что проверяется точным unit-сравнением.
Архивный синхронный delay1: t0 получает0, t1..99 использует input0..98.
LIF leak=exp(-1/150),threshold.38,rest/reset0,refractory2,dt1; input копируется,
чтобы in-place refractory masking не испортил LUT. E/I posthook архива фактически
пропускался из-за undefined connection; его не исправляем скрыто.
`.cb` stage snapshot — последняя штатная сериализация; служебный lastR в RAM
после дальнейших чтений может отличаться.

## Бюджет, артефакты и ограничения результата

≤332800pulse-decisions,≤1683200programming calls +9600final readback.
LUT cap после inventory: `16*8*3*Σ_masks(1+popcount(mask))`.
Actual Connector calls отдельно от cached logical filter-MVM. Полный walltime
будет оценён по пилоту, не обещается заранее. Все raw/INI/initial+stage arrays,
исходные W/readbackG/alpha/LUT/features/scaler/head/predictions/sourceSHA сохраняются.

Старый head/scaler69.65% отсутствуют: это новый readout с фиксированным seed42,
SGD lr.001/wd.0001/batch512/25epochs. `randperm` shuffle потребляет RNG иначе,
чем исторический DataLoader; streaming scaler имеет float64 накопление и float32
log1p/sampleSD. Bit-identical69.65% не обещается. Final epoch25 фиксирован до test.
Conventional argmax приведён как архивная метрика; strict считает silent/tied
неверным ответом, **не убирая** эти изображения из знаменателя.

Численный алгоритм после пилота не подстраивался по test: mapping, guards,
LIF, scaler и SGD неизменны. Test labels открываются только для финальной оценки
сохранённого readout. Достижение accuracy выше 50% не гарантируется; эта цель
не используется как условие подбора параметров по test.

После пилота исправлено только журналирование команды: Connector получает
отдельную копию списка напряжений, чтобы штатное SI→DAC преобразование не
переписывало `result.command_si`. Передаваемые напряжения и native вычисления
прежние. Исходный raw пилота с DAC в этом поле остаётся неизменным.
