# Новое online STDP в MemriBoard SIM

Это **новое обучение**, без `model.pt` и других готовых весов. Сохранённые
64,29% предыдущего опыта — отдельный неизменный контроль. Hardware CLI нет.

Из корня репозитория, в существующем Python3.11 окружении:

```powershell
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_online_stdp64_20261021/experiment.py --smoke
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_online_stdp64_20261021/experiment.py --run-simulator --smoke
```

Первая команда — preflight без Connector/нейросети, вторая — **только4filter,
image0,100ticks**, pre/post frozenprobe. Это не64filter accuracy. Полный маршрут
запускается без `--smoke` только после оценки времени smoke оператором:

```powershell
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_online_stdp64_20261021/experiment.py --run-simulator
```

## Что и где происходит

1. CPU: новое uniform[0,.3] состояние64×5×5, seed42, нормировка каждого
   фильтра к L1=78.4. Исходный массив и clip[0,1] сохранены отдельно: clipping
   первого нулевого timestep архива не выдаётся за обучение.
2. Public Connector→native SIM: один32×8 массив,25BL×8WL;4filter по двум
   дифференциальным ногам. Gminus=L, Gplus=L+span×clippedW. L/span только из
   прежней label-free характеристики SIM. Инициализация конечная:8RESET
   .3/.6/.9/1.2/1.5/2/2.5/3V×125µs, затем≤96feedback импульсов1.7SET/1.4RESET
   ×10µs. Это модельные125/10шагов, не аттестация физической длительности.
3. Для каждого изображения CPU выдаёт архивный content-deterministic Poisson:
   rate.0015, seed123,100ticks,5×5stride1,576spatial positions. Метки не нужны.
4. Каждый timestep:200ног×3nativeR→G=mean(1/R); native mode10 для всех
   различных **полных25bit masks** предыдущего входа. Для каждого mask/WL
   три exact прохода; ADC/V должны совпасть. FreshR/current/headroom guards
   перед каждым запросом. CPU только раскладывает измеренные ответы по
   повторяющимся spatial позициям, вычитает ноги и измеренный zero offset.
   Cache не переносится между timestep; любой импульс его инвалидирует.
5. Fixed gain=1/(330Ω×decodedDAC(.05V)×span[S]) переводит measuredΔV в nominal
   W units. Никакого refit по desiredW после обучения. NativeR≈.3V и MVM.05V
   могут давать разные эффективные проводимости: шкала — объявленная модель,
   не доказательство точного линейного переноса. Forward не clip.
6. BindsNET на CPU: LIF leak exp(−1/150),threshold.38,refrac2,rest/reset0;
   traces decay exp(−1/20),nonadditive. MVM использует pre[t−1], но STDP после
   обоих forward использует pre[t]/trace[t],post[t]/trace[t]. Native
   WeightDependentPostPre суммирует576позиций в25shared weights каждого
   фильтра. Его цифровая connection используется **только для update**,
   `compute`/идеальная свёртка не вызываются.
7. WDP получает clip((Gplus−Gminus)/span,0,1), оба unclipped/clipped массива
   записаны. Ставки+.001pre/+.0001post; LTD уже со знаком минус внутри правила.
   Target ограничен[0,1]. При |target−rawMeasuredW|>.002 — максимум один
   SET1.7V либо RESET1.4V×10µs наplusleg заtick. Далее3nativeR, mismatch/знакΔG
   сохраняются. Guard-skipped не считается импульсом; идеального shadowW нет.
   Out-of-range measuredW может вызвать коррекцию границы даже при нулевом
   корреляционном update; это видно в raw и счётчиках.
8. После всехN изображений блока — frozen native LUT train/test без записей,
   цифровые LIF counts. Затем следующий блок4filter на **том же массиве**.
   Нет скрытых16независимых физических массивов. Математическая независимость
   следует из неработавшего E/I/postnorm hook архива; WTA не добавляется.
   Порядок SIM noise новый, побитная идентичность старому64interleaved не обещается.
9. Только после всех64filter: новый CPU readout (log1p,train-only sampleSD,
   floor1e−6,Linear+bias,SGD25,seed42). Test labels открываются после сохранения
   head/scaler. Strict/argmax, silent/ties и10000denominator как в baseline.

SET/RESET накопительно изменяют синаптическую проводимость. RESET устройства
не равен сбросу membrane LIF. Сам LIF здесь цифровой; нейронные схемы с
мемристорами и конденсаторами не отрицаются, но текущий crossbar mode10 их
не реализует. Ни один этап этого маршрута не является аппаратным запуском.

## Конечные бюджеты и сохранение

На блок initialization≤105800calls/20800pulses. На изображение100ticks:
60000nativeR calls +≤50000learning calls,≤10000learning pulses. Дополнительно
точный MVM budget по входным masks:24×Σ(1+popcount(mask)), включая zero, для
каждого timestep и frozenprobe/LUT. Он считается до открытия Connector.
Для full64×10000 только freshR — **9,6млрд calls**, до MVM/записей. Поэтому
время full не обещается; smoke отдельно измеряет init/preprobe/online/postprobe
seconds/calls/pulses. При экстраполяции умножается online phase, не init.

Raw expanded cap256GiB, chunks8MiB, reserve2GiB, initial free≥8GiB. Это
fail-closed ограничения, не гарантия, что полный10k поместится. Ошибка/полный
диск/неповторяемый MVM останавливают опыт с сохранением префикса и close_port.
Нет resume, которое выдаёт восстановление старой проводимости за состоявшийся
факт. Results: `results/mnist_online_stdp64_20261021/<UUID>/`.

Raw содержит SI/native запросы/ответы, G, current/trace inputs, post spikes,
targets, pulse guards/admission, postR/ΔG/residual. Manifest содержит counters
LTD/LTP/deadband/skipped/out-of-range и отдельные phase timings. `.cb` snapshots
отражают последнюю штатную сериализацию SIM, не точный live last_resistance.
Pulse/deadband адаптация **не точный цифровой архивный STDP**; пропущенные
малые изменения и overshoot не скрываются и не исправляются по test accuracy.

Строгая типизация: `typing/mnist_online_stdp64/mypy.ini` проверяет все новые
Python-файлы. Узкие `.pyi` описывают фактические untyped BindsNET API; старые
public MemriBoard/Writer границы переиспользованы. Они не исключают новый код
из проверки. Unit golden сравнивает timestep/spikes/traces/weights с настоящим
BindsNET Network, а идеальная conv используется только внутри этого теста.
