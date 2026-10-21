# MNIST64 block STDP — только MemriBoard Simulator

Это **новый** протокол с отложенной записью, не повтор timestep-STDP и не
прежний результат64,29% на готовых весах. Fresh initialization64×5×5,
10000train,10000officialtest. Один32×8массив последовательно хранит16групп
по4фильтра. Hardware-entry отсутствует.

1. CPU генерирует freshweights seed42, архивную initial normalization и clip.
   Public `Connector` программирует пары WL с native SET/RESET/readback.
2. Перед блоком32/64/128/256изображений читаются все200ног, по3R.
   `rawW=(Gplus-Gminus)/span`, `target0=clip(rawW,0,1)`.
3. Все встреченные25bitmasks измеряются через native mode10 полного32vector.
   Три прохода требуют точного совпадения ADC/V. CPU не считает W@input.
   Фиксированный gain `1/(330ohm*decodedVin[V]*span[S])` и разность двух WL
   переводят выход в единицы LIF. Во время блока проводимости неизменны.
4. CPU последовательно выполняет100ticks каждого изображения: native-LUT
   forward, цифровой LIF/traces, штатный WeightDependentPostPre. **Цифровой
   target переносится между изображениями и clip выполняется каждый tick**;
   target используется только в STDP weight-dependent factors, не в forward.
   LIF/traces сбрасываются перед каждым изображением. Нет усреднения batch.
5. После блока каждый mode10 повторяется для exact end-check. Новый readback
  200ног перед решением записи; на каждом из100plus-coefficients максимум
   одинSET1,7V илиRESET1,4V,10modelsteps. Deadband0,002span. Native guard перед
   импульсом,3postR, затем весь массив вновь измеряется. Кеш инвалидируется.
   Clipping correction, корреляционная plasticity и readback drift записаны
   отдельно; correction не выдаётся за корреляционное обучение.
6. После обучения группы frozen native LUT сохраняется. После всех16групп
   CPU объединяет измеренные LUT64каналов, вычисляет LIFfeatures, новый
   train-only scaler/SGD25readout. Test labels открываются после savedhead.

Current/voltage/ADC guards сохранены. Никаких обращений к private driver,
state_variable или `.interface`. Чтения не кешируются; mode10 reuse разрешён
только для проверенного frozen состояния. SET/RESET меняют проводимость,
не цифровой membrane LIF. Baseline-ноги после инициализации не записываются.

## Запуск из корня репозитория

Окружение `.venv-mnist-pilot`, зависимости/данные как у архивного маршрута.
Все пути относительны. Сначала runtime benchmark, не подбор по accuracy:

```powershell
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_block_stdp64_20261021/experiment.py --run-simulator --benchmark --block-size 32
```

Два блока можно измерить на более плотном заранее выбранном train участке:
`--benchmark-start 8960`. Benchmark использует первые128train/test для
измерения конечной CPU стоимости4filterfeatures и fit; **test labels не
читает, accuracy не вычисляет**. Нельзя линейно приписывать его4filteraccuracy
полной сети. Размер блока выбираем по времени/памяти, не меткам или точности.

```powershell
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_block_stdp64_20261021/experiment.py --run-simulator --block-size 32 --wall-hours 8
```

CLI без `--run-simulator` только preflight. Верхний импульсный бюджет:
`16*(20800+100*ceil(10000/block_size))` решений; calls ограничены конечным
верхом по union входных masks. Wall-budget8h проверяется перед nativecall и
каждым изображением/CPU progress, closePort выполняется при остановке.
`status=failed,error_kind=wall_budget_exhausted` — **partial**, не завершённая accuracy. Это cooperative
deadline; текущий публичный вызов/CPU batch может закончиться чуть позже.
Для выбора full aim≤6,4h из8h; учитывать реальные плотные блоки,16init и
финальные frozen features/readout, а не умножать только sparse image0.

Результаты `results/mnist_block_stdp64_20261021/<UUID>/`: manifest, per-block
G/target/spikes/stats/timing, frozenLUT,counts,head,scaler,predictions. Native
raw сохраняется полностью для первого учебного блока каждой группы;
прочие низкоуровневые request/result/read/debug свёрнуты в counters+SHA.
Это **sampled evidence**, не lossless/fullrawreplay. Все pulse decisions и
readback summaries сохраняются. Старые raw не удаляем. Максимум64GiBexpanded
нового журнала, reserve4GiB. SHA пропущенного потока не позволяет восстановить
его значения. Source/INI/protocol сохранены; private runtime не переаттестуется.
