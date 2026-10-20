# Native MemriBoard SIM: запись/чтение → hybrid inference

Отдельный маршрут после clean4seed. Все 4800 signed Conv1 coefficients каждого
checkpoint проходят native MemriBoard Connector, mode7, штатные converters и
INI. Прямых BoardSimulator/HardCore/interface вызовов нет. Остальная сеть,
а также применение восстановленного Conv1 к изображениям, — PyTorch.
**Это hybrid programming/readback experiment, не literal analog MVM.**

## Mapping до тестовых меток

Characterization `f237904b-0560-4a9a-8c33-f22c69345650`: 8 фиксированных ячеек
`(WL,BL)=(w,4w)`, два цикла, каждый RESET8→SET8, напряжения
`[.3,.6,.9,1.2,1.5,2,2.5,3]` В. SIM pulse_width125µs соответствует125
модельным шагам, не утверждённой длительности физической платы.
Перед каждым импульсом свежий nativeR: RESET≤.04A, SET≤.002A; после решения
три nativeR чтения. 1304 impacts/256 pulses, независимый raw replay PASS.

G здесь строго **1/R, Siemens**, а не GUI weight и не proxy330/R.
Медиана16 RESET endpoint G =9.969779604µS,
медиана16 SET endpoint G =169.991993377µS.
Центральные80% зафиксировали до просмотра новых hybrid test scores:
L=25.972000981µS, H=153.989772000µS, span=128.017771018µS.
Это типичный calibration диапазон, не гарантия достижения каждой ячейкой.

Для фильтра f с `a=max(abs(w_f))`:

```
Gplus_target  = L + max(w,0) * span/a
Gminus_target = L + max(-w,0) * span/a
w_measured   = (Gplus_readback-Gminus_readback) * a/span
```

Дифференциальные пары WL(0,1),(2,3),(4,5),(6,7), BL0..31;
64filters×75taps покрываются48 тайлами (четыре фильтра, три участка75taps).
На каждом leg ровно8 RESET решений, затем до8 SET; остановка SET при
crossing или `abs(G-target)≤max(.01span,3SE)`. Новых попыток/подстройки
mapping после результата нет. Пропуск по current guard сохраняется с
причиной, readback остаётся измеренным. Никакой ideal/zero fill.
Final readback тройками выполняется после **всех** записей текущего тайла.

Максимум825600 native calls /153600 pulses на checkpoint.
Сохраняются initial/native persisted array,48 native persisted tile arrays,
полный request/result/driver_output, targets, measurements, guard/skip decisions,
ошибки/знаки и4800 reconstructed weights. `.cb` — сохранённое native driver
состояние; сами readback отсчёты проверяются по raw, не заменяются `.cb`.
Random initial-state seeds:20261020+checkpoint_seed; characterization20261021.
Штатный noise модели не переопределяется через её внутренности.

## Команды

Characterization уже выполнена, повторять для выбора лучшего результата не надо.

```console
python -B experiments/cifar_confirmatory_20261020/native.py --seed 0 --characterization results/cifar_confirmatory_native_20261020/f237904b-0560-4a9a-8c33-f22c69345650
python -B experiments/cifar_confirmatory_20261020/hybrid.py --session results/cifar_confirmatory_native_20261020/UUID --output-dir reports/2026-10-20/cifar-confirmatory-native/seed0
```

Повторить фиксированный programming protocol для seeds1,2,42. Native CLI
вообще не открывает test images/labels. Hybrid сначала независимо проверяет
все команды/guard/readback/reconstruction и только затем открывает test1000.
Все четыре результата сохраняются независимо от accuracy; нет выбора seed,
ячейки или mapping по test. Conventional FS/TWD нужны для сравнения с clean,
strict silent/tied-abstention также сохраняется со знаменателем1000.

Это один bounded approximate programming опыт на seed. Он не воспроизводит
исторический HardCore mean-of-five/readback режим и не доказывает аппаратную
accuracy, межсеансовую retention или аналоговую сумму токов при плотных MVM.
