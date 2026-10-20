# Компактный CIFAR100 subset10: RGB48 → 8 LIF → 10

Статус: specification для независимого review; обучение, native SIM и плата
этим документом не запускаются. Пользователь выбрал новую компактную модель
и порядок «сначала плата». Сначала отдельно квалифицируем свежую калибровку
по `FRESH_STABLE_CELL_CALIBRATION_20261020.md`, получаем аппаратную stable map,
затем фиксируем конкретный mapping/programming protocol и готовим новый
checkpoint. Это новая сеть, не исходная Conv1/BNTT сеть и не обещание её accuracy.

## Данные и разделение

Официальный CIFAR100 Python dataset уже найден и побайтово проверен локально.
Рабочие пути после отдельного копирования: `data/cifar-100-python/train`,
`test`, `meta`; абсолютные пути в код/manifest не включать. Новая загрузка не нужна.

| Файл | Байты | SHA256 |
|---|---:|---|
| train |155249918|735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b|
| test |31049707|4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864|
| meta |1473|a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9|

Выходной порядок fine labels `[0,1,2,3,5,8,13,14,17,19]`, remap0..9:
apple, aquarium_fish, baby, bear, bed, bicycle, bus, butterfly, castle, cattle.
Из official train берём все5000 изображений этих классов в исходном порядке.
Позиции внутри этого отфильтрованного списка делим существующим
`weights/split_manifest.json` SHA
`0967934d6f3fe1c6a8dbbaf42e32d3eb1416428ba47579a64734d37d25cc0690`:
4500 train,500 validation. Не путать эти позиции с original official indices.
Новый manifest сохраняет оба вида индексов, labels и imageSHA; проверяет
раздельность, полноту5000 и отсутствие official-test в обоих split.

Test pilot100 фиксируется **до обучения/подбора checkpoint**: для каждого
fine label первые10 изображений в official-test order, исключая все10
исторических fixture indices `{9,132,54,396,334,27,94,7,49,122}`.
Исключения привязаны к `weights/cifar100_conv1_fixture.json`, поля
`cases[].original_test_index`, а не к ошибкам/успехам старой сети.
Сериализация pilot — по возрастающему original test index;10/class ровно.
Это детерминированный пилот, не случайная выборка и не весь test1000 subset.
Другие test890 и исключённые10 не участвуют в tuning.

Подготовка sealed manifest вправе читать labels только для стратификации.
Acquisition fixture содержит image/source/index/pooling/spike SHA, но не labels.
Labels находятся в отдельном evaluation-файле; физическая запись, выбор ячеек,
gain/C/thresholds не получают их. Dataset читать только после exactSHA:
предпочтителен существующий ограниченный opcode parser без pickle callbacks.
Новый checkpoint, image/split manifest и protocol фиксируются до test forward.

## Точный preprocessing и модель

Без crop, flip, channel centering, augmentation и обучаемой нормализации.
RGB uint8 CHW32×32 разделить на неперекрывающиеся8×8 блоки.
Получаем3×4×4=48 признаков, порядок `input=16*c+4*y+x`.
Для блока integer sum `S` в `[0,16320]`, интенсивность `S/16320`.
TTFS T12 определяется целочисленно:

* `S<1632`: spike отсутствует;
* иначе `t=floor(11*(16320-S)/16320)`, ровно один spike.

Это единая арифметика CPU/nativeSIM/HW, без промежуточного uint8 округления.
Нули/белое/порог/границы времени покрываются тестами; schedule сохраняется.

Входная матрица `W[8,48]`, signed, bias отсутствует,384 коэффициента.
На шаге t для нейрона j:
`i[t,j]=sum_k W[j,k]*s[t,k]`;
`u=.5*v+i`; `z=1[u>=1]`; `v=(1-z)*u`.
Начальное v=0 для каждого изображения; отрицательные значения сохраняются.
T12 полных шагов, даже когда входных spikes нет; hard reset в ноль.
Никаких дополнительных leak/reset между физическими microvectors/тайлами.

Readout цифровой: `features[j]=sum_t z[t,j]/12`,
`logits=A[10,8]@features+b[10]`. Его80 weights и10 bias **не являются
физическими384 weights**. Основной decoder — argmax logits; не добавляем
после test альтернативные обучаемые декодеры или temporal weights.

## Конечный CPU training protocol — после аппаратной карты

Новая отдельная реализация, не изменение frozen CIFAR/MNIST acquisition.
Один seed20261020 для Python/NumPy/Torch; отдельный Torch generator с тем же
seed для train permutation. CPU float32,2threads, deterministic algorithms,
зафиксированные версии; без silent GPU fallback и без sweep по seeds.

W инициализируется uniform[-1/sqrt(48),1/sqrt(48)].
A uniform[-1/sqrt(8),1/sqrt(8)], b=0.
Adam lr=.001,betas(.9,.999),eps1e-8,weight_decay0,amsgradFalse;
cross-entropy mean;50epochs; batch128,shuffle train only,drop_lastFalse,
num_workers0. После каждого optimizer step W проецируется clamp[-1,1].
Это36updates/epoch,1800updates всего, последний train batch20.
Validation500 целиком после каждого epoch, без gradient/optimizer updates.
Checkpoint — минимальный validation mean CE, вычисленный как сумма loss
по500/500; при точном равенстве earliest epoch. Test не вызывается до freeze.

Hard threshold forward остаётся указанным выше. Backward surrogate:
для a=u-1 производная spike `1/(1+5*abs(a))²`;
hard-reset multiplier `(1-z.detach())`, то есть без градиента через reset mask.
Полный BPTT12, без truncated-state carry между изображениями.
Обе детали проверяются отдельно и сохраняются в model manifest.
Это supervised surrogate-gradient обучение цифрового эталона, не STDP
и не обучение мемристора на test-изображениях.

Сохраняем init, все50 epoch metrics/permutation hashes, final optimizer/RNG,
selected checkpoint+epoch, W/A/b, exact train/val IDs и source/environment pins.
Для небольших tensors допустимы npy+JSON; Torch checkpoint restricted loading.
Никаких повторных запусков с изменёнными гиперпараметрами после test.
Если pilot100 оказался слабым, это результат данного протокола; новая версия
должна иметь новый заранее выделенный test и отдельное разрешение.

## Полное физическое представление и сумма перед LIF

Stable rectangle из calibration имеет m disjoint WL pairs и K общих BL,
1≤m≤4,1≤K≤32. Его выбор использует только calibration, не CIFAR labels,
checkpoint accuracy или test responses. Необходима qualified signed B map;
flat reference leg не превращается в programmable pair.

Output groups `G=ceil(8/m)`, input tiles `Q=ceil(48/K)`.
Порядок output groups, затем input tiles — возрастающий.
Каждый из384 W[j,k], включая нулевой target, получает физическую пару и
новую measured provenance. Tail BL не активируются; unused WL pair не
используется как запасной CPU канал. Геометрия и все384 mappings фиксируются
перед первым programming. При любой недостаточной geometry не запускать
усечённую сеть под именем48→8.

В каждом tile: bounded программирование своих targets; fresh native R,
post-write singleton basis и независимые fit/selection/heldout измерения на этом
состоянии. Fit/selection/оценка C выполняются на Raspberry по этим данным.
C target — сумма **измеренных** singleton responses, не W из CPU.
Никаких per-weight digital corrections или fitted target/measured ratios.
Входное напряжение .05V, density≤4; активные BL делятся по возрастающему BL
на microvectors≤4. На каждом t суммируются их измеренные вклады, затем все
input tiles48, и только после этого выполняется один LIF update.
Все логические T12×8×100 currents имеют полный набор физических вкладов.
Пропуск измерения/guard refusal — missing evidence, не численный ноль.
Для пустого t берётся actual zero/zero/zero bracket; silence не разрешает
пропуск измерения. Общие zero brackets между t в этом новом маршруте не
вводятся без отдельного review.

Raw и C-corrected пути публикуются оба, используют один фиксированный
V-per-checkpoint-weight gain внутри данного source и одинаковый A/b/LIF.
Разница gain/state между nativeSIM и HW указывается явно.
Gain, common-mode/target mapping, конечный pulse budget, допускаемые
approximation errors и retention schedule надо **отдельно зафиксировать
после фактической карты и до test**. Этот draft не выдаёт произвольную
conductance map за умение писать learned weights и не открывает HW gate.
На длительности100 изображений нужна периодическая проверка сохранности;
старый calibration C на checkerboard не переносится на новые weights.
CIFAR responses не используются для перенастройки C, gain, карты или weights.

Pi выполняет физические calls/online guard/checkpoints, статистику стабильности,
ranking/выбор rectangle и C fit/selection. Calibration A→selection→B→C идёт
единым ограниченным потоком на Raspberry. Обучение цифрового эталона/readout,
сбор финальных currents, digital LIF/readout inference, тяжёлый numerical
replay и сравнение — локально на PC. Raw и результаты переносятся через Git;
далее отдельный learned-weight protocol. Паузы/переносы не считаются
сохранением физического состояния: fresh basis/retention evidence и
актуальность C должны быть явно включены в следующий reviewed protocol.
Только public MemriBoard Connector, native converters и INI.
Private runtime provenance — `private_runtime_not_independently_reverified`;
никаких новых ручных driver receipts. Public source/INI/raw checks сохраняются.

## Число операций и предварительный ресурсный контракт

На изображение:48 TTFS inputs (не более48 spikes),12×8=96 digital LIF updates,
384 weights и80 digital readout MAC. Dense CPU reference считает
12×48×8=4608 input MAC/image. На100:9600 LIF updates,460800 dense input MAC,
8000 readout MAC. Это не количество MemriBoard impacts.

Для input tile длины k и конкретного изображения:
`M=sum_t max(1,ceil(active_count[t]/4))`.
На каждый из2m используемых WL:3 mode10 calls на каждый microvector/bracket,
плюс один fresh mode7 R на каждое активное BL в этом microvector.
Inference impacts ≤`sum_groups,tiles,images 2*m_g*(3*M+k)`.
Фактический точный budget вычисляется из sealed unlabeled schedules и
mapping до launch; программирование/basis/C/retention добавляются отдельно.
Для максимально широкой m4/K32: G2,Q2, k32+16,
грубая безопасная граница M≤12+ceil(k/4) даёт≤249600 inference impacts/100.
Для более узкой stable map цена выше; не обещать её время до карты.
Для каждого nonempty vector mandatory current guard может остановить
полноту опыта; signed/mixed result не компенсируется CPU.

CPU train volume50×4500×12≈2.7M sample-time steps,24M scalar LIF updates
включая50 validation passes; измерить bounded smoke runtime/memory после
реализации, не выдавать число операций за ETA. Dataset около186MB;
точные raw/storage/deadline/pulse caps следуют отдельному mapping protocol.
На этом этапе тяжелое обучение и hardware classification не запускать.

## Оценка и qualification

Одинаковые sealed100 images, preprocessing, W/A/b, T12 и LIF для CPU,
nativeSIM и HW. CPU baseline до HW report; nativeSIM — собственный native
массив/initial-state provenance, не файл с CPU ответами.
HW все384 коэффициента/все выбранные tiles, без CPU fill/cache lookup.
Частичный сбор сохраняется, но dataset accuracy полного маршрута не объявляется.

Для raw/C/CPU отдельно: confusion10×10, conventional top1 numerator/100,
per-class count/10, per-image logits/prediction, spike counts/first times,
all-eight-silent, exact ties и near-ties (top1−top2≤1e-6).
Механический argmax при tie сохраняется как индекс, а не уникальный ответ.
Head bias может дать уникальный правильный ответ при полной тишине:
conventional top1 это учитывает, но informative coverage и дополнительный
`informative_correct/100` исключают silence и near-ties. Ничего не удалять
из denominator100, не называть bias-only ответ доказательством полезного MVM.
95% Wilson interval для conventional top1 — номинальный биномиальный
descriptive interval. First10/class является детерминированной стратифицированной
выборкой, а не IID random sample; это не design-based confidence interval
качества на популяции. Он не доказывает качество на всех1000/10000 test
или на исходной64-filter сети.

Проверки до запусков:

1. Dataset SHA/indices/remap/exclusions, pool/TTFS integer boundary tests,
   no train/val/test overlap, acquisition fixture без labels.
2. Pure forward и surrogate/reset gradients; воспроизводимый init/permutation;
   validation selection; test callable недоступен до checkpoint freeze.
3. Variable m/K exact384 mapping, tail/no duplicates; microvector/tile
   summation до одного LIF update, including zero inputs/negative currents.
4. Independent raw audit, missing/duplicate/guard/ref/source mutations,
   pulse/raw caps, close/error/crash evidence; no partial-as-full classifier.
5. Реальный nativeSIM на том же новом programming protocol и независимый audit,
   после review. Успех calibration SIM не означает успех learned-weight SIM.
6. Root выполняет reviewed physical route из Git checkout; затем локальные
   audit/classification и paired per-image CPU/SIM/HW отчёты. Никакой
   автоматической смены subset/threshold/model ради лучшего test результата.
