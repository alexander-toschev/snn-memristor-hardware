# MNIST32→8: отдельный физический пилот128

Архитектуру **32→8 с обучением проводимостей на плате** выбрал пользователь
19 октября. Scientific review основного протокола разрешает реализацию;
затем обязательны code review, unit tests, native SIM и независимый raw audit. Это не разрешение нового HW
сегодня: сейчас только уже идущая CIFAR-партия. Пилот128 не является full10k,
архивной Conv-SNN или воспроизведением69,65%; точность здесь не измеряется.

## Представление и разделение данных

Все данные из закреплённого train-images IDX MNIST; labels не загружаются.
Источник: `experiments/mnist_baseline_20261019/data/MNIST/raw/train-images-idx3-ubyte.gz`,
SHA256 `440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609`.
На ПК отдельный stdlib extractor создаёт переносимый пакет176 уникальных
изображений с pixel bytes/IDs/source SHA/per-image SHA и exact schedules:
128train+32fit+16probe. Labels и полный ignored IDX на Pi не нужны; provenance
пакета проверяется перед Connector.open. Порядок диапазонов возрастающий,
без shuffle и без отбора по ответам.

| Назначение | Train IDs | Проходы | Обновления |
|---|---|---:|---|
| Gain-fit | 11000–11031 | 1×32 | Нет |
| Обучение | 0–127 | 1×128 | Только WL0–3 |
| Независимые probes | 12000–12015 | pre/post/retention:3×16 | Нет |

Всего208 измеренных изображений. Probes не участвуют в выборе gain/амплитуд/
правила и не становятся test accuracy. Первое сравнение probes только после
заморозки gain; последнее после60с без записи, отсчёт после post-probe checkpoint.

Pooling28×28→4×8: границы строк `[0,7,14,21,28]`, столбцов
`[0,3,7,10,14,17,21,24,28]`; BL=8*block_row+block_column. Для блока с целой
суммой пикселей S и площадью A событие отсутствует при S=0; иначе
`t=min(31,(32*(255*A-S))//(255*A))`. T32, один pre-spike на BL максимум.
Сохранять S/A, исходные IDs, SHA входного файла и exact schedule SHA.

## Размещение, измерение и безопасность

Все BL0–31×WL0–7 используются как256 положительных проводимостей. WL0–3
обучаются; WL4–7 — контроль без обучающих SET/RESET, пары `(j,j+4)`.
Это256 измеряемых синапсов, но только128 обучаемых в данном контрольном пилоте.
Full10k со всеми8 обучаемыми WL потребует отдельного протокола/допуска.

Native SIM seed20261019 для Python random и NumPy; сохранить SHA начального
.cb-массива. У HW начальное состояние фактически измеряется, не приравнивается
к SIM. Начальная подготовка всех256: ровно4 решения RESET1,4В/sign0, затем4 решения
SET1,7В/sign1 на ячейку. Нет q-target, дополнительных попыток или поиска
напряжений. Проходы decision0–7, затем BL0–31, WL0–7; направление WL менять
при нечётном decision+BL. Оба банка получают одинаковый порядок воздействий,
но равенство достигнутых G не предполагается. До/после admitted импульса
измерять fresh nativeR. Полный RESET перед каждым обучающим update не нужен.

Каждый ненулевой вклад — новое измерение public MemriBoard Connector.
Native converters, драйвер только через INI; mode7 read/SET/RESET и mode10 MVM.
Без прямых driver/.interface/GPIO/SPI/serial/FPGA или dynamic module.
MVM0,05В, не более4 активных BL в микрокоманде, BL сортировать/делить подряд.
Перед каждой signal-командой fresh nativeR каждой её активной ячейки;
суммарный V/R≤5мА и330*I<3,3В. V для этого guard —
max(requested .05В, native DAC→V для штатного DAC-кода .05В). Перед импульсом GUI guard:
1,4/R≤40мА RESET,1,7/R≤2мА SET. Guard skip — записанное отсутствие импульса,
без retry. Невалидные R/ADC, MVM current/headroom violation — глобальный stop.
Как reviewed STDP: nativeR finite>1e-8, read ADC20..16382, mode10 ADC0..16382,
native converted voltage finite≤4В. Команды/ответы писать до валидации.
HW: `pulse_width=0` в SI-команде, физическая длительность неизвестна; поле0
не означает0с. Native SIM: явно `pulse_width=0.0001`с SI, как reviewed STDP
(100 модельных шагов в закреплённом симуляторе). Это не установленная
длительность HW и не доказательство эквивалентности импульсов. Сохранять
полную source-specific SI-команду и фактический ответ, без defaults.

Global image_ordinal: fit0–31, preprobe32–47, train48–175, postprobe176–191,
retentionprobe192–207; micro_index начинается с0 на каждом t.
Изображение: zero_before для8WL, все непустые t/microvectors/8WL,
zero_after для8WL **до любого LIF/E или SET/RESET**. Порядок WL прямой на
чётных image_ordinal+t+micro_index, обратный на нечётных; нулевые векторы
по image_ordinal, после — обратным порядком. Каждый signal центрируется
`signal-(zero_before+zero_after)/2` своей WL; микровклады суммируются до
единственного LIF-step соответствующего t. Пустой t имеет математический0,
без lookup/CPU-матрицы. Не clamp отрицательный centered signal.
Общий offset через всё изображение — допущение; сохранять zero drift, не
скрывать его и не объявлять центровку доказательством стационарности.

## Цифровой feedback и заморозка gain

На каждом изображении сбрасывать u, pre/post traces и E. Восемь LIF на Pi
— небольшой online feedback, тяжёлый replay/head на ПК. Проводимости хранятся
и изменяются физически. Нет цифрового target-weight или замены измерения.

`u*=0.5`, затем `u+=gain*measured_voltage`; post=1 при u≥1 и reset u=0,
иначе u сохраняется без нижнего clamp. Traces decay `exp(-1/4)`; затем
`E_i += pre_trace_i*post -1.05*post_trace*pre_i`, затем добавить текущие
pre/post в traces. Так zero-lag не входит; T32 — логические шаги, не реальные
pulse-overlap интервалы. E накапливается только внутри текущего изображения.

Точный порядок:32 свежих fit-изображения → after_fit checkpoint → цифровой
replay сохранённых fit-данных → gain-freeze event → preprobes. Read-only
checkpoint также может менять G; состояние после него не объявляется
тождественным состоянию fit. Выбор **только WL0–3** по grid
`[1,2,4,8,10,20,40,80,100,200,400,800,1600]` В⁻¹. Нижние4 значения
утверждены до новых данных: сумма32 входов может насыщаться при старом gain10. Candidate допустим, если для каждой
из4 WL: суммарно8..64 posts/32fit, ≥4 изображений с posts, ≤8 posts на каждом
изображении, и ни одного post на t с полностью пустым входом. Выбор минимум
`(abs(total_posts_across_four_WL-128),gain)`. Один gain замораживается для
всех8WL. WL4–7, probes, train и labels в выборе не участвуют. Сохранять все
13 candidate trajectories/criteria и ссылки на исходные raw; никакого best-test.
Нет допустимого gain → `uninformative_calibration`, exit3 до probes/train;
grid не расширять. Молчание/насыщение контроля или последующего обучения
отдельно отмечать, не подгонять gain и не считать успехом.

После полного train-изображения для каждой WL0–3 выбрать BL с максимальным
|E|≥0,25, tie по меньшему BL. Максимум одна попытка на WL/изображение:
E>0 SET1,7/sign1; E<0 RESET1,4/sign0. Чётные train IDs: WL0→3, нечётные3→0.
Перед fresh train guardR снять paired control R той же BL,WL+4; после admitted
pulse снять train postR, затем control postR. Skip: оба guardR остаются,
postR не требуется. Ни один импульс контроля не вызывается во время обучения.
Все8WL E/shadow-choice сохранять, но control choices не исполнять.

## Точный конечный бюджет и checkpoint

Полный checkpoint точно как reviewed conductance STDP: каждый из256 адресов,
3nativeR и3 повторения `(zero_before, fresh nativeR guard, singleton .05В signal,
zero_after)` =15calls/cell =3840calls. В каждом повторении новая guard-R
непосредственно перед signal. Это не survey-протокол с .1В. Обход адресов
WL0–7, затем BL0–31, repeat0–2; нулевые вызовы отдельные для каждой ячейки.
Токи по max(requestedV,native decoded DAC V), как и у основных signal-команд.

14 checkpoint: initial; after_preparation; after_fit; after_preprobe;
после каждых16train (8); after_postprobe; after_retentionprobe.

| Раздел | Верхний impact | Верхний pulse |
|---|---:|---:|
|208 изображений×(272mode10+256guardR)|109824|0|
|Подготовка256×8×(guardR+pulse+postR)|6144|2048|
|Train128×4×(controlR+guardR+pulse+postR+controlPostR)|2560|512|
|14 полных checkpoint|53760|0|
|**Всего**|**172288**|**2560**|

Нет gain-кандидата: initial/after_preparation/after_fit и32fit;
cap34560impacts/2048pulses. Фактические skip уменьшают счётчик, не разрешают
расходовать остаток на новые действия. Close/open не mode-impact; все native
внутренние аппаратные операции за одной командой не выдавать за этот счётчик.

## Raw, контроль и критерии результата

Логировать каждый request/result, native stdout, guard, ADC/nativeR, временные
метки, image/t/WL/BL, source/INI/input/SPEC SHA, gain-freeze до probes, LIF/E,
решение/pulse/postR, причины skip и все нулевые отсчёты. Durable chunk/index
fsync не реже каждого завершённого изображения и после каждой update-пары;
прогресс отдельно, нельзя переписывать исходные данные. До нового запуска —
достаточный заранее рассчитанный disk cap, без удаления прежних raw.

Контроль парный, но начальные G/пороги ответов разные: сравнивать изменение
от собственных pre-baselines, same-image pre/post/retention, ΔG знак и размер,
zero drift/variance, posts/E/coverage и train-vs-control descriptively. Sham
соответствует явным guard/post reads, не внутреннему read-dose драйвера.
Повторные read-only probes/контроль тоже могут disturb G; это измерять.
Обязательная диагностика collapse: per-image/WL post-traces/counts, доля
молчания, распределение counts, число уникальных8-нейронных выходов, pairwise
точное совпадение всех traces и отдельно conditioned-on-any-active;
Hamming/Jaccard (пустой union → undefined, отдельно от молчания), occupancy выбранных BL и SET/RESET по WL, разнообразие
G до/после (mean/CV и попарное сходство строк), без порогов, выбранных после данных. Отдельно флаг: все4 learning-WL дают одинаковые post-traces на всех
128 train. Это отсутствие различия выходов в этой выборке, не доказательство
идентичности физических весов. Нет WTA, homeostasis или скрытого rescale.
Общая подготовка и отсутствие конкуренции могут привести к сходным нейронам;
такой результат сохраняется, а не исправляется подбором по probes.
Нет обещания causality/accuracy/полезного обучения по одному пилоту.

Механизм информативен, если есть posts, eligibility, admitted training pulses
и связанный native postR; иначе результат явно partial/неинформативный по
соответствующему звену, а не pipeline failure автоматически. Zero-input-post=0
при подставленном математическом0 и leak/reset в основном структурное свойство,
а не экспериментальное доказательство отсутствия offset/noise. Отдельно сообщить
положительные/отрицательные E, SET/RESET, знак ΔG и неподвижные/шумные ячейки.
Отдельно считать per-BL input-event count и per-cell участие в image-MVM,
checkpoint/preparation coverage и update occupancy. Нулевые pooled-блоки
могут ни разу не участвовать во входе, хотя их ячейки обследованы.
Все256read не означают256 image-contributing или256обучены. MVM probes показывают накопленный эффект
окна, не изолируют MVM-эффект каждого импульса. SIM не доказывает HW retention.

После аварии нельзя восстановить G из цифрового checkpoint и автоматически
повторить изображение/updates. Сохранить raw, остановить, отдельный state-read
план/review перед продолжением. Native SIM: основной тот же protocol; дополнительный заранее маркированный
reference без training pulses, seed20261019 и SHA исходного SIM-массива должны
совпасть; те же preparation/inputs/selected gain
из основного fit с SHA-ссылкой на его raw и gain-decision. Reference нужен
только если основной gain найден; его собственные fit-данные не меняют gain.
Shadow selection определяется собственными измерениями/LIF/E reference,
не копируется из изменившегося train-банка основной сессии. Для выбранных
попыток reference выполняет4 чтения
(controlR, trainR, trainpostR, controlpostR), без pulse: cap171776impacts/2048
preparation pulses. Это отдельная сессия, не скрытая часть HW172288cap; её
сравнение не доказывает, что начальное аппаратное состояние воспроизведено.
Reference допускается только в SIM: hardware source отклоняется до Connector;
это отдельный позднее проверяемый режим, не условие готовности основной реализации.
Одинаковые seed/preparation не сохраняют одинаковые случайные draws после
расхождения путей с обучающими импульсами и без них. Перед платой необходимы
source review, fake tests, actual native SIM и независимый natural-raw audit.
