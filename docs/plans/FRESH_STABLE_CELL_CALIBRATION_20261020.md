# Свежая калибровка стабильных ячеек и знаковых пар, 20 октября 2026

Статус20октября: **новый исполняемый маршрут подготовлен, аппаратного запуска
ещё не было**. Пользователь выбрал новую компактную CIFAR100 subset10
RGB48→8 LIF→10, пилот100, сначала свежая аппаратная калибровка.
Последнее уточнение: лёгкая калибровка, ranking и матрица C считаются на
Raspberry в одном конечном foreground A→selection→B→C; readout/training/
accuracy и независимый аудит выполняются на локальном ПК. Промежуточные
Git roundtrip для вычисления карты не требуются. Пользователь отменил
новые source-approval/grant/receipt барьеры: public source SHA записываются
как provenance, не требуют ручной регистрации; секреты проверяются перед Git.
Защита тока/напряжения, invalid ADC/native R, конечные бюджеты и close остаются.
Предыдущая научно рассмотренная версия документа имела SHA256
`f404bab23f1d9be6d4019dd89110475bd27826e3dd9ca73bc3cad248637b6e3f`;
это историческая ссылка, не SHA текущего текста. Старые25 CIFAR-stages
остаются partial baseline, продолжение all256 не следует из этого документа.
Применимость карты — только0,05В и максимум4 активных BL: иное напряжение
или плотность требует отдельной проверки; карта не доказывает accuracy.

### Заранее фиксированный diagnostic fallback

Основной выбор остаётся прежним: quality-qualified rectangle с максимальным
m×K. Если его нет, рассматриваются все одиночные WL-пары; для каждой берём
до8 общих hard-eligible BL с наименьшим max(normalized quality score обеих
ячеек), при равенстве меньший BL. Пару выбираем по максимуму K, затем
минимуму worst score, затем lexicographic WL-pair/отсортированные BL.
Hard-eligible требует полных валидных измерений, admitted SET/RESET в обоих
циклах и lower span≥max(10ADC,5SD). Noise/drift/zero-shift/cross-cycle и
немонотонность M остаются quality flags с неизменными порогами. Такой выбор
явно diagnostic_best_available, qualified=false. В B missing/wrong-sign или
неразличимые обе знаковые ориентации останавливают путь без rerank; только
soft noise/drift fail допускает диагностические C-измерения. Порог не
расширяется после данных, нет повторов/подбора по CIFAR labels. C qualityfail
также diagnostic, не заявление, что аппаратные вычисления невозможны.

## Граница измерения и защиты

Кандидаты — все256 адресов: WL0..7 ×BL0..31. Единственный I/O — публичный
`manager.board.Connector`, штатные converters, SI task_format и драйвер через
`configs/raspberry_board.ini`; root оператор. Без прямого driver/.interface,
GPIO/SPI/serial, динамической загрузки или подмены измеренных данных CPU.
Проверка tracked source/INI по действующему Git-процессу; согласно прямому
указанию пользователя private drivers не требуют свежего operator receipt.
Ограничение provenance: `private_runtime_not_independently_reverified`;
Git не подтверждает untracked private bytes.

Чтение MVM — mode10, активный BL0,05В, остальные0В, штатный DAC/ADC converter.
Native mode7 read использует штатный vol_read0,3В: оно тоже физическое
воздействие, не обещается неразрушающее чтение. Перед каждым ненулевым MVM
свежий mode7 каждого активного BL на соответствующем WL; прогноз
`sum(max(0.05,native_decoded_voltage)/R)` ≤0,005А и330×current<3,3В.
Нативный R finite>1e-8, mode7 ADC20..16382; mode10 ADC0..16382 и выход≤4В.
Невалидные значения/ID/транспорт — global stop с сохранением prefix.
Ожидаемый отказ guard — recorded skip, не нулевой измеренный результат.
Отказ MVM guard исключает квалификацию соответствующего checkpoint, но
остальные разрешённые адреса можно продолжать в конечном бюджете.

SET sign1, RESET sign0. Перед каждым решением свежий R; writeguard
`requested_voltage/R` ≤0,002А для SET, ≤0,04А для RESET. Амплитуды известного
CIFAR-расписания: [0.3,0.6,0.9,1.2,1.5,2.0,2.5,3.0]В. Длительность
nativeSIM0,000125с/125modelsteps; на hardware submitted0, физическая длительность
неизвестна. Они не считаются эквивалентными. Не увеличивать напряжение,
число попыток или диапазон после неудачного измерения.

Один singleton bracket: zero_before(mode10), fresh R(mode7),
signal0,05В(mode10), zero_after(mode10) =4 impact. Центрированный сигнал
`signal − (zero_before+zero_after)/2` в ADC и converter-volts; отрицательные
значения сохраняются.10 repeats —10 новых brackets, не lookup/reuse.

## Фаза A: все256, воспроизводимость состояния и поздние reread

1. До любых записей10 singleton brackets каждого адреса.
2. Два одинаковых цикла, каждый последовательно проходит три состояния:
   R — все8 RESET-решений; M — первые4 SET-решения после R;
   H — оставшиеся4 SET-решения после M. Здесь нет достижения q-target или
   утверждения монотонности: записываются фактические достигнутые значения.
   Все решения конечны; guard-skips не выдаются за попытки записи.
3. В каждом состоянии адреса идут WL-major/BL-minor. Сразу после его
   собственного программирования —10 brackets данного адреса. После всех
  256 адресов — ещё три полных10-repeat reread-pass, стартующие не раньше
  0/10/60с от последнего programming decision состояния. Если предыдущая
   работа заняла это время, ждать дополнительно не надо; actual time каждого
   чтения обязателен. Это lower-bound задержки, не одновременный снимок.
4. После второго H — финальный10-repeat pass всего массива.

Изменение после операций других адресов обозначается temporal/cross-operation
change; без отдельного эксперимента причина соседнего disturbance не доказана.
60с не доказывают retention на многочасовом CIFAR. Карта привязана к датам,
состояниям/рецептам и0,05В, а не универсальна для будущих записанных весов.

## Предварительно заданные критерии и выбор прямоугольника

Все вычисления только из фазы A, без CIFAR image responses/labels/predictions,
без CPU accuracy и без будущего heldout. Для каждой ячейки на каждом состоянии
и checkpoint сохраняются среднее центрированного ADC, sample SD, mean/SD нулей,
R/G/q (`q=330/R` — proxy, не GUI вес), actual задержки и guard-skip counts.

Определения проекта:
- span = минимум по двум циклам(μH immediate − μR immediate −3×SEdiff),
  где SEdiff=sqrt(SD_H²/10+SD_R²/10); это conservative uncertainty margin,
  не выбранный максимум единичных отсчётов. Для каждого цикла span должен
  быть ≥max(10ADC,5×sqrt(SD_H²+SD_R²)).
- Все state/checkpoint SD≤max(2ADC,0,05×span).
- Отклонение каждого позднего mean от собственного immediate того же состояния
  ≤max(3ADC,0,05×span); нулевой сдвиг≤max(2ADC,3×SE zero-difference).
- Межцикловое различие mean каждого одноимённого состояния
  ≤max(4ADC,0,10×span). M должен лежать между R и H с этим допуском;
  совпадение M с endpoint означает наблюдаемые2 уровня, не3.
- В каждом из двух циклов имеется хотя бы один admitted RESET и хотя бы
  один admitted SET (M/H суммарно), иначе programmable не подтверждено.
  Все обязательные измерения состояний получены; guard-skipped programming
  остаётся описанным, но критерии оценивают достигнутые состояния. Никакого
  исключения только из-за недостижения0,005/0,03/0,05 по q.

Пороговая сетка — proposal, должна быть заморожена после review ДО новых
измерений. Не расширять её по результатам. Слабый стабильный R допустим как
reference leg; SNR отдельно требуется для span, не для каждого около-zero
состояния. Native G drift — отдельная метрика/guard, не автоматическая замена
критерия стабильности фактического MVM.

Рассмотреть все28 WL-пар и все BL. По состояниям каждой ножки оценить
воспроизводимость пары; предварительная верхняя оценка SD разности равна
SD_a+SD_b, для SE — SE_a+SE_b. Из раздельных измерений не восстанавливать
неизвестную covariance. Нельзя считать хороший одиночный CV доказательством
низкого шума разности; фактическая разность проверяется отдельно в B. Для фаз B/C выбрать disjoint WL-pairs с общим набором
BL, максимизируя число coefficient slots m×K (≤128); это2mK физических ячеек.
Общий BL-набор позволяет измерять одинаковые входы по всем выбранным WL и
корректно применять C. Из равных по размеру кандидатов выбрать меньший worst
нормированный шум/дрейф ножек: максимум по всем ножкам/A-checkpoints отношений
  SD/max(2,0.05span), abs(drift)/max(3,0.05span),
  abs(crosscycle)/max(4,0.10span). Окончательный tie — лексикографический порядок
пар, затем BL. Сохранить весь ranking и причины исключения каждого адреса.
Непрямоугольные маршруты оставлены будущему отдельному протоколу.

## Фаза B: независимая физическая проверка выбранных знаковых пар

Кандидатный прямоугольник заморожен после A. Четыре конфигурации по очереди:
+ =H/L, − =L/H, zero =M/M, checkerboard =H/L при чётном индексе BL и L/H
при нечётном. Ножка L: reset8; M: reset8+set4; H: reset8+set8.
Эти mapping recipes фиксированы: L всегдаR, M всегдаS1, H всегдаS2;
  нельзя заменить L наS1 после анализа. Отсутствие воспроизводимого R/Hspan
  бракует programmable leg; стабильная flat-leg сама по себе не signed pair.
  В каждой конфигурации24 programming decisions на coefficient pair.
После программирования каждой пары10 paired brackets (оба WL отдельно),
затем после программирования всего прямоугольника и задержки≥60с ещё10.
Actual интервалы/порядок ножек сохраняются; измерения не атомарные.

Для каждой пары pair_span=min(A_span_a,A_span_b), вычисленный только из A
и замороженный до B. Проверить SD/дрейф differential signal по тем же
span-based порогам с этим pair_span; + и −
должны иметь соответствующий знак и модуль≥max(5ADC,5×SDdiff).
Измеренный zero сохраняется как residual offset, а не насильно обнуляется.
Провал B не вызывает новый подбор пар/порогов, дополнительные writes или
повтор всей фазы. Mandatory failure останавливает путь; soft quality failure
сохраняется как diagnostic_best_available с прежними порогами.
Последняя checkerboard-конфигурация — состояние проверки C, не CIFAR-веса.

## Фаза C: совместный MVM, fit/selection/heldout раздельно

Амплитуда0,05В неизменна,1..4 BL, те же свежие guards. Перед/после C свежий
базис: zero+K singletons,3 repeats, по всем2m выбранным WL. Для singleton
используется bracket4calls. Для zero в КАЖДОМ из3 repeats — три реальные
mode10 zero/zero/zero без mode7, то есть9 zero reads наWL заbasis-pass,
а не3 суммарно. Поэтому6288 — точный maximum для двухbasis-pass.
Для каждой fit/selection/heldout-маски3 полных zero/signal/zero brackets по
всем выбранным WL. Максимум mask budget48fit+24selection+24heldout.

Маски: перечислить `itertools.combinations(sorted(BL), k)` в порядке
k=1..min(4,K), внутриk лексикографически; затем единственный
`random.Random(20261020).shuffle(list)`; взять первыеmin(96,len(pool)).
При96 элементах split48/24/24 последовательными срезами. Иначе использовать
все выбранные элементы и split floor(0.50*n)/floor(0.25*n)/остаток, также
последовательными срезами. Маски не пересекаются, метки не используются. Если не получается минимум1 mask в каждом split, C unavailable;
сохранить singleton-only qualification, без заявления о совместных входах.

Target — сумма свежих singleton-ответов того же состояния. Не обученные веса,
не target/measured coefficient correction. Для basis h_b взять среднее3
centered bracket responses соответствующего BL; zero subtraction внутри
каждого bracket. Все расчёты C float64 в volts, полученных штатным
`convert_adc_to_volt(1,14,5,ADC)` (его round5decimals сохраняется).
Каждый repeat каждой fit-mask — отдельная строка X, до144строк ×d=2m;
не усреднять repeats передfit. Y строки — сумма h_b по BL этойmask.
Собственный intercept отсутствует: measured bracket-centering уже выполнен.

Кандидаты: identity C=I; diagonal C_jj=sum(X_j*Y_j)/(sum(X_j²)+lambda);
full C=solve(X.T@X+lambda*I,X.T@Y).T, lambda=0.0001volts². Ridge применяется
к исходным volts, без предварительного нормирования. Normalization используется
ТОЛЬКО для condition diagnostic. Для X и Y SVD tol=eps_float64*max(shape)*smax;
rank=sum(s>tol). Full только при rank(X)=rank(Y)=d и smin≥16tol в обеих
матрицах, и condition(X с нормированными L2 columns)≤1e6; нулевой column
full исключает. Сохранять spectra/tol/ranks/condition и причины исключения.
Все C finite и operator infinity norm=max_j(sum_k(abs(C_jk)))≤10;
никакого молчаливого pseudoinverse или выбора ridge поданным.

Selection также использует отдельные repeat-строки и RMSE по всем scalar
responses `Xselection @ C.T − Yselection`. Minimum RMSE округлить12decimals;
tie предпочитаетidentity, затем имя. Любой пропущенный обязательный fit или
selection response =>C unavailable, без заполнения и без silent subsetfit.
Heldout следует после freeze, не участвует в refit, выборе пар или C.
Raw и calibrated сохраняются обе.

Heldout differential residual дляpair(a,b):
`((C*x)_a−(C*x)_b)−(target_a−target_b)`, по отдельным повторам.
RMSE/RMS считать вvolts, затем делить на LSB=5/16383V для ADC-equivalent units;
это не заявление об отсутствии native rounding. Отдельно вывести pair×density.
Предварительный acceptance порог≤max(4ADC-equivalent,0.10×RMS targetdiff)
для выбранной C, и отсутствиеguard/missing. RMS targetdiff считается наheldout
по заранее заданной формуле нормализации; это не подборthreshold послеответов
и не selection/refit. При пустом density stratum писатьnot-tested.
Провал совместного теста означает joint qualification failed;
singleton evidence сохраняется. Переход к singleton-only CIFAR возможен только
как явно выбранный следующий маршрут, не скрытая замена эксперимента.

## Конечный бюджет и остановки

| Фаза | Максимум impacts | Максимум programming pulses |
|---|---:|---:|
| A: baseline+final,2×3×4checkpoint,10repeats |282624|8192|
| B:4configs, максимум128pairs |106496|12288|
| C: дваbasis-pass и96mask×3repeat |22416|0|
| Итого |411536|20480|

A:256×2×3×4×10×4=245760 checkpoint impacts; baseline/final20480;
8192guard+write pairs≤16384. B:128×4×24×2=24576 writeguard/impact;
128×4×2checkpoints×10×8=81920. C:2×3×(32×8×4+8×3)=6288 basis;
96×3×8×(3+4)=16128 mixed-mask impacts. Это upper bounds при всех256;
skips и меньший прямоугольник уменьшают фактические calls.

Дополнительный лимит:3600с soft acquisition deadline, проверка перед каждым
новым воздействием и ожиданием; не начинать очередной block если остаток
времени исчерпан. Нативный blocking call не имеет обещанного жёсткого timeout,
поэтому это не гарантированный wall-time cap. Explicit waiting≤600с по
расписанию A/B; actual I/O может поглотить часть ожидания. При наблюдённых
365calls/s максимумcalls≈18,8мин, плюс ожидания≤10мин и файловые накладные
расходы; сценарно≈30мин, не ETA и не гарантия. Driver freeze/transport может
потребовать операторской остановки, не обхода Connector.

Raw limit1GiB expanded, durable chunks≤8MiB, минимум4GiB initial disk и1GiB
reserve; проверять между blocks. Сохранять requests/results/driver stdout,
source-specific actual SI, ADC/R/ID, guards/skips, originalbytes и fullmap256.
Checkpoint после каждой целиком измеренной ячейки и каждой пары; fsync/index
atomic. Interrupt/невалидность =>failed prefix и missing причины; аппаратное
состояние не восстанавливается цифровым checkpoint. Никакого auto-resume SET.

## Повторное использование и результат

Frozen `survey256` не исполняет это расписание:0,1В,3repeats,55-step ramps
и иные checkpoint/selection rules. Frozen STDP8cell ограничен8адресами и
не поддерживает256/совместные входы; его constants не monkeypatch.
Из существующего кода можно обычным import переиспользовать public connector
factory, converters, pure stats/ADC helpers и durable writer с явными caps.
Требуется отдельный небольшой scheduler/collector и независимый replay для
нового расписания; не копировать/менять исторические экспериментальные файлы.
Source-approval/receipt/grant барьеры отменены последним указанием пользователя.
Один конечный nativeSIM даёт функциональное сравнение после offline tests;
аппаратный запуск выполняет root, секреты проверяются перед публикацией.

Результат: полная карта256 с причинами/сырыми ссылками; observed stable ranges
R/M/H, candidate/final pair map иmatrix-fit/report, missing/failed отдельно.
Карта не доказывает возможность записать arbitrary trained weight. После
размещения новых CIFAR-весов обязательны fresh basis/calibration/heldout,
диапазон/quantization check и периодические reread на длительности самого
inference. При исходной сети все75 taps суммируются доBN/LIF; при новой модели
сохраняется её явно определённая полная сумма, без CPU заполнения вкладов.
Никакая CIFAR accuracy этим калибровочным опытом не измеряется.
