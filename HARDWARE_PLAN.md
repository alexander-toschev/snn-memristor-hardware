# CIFAR Conv1: четыре физические фазы

Пользователь согласовал native-импульс неизвестной длительности и один массив
32×8 с последовательной перезаписью плиток. Точка входа — `experiment_hardware.py`.
Старые simulator scripts, результаты и `read_all.py` не меняются. Аппаратного
прогона новой реализации при подготовке ещё не было.

```text
python experiment_hardware.py --dry-run
python experiment_hardware.py --run-hardware --phase phase1
python experiment_hardware.py --run-hardware --phase phase2 --phase1-session results/hardware_four_phase/<uuid>
python experiment_hardware.py --run-hardware --all
```

Без флага подключения нет. По умолчанию phase1. Для phase2/3/4 нужна завершённая
физическая phase1 с совпадающими SHA кода, INI, плана и host; `--all` проверяет
условия phase1 перед mapping. `--max-pulses` уменьшает общий бюджет nonzero mode7.
Exclusive lock общий с read_all.py; он не блокирует сторонние приложения платы.

## Фазы

1. 63 baseline impact: ячейки `(WL,BL)=(0,2),(2,6),(2,4)`, одно чтение и 20
   повторов каждой. Затем 75 попыток: три ячейки × пять q-целей × пять циклов,
   пять post-read после каждой записи. q-цели .01/.02/.03/.04/.05.
2. 14 фильтров `[63,41,32,30,38,43,60,47,2,9,46,12,0,1]`.
3. Все 64 фильтра Conv1 отдельным проходом.
4. Фильтр 41: три отдельные записи, средние K=1,2,3. Статистическая независимость
   записей не установлена; улучшение не гарантируется.

Веса `weights/conv1_weights.npy`, float32 `[64,3,5,5]`, проверяются по SHA256.
75 taps каждого фильтра разбиты на 32/32/11. Знаки размещаются на WL0/WL2:
q±=.03±.02*w/maxabs(w). WL1 в programming не используется. Все MVM текущей
плитки измеряются до перезаписи следующим tile. После каждой записи — три
immediate-readback; после завершения всех записей плитки — ещё три чтения
каждой её ячейки. Последняя карта G используется для reconstructed prediction
и оценки тока перед MVM; это дополнительные 450 impact на filter-run.

В каждой mapping-фазе 92 входа: zero, 75 basis, 16 sparse с восемью активными
taps из 75. Физический вектор: 32 входа nominal 0/.1 В, не более восьми активных.
Seed=20261014 сбрасывается для каждой mapping-фазы: 16 sparse одинаковы в
физических phase2/3/4. В историческом simulator RNG продолжался между фазами;
сравнивать следует сохранённые векторы. Это эксперимент с первым слоем и
синтетическими входами, не accuracy полной SNN на CIFAR.

## Native команды и конвертации

Все аппаратные обращения — публичные Connector open_port/impact/close_port.
Эксперимент не импортирует physical driver напрямую. До open_port проверяются
fingerprints исходников, INI, веса, план и зависимости.

RESET/SET staircase: .3..3 В, шаг .01 В, не более 271 команд каждого направления.
`pulse_width=0` означает выбранный native режим, а не нулевую физическую
длительность: установленный mode7 игнорирует tms/tus, включает DAC и сразу
выключает его. Исторические 125 мкс — только reference metadata. Реальная
длительность неизвестна; sleep .001 после операции её не задаёт. Mode7
читает с DAC246 и возвращает среднее десяти ADC: журнал хранит native aggregate,
не десять отдельных отсчётов. INI soft_cc не доказывает аппаратное ограничение тока.

`native_conversions.py` вызывает штатные converters MemriBoard:

- SI/DAC: v2d и d2v, включая штатное округление d2v до трёх знаков.
- Readback R: a2r при nominal vol_read=.3; G=1/R, q=sum_gain/R как GUI.
  ADC<20 и sentinel неположительного R не используются как валидные оценки.
- MVM Vout: a2v(1, adc_bit, vol_ref_adc, ADC), как GUI: знаменатель 16383,
  пять десятичных знаков. ADC=0 допустим; saturation — ошибка. Native R
  из ответа Connector для MVM сохраняется без интерпретации.

WR-converters r2w/w2r не применяются к MVM q. Каждый impact получает новый
список напряжений: native SI converter мутирует его. SI-команды, DAC-коды,
native tuples, ADC и ошибки сохраняются. Физическая калибровка не заявляется.

## Остановка и интерпретация

Первый недостигнутый RESET endpoint q≤.005 или SET crossing, ошибка ADC/
транспорта/ID, исчерпание бюджета либо превышение MVM bounds останавливают run.
Предварительные bounds по последней модели G: Σ(V*G)≤5 мА и прогноз Vout≤80%
ADC reference. Наблюдаемый Vout также ограничен 80% reference. Это не измеренная
калибровка тока; амплитуды не адаптируются для принудительного прохождения.

Post-read mean ниже цели не останавливает опыт: отрицательные ошибки,
bias/SD/RMSE и flag сохраняются без отбора по знаку. Условия phase1 означают
достижимость на трёх ячейках, не точность всего массива; precision_qualified=false.
Zero-input сохраняется отдельно на обеих WL каждой плитки. Offset correction
и fitted scale к идеальным весам не применяются. Общая ошибка MVM может включать
систематический offset; она не называется шумом записи.

Сессии: `results/hardware_four_phase/<uuid>/`. Request flush перед impact,
result flush в finally, включая Ctrl+C. Partial programming attempts и tile
outputs сохраняются сразу. Итоговые manifest/summary — и при ошибке, с pulse
attempt counts, source/result SHA256. После ошибки фактическая доставка импульса
неизвестна. close_port не гарантирует аппаратный сброс после исключения.

Offline: `python -m unittest test_hardware_suite test_native_conversions -v`.
Тесты используют FakeConnector и реальные converters, без аппаратных вызовов.
