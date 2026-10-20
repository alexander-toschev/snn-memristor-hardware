# MNIST pooled32 → 8: контрольный conductance-STDP пилот

Новая архитектура по выбору пользователя: 256 физических положительных
синапсов, цифровые восемь LIF и window eligibility. Это не архивная Conv-SNN,
не повтор её 69,65% и не полный опыт на 10 000 изображениях. Физические
начальные веса определяются измеренным состоянием и ограниченной подготовкой;
чужой checkpoint не загружается. Пилот не измеряет accuracy и не обучает head.

Основной контракт: `docs/plans/MNIST_PHYSICAL_PILOT128_20261019.md`.
Статус: независимый source/auditor review разрешил регистрацию только native
SIM. Фактическая квалификация main/reference ещё не выполнена. Аппаратный
запуск в этом entrypoint запрещён явно; grant для него не создан.

## Данные и физическая граница

`fixture.json` содержит ровно 176 уникальных train-изображений, их исходные
индексы, точные uint8 pixels, integer pooling/TTFS32 и SHA. Labels и test
не читаются. Полный IDX на Raspberry не нужен. Фактический fixture подготовлен
отдельным проверенным extractor и закреплён SHA; повторно генерировать его
не требуется. Поля train/fit/probe не выбираются по ответам.

- Gain-fit: train 11000–11031, только чтение.
- Train: 0–127, learning WL0–3, парный no-update control WL4–7.
- Probes: 12000–12015, pre/post/retention без записи.

Все 256 адресов обследуются и получают подготовку. BL0,16,24 молчат во всех
176 исходных изображениях: 24 ячейки имеют только checkpoint/preparation
exposure, максимум 232 ячейки участвуют в image MVM. Среди 128 learning cells
12 не получают входных spikes/eligibility. Это свойство входов, не дефект
ячеек. Искусственные spikes и исключение адресов ради красивого охвата запрещены.
Фактические updates отдельно считаются по каждому адресу.

Все native операции — только `manager.board.Connector.impact`, штатные
converters и выбор драйвера через INI. Прямых transport/driver/.interface
обращений нет. Native SIM также вызывается через public Connector. Микровекторы
содержат максимум четыре BL; свежий native R перед каждым сигналом/импульсом,
GUI current guards и ADC/headroom limits обязательны. Длительность аппаратного
импульса неизвестна; SI width=0 не означает физические 0 секунд. Native SIM
явно получает 0,0001 секунды, без заявления эквивалентности плате.

Изображение полностью измеряется с zero-before/after до цифрового LIF/E и
обновлений. Все микровклады одного t складываются до одного LIF-step; никакой
CPU-матрицы весов, подстановки целевых весов или lookup. Четыре learning WL
получают максимум одну попытку на изображение; control получает только парные
guard/post reads. Все восемь shadow choices сохраняются.

## Конечные бюджеты и воспроизводимость

Main: максимум 172 288 impact, 2 560 импульсов, 208 изображений, 14 полных
checkpoints. Нет допустимого gain из 13 фиксированных кандидатов — exit3
после fit, максимум 34 560 impact / 2 048 подготовительных импульсов; train и
probes не запускаются. Молчание/одинаковые выходы/неподвижные G сохраняются
как результат. Необходимость дополнительного варианта требует нового протокола.

Raw — lossless gzip level1, chunks до 8 MiB expanded, общий cap 2 GiB,
минимум 2 GiB свободного места во время записи, initial reserve 6 GiB.
Используется неизменный SHA-pinned durable writer: fsync закрытых chunks/index
после каждой update-пары, каждого изображения и строки checkpoint. Наличие
progress.json не разрешает повторить физические updates после аварии.
Нужны сохранение partial raw, проверка фактического состояния и отдельное review.

Source/fixture/INI/protocol/environment и все result SHA сохраняются в сессии.
Полные fit trajectories, raw response до проверки, guards, posts/traces/E,
выборы/пропуски и postR остаются в raw. `observations.json` содержит связанные
с ними компактные summaries, collapse/coverage/Δinverse-R и контрольные ряды.
Сам raw audit и тяжёлый readout выполняются на ПК; Pi хранит данные и выполняет
только public MemriBoard acquisition с небольшим online feedback.

## Команды после review

Проверка файлов, без Connector:

```sh
python -B experiments/mnist_conductance_20261019/experiment.py --dry-run --preflight-source simulator
```

После регистрации reviewed SIM SPEC основной foreground-запуск:

```sh
python -B experiments/mnist_conductance_20261019/experiment.py --run-simulator
```

Для SIM-only no-update reference к этой же команде добавляется
`--reference-main` с реальным repo-relative путём завершённой main-сессии.
Несуществующий UUID здесь не выдумывается. Проверяются полная main inventory,
source/input/protocol, SHA начального `.cb` и выбранный main gain. Reference
использует неизменный `array_initial.cb`, сохранённый до создания Connector;
`array.cb` остаётся изменяемым runtime/final артефактом симулятора. Оба файла
входят в итоговую SHA inventory. Reference
имеет собственные свежие измерения/eligibility/shadow choices, но на выбранную
попытку четыре чтения без обучающего импульса. Его cap 171 776 impact / 2 048
подготовительных импульсов; это отдельная сессия, не скрытая часть main cap.
Собственные fit-данные reference не меняют gain. Сравнение SIM reference не
доказывает воспроизводимость начального аппаратного состояния.

Среда: обычный Python, существующие repo-local `.deps` и MemriBoard runtime.
Torch/BindsNET/CUDA не используются. Фактическое время native SIM пока неизвестно;
старые средние задержки платы не являются ETA нового протокола и fsync/logging.

Offline fake tests:

```sh
python -B -m unittest tests.test_mnist_conductance_fixture tests.test_mnist_conductance -v
```

Fake tests проверяют программные правила; они не заменяют native SIM,
независимый raw audit или аппаратные данные.
