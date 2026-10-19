# MNIST STDP: допуск свежего online-learning smoke128

**GO только для нового CUDA smoke на первых 128 train-примерах, batch1, T=100: 12 800 model timesteps, максимум 30 минут.** Это STDP-обучение свежих весов, не frozen-feature пилот и не повтор обучения readout на старом checkpoint. Полный маршрут 30 000 train-примеров и последующей оценки пока остаётся закрытым отдельным gate. Модель рецензент не запускал.

Протокол восстанавливает родительский опыт `20260425T113337Z_5b89f013f42b`; поздние 69,65% относятся к последующей readout-оценке его весов. Старый checkpoint здесь не используется для инициализации. Точный source snapshot и регистрация — [`SOURCE_FREEZE.json`](SOURCE_FREEZE.json).

## Проверенные механизмы

- Сохранённый parent summary задаёт положительные `nu_minus=0.001` и `nu_plus=0.0001`. Это pre-depression и post-potentiation native BindsNET `WeightDependentPostPre`, соответственно. Нельзя брать отрицательный default `nu_minus` из позднего evaluation checkpoint.
- Проверен установленный SHA-pinned native Conv2d update: обе корреляционные части вычисляются с текущими весами, суммируются, затем применяется clamp[0,1]. Traces и порядок шагов выполняет штатный BindsNET; отдельная переписанная реализация обучения не внедрена.
- `net.train(True)`, network/layer flags, точный класс native rule, rates, bounds, отсутствие norm/Dale transform и неизменность thresholds/bias проверяются перед/после окон. Это реальное выполнение model STDP, но не физическое переключение мемристоров.
- Fresh builder с новым явно заданным seed42 создаёт Conv1=64×1×5×5. Его uniform[0,0.3] нормируется до суммы78,4 на фильтр: средний initial weight3,136 выходит за wmax1. Сохраняются initial W и100 post-update состояний первого окна. Step0 обязан быть ровно `clamp(initial,0,1)`, последний recorded W — равен post-window W. Так первоначальное clipping отделено от последующей пластичности.
- После каждого изображения сохраняются все1600 весов и input/output spike counts, input/weight SHA, изменения, время. Перед следующим изображением сбрасываются состояния сети. End-of-image trajectory имеет128 записей; новый checkpoint сохраняется и проверяется через `weights_only=True` roundtrip.
- CPU Monitor читается после `torch.cuda.synchronize()`. Полный spike count окна нельзя заменять последним шагом. При неверной форме/нечисловом состоянии/нарушении bounds сбор прекращается с failed-результатом.
- Smoke останавливается до readout, frozen extraction и test evaluation. Shared preflight хеширует train/test gzip для provenance; в самих модельных окнах используются только train IDs0–127. Старый weight-файл также может проверяться helper-функцией по SHA, но training его не загружает.

## Что не следует выдавать за воспроизведение

Архивная CSNN-ветка не вызывала seed initialization, а её RNG/dependency state не аттестованы. Поэтому новый seed42 и закреплённое окружение определяют **новую реконструкцию**, не побитовый исторический replay. `construction_rng_*` сохранены после builder; исходный W — фактическая запись инициализации.

Сохранён дефект reference post-window hook: обращение к неопределённому `connection` перехватывается и пропускает последующие E/I/renorm действия. Наличие `ei_enable=True` не означает работающую конкуренцию; inhibitory population существует без E/I-connections. Новые aliases и исправление этого дефекта не подмешаны. Стандартный synchronous `one_step=False` сохраняет задержку входа на один шаг.

Аргумент native `weight_decay=1.0` превращается в внутреннее0; штатная условная ветвь умножения тогда пропускается. Это **не обнуление всех весов** и не отдельный источник последующих изменений. Изменение initial→final без отделения первого clipping не доказывало бы корреляционное обучение.

## Что потребуется после фактического smoke

Независимо проверить SHA/формы/IDs всех артефактов, init norm, native first-step clipping, конечные bounds, per-image trajectory hashes и изменения **после** initial clipping. Оценить долю весов у границ, количество активных изображений, нулевые/вырожденные результаты и реальные batch1 timesteps/second. Изменение весов подтверждает работу online update; качество представления подтверждается только дальнейшей независимой оценкой новых весов.

Если STDP не изменит веса после clipping, это результат, а не разрешение подстроить rates во время того же опыта. Full30k должен начать заново с fresh seed42, не продолжать smoke и не импортировать прежние обученные W. Его стеновой бюджет и новый readout требуют отдельного допуска после анализа actual smoke. Frozen batch128 throughput не используется как оценка online STDP runtime.

Автор сообщил 8/8 offline synthetic/source tests (4 STDP и4 recovery math/coverage); они не создавали и не запускали SNN. Source review не подтверждает ещё ни фактическое успешное обучение, ни исторические69,65%, ни аппаратный STDP. Все будущие реальные операции с мемристорами остаются исключительно через MemriBoard.
