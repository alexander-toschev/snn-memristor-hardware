# Confirmatory wide BNTT, CIFAR100 subset10

Здесь сохранены четыре архивных checkpoint `best_fs.pt` с cfg, split,
summary и metrics: seeds 0/1/2/42, 64/128/256 conv + FC512, T12,
tau0.5, threshold1, BNTT, use_twd=true, twd_weight0.1, memristor_sigma0.
`TWD=1` означает включённый decoder, не weight=1.

Исходные checkpoint выбраны по validation FS; новый подбор/обучение не делаем.
Архивные FS: .725/.739/.741/.754; TWD: .764/.770/.767/.782.
Это **clean PyTorch**, не MemriBoard native simulator и не плата.
1000 official test примеров классов 0,1,2,3,5,8,13,14,17,19; исходный порядок.
Повторный clean eval сохраняет conventional архивные метрики, отдельно strict
метрики (silent/tied abstain), predictions и полные output-spike rasters.

```console
python -B experiments/cifar_confirmatory_20261020/evaluate.py --dry-run
python -B experiments/cifar_confirmatory_20261020/evaluate.py --seeds 0 1 2 42
```

Запуск eval следует только после MNIST accuracy по указанию оператора.
Никаких CUDA/remote jobs, обучения или платы. Torch берётся из repo-local
`.deps-classifier`, NumPy из `.deps`; CIFAR bytes — `data/cifar-100-python`.
Имена старого проекта/путей не нужны runtime.

`temporal_snn/{model,lif,surrogate}.py` — неизменённые локальные копии исходников
проекта snn-mnist, импортированные 20 октября. Checkpoint SHA сверяется с
историческим summary. Точного source commit обучения архив не удостоверяет;
фактическое совпадение исторических test-counts проверяется новым eval.

Следующий nativeSIM маршрут должен быть отдельным: только public MemriBoard
Connector + INI, bounded programming всех 4800 signed Conv1 коэффициентов,
native readback, затем **hybrid Torch inference** с measured weights. Это
характеризует programming/readback и не подтверждает literal analog MVM.
Нельзя использовать прежние прямые BoardSimulator/HardCore/dynamic-import
скрипты или тихо подставлять ideal weights вместо отсутствующего readback.
Старые hybrid .653 (64 фильтра) / около .706 (10 фильтров) — исторические
результаты другого измерительного маршрута, не ожидаемые гарантии нового.

Для native route ещё нужен явный mapping в достижимый диапазон штатного SIM:
старое proxy-окно .01–.05 не переносится автоматически. Никаких новых native
запусков этот clean evaluator не делает. Full literalMVM всего test-набора
этим этапом не заявляется.
