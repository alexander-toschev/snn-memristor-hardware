# Аппаратный MNIST32→8: пилот128

Новый launcher использует прежние core/SPEC/label-free fixture без изменения
численного протокола. 32 входа,8 LIF: WL0–3 обучаются на128 MNIST-изображениях,
WL4–7 — парные контроли без training writes. Все256 ячеек измеряются;
128 потенциально обучаемых ячеек. Это проверка операций/STDP/разнообразия,
не измерение accuracy, не архивная69.65% сеть и не full10000.

Всего176 уникальных изображений:32 отдельныеfit,128train,16probe.
С повторными pre/post/retention probe —208 измерений изображений.
Общий gain выбирается по32fit без labels, фиксируется до probes/train.
Отсутствие допустимого gain завершает `uninformative_calibration` без
подбора новых параметров. NativeSIMmain/reference уже выполнены; повтор
симуляции не нужен. MainSIM:107728impacts/2400pulses,9мин44с;
это ориентир программного прогона, не обещание длительности платы.

Конечные пределы:172288impacts/2560pulses,14полных checkpoints,
retention60с, raw≤2GiB, начальный свободный диск≥6GiB/резерв2GiB.
Чтение.05V, максимум4 активных BL; RESET1.4V/SET1.7V, nativeR/ADC/current
checks прежние. Реальный импульс: submitted0, длительность неизвестна.
Только публичный MemriBoard Connector, converters и драйвер через INI.
Lock исключает параллельный аппаратный опыт; finally закрывает Connector.

Команды на Raspberry из корня checkout, по одной:

```sh
../../.venv/bin/python -B experiments/mnist_conductance_20261019/portable.py --dry-run
```

```sh
../../.venv/bin/python -B experiments/mnist_conductance_20261019/portable.py --run-hardware
```

Результаты: `results/mnist_conductance_20261019/<UUID>/`.
Terminal completed/no-gain/failed сохраняется целиком; цифровой checkpoint
не восстанавливает проводимости после обрыва. Не продолжать частично
выполненные updates как будто физическое состояние восстановлено.

После публикации и переноса на ПК:

```sh
python -B tools/audit_mnist_conductance_portable.py --session results/mnist_conductance_20261019/UUID --output reports/2026-10-20/mnist-hardware/UUID_AUDIT.json
```

Auditor напрямую использует прежний независимый Replay/durable parser,
проверяет raw→commands/guards→gain/LIF/E/updates/control, fixture, итоговые
счётчики и наблюдения. Public recorded bytes/INI проверяются; private
runtime не аттестуется: `private_runtime_not_independently_reverified`.
Ручные grants/receipts/private SHA barriers отменены пользователем.
Прежние SIM результаты и исторические отчёты остаются неизменными.
