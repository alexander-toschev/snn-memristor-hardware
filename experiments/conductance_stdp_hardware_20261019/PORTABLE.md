# STDP8 через public MemriBoard

Новый launcher выполняет прежний conductance STDP v2 без изменения collector,
протокола или старых raw. BL6/8/9/14, WL5 training, WL7 paired no-update control.
Максимум6712 воздействий и128 импульсов; свежие2fit-окна выбирают общий gain
для обеих WL. Нет подходящего gain — `uninformative_calibration`, выход3,
без повторов или расширения grid. Импульсы RESET1.4V/SET1.7V, штатные guards
.04A/.002A и nativeR/ADC, чтениеMVM.05V. Длительность реального импульса
неизвестна; драйверу передаётся0. Это алгоритмический STDP проводимостей с
цифровым LIF, не доказательство intrinsic delta-t STDP и не accuracy.

Код выбирает драйвер из INI через публичный `manager.board.Connector` и
штатные converters. Ручные grants, private SHA проверки и receipts отменены
пользователем. Public hashes записываются для воспроизводимости;
`private_runtime_not_independently_reverified` прямо обозначает ограничение.
Сохранены аппаратный lock, конечный бюджет, raw и закрытие Connector.
Сначала дождаться завершения CIFAR; никаких параллельных аппаратных опытов.

Из корня репозитория, команды по одной:

```sh
../../.venv/bin/python -B experiments/conductance_stdp_hardware_20261019/portable.py --dry-run
```

```sh
../../.venv/bin/python -B experiments/conductance_stdp_hardware_20261019/portable.py --run-hardware
```

После публикации результатов перенести `results/conductance_stdp_hardware_20261019/UUID`
на ПК и выполнить локальный численный replay:

```sh
python -B tools/audit_conductance_stdp_portable.py --session results/conductance_stdp_hardware_20261019/UUID --output reports/2026-10-20/conductance-stdp-hardware/UUID_AUDIT.json
```

Auditor проверяет записанные public source/result bytes, INI, protocol,
полный raw, команды/guards, fresh-fit gain, LIF/eligibility, ownership
импульсов и контроль. Численный replay — статическая точная копия ранее
проверенного replay; тест подтверждает равенство текста. Private bytes
не читаются; старый strict-attestation PASS не заявляется. Исторический
nativeSIMd27da87e… прошёл новый локальный replay (6616calls/32trainingpulses),
без нового запуска модели. Фактический HW сеанс ещё необходим.
