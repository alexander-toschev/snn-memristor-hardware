# Независимый аудит повторов интервалов

Сессия `0ab02a19-b10c-4d4f-b7a5-6152a28f290a`, hardware; сбор **completed**, аудит **passed**. 602480 проверок, 0 ошибок.

Actual impacts: 4026; programming decisions: 2999; реальные попытки импульса: 2999; skipped: 0. Полный replay: True.

Targets: {'hit': 22, 'set_ramp_exhausted': 5}; наблюдённый interval+MVM pass: 16. Источник hit: {'programming_response': 22}.

Проверены source/result SHA, native conversions, последний mode7 R перед current guard, отдельные skipped decisions, inclusive R bands и происхождение evidence. Для completed воспроизведены все 9 циклов, общий SET budget, все pre/post/delayed samples, MVM-screen, delay и итоги.

Аудитор не создаёт Connector и не вызывает аппаратуру. Passing отдельных точек не подтверждает непрерывный диапазон, все256 ячеек или CIFAR accuracy. Аппаратная длительность импульса неизвестна.

```sh
python -B audit_cell_range_repeat.py --repo . --session results/cell_range_repeats/0ab02a19-b10c-4d4f-b7a5-6152a28f290a --output-dir results/cell_range_repeats/0ab02a19-b10c-4d4f-b7a5-6152a28f290a/offline-audit-recheck
```
