# Независимый аудит повторов интервалов

Сессия `82982625-1bd9-470a-a512-2b7c506452a1`, simulator; сбор **completed**, аудит **passed**. 216298 проверок, 0 ошибок.

Actual impacts: 2011; programming decisions: 1801; реальные попытки импульса: 1801; skipped: 0. Полный replay: True.

Targets: {'overshoot': 24, 'hit': 3}; наблюдённый interval+MVM pass: 3. Источник hit: {'programming_response': 3}.

Проверены source/result SHA, native conversions, последний mode7 R перед current guard, отдельные skipped decisions, inclusive R bands и происхождение evidence. Для completed воспроизведены все 9 циклов, общий SET budget, все pre/post/delayed samples, MVM-screen, delay и итоги.

Аудитор не создаёт Connector и не вызывает аппаратуру. Passing отдельных точек не подтверждает непрерывный диапазон, все256 ячеек или CIFAR accuracy. Аппаратная длительность импульса неизвестна.

```sh
python -B audit_cell_range_repeat.py --repo . --session results/cell_range_repeats/82982625-1bd9-470a-a512-2b7c506452a1 --output-dir .tmp/interval-independent-audit-recheck
```
