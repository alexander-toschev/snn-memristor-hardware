# Независимый аудит повторов интервалов

Сессия `116011c8-6d40-4657-b09c-32074177efa8`, simulator; сбор **completed**, аудит **passed**. 216559 проверок, 0 ошибок.

Actual impacts: 2011; programming decisions: 1801; реальные попытки импульса: 1801; skipped: 0. Полный replay: True.

Targets: {'overshoot': 21, 'hit': 3}; наблюдённый interval+MVM pass: 3. Источник hit: {'programming_response': 3}.

Проверены source/result SHA, native conversions, последний mode7 R перед current guard, отдельные skipped decisions, inclusive R bands и происхождение evidence. Для completed воспроизведены все 9 циклов, общий SET budget, все pre/post/delayed samples, MVM-screen, delay и итоги.

Протокол: three-cell-native-R-interval-v2-refined. Остановка hit/overshoot проверена по stop band; подтверждение readbacks — по outer ±1%. Отличия текущих неимпортируемых файлов от сохранённого Git: 0; детали в JSON. Все импортируемые vendor Python файлы проверены по локальным байтам.

Аудитор не создаёт Connector и не вызывает аппаратуру. Passing отдельных точек не подтверждает непрерывный диапазон, все256 ячеек или CIFAR accuracy. Аппаратная длительность импульса неизвестна.

```sh
python -B audit_cell_range_repeat.py --repo . --session results/cell_range_repeats/116011c8-6d40-4657-b09c-32074177efa8 --output-dir .tmp/audit-refined-compat-116011c8-6d40-4657-b09c-32074177efa8-recheck
```
