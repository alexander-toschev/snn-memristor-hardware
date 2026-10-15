# Независимый аудит повторов интервалов

Сессия `977aa0af-b7f6-4752-bb63-09f92082031e`, hardware; сбор **completed**, аудит **passed**. 578703 проверок, 0 ошибок.

Actual impacts: 3647; programming decisions: 2577; реальные попытки импульса: 2577; skipped: 0. Полный replay: True.

Targets: {'hit': 23, 'overshoot': 1}; наблюдённый interval+MVM pass: 16. Источник hit: {'programming_response': 23}.

Проверены source/result SHA, native conversions, последний mode7 R перед current guard, отдельные skipped decisions, inclusive R bands и происхождение evidence. Для completed воспроизведены все 9 циклов, общий SET budget, все pre/post/delayed samples, MVM-screen, delay и итоги.

Протокол: three-cell-native-R-interval-v2-refined. Остановка hit/overshoot проверена по stop band; подтверждение readbacks — по outer ±1%. Отличия текущих неимпортируемых файлов от сохранённого Git: 0; детали в JSON. Все импортируемые vendor Python файлы проверены по локальным байтам.

Аудитор не создаёт Connector и не вызывает аппаратуру. Passing отдельных точек не подтверждает непрерывный диапазон, все256 ячеек или CIFAR accuracy. Аппаратная длительность импульса неизвестна.

```sh
python -B audit_cell_range_repeat.py --repo . --session results/cell_range_repeats/977aa0af-b7f6-4752-bb63-09f92082031e --output-dir results/cell_range_repeats/977aa0af-b7f6-4752-bb63-09f92082031e/offline-audit-recheck
```
