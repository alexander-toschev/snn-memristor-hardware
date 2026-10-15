# Независимый аудит discovery

Сессия `f32eab10-9abd-4d8c-a4c5-3ed49a3cb8fb`, источник **hardware**, сбор **completed**, охват **complete**. Целостность журнала: **passed**; 17629650 проверок, 0 расхождений.

Проверено 114090 пар request/result и 78709 nonzero mode7. Checkpoint 967. Полный независимый replay алгоритма: True. Для partial проверены сохранившиеся пары, преобразования, префиксы начального/конечного чтения, checkpoint, пределы и pulse ledger; полного охвата не заявляем.

RESET crossing: 232; RESET confirmed: 223; SET attempted cells: 223; q=.05 crossing: 13.

Записанный commit `f3f57fe49a0c79bd7c72d6e3406b19e666970f90`, dirty_tracked=False. Source blobs сверены с `f3f57fe49a0c79bd7c72d6e3406b19e666970f90`. Приватные драйверы сверены по локальным байтам и фиксированным SHA, не публикуются.

Один цикл не устанавливает стабильный диапазон, повторяемость, retention, endurance или MVM нескольких фильтров. .05 — граница исследуемого окна. Hardware duration неизвестна;125 model steps применяются лишь к simulator. Исторический checkpoint не обязательно соответствует текущему состоянию массива.

Аудитор не создаёт Connector и не вызывает аппаратуру; обычный импорт пакета используется только для штатных численных converters.

```sh
python -B audit_calibration_generic.py --repo . --session results/cell_calibrations/f32eab10-9abd-4d8c-a4c5-3ed49a3cb8fb --output-dir results/cell_calibrations/f32eab10-9abd-4d8c-a4c5-3ed49a3cb8fb/offline-audit-recheck --source-commit f3f57fe49a0c79bd7c72d6e3406b19e666970f90
```
