# Независимый аудит discovery

Сессия `8fb315b9-3750-41d0-a810-01a787b48efc`, источник **simulator**, сбор **completed**, охват **complete**. Целостность журнала: **passed**; 26784168 проверок, 0 расхождений.

Проверено 114071 пар request/result и 55975 nonzero mode7. Checkpoint 1616. Полный независимый replay алгоритма: True. Для partial проверены сохранившиеся пары, преобразования, префиксы начального/конечного чтения, checkpoint, пределы и pulse ledger; полного охвата не заявляем.

Протокол `cell-conductance-discovery-v2-gui-write-guard`. GUI write guard проверен: True. Решений записи: 55975; пропусков по модельному току: 0. Пропуск не является аппаратной командой и не создаёт ADC/новое измеренное состояние. Для v1 защиту v2 ретроспективно не предполагаем.

RESET crossing: 256; RESET confirmed: 256; SET attempted cells: 256; q=.05 crossing: 253.

Записанный commit `d2bca6688b1b0bb886f954bab47aee49a356681a`, dirty_tracked=False. Source blobs сверены с `d2bca6688b1b0bb886f954bab47aee49a356681a`. Неимпортируемых root/config файлов с отличающимися текущими байтами: 0; они проверены по сохранённому Git blob и перечислены в JSON. Все локальные vendor Python sources проверены до импорта. Приватные драйверы сверены по fixed PIN и локальным байтам при наличии; недоступных локально: 0. Приватные файлы не проверяются через Git и не публикуются.

Один цикл не устанавливает стабильный диапазон, повторяемость, retention, endurance или MVM нескольких фильтров. .05 — граница исследуемого окна. Hardware duration неизвестна;125 model steps применяются лишь к simulator. Исторический checkpoint не обязательно соответствует текущему состоянию массива.

Аудитор не создаёт Connector и не вызывает аппаратуру; обычный импорт пакета используется только для штатных численных converters.

```sh
python -B audit_calibration_generic.py --repo . --session results/cell_calibrations/8fb315b9-3750-41d0-a810-01a787b48efc --output-dir .tmp/independent-v2-audit-final-recheck --source-commit d2bca6688b1b0bb886f954bab47aee49a356681a
```
