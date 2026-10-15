# Независимый аудит discovery

Сессия `3380d497-ded0-4881-9df2-85ef1c7e32d9`, источник **hardware**, сбор **completed**, охват **complete**. Целостность журнала: **passed**; 21714230 проверок, 0 расхождений.

Проверено 120296 пар request/result и 82745 nonzero mode7. Checkpoint 1029. Полный независимый replay алгоритма: True. Для partial проверены сохранившиеся пары, преобразования, префиксы начального/конечного чтения, checkpoint, пределы и pulse ledger; полного охвата не заявляем.

Протокол `cell-conductance-discovery-v2-gui-write-guard`. GUI write guard проверен: True. Решений записи: 82745; пропусков по модельному току: 0. Пропуск не является аппаратной командой и не создаёт ADC/новое измеренное состояние. Для v1 защиту v2 ретроспективно не предполагаем.

RESET crossing: 222; RESET confirmed: 214; SET attempted cells: 214; q=.05 crossing: 15.

Записанный commit `34b8819bf3d501f337b4a0ad401204cb8c5397cd`, dirty_tracked=False. Source blobs сверены с `34b8819bf3d501f337b4a0ad401204cb8c5397cd`. Неимпортируемых root/config файлов с отличающимися текущими байтами: 0; они проверены по сохранённому Git blob и перечислены в JSON. Все локальные vendor Python sources проверены до импорта. Приватные драйверы сверены по fixed PIN и локальным байтам при наличии; недоступных локально: 0. Приватные файлы не проверяются через Git и не публикуются.

Один цикл не устанавливает стабильный диапазон, повторяемость, retention, endurance или MVM нескольких фильтров. .05 — граница исследуемого окна. Hardware duration неизвестна;125 model steps применяются лишь к simulator. Исторический checkpoint не обязательно соответствует текущему состоянию массива.

Аудитор не создаёт Connector и не вызывает аппаратуру; обычный импорт пакета используется только для штатных численных converters.

```sh
python -B audit_calibration_generic.py --repo . --session results/cell_calibrations/3380d497-ded0-4881-9df2-85ef1c7e32d9 --output-dir results/cell_calibrations/3380d497-ded0-4881-9df2-85ef1c7e32d9/offline-audit-recheck --source-commit 34b8819bf3d501f337b4a0ad401204cb8c5397cd
```
