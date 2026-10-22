# Диагностическое чтение на нескольких напряжениях

Один foreground-сценарий через публичный MemriBoard Connector/converters/INI:
WL0/7 × BL14,15,16,17,20,30,31; напряжения 0,025; 0,05; 0,075; 0,1 В.
Каждый адрес/уровень измеряется mode9 и mode10 по три раза, порядок режимов
чередуется между повторами. Каждый bracket: zero → fresh mode7 R → signal → zero.
**1344 вызова, 0 SET/RESET**, wall budget600с, raw64MiB, резерв диска512MiB.

Консервативный токовый guard: max(0,1 В, nativeDAC→V)/R≤5mA;330I<3,3В;
измеренный выход ADC→V≤4В. Все запреты и guards проверяются до signal.
Штатный mode7 читает по INI; mode9/10 не получают напряжений выше0,1В.
Прямого драйвера/GPIO/SPI доступа, CPU-подстановки V/R вместо MVM нет.

Цель — различить порог/смещение и отсутствие отклика в диапазоне напряжений.
Даже сохранение нулей не доказывает конкретную аппаратную причину.
Аналоговые пути mode9/10 разные, равенство амплитуд не предполагается.
Графики/таблицы не переименовываются в калибровку или MNIST accuracy.

```bash
.venv/bin/python -B experiments/mvm_voltage_diagnostic_20261022/experiment.py --run-hardware --publish
```

Локально доступны `--dry-run`, `--run-simulator` с тем же конечным расписанием.
Результаты в `results/mvm_voltage_diagnostic_20261022/<UUID>`.
Автопубликация после закрытия Connector: byte-verified archive, два secret scans,
exact archive/receipt/summary/HTML, обычные Git hooks/LFS. Credential helper
существующий; Git log/pending остаются в `.tmp`, без интерактивного запроса.
Ошибка не повторяет аппаратные измерения; `--publish-session UUID` сохраняет
только уже закрытую сессию. Старые результаты не меняются.
