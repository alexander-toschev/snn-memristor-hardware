# Независимый аудит simulator discovery

Сессия `9fa4e745-cd50-44e9-97b0-b650292e4db8`: **passed**, 24813305 проверок, 0 расхождений.

Проверено 114071 пар request/result, 55975 nonzero mode7. Все256 адресов, initial/final по3 чтения, ограничения амплитуды/бюджета, native conversions, ID, порядка и checkpoint dedup проверены. Checkpoint 1616, переиспользований 289.

RESET crossing: 256/256; RESET confirmed: 256/256; SET attempted: 256; верхняя q=.05 достигнута: 253.

Provenance: manifest честно сохраняет dirty working tree на `7b73686c19febfcb638ea826c5246e582a451aa3`; 65 записанных source SHA совпадают с локальными файлами и Git-объектами более позднего `8d37e45f59e5e2c313963bc62b254038817f016b`. Это не чистый запуск позднего коммита.

Это проверка исполнения алгоритма на штатном симуляторе, не физическая калибровка. Один цикл не устанавливает стабильный диапазон, повторяемость, retention, endurance или MVM нескольких фильтров. .05 — граница исследуемого окна. Старое удачное checkpoint-состояние не обязательно оставлено на массиве.

```sh
python -B audit_calibration.py --source-repo . --source-commit 8d37e45f59e5e2c313963bc62b254038817f016b --session results/cell_calibrations/9fa4e745-cd50-44e9-97b0-b650292e4db8 --output-dir results/cell_calibrations/9fa4e745-cd50-44e9-97b0-b650292e4db8
```
