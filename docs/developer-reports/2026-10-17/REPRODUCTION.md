# Воспроизведение: no-BL8 Conv1 и параллельный MVM

Проверка исходников 17.10.2026; команды ниже сверены чтением CLI и кода,
но в этой проверке не выполнялись. Никаких Connector/driver imports и измерений.
Все команды выполняются из корня соответствующего полного checkout. Пути
относительные; `python` означает его подготовленный Python с зависимостями.
На Raspberry обычно `.venv/bin/python`, на Windows `.venv/Scripts/python.exe`.
Private native-файлы не входят в этот отчёт.

## Что именно сравнивается

| | Fresh Conv1 без физической BL8 | Параллельный calibrated MVM |
|---|---|---|
| CLI flag | `--test-cifar-fresh-conv1-no-bl8` | `--test-calibrated-mvm` |
| Runtime / plan | `fresh_conv1_no_bl8.py`, `configs/fresh_conv1_no_bl8.json` | `all256_mvm.py`, `configs/calibrated_mvm.json` |
| Protocol | `fresh300-jit-capacity-no-bl8-fixed-gain-v1` | `all256-preparation-mvm-lif-v2` |
| Операции | mode7 RESET/SET/readback и mode10 | только mode9/readback и mode10; SET/RESET отсутствует |
| Вход mode10 | 32 значения, одно активное BL с командой 0.1 V; остальные 0 | бинарная маска 0/0.1 V, максимум 4 активных BL |
| Scope | 300 свежих весов, 4 Conv1-фильтра, 44 тайла шириной до7; максимум530 candidate-pair attempts | все256 адресов, свежие одиночные ответы и смешанные входы |
| Повторы | 5, 10 настоящих CIFAR100 patches, T12; 2400 Conv/BN/LIF outputs | 5 mode10, 3 mode9 до/после; synthetic T12 |
| Общий decoder gain | HW0.25 V/единицу веса, SIM0.03; фиксирован до измерений | нет такого trained-weight gain; correction `C @ (ADC-zero)+d` |
| Калибровка | C2, d=0; собственные measured anchors/train/selection до CIFAR | identity/diagonal/full C8, отдельные fit/selection/test |
| Бюджет | <=1686292 impacts, <=91000 ненулевых mode7 | <=18576 impacts, 0 writes; cap4 даёт <=13736 фактически отправленных calls |

Fresh Conv1 не является параллельным 32-input MVM или полным классификатором.
Сумма всех75 taps выполняется до обученного BN[t], затем один цифровой LIF:
tau=0.5, spike при V>=1, hard reset0, отрицательный потенциал сохраняется.
Parallel MVM использует synthetic LIF с identity BN, не trained BN и не CIFAR.
Каждый WL в обоих сценариях читается отдельным Connector.impact; это не восемь
выходов за один аппаратный вызов. Разные HW/SIM gain не делают состояния,
шум или физику источников одинаковыми. Fitted alpha в Fresh остаётся диагностикой,
не используется для скрытого масштабирования весов.

## Штатный Connector, INI и единицы

`read_all.py:107–116`: только обычный `manager.board.Connector`,
`cb_type='real'`, `board_type='rp5_fpga_python'`, `task_format='SI'`.
Simulator создаётся тем же public Connector с `cb_type='simulator'`, тем же
board_type, новой native32x8 array и фиксированным seed20261016.
Прямых вызовов приватного драйвера из сценария нет.

`configs/raspberry_board.ini` и simulator board-settings:
DAC12bit/5V; ADC14bit/5V; res_load3000Ω; res_switches10Ω;
read gain11; sum_gain330; vol_read0.3V; soft_cc0.002A;
read_time1200, blank_time5, blank_time_between55000 (значения INI,
не независимо измеренные длительности). `native_conversions.py` использует
штатные converters. Mode10 переводится через a2v с gain1, как в GUI,
а не gain11 и не дополнительным множителем330.

Fresh RESET: native sign0, 55 уровней0.3…3V с шагом0.05;
SET: sign1, 109 уровней0.3…3V с шагом0.025. Знак trained weight выбирает
положительную/отрицательную ногу; знак напряжения не задаётся отрицательным float.
Перед ненулевым mode7 требуется свежий валидный native R этой ячейки:
V/R<=0.04A для sign0, <=0.002A для sign1. Нарушение останавливает Fresh;
это расчётный guard, не доказанная аппаратная стабилизация тока.
Hardware mode7 передаёт pulse_width=0, physical_pulse_width_s=None.
Фактическая длительность аппаратного импульса неизвестна. SIM получает
0.000125s/125 модельных шагов; нельзя приписывать эти125µs аппаратуре.
Mode7 readback использует native0.3V; в журнале HW указывается10 ADC samples.

В parallel mode9 команда0.1V, для производного R явно используется0.1V.
Native R в возвращаемом tuple использует INI0.3V; эти величины не смешиваются.
ADC<20 для производного R, невалидный/насыщенный ADC и ошибки протокола не
подменяются числом. Mode10 ADC0 допустим, но не доказывает нулевую проводимость.
Parallel guards: model current<=5mA, GUI prediction<3.3V, fresh-basis
эвристика2×signal+3SD до80%ADC, фактический выход<=4V. Плотности8/16/24/32
явно пропускаются без impact/подставленного ADC. Это не результат для dense MVM.

## Офлайн: сохранённый Fresh no-BL8

SIM session: `28aa7248-72db-428a-b0eb-92b1712a25af`.
HW session: `8c59eb76-c4e1-4d6f-8238-2b6622e226f6`.
Полные SIM raw доступны в полном checkout ветки `codex/full-cifar-diagnostic`. Полные HW raw
остаются на Raspberry; компактный опубликованный отчёт не заменяет raw.
Полный аудит требует также сохранённых manifest/source/capacity dependencies.
Новый output должен отсутствовать; исторический аудит не перезаписывать.

```sh
python -B audit_fresh_conv1_no_bl8.py --repo . --session results/cifar_fresh_conv1_no_bl8/28aa7248-72db-428a-b0eb-92b1712a25af --output results/cifar_feedback_reviews/no_bl8_simulator_reaudit_20261017
```

На Pi с полными HW файлами, также строго без аппаратного I/O:

```sh
python -B audit_fresh_conv1_no_bl8.py --repo . --session results/cifar_fresh_conv1_no_bl8/8c59eb76-c4e1-4d6f-8238-2b6622e226f6 --output results/cifar_feedback_reviews/no_bl8_hardware_reaudit_20261017
python -B export_fresh_trace.py --repo . --session results/cifar_fresh_conv1_no_bl8/8c59eb76-c4e1-4d6f-8238-2b6622e226f6 --audit results/cifar_feedback_reviews/no_bl8_hardware_reaudit_20261017/audit.json --output reports/2026-10-17/no-bl8-hardware-trace
```

Для SIM экспортируется такой же trace с его session и новым аудитом:

```sh
python -B export_fresh_trace.py --repo . --session results/cifar_fresh_conv1_no_bl8/28aa7248-72db-428a-b0eb-92b1712a25af --audit results/cifar_feedback_reviews/no_bl8_simulator_reaudit_20261017/audit.json --output reports/2026-10-17/no-bl8-simulator-trace
python -B compare_fresh_traces.py --repo . --first reports/2026-10-17/no-bl8-hardware-trace --second reports/2026-10-17/no-bl8-simulator-trace --output reports/2026-10-17/no-bl8-direct-comparison
```

Последняя команда выполняется там, где имеются оба компактных trace-v2 каталога.
Она сравнивает actual Conv/BN/LIF, а не выводит HW–SIM отличие из двух отдельных
RMSE относительно эталона. Проверяет2400 ключей, SHA, fixture/weights/route;
не дополняет пропуски CPU. Разные CRLF/LF route связываются по canonical JSON SHA.

## Офлайн: сохранённый parallel MVM

Последний HW: `cda1910c-6e20-4be3-8cd9-1519fdb26c36` (полные raw на Pi).
Native SIM: `3ed62521-1a3a-495f-9cd3-f4bbcf807c90` (полный checkout ветки `codex/full-cifar-diagnostic`).
`mvm_calibration.py --session ...` сверяет raw/compact/result SHA и расписание,
затем пишет `analysis/`. Это raw reconciliation, не равнозначный независимому
Fresh CIFAR raw-аудитору. Команда откажет, если `analysis/` уже существует.
Для повторного анализа не удалять/перезаписывать прежний результат: скопировать
сессию в новую папку, исключив только производный `analysis/`.

```sh
python -B -c "from shutil import copytree,ignore_patterns; copytree('results/calibrated_mvm/3ed62521-1a3a-495f-9cd3-f4bbcf807c90','results/calibrated_mvm/reanalysis_sim_20261017',ignore=ignore_patterns('analysis'))"
python -B mvm_calibration.py --session results/calibrated_mvm/reanalysis_sim_20261017
```

На Pi аналогично заменить исходный UUID на HW cda1910c… и назначение на
`results/calibrated_mvm/reanalysis_hw_20261017`. Содержимое исходных
manifest/raw/observations/protocol/schedule сохраняется без изменений.
Analyzer требует NumPy. Frozen C выбирается только на selection;
heldout test и reserved path probes в подгонке не участвуют.

## Исторические команды сбора — не запускать сейчас

Точная форма CLI 16 октября:

```sh
python -B experiment.py --test-cifar-fresh-conv1-no-bl8 --dry-run
python -B experiment.py --test-cifar-fresh-conv1-no-bl8 --run-simulator
python -B experiment.py --test-cifar-fresh-conv1-no-bl8 --run-hardware
python -B experiment.py --test-calibrated-mvm --dry-run
python -B experiment.py --test-calibrated-mvm --run-simulator
python -B experiment.py --test-calibrated-mvm --run-hardware
```

Это документирование исторического invocation, не инструкция исполнить блок.
Fresh no-BL8 имеет жёсткий cutoff **2026-10-16 18:50UTC =21:50MSK**;
17 октября оба run-режима истекли. Проверка до backend/open_port и перед
impact; dry-run не открывает порт, но требует source/INI/private-file preflight.
Просто заменить дату/отключить проверку нельзя: новый опыт требует нового
reviewed/versioned протокола, замороженных параметров, source commit,
native-SIM проверки и отдельного разрешённого аппаратного запуска.
Parallel CLI технически не содержит этого cutoff, однако разрешение16 октября
не является разрешением нового аппаратного опыта. Новый запуск также должен
иметь явно согласованные текущие условия/границы. В рамках данного отчёта
допустимо воспроизводить только анализ уже сохранённых данных.

## Точность формулировок

Исключена физическая BL8, не логический коэффициент8; это не диагноз ячеек.
No-BL8 изменил160/300 primary pairs, поэтому сравнение не строго однофакторное.
Audit PASS означает согласованность сохранённых команд/данных/математики,
не хороший физический вес, стабильность платы или accuracy классификатора.
Физическая причина HW–SIM расхождения, реальная длительность импульса,
вольтаж на ячейке и аппаратный current compliance этими журналами не измерены.

Source evidence: `experiment.py:196–212,270–277`; `fresh_conv1_no_bl8.py:24–37,55–77,98–109`;
`fresh_conv1.py:130–165,178–278,320–361`; `cell_calibration.py:65–127`;
`native_conversions.py:10–49`; `all256_mvm.py:50–95,115–214`;
`mvm_calibration.py:22–90,295–365`; `read_all.py:107–116`.
