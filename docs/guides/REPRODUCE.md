# Воспроизведение на Raspberry

## Получить данные Git LFS

После clone или обновления ветки загрузите настоящие файлы весов и результатов:

```sh
git lfs install --local
git lfs pull
```

Git LFS должен быть установлен в окружении. До запуска проверок убедитесь, что
веса и журналы загружены, а не остались текстовыми LFS-pointer файлами.
`git lfs ls-files` показывает объекты LFS; штатные SHA-проверки протоколов
проверяют содержимое загруженных файлов. После следующего `git pull` повторите
`git lfs pull`, если автоматическая загрузка отключена или была прервана.


Все команды выполняются из корня репозитория. Актуальный конечный пакет — [резерв A](../protocols/CIFAR_FEEDBACK_RESERVE.md) и [резервы B/C](../protocols/CIFAR_REMAINING_RESERVES.md), единая точка входа `experiment.py`. Рабочая ветка — `codex/full-cifar-diagnostic`.

[Итог HW/SIM](../../results/cifar_feedback_reviews/remaining_hardware_vs_simulator_20261016/REPORT.md) · [Карта участвовавших ячеек](../../results/cifar_feedback_reviews/remaining_hardware_vs_simulator_20261016/hardware_contributing_map.svg) · [Аппаратный review](../../results/cifar_feedback_reviews/remaining_hardware_20261016/INDEPENDENT_REVIEW.md).

Измерены четыре Conv1-фильтра на десяти фиксированных настоящих CIFAR-патчах, с обученным BN и цифровым LIF. Полный CIFAR-классификатор не запускался; завершение сбора и точность весов — разные результаты.

## Fresh300 и полный CPU: профили VS Code

В `.vscode/launch.json` после профиля справки доступны профили fresh300 preflight, fresh300 native SIM, CPU CLI help и CPU reference. Ни один не запускает аппаратуру. MemriBoard-профили используют repo-local `.venv`, CPU-профили — `.venv-classifier`; установка PyTorch описана в [FULL_CIFAR_CLASSIFIER.md](FULL_CIFAR_CLASSIFIER.md).

Точные команды датированного [fresh300](../protocols/FRESH_CONV1.md):

```sh
python experiment.py --test-cifar-fresh-conv1 --dry-run
python experiment.py --test-cifar-fresh-conv1 --run-simulator
```

У протокола неизменяемый cutoff **2026-10-16 21:50 МСК / 18:50 UTC**, проверяемый до открытия backend и перед каждым воздействием. Он относится и к simulator. После cutoff запуск сбора откажет; профиль не является способом продлить дату. Для другого дня требуется отдельный reviewed protocol.

CPU-only команды (выберите Python из `.venv-classifier`):

```sh
python full_cifar_classifier.py --help
python full_cifar_classifier.py --cpu-reference --case-index 0 --output results/full_classifier_cpu/new_case0.json
```

Профиль CPU `--help: справка (без inference)` только загружает CLI/PyTorch и показывает справку; это **не** проверка SHA checkpoint или запуск inference. Флага preflight у этого CLI нет. Профиль reference проверяет pinned входные файлы, выполняет полный CPU forward и сравнивает его с исходной моделью. Он предлагает индекс изображения 0..9 и новый относительный выходной JSON; существующие результаты защищены от перезаписи. Это FC10 для fine labels `[0,1,2,3,5,8,13,14,17,19]`, не100-классовая сеть. `.venv-classifier` может отсутствовать на вашем компьютере: создайте его по инструкции зависимостей; профили не устанавливают пакеты автоматически. Аппаратный measured import остаётся отключённым: [граница bridge](FULL_CIFAR_NATIVE_LINK.md).

## A и B/C: точные команды

Preflight без подключения:

```sh
python experiment.py --cifar-feedback-reserve --dry-run
python experiment.py --cifar-feedback-remaining-reserves --dry-run
```

Новый native SIM:

```sh
python experiment.py --cifar-feedback-reserve --run-simulator
python experiment.py --cifar-feedback-remaining-reserves --run-simulator
```

Только для намеренно разрешённого оператором нового аппаратного опыта:

```sh
python experiment.py --cifar-feedback-reserve --run-hardware
python experiment.py --cifar-feedback-remaining-reserves --run-hardware
```

A ограничен 24323 вызовами / 3232 решениями записи; B/C — 40378 / 5260. Запуски создают новые UUID. B/C не подхватывает последний A автоматически: оба backend привязаны к точным опубликованным completed-A manifest/audit SHA. Замена родителя требует отдельного review.

Сохранённый аппаратный B/C получен из clean tracked commit `418c4ad4ca1551d7733ccc4718ae85aab55937fa`; source commit и поздний commit публикации данных различаются. Проверка полученного результата без аппаратуры:

```sh
python audit_cifar_feedback_remaining_reserves.py --session results/cifar_feedback_remaining_reserves/58ca618b-c37f-454e-be25-db21d1d25f60 --output results/cifar_feedback_reviews/remaining_hardware_REAUDIT
```

Выходной каталог должен быть новым. Повторный физический запуск для проверки опубликованных данных не нужен. Итог: 20 новых B/C + 13 ранее измеренных A + 267 исходных коэффициентов, а не одновременно записанные 300 весов. Ниже сохранены настройка окружения и исторические сценарии.

## Получение проверенной версии

```sh
 git clone --branch codex/full-cifar-diagnostic https://github.com/alexander-toschev/snn-memristor-hardware.git
 cd snn-memristor-hardware
```

Обновление существующего checkout с чистыми tracked-файлами:

```sh
 git status --short
 git fetch origin
 git switch codex/full-cifar-diagnostic
 git pull --ff-only origin codex/full-cifar-diagnostic
 git log -5 --oneline
 git rev-parse HEAD
 git status --short
```

Выведенный `git rev-parse HEAD` должен совпасть с полным SHA проверенного
опубликованного коммита. Используйте ветку `codex/full-cifar-diagnostic`;
выполняйте команды по одной.
Последний коммит не становится проверенным автоматически. При расхождении
веток или локальных изменениях остановитесь и разберите различия; не удаляйте
данные через reset/clean. Ветка содержит код и опубликованные результаты; приватные файлы драйвера
остаются локальными. На Raspberry выполняется код именно этого checkout.

## Окружение и локальный драйвер

Наблюдавшееся окружение Raspberry: Python 3.13.5; системные NumPy 2.2.4 и
 typing_extensions 4.13.2; SQLAlchemy 2.0.51 и greenlet 3.5.6 внутри venv.
Это описание окружения, не универсальный lock. Системные аппаратные зависимости
готовит оператор как для штатного MemriBoard. Windows .venv/.deps на Pi не копировать.

```sh
python3 --version
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install SQLAlchemy==2.0.51 greenlet==3.5.6
python setup_driver.py --source ../MemriBoard
git check-ignore vendor/MemriBoard/MemriCORE/rp5_fpga_python/rpi_FPGAed.py vendor/MemriBoard/MemriCORE/rp5_fpga_python/FPGA_SIM.py
python experiment.py --cifar-feedback-remaining-reserves --dry-run
```

../MemriBoard — установленная оператором соседняя директория. Helper проверяет
SHA256 обоих файлов до копирования, копирует только rpi_FPGAed.py и FPGA_SIM.py
и ничего не импортирует. Несовпадающий существующий файл не перезаписывается.
Файлы остаются локальными и игнорируются Git: нельзя применять git add -f или
публиковать их содержимое. Если SHA не совпал, проверьте версию с оператором,
не ослабляя preflight. INI платы: configs/raspberry_board.ini.

Dry-run проверяет exact fingerprints Connector/converters/драйвера, INI, веса
по сохранённым файлам. Импортируемость программных и аппаратных зависимостей dry-run не проверяет. Это не проверка проводки и не физическая калибровка.
Актуальные команды A/B/C приведены выше; исторический адаптивный сценарий описан в [CIFAR_FEEDBACK_ADAPTIVE.md](../protocols/CIFAR_FEEDBACK_ADAPTIVE.md).

## Исторический четырёхфазный эксперимент

Без флагов — справка. Отдельный simulator: python experiment.py --run-simulator.
Историческая команда первой фазы (не текущий пакет A/B/C):

```sh
python experiment.py --run-hardware --phase phase1
```

Phase1: 63 baseline impact и до 75 попыток программирования. Недостижимый
RESET/SET endpoint, ADC/transport error или превышение bounds останавливают
run с сохранением частичных данных. Окно q=.01–.05 пока не квалифицировано
для физического mapping. Обновление launcher не исправляет это ограничение.
Остановленный run не является успешной CIFAR inference.

Продолжение требует завершённую phase1 той же версии, INI, плана и host:

```sh
python experiment.py --run-hardware --phase phase2 --phase1-session results/hardware_four_phase/SESSION_UUID
python experiment.py --run-hardware --phase phase3 --phase1-session results/hardware_four_phase/SESSION_UUID
python experiment.py --run-hardware --phase phase4 --phase1-session results/hardware_four_phase/SESSION_UUID
```

--run-hardware --all выполняет четыре фазы с gate после phase1. --max-pulses N
уменьшает общий бюджет, не меняя научные параметры. Native длительность
импульса неизвестна, физический массив один. Аппаратный доступ выполняет
только публичный manager.board.Connector. Не вызывайте GPIO/SPI/драйвер напрямую.

## Provenance и результаты

Сохраните целиком UUID-каталог с raw, manifest, attempts, summary, INI/plan
snapshots. Manifest содержит hashes experiment.py и runtime sources, weights,
Git commit, dirty_tracked, отдельные untracked пути, process_argv и hardware_argv.
Git считывается до создания новой сессии. Untracked generated results не делают
tracked-код dirty; private ignored файлы проверяются отдельными fingerprints.
При недоступном Git сохраняется null/error. Для воспроизводимости нужен clean
reviewed commit; provenance не заменяет ревью. Данные не публиковать без разрешения.

Offline проверки, без аппаратуры:

```sh
python -m unittest tests.test_launch tests.test_hardware_suite tests.test_native_conversions -v
```

## Историческая проверка MVM без записи весов

```sh
python experiment.py --dry-run --check-mvm
python experiment.py --run-hardware --check-mvm
```

Проверяются существующие проводимости: WL0/2, BL2/3, по три mode9 чтения
nominal 0.1 В до и после (24 impact). Mode10: zero, BL2, BL3, BL2+BL3,
nominal 0.1 В, пять повторов каждой четвёрки на каждой WL (40 impact).
Векторы чередуются внутри повтора. RESET/SET/mode7 и запись весов отсутствуют.
Это аналоговая диагностика, не CIFAR inference. Всего 64 impact.

Native mode9 возвращает ID=0. Его native R использует INI vol_read=.3 и
сохраняется отдельно; оценки R/G рассчитывает native a2r с read_voltage=.1.
Mode10 преобразуется native a2v с gain=1; ADC=0 допустим. Код DAC .1 В — 82;
native d2v округляет до .1 В. Это номинальная модель, не калибровка напряжения.
Перед MVM проверяется модель тока <=5 мА и выхода <=80% ADC reference;
наблюдаемый выход также ограничен 80%. ADC/transport errors останавливают run.

Результаты: results/mvm_checks/<uuid>/{raw.jsonl,summary.json,manifest.json}.
Сохраняются SI/DAC, native tuples, ADC, все request/result и ошибки; общий lock
с read_all/four-phase. close_port не гарантирует сброс после исключения.
Для каждой WL публикуются mean/sample SD ADC и V по четырём векторам,
epsilon=mean(AB)-mean(A)-mean(B)+mean(zero) в ADC и V. Это отдельная описательная
метрика суперпозиции с поправкой на zero; исходные данные не корректируются,
порог приемлемости не задаётся. Pre/post G differences не доказывают запись
или физический drift. No pass/fail noise threshold, no fitted scale.

Offline тест: python -m unittest tests.test_mvm_check -v.
## Проверка в порядке GUI «Математика», весь массив 32×8

```sh
python experiment.py --dry-run --check-mvm-gui
python experiment.py --run-hardware --check-mvm-gui
```

Этот режим был создан для проверки «как Математика». Прежний
--check-mvm (64 обращения) сохраняется. Новый режим читает все 256 ячеек
mode9 по три раза до и после: 1536 impact. Затем/между этими readback-блоками
MatMul: пять повторов входов zero, A(BL2=.1), B(BL3=.1), AB и ALL32([.1]*32),
для каждого входа все WL0..7 подряд. Всего 200 mode10, вместе 1736 impact.
ALL32 идёт после четырёх меньших входов в каждом повторе и охватывает все BL.
Mode10 task ровно как GUI: mode_flag, vol, id=0, wl; список vol всегда новый.
Локальный task_id журнала остаётся уникальным. Маска all ones, без scaling,
correction и дополнительной quantization; штатное округление SI/DAC и a2v
сохраняется. Никаких mode7, RESET/SET и записи весов.

GUI math.py MatMul использует input outer/WL inner, a2v(gain=1). Функция
read_current_weights_matrix берёт round(sum_gain/R,4) из GUI all_resistances
cache, сама не измеряет плату. GUI при predicted>=3.3 В пропускает impact и
подставляет ADC0. Здесь cache недоступен: predicted строится по свежим mode9
readback, включая отдельную оценку с округлением q как GUI. Перед ЛЮБЫМ MVM
проверяются все 40 комбинаций WL/input: при модельном токе >5 мА, выходе
>80% ADC reference либо GUI prediction>=3.3 В run останавливается. Нулевые
ответы для пропущенных измерений не выдумываются. Наблюдаемый выход >80%
reference тоже останавливает run. Модельный ток не является физическим
измерением или аппаратным ограничителем. Это явное отличие от GUI skip.

Manifest сохраняет GUI source SHA и различия guard/cache; raw и summary —
actual ADC, native conversion, пять векторов по каждой WL и pre/post G по
всем ячейкам. Проверка суперпозиции A/B не охватывает все возможные комбинации;
ALL32 даёт отдельный dense-отклик, а не точность CIFAR inference.

## Карта кандидатов перед экспериментом, без программирования

```sh
python experiment.py --dry-run --map-cells
python experiment.py --run-hardware --map-cells
```

Исторический сценарий карты. Все 256 ячеек читаются
mode9 nominal .1 В по три раза до и после (1536 impact). Пять повторов
mode10: zero, затем BL0..31 с .05/.1 В; каждый вход проходит WL0..7.
Максимум 2600 mode10, всего 4136 impact. Только один BL активен; веса не пишутся.
SI/DAC native преобразования: .05 В→41, .1 В→82. Локальные task_id уникальны;
mode10 native id=0. Прежние MVM проверки и четырёхфазный план не изменяются.

Предварительные критерии в configs/cell_map.json редактируются отдельно и
сохраняются вместе с SHA в сессии. Для каждого уровня signal=mean(input)-mean(zero),
noise=max(1 ADC code,sqrt(SDinput²+SDzero²)); положительный signal/noise>=5.
CV(.1)=sqrt(SDhigh²+SDzero²)/signalHigh <=5%. Linearity epsilon=meanHigh−2meanLow+meanZero,
|epsilon|<=max(5%|signalHigh|,3*SEepsilon,2 ADC codes), где
SEepsilon=sqrt(SDhigh²/5+4*SDlow²/5+SDzero²/5).
Pre/post conductance: |meanPost−meanPre|/meanPre <= max(5%,3*SEdelta/meanPre),
SEdelta=sqrt(SDpre²/3+SDpost²/3). Это screening defaults, не валидированная гарантия.
Общий zero используется для WL; его шум учтён, а зависимость оценок разных
ячеек через общий zero не доказывает их статистическую независимость.

Невалидная модель R/G не заменяется числом: ячейка помечается reject,
её MVM явно записывается как skipped без fabricated ADC, остальные продолжаются.
Для single-cell модели проверяются 5 мА, 80% ADC reference и GUI prediction<3.3 В;
непрошедшая ячейка исключается. Это защитные модельные bounds, не калибровка.
Ошибка транспорта/формата ADC, saturation или фактический выход >4 В прекращают
весь run с partial map. ADC=0 MVM допустим, но нулевой signal не станет candidate.

Результаты в results/cell_maps/<uuid>: raw.jsonl, criteria.json, manifest.json,
summary.json, cells.json, cell_map.csv, map.html. Зелёный candidate, красный reject,
серый unmeasured; HTML tooltip содержит метрики. candidate_cells.csv содержит
строго заголовок wl,bl и координаты кандидатов, совместимые с маской GUI.
При отсутствии кандидатов сохраняется один заголовок. Это маска текущего
состояния при .05–.1 В, НЕ разрешение программировать ячейки. После записи
весов карта недействительна. Pair additivity/полный выбранный MVM нужно
проверить отдельно. Автовыбор или запись весов здесь не выполняются.

Offline: python -m unittest tests.test_cell_map -v.

При общей остановке HTML явно показывает failed, summary.mask_usable=false,
а candidate_cells.csv остаётся пустым (только wl,bl), даже если несколько
первых ячеек успели завершить все измерения. Их provisional метрики сохраняются
в cells.json/cell_map.csv, но частичная карта не выдаётся как готовая GUI-маска.
