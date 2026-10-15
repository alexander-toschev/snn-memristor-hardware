# Воспроизведение на Raspberry

Все команды выполняются из корня репозитория. Получение кода и dry-run не
запускают эксперимент. Сейчас пользователь разрешил только диагностический --check-mvm;
четырёхфазные программирующие команды ниже остаются справочными.

## Получение проверенной версии

```sh
 git clone https://github.com/alexander-toschev/snn-memristor-hardware.git
 cd snn-memristor-hardware
```

Обновление существующего checkout с чистыми tracked-файлами:

```sh
 git status --short
 git fetch origin
 git switch main
 git pull --ff-only origin main
 git log -5 --oneline
 git checkout --detach REVIEWED_COMMIT
 git rev-parse HEAD
 git status --short
```

REVIEWED_COMMIT заменяется полным SHA проверенного опубликованного коммита.
Последний коммит не становится проверенным автоматически. При расхождении
веток или локальных изменениях остановитесь и разберите различия; не удаляйте
данные через reset/clean. Публичная ветка содержит код без прежних
неопубликованных результатов. На Raspberry выполняется код именно этого checkout.

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
python experiment.py --dry-run
```

../MemriBoard — установленная оператором соседняя директория. Helper проверяет
SHA256 обоих файлов до копирования, копирует только rpi_FPGAed.py и FPGA_SIM.py
и ничего не импортирует. Несовпадающий существующий файл не перезаписывается.
Файлы остаются локальными и игнорируются Git: нельзя применять git add -f или
публиковать их содержимое. Если SHA не совпал, проверьте версию с оператором,
не ослабляя preflight. INI платы: configs/raspberry_board.ini.

Dry-run проверяет exact fingerprints Connector/converters/драйвера, INI, веса
по сохранённым файлам. Импортируемость программных и аппаратных зависимостей dry-run не проверяет. Это не проверка проводки и не физическая калибровка.
После обновления checkout разрешён диагностический MVM, описанный ниже.

## Команды эксперимента после разрешения

Без флагов — справка. Отдельный simulator: python experiment.py --run-simulator.
Физическая первая фаза, только после нового явного разрешения:

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
python -m unittest test_launch test_hardware_suite test_native_conversions -v
```

## Текущая разрешённая проверка MVM без записи весов

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

Offline тест: python -m unittest test_mvm_check -v.