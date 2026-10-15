# CIFAR Conv1 — MemriBoard

Единая точка входа — **experiment.py**. Четыре фазы: характеристика записи,
14 фильтров, все 64 фильтра, три записи фильтра 41 и средние K=1,2,3.
Это эксперимент первого слоя, не accuracy полной SNN на CIFAR.

```sh
python experiment.py --help
python experiment.py --dry-run
python experiment.py --run-simulator
```

Без флагов показывается справка. Dry-run проверяет файлы исходников, INI и веса
без импорта физического драйвера и подключения. Без локальных
operator-provided файлов драйвера dry-run ожидаемо сообщает об их отсутствии.
Несовместимые CLI режимы отвергаются до подключения.

Физический сценарий: `--run-hardware --phase phase1`. Сейчас пользователь
запретил физические запуски; команды в документации не разрешают запуск.
Окно q=.01–.05 не квалифицировано для физического mapping. Алгоритм и
научные пороги в обновлении launcher не изменены.

- [Воспроизведение на Raspberry через GitHub](REPRODUCE.md).
- [Физический протокол и условия остановки](HARDWARE_PLAN.md).
- Физические настройки: configs/hardware_four_phase.json и configs/raspberry_board.ini.
- Реализация: experiment_hardware.py, hardware_suite.py, native_conversions.py.
- Настройки simulator находятся сверху experiment.py; он использует два
  массива A/B. Физический протокол использует один массив 32×8 последовательно.
- Веса: weights/conv1_weights.npy; штатный backend: vendor/MemriBoard.

Все аппаратные I/O — только публичный MemriBoard Connector open_port/impact/
close_port. Длительность native write неизвестна: не называть её 125 мкс.
Native converters используются для SI/DAC/ADC; исходные команды, ADC и ошибки
сохраняются без исправления невалидных значений.

Сессии сохраняются в новых UUID-каталогах results/simulator или
results/hardware_four_phase. Аппаратный manifest содержит hashes, Git commit,
tracked dirty state, untracked paths и CLI args. Старые сессии не перезаписываются.
Частные драйверы и неопубликованные результаты не входят в публичное обновление.

Offline тесты, без аппаратных обращений:

```sh
python -m unittest test_launch test_hardware_suite test_native_conversions -v
```

VS Code: первый профиль F5 — прежний симулятор; отдельный профиль
«Аппаратный preflight: без I/O» запускает только --dry-run.

Локальное окружение симулятора (Python 3.12; Linux/Raspberry activation: . .venv/bin/activate):

```sh
python3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python experiment.py --help
```

На Linux используйте .venv/bin/python вместо .venv/Scripts/python.
