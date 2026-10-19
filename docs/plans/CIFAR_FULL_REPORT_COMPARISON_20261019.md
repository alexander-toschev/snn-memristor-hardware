# Сравнение полного CIFAR case0: SIM и плата

Утилита `tools/compare_full_cifar_reports.py` работает только с завершёнными
отчётами штатного full-classifier и явно указанными сессиями. Это отдельная
проверка/сводка сохранённых отчётов, не повторный raw audit и не запуск модели.
Нужен только Python standard library. Frozen seven-source CIFAR не изменяется.

Проверка программной логики, только синтетические JSON:

```powershell
& .venv-mnist-pilot/Scripts/python.exe -B -m unittest tests.test_compare_full_cifar_reports -v
```

Основную команду выполнять на Raspberry из проверенного Git checkout после
завершения обоих classifier reports и независимого ревью. Windows checkout
не содержит приватный rpi_FPGAed.py: проверка там обязана остановиться.
Для Pi нужны оба отчёта и manifest/protocol sidecars после обновления Git,
но скачивать raw SIM (сотни МБ) для этой утилиты не требуется.
Ниже команда для текущей аппаратной сессии
`f21e608c-059f-4dce-afe0-6edf5c26133c`. Её данные ещё не готовы:
команду выполнять только после статуса `completed`, успешного независимого
аудита и создания `NATIVE_HW_FULL_CLASSIFIER.json`. Указание имени будущего
отчёта не означает, что результат уже существует.

```bash
python3 -B tools/compare_full_cifar_reports.py --sim-report reports/2026-10-19/cifar-full-case0-review/NATIVE_SIM_FULL_CLASSIFIER.json --hardware-report reports/2026-10-19/cifar-full-case0-review/NATIVE_HW_FULL_CLASSIFIER.json --sim-session results/cifar_full_case0_20261019/5ee64f20-d499-40df-a742-95a76d2dc386 --hardware-session results/cifar_full_case0_20261019/f21e608c-059f-4dce-afe0-6edf5c26133c --output reports/2026-10-19/cifar-full-case0-review/FULL_SIM_HARDWARE_COMPARISON.json
```

Выходной файл создаётся только новым; история не перезаписывается. Утилита
читает manifest.json/protocol.json исключительно указанных завершённых сессий,
а также локальные исходники для SHA-сверки. Она не ищет сессии, не читает
raw/events/start manifests, не открывает MemriBoard, не обращается к сети.
Все записанные исходники, включая приватные MemriCORE, обязаны присутствовать
и совпадать с SHA. Отсутствие не пропускается и не заменяется аттестацией.

Проверки: те же case0, checkpoint, fixture, protocol, 4800 коэффициентов,
48 этапов и 1 806 336 вкладов Conv1; полный filter/tap/cell mapping,
совпадение SHA отчёт→manifest и цепочки зарегистрированного classifier.
`status=pass` означает согласованность отчётов/CPU controls, а не успешное
распознавание или физическую эквивалентность. Число impact и импульсов обязано
укладываться в закреплённый конечный budget протокола.

В результате представлены:

- Ошибки непосредственного, после всех записей и конечного веса: raw/C,
  RMSE, relative L2, число попаданий в исходный voltage tolerance,
  точных нулей и неправильных знаков. Индивидуальная подгонка запрещена.
- Heldout RMSE по всем 48 этапам, средние, число улучшившихся/ухудшившихся/
  неизменных этапов, retention и дрейф нуля.
- Conv1 RMSE/relative L2; выходные спайки всех десяти классов и всех 12 шагов,
  первая активность, мембраны, TWD, все три механических декодера и их ties.
- Отдельно `valid_unique_decision` и `valid_correct`: при тишине first-spike
  и TWD не имеют валидного решения, даже если механический argmax равен
  истинному классу0. Мембранный readout обозначен отдельно и может иметь
  уникальный аналоговый ответ без выходного спайка.
- Проверка CPU-компараторов между платформами с фиксированными atol=rtol=1e-5;
  спайки должны совпадать точно. Отличия вне допуска или разные предсказания
  дают `review_required`, exit3 и сохраняются для разбора.

Один case0 — пример инференса, а не оценка test accuracy. Общий protocol
использует разные nominal gains: SIM0.015 V/weight и HW0.125 V/weight.
Это ограничивает прямое сравнение физических свойств; коэффициенты не являются
измеренным доказательством физической эквивалентности. BN/LIF и остальная
сеть выполняются цифровым кодом. Калибровка C исправляет аддитивность
измеренного состояния, но не подменяет обученные веса.
TWD scores берутся из зарегистрированного classifier; утилита проверяет
argmax/ties, но не переисполняет learned softplus(alpha) и не читает checkpoint.
