# Документация

Начните с [плана на понедельник, 19 октября](plans/EXPERIMENTS_20261019.md), [готовности полного CIFAR](plans/FULL_CIFAR_READINESS_20261016.md) и [воспроизведения окружения](guides/REPRODUCE.md).

[Карта репозитория](REPOSITORY_LAYOUT.md) · [README проекта](../README.md) · [офлайн-инструменты](../tools/README.md) · [тесты](../tests/README.md)

Документы 14–16 октября описывают конкретные завершённые протоколы. Старые команды и cutoff не являются разрешением нового запуска. Исторические результаты остаются в `results/` и `reports/` без переписывания.

## Отчёт разработчику

- [Операции HW и SIM: результаты и ограничения, 17 октября](developer-reports/2026-10-17/REPORT_RU.md)
- [Воспроизведение измерений и анализа](developer-reports/2026-10-17/REPRODUCTION.md)

## Окружение, воспроизведение и инструменты

- [Offline-сравнение B/C и конечная карта резервов](guides/CIFAR_REMAINING_COMPARISON.md)
- [Offline-сравнение резервного маршрута A](guides/CIFAR_RESERVE_COMPARISON.md)
- [Runbook — local simulator, four phases](guides/CLAUDE_HARDWARE_RUNBOOK.md)
- [Прямое HW–SIM сравнение сохранённых выходов](guides/COMPARE_FRESH_TRACES.md)
- [Прямое сравнение HW и native SIM](guides/EXPORT_FRESH_TRACE.md)
- [Offline-анализ fresh300 и варианта без BL8](guides/FRESH_CONV1_STABILITY_ANALYSIS.md)
- [Причины отказа качества и расхождения LIF](guides/FRESH_FIDELITY_DIAGNOSTICS.md)
- [Full Conv1 bridge preparation (software only, not enabled hardware import)](guides/FULL_CIFAR_BRIDGE.md)
- [Full CPU classifier and future measured Conv1 boundary](guides/FULL_CIFAR_CLASSIFIER.md)
- [Native linkage extension: verified historical300, full-image import still closed](guides/FULL_CIFAR_NATIVE_LINK.md)
- [MemriBoard configuration audit - REV3, 2026-10-14](guides/HARDWARE_REQUIREMENTS.md)
- [Разработчику MemriBoard: запуск и разбор результатов](guides/MEMRIBOARD_DEVELOPER_REPRO.md)
- [Компактная публикация карты](guides/PACKAGE_FRESH_REPORT.md)
- [Первое чтение физической платы](guides/READ_ALL.md)
- [Воспроизведение на Raspberry](guides/REPRODUCE.md)

## Протоколы ограниченных опытов

- [Бинарный MVM и цифровой LIF](protocols/BINARY_MVM.md)
- [Бинарные входы и LIF: что можно заключить offline](protocols/BINARY_SPIKE_NOTES.md)
- [Подготовленные операции MVM, CIFAR Conv1 и LIF](protocols/CALIBRATED_MVM_OPERATIONS.md)
- [Discovery SET/RESET и проводимостей всех 256 ячеек](protocols/CELL_CALIBRATION.md)
- [Повтор интервалов: refined v2](protocols/CELL_RANGE_REFINED.md)
- [Повтор попадания в интервалы: три ячейки](protocols/CELL_RANGE_REPEAT.md)
- [CIFAR: ограниченные микротайлы](protocols/CIFAR_FEEDBACK_ADAPTIVE.md)
- [Полный диагностический CIFAR с фиксированным масштабом](protocols/CIFAR_FEEDBACK_DIAGNOSTIC.md)
- [Один ограниченный опыт с резервной WL7/BL15](protocols/CIFAR_FEEDBACK_RESERVE.md)
- [CIFAR: запись с обратной связью по измеренному mode10](protocols/CIFAR_FEEDBACK_RUN.md)
- [C9: диагностическая запись пары непосредственно после RESET](protocols/CIFAR_JIT_PROGRAMMING.md)
- [CIFAR Conv1: запись весов, матрица коррекции и измеренные суммы](protocols/CIFAR_MEMRISTOR_RUN.md)
- [B/C: конечный пакет оставшихся резервов](protocols/CIFAR_REMAINING_RESERVES.md)
- [Новый сбор Conv1 с подтверждением RESET непосредственно перед записью](protocols/FRESH_CONV1.md)
- [Fresh Conv1 без физических адресов BL8](protocols/FRESH_CONV1_NO_BL8.md)
- [Последовательные измеренные вклады и цифровой LIF](protocols/MAPPED_SEQUENTIAL_MVM.md)

## Планы и границы готовности

- [Offline-план 75 коэффициентов Conv1](plans/CIFAR75_CANDIDATE_ROUTING.md)
- [Завершение физического CIFAR: проверяемый результат и охват ячеек](plans/CIFAR_COMPLETION_CRITERIA.md)
- [Эксперименты на понедельник, 19 октября 2026](plans/EXPERIMENTS_20261019.md)
- [Full-image CIFAR readiness, 16 October 2026](plans/FULL_CIFAR_READINESS_20261016.md)
- [CIFAR Conv1: четыре физические фазы](plans/HARDWARE_PLAN.md)
- [Резервный вариант: MVM из реально измеренных частичных вкладов](plans/SEQUENTIAL_MVM_PROPOSAL.md)
- [Two-crossbar plan — four simulator phases](plans/TWO_CROSSBAR_PLAN.md)

## Исторические аудиты и лабораторные записи

- [Independent B/C audit  final source freeze](reviews/AUDIT_REMAINING_RESERVES_DRAFT.md)
- [Аппаратное исследование проводимостей и новая карта](reviews/CALIBRATION_RESULTS_20261015.md)
- [Контроль готовности CIFAR и цифрового LIF](reviews/CIFAR_GOAL_AUDIT.md)
- [Независимая проверка JIT-C9](reviews/CIFAR_JIT_AUDIT.md)
- [Проверка ограничения прогнозного тока: v2](reviews/CURRENT_GUARD_RESULTS.md)
- [Fresh300: независимый независимый аудит](reviews/FRESH_CONV1_AUDIT.md)
- [Независимый аудит fresh300 без BL8](reviews/FRESH_CONV1_NO_BL8_AUDIT.md)
- [No-BL8 source review — 16 October 2026, 20:20 MSK](reviews/FRESH_CONV1_NO_BL8_SOURCE_REVIEW.md)
- [Научный журнал: CIFAR, мемристоры и калибровка](reviews/SCIENTIFIC_LAB_NOTES.md)

Новые отчёты для разработчиков размещаются в `docs/developer-reports/YYYY-MM-DD/`; ссылка добавляется после появления и проверки документа.
