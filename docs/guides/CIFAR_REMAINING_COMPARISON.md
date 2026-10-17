# Offline-сравнение B/C и конечная карта резервов

`compare_cifar_remaining.py` читает опубликованные результаты и связанные
независимые аудиты. Коллекторы, драйверы и аудиторы не импортируются; Git и
плата не вызываются. Чистые функции прежних компараторов загружаются только
после проверки закреплённых SHA. Сам компаратор не выполняет raw replay.

Поддержан frozen B/C protocol `cifar-feedback-remaining-reserves-v1`,
auditor SHA `4f2f73eb96fd286aabde4e0513b7aac2ef96e57ef873ce36573a73a29a3b0525`,
с actual completed-A parent каждого backend. Другая версия или fallback
после failed-A требуют отдельного review и не принимаются неявно.

Из корня репозитория, когда опубликованы оба B/C audit PASS:

```text
python compare_cifar_remaining.py --sim-session results/cifar_feedback_remaining_reserves/99631af1-c73e-42ed-abcd-20d4e580f691 --sim-audit results/cifar_feedback_reviews/remaining_simulator_20261016/audit.json --hardware-session results/cifar_feedback_remaining_reserves/58ca618b-c37f-454e-be25-db21d1d25f60 --hardware-audit results/cifar_feedback_reviews/remaining_hardware_20261016/audit.json --output results/cifar_feedback_reviews/remaining_hardware_vs_simulator_recheck
```

Аргументы указывают на завершённые SIM и аппаратный B/C; соответствующие
аудиты PASS604836/0 и PASS669378/0. Сохранённый отчёт находится в
`results/cifar_feedback_reviews/remaining_hardware_vs_simulator_20261016/`.
Команда выше создаёт отдельную проверочную папку. Output — только новая папка;
нельзя писать внутрь сессии или измеренной ancestry. Пути относительны репо.

Файлы результата:

- `comparison.json`: связывание audit/manifest/result hashes и ancestry;
  fixed gains, счётчики, per-filter Conv/BN/LIF, различия спайков и повторы.
- `REPORT.md`: полнота, все десять заранее заданных резервов и их disposition.
- `simulator_contributing_map.svg`, `hardware_contributing_map.svg`: карта
  8 WL ×32 BL с цветами assigned/quiet unused/screened not quiet/unassessed
  внутри contributing измерений. Это union разных моментов, не финальная
  проводимость, состояние всех ячеек или одновременное хранение весов.

20 новых B/C и13 ранее измеренных A отдельно от267 исходных коэффициентов.
Final33=A13+BC20 не называется33 новыми измерениями в B/C. Для блоков B/C
сохраняются отдельные weight/fidelity и pre-BN/LIF partial-метрики.
Общий2400 выход вычислен только после суммы всех75 taps; запрещено выдавать
один fresh block за отдельный полный BN/LIF сети.

При научной capacity-неудаче одного блока сохранены только retained280 и
завершённые новые блоки. Полного300/2400 сравнения нет; исключённые старые
B/C не возвращаются автоматически. Техническая/transport-неудача, не
прошедшая независимый audit, отклоняется. Успех требует explicit completion.

Карта и unique-address union пересчитываются после замены, а добавленные
и вытесненные адреса перечисляются отдельно. Quiet, assigned, SET-attempted
и accurate — разные критерии. Десять резервов не доказывают абсолютный
оптимум массива среди всех состояний и режимов.

Проверка только offline:

```text
python tests/test_compare_cifar_remaining.py
```

16 тестов прошли на реальном native SIM B/C `99631af1…` и проверенном A:
exact300/2400,33/267 provenance, partial280/291, duplicate rejection,
auditor/manifest/ancestry/gain нарушения, неизвестные/quiet/неquiet резервы,
маркировка частичной карты. SIM audit PASS604836/0. Из этого не следует
аппаратная программируемость резервов или classifier accuracy.
