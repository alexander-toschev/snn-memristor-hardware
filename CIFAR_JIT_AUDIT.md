# Независимая проверка JIT-C9

`audit_cifar_jit_programming.py` работает offline: не импортирует collector,
MemriBoard, драйвер или Connector. Он проверяет SHA замороженных независимых
аудиторов и заново разбирает исходные команды/ответы native mode7/mode10.

Проверяются точный маршрут девяти весов C9, два цикла, RESET обеих ячеек до
совместного quiet gate, нативные преобразования и current guards, отсутствие
SET при отрицательном gate, sentinel WL3/BL8, пять bracket-измерений каждой
пары, retention не менее1 секунды, карта всех девяти пар после каждого решения,
счётчики и пределы29934 impact /3942 SET+RESET. Каждый исходный task должен
быть использован ровно в предусмотренной позиции. Протокол, INI, provenance,
Git blobs и hashes всех result-файлов связаны с manifest.

```powershell
python audit_cifar_jit_programming.py --repo . --session results/cifar_jit_programming/SESSION_UUID --output results/cifar_feedback_reviews/JIT_REVIEW_NAME
```

Выходная папка должна быть новой. Аудитор не изменяет исходную сессию.
Прерванные/transport/current-stop сессии сейчас отклоняются fail closed;
для них PASS полноты не выдаётся. Нормальный полный сбор с quiet gate failure
допускается как диагностический результат: это не успешная запись.

`test_audit_cifar_jit_programming.py`:16 offline tests. Collector используется
только для генерации явно синтетических FakeConnector fixtures. Проверены
успешный захват двух циклов, разрушение первой quiet-ячейки вторым RESET,
подмена gate/target/current prediction/converter, удаление решения/sentinel,
изменение маршрута/бюджета, слишком короткая retention и неизвестные/stop events.
Эти тесты не являются native SIM или аппаратным экспериментом.

PASS аудитора означает согласованность команд и данных. Точность конечного
состояния, факт программирования всех18 решений и повторяемость различаются.
Два фиксированных цикла не доказывают долговременной стабильности и не
квалифицируют остальные исторические114 assigned ячеек.
