# Review отдельного завершения MNIST

19 октября 2026. **Source GO условный:** маршрут восстановления допускается
к регистрации. Запуск root разрешён этим review только после успешного
обычного runtime preflight. Сохранённый preflight сейчас завершился отказом
по доступной RAM <8GiB; порог не изменялся. Этот отказ не заменяется успехом
проверки входных данных.

Проверены experiment.py и qualification.py нового
`experiments/mnist_stdp_finish_20261019`, отдельный
`tools/audit_mnist_stdp_finish.py`, документация и тесты. Независимо пройдены
7/7 synthetic offline-тестов, включая порядок обоих префиксов, запрет
двойного test, закрытый launch gate, контроль первых пяти эпох и точное
слияние source inventory с отклонением конфликтующих SHA.
Окончательные зарегистрированные SHA фиксируются отдельно в
`RECOVERY_SOURCE_FREEZE.json` после gate-only registration.

Основание — независимая проверка сохранённого префикса в
[INTERRUPTED_PREFIX_INDEPENDENT_AUDIT.json](INTERRUPTED_PREFIX_INDEPENDENT_AUDIT.json)
и CPU-only qualification автора в
[RECOVERY_INPUT_QUALIFICATION.json](RECOVERY_INPUT_QUALIFICATION.json).
Последняя дополнительно повторила train-only scaler точно и ограниченно
десериализовала новый checkpoint (`weights_only`, CPU), сверив W/threshold/cfg.
SHA qualification.py `fd211da4f6f0140a34949bf70866cb61fd182bbf4d03da552e7ee5b21725ad71`.
В отчёте автора записан более ранний entry SHA `87fb6215…`: после его старта
менялось только объединение source metadata; неизменённый qualification
проверял данные. Это не полный runtime preflight окончательного entry.

Согласованный протокол:

- Исходные 29 файлов незавершённой сессии остаются неизменными и закреплены
  точным inventory/SHA. Старый final manifest не создаётся задним числом.
- STDP30k и train-features30k не повторяются. Head обучается заново ровно
  25 эпох/1475 updates из seed42. Это новый head replay, не optimizer resume.
- Initial head, пять первых перестановок/RNG и RNG перед шестой эпохой
  совпадают побайтово. Потери первых пяти эпох должны совпасть в заранее
  заданных atol1e-6/rtol1e-5. Несовпадение останавливает путь до test.
- Новый head/scaler/checkpoint фиксируются до test. Feature network
  восстанавливается из нового STDP checkpoint и обязана иметь тот же
  W/bias/threshold frozen state, что все235 сохранённых train batches.
  Пластичность выключена; norm и Dales transforms отсутствуют.
- Единственный test10k/T100 выполняется в79 batches, затем единственная
  оценка final head25. Параметры и исходный snapshot сверяются после test.
- Отдельный auditor связывает две сессии, проверяет точный source/result
  inventory, 25 перестановок и RNG transitions, test coverage/newW/freeze,
  official labels, logits→predictions/confusion/accuracy и ограниченный CPU
  linear replay с прежними допусками. SNN и optimizer аудитом не запускаются.
  Standalone auditor фиксирует CPU threads2 перед exact scaler replay.

Runtime: только локальная изолированная MNIST-среда; Raspberry для этого
обучения/анализа не используется. Сохранены исходные preflight guards:
18GiB disk,8GiB available RAM,4GiB available GPU. Новый execution budget:
2ч,3GiB outputs,2GiB disk reserve. Старые acquisition и full auditor
не изменены и не объявляются прошедшими для другой схемы сессий.

Не воспроизводились STDP/энкодер/feature forward независимо; повторённые
перестановки не доказывают градиентные updates. Финальная точность требует
фактического завершения recovery и отдельного итогового аудита. Ни исторические
69,65%, ни аппаратная точность этим review не подтверждаются.
