# Сравнение полного CIFAR continuation с исходным native SIM

Новый инструмент: `tools/compare_cifar_continuation.py`. Он нужен, потому что
исходный full SIM имеет одну completed-сессию, а hardware continuation собирает
48 тайлов из исходного verified prefix и новых completed stage sessions.
Эти схемы остаются разными: адаптер не создаёт фиктивный legacy hardware report.

Статус подготовки 19 октября: выполнены только software/synthetic tests. Полного
аппаратного 48-stage classifier report пока нет. Ни выводы о hardware accuracy,
ни фактическое двустороннее сравнение сейчас не получены. Требуется независимый
scientific review перед использованием нового инструмента.

Адаптер выполняется локально. Он не запускает raw replay, нейросеть, обучение,
native SIM, MemriBoard или сеть. Он читает два classifier JSON, завершённый SIM
manifest, SHA-bound stage audit JSON, manifests/start manifests, source/INI/
grant metadata и небольшие float32 payloads. Источник частного кода Pi может
подтверждаться только уже reviewed explicit receipt для конкретной сессии;
отсутствующий файл не считается автоматически проверенным локально.

Проверки:

- Ровно 48 уникальных полных тайлов, 4800 коэффициентов, 1 806 336 вкладов;
  одинаковые case0/image/checkpoint/fixture/physical protocol.
- SHA каждого classifier/auditor/composer, stage certificate, source inventory,
  execution grant и payload; для новых сессий обязателен completed manifest.
- Session ID сертификата должен совпадать с report/start manifest. По каждому
  тайлу нужна одна numerical diagnostic: strict PASS либо явно зарегистрированная
  condition-only policy с неизменным решением и устойчивым rank margin 16.
  Здесь проверяется согласованность метаданных, новый SVD/raw replay не выполняется.
- Повторное byte interleave всех 48 stage exports должно совпасть с сохранёнными
  raw/calibrated composite tensors. Нет CPU/zero fill и подбора лучшей попытки.
- Веса: immediate/post-all-write/final raw/C, RMSE, tolerance count, wrong sign,
  нулевой effective weight. Heldout/retention сохраняются по всем 48 тайлам.
- CPU comparator должен согласоваться в объявленном допуске 1e-5 abs/rel;
  spikes — точно. Нарушение возвращает `review_required`, а не тихий PASS.
- Выходные spikes/membranes/TWD по 10 классам и все три decoder. При тишине
  first-spike/TWD mechanical argmax остаётся виден, но `valid_correct=null`.
  Численные ничьи также не признаются успешным распознаванием.

`consistent_reports` означает согласованность отчётов/источников/payloads и CPU
controls, а не правильный ответ, test accuracy или эквивалентность платы SIM.
Один case0 остаётся одним примером. После физического Conv1 остальные слои,
BN и LIF цифровые; три tile contributions складываются до одного LIF step.
SIM gain 0.015 и hardware 0.125 V/weight различаются. Исходный full SIM использовал
одну непрерывную сессию массива; continuation имеет исходный prefix плюс новые
Connector/programming/calibration contexts. Квалификация восстановления с новым
SIM-массивом на каждый stage не подменяет исходный full 48-stage SIM baseline.
Неполный старый tail учитывается как неизвестный расход, не как ноль.

CLI имеет четыре обязательных параметра:

```text
--sim-report          исходный NATIVE_SIM_FULL_CLASSIFIER.json
--sim-session         его completed acquisition directory
--hardware-composite  будущий полный COMPOSITE_REPORT.json после local raw replay/classifier
--output              новый путь JSON внутри reports/
```

Известные SIM-входы:
`reports/2026-10-19/cifar-full-case0-review/NATIVE_SIM_FULL_CLASSIFIER.json` и
`results/cifar_full_case0_20261019/5ee64f20-d499-40df-a742-95a76d2dc386`.
Hardware путь и конечную команду задаём только после фактического complete 48-stage
classifier. Текущий 3-stage qualification JSON и любая неполная аппаратная
партия отвергаются. Существующие отчёты, source registry/grants, acquisition
исходники и raw остаются неизменными.

Software verification:

```sh
python -B -m unittest tests.test_compare_cifar_continuation -v
```

Синтетические fixtures существуют только во временных каталогах тестов;
они не публикуются как измерения платы.
