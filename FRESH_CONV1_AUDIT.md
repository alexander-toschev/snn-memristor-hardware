# Fresh300: независимый независимый аудит

Scientific validator. Только offline. Collector не импортируется и не является
oracle. Exact phase names, config/route/evidence и runtime связаны с финальными SHA.
15 offline tests PASS; отдельные Fake fixtures: success143625 checks и
quiet failure25105 checks, ошибок0. Native SIM/HW ещё требуют actual audit.

1. Исходные данные: exact300 trained weights и75tap/filter, новый замороженный
   capacity evidence_v2 SHA `f1c109655557e8186d7356250c80194facd7f2e5ab209fdacfe5ab0cb38c2d72`,
   exact frozen44tile primary/reserve routes. Сохранить все candidate attempts,
   включая неуспешные, отдельно от реально contributing cells. Проверить
   отсутствие импортированных старых partials/выходов и повторного tap.
2. Использовать только проверенный SHA независимого native raw helper:
   source/session/ID, SI commands, native converters, current/ADC guards,
   unknown hardware width, штатный SIM125steps. Никакого collector/backend import.
   Глобальные transport/current/deadline interruptions fail closed; отдельный
   quiet/finiteprogramming failure не скрывать как техническую ошибку.
3. RESET replay хранит index0..54 для каждой ноги конкретной candidatepair.
   После single crossing идут5 подтверждений. Failed mean/SD продолжает
   оставшийся ramp; счётчик не обнуляется. После обеих ног — fresh joint gate.
   Не более2 repair rounds. Каждый repair обязан израсходовать хотя бы один
   следующий RESETstep;55/leg не превышается. Ни одна SET до совместного gate.
4. Candidate fallback: не более2 заранее описанных pair attempts/tap;
   только последующий разрешённый candidate после первого failed gate;
   не красть primary другой пары и не дублировать адрес в contributing tile.
   Число возможных attempts/impacts/writes независимо вывести из frozenplan,
   а не принять произвольные поля collector. Частичный tile не выдавать за75tap.
5. Active SET leg фиксирована знаком trained weight и capacity screen.
   При начальном D0 с residual противоположного направления за stopband:
   `initial_overshoot`,0SET, fresh mode7/paired confirmations, фактический
   вклад остаётся в последующих измерениях/ошибке. Не программировать
   противоположную неактивную ногу. Если already withinband — hit с0SET.
   Максимум109SET из зафиксированного ramp, stopband и post-five проверяются.
6. Матрица C обучается только по fresh synthetic masks и measured H;
   fixed offset0, independent selection, singular/gain/rank guards, freeze
   раньше любого CIFARrequest. Need own independent .25-gain fidelity/math:
   старые helper fixeddecode/fidelity принимают лишь1/.03 и не могут
   молча использоваться как oracle. New nativeSIMgain=.03, HWgain=.25.
7. CIFAR: каждый partial требует новых raw impact, serialonehot sums,
   zero brackets; никаких lookup H или CPU W*x вместо измерений.
   Все75 contributions/filter складываются перед обученным BN[t] и одним
   цифровым LIFstep/logicalt. Exact4filters×10patches×5repeats×12t=2400.
   Trained reference и own measuredW_eff reference отдельно; zero spikesdiff
   к W_eff не означает точных trainedweights/полного классификатора.
8. Final manifest связывает runtime/config/inputs/INI/native sources,
   clean source Git provenance, все файлы/hash и counters. Raw consumption
   exact; все candidate attempts в total, только final selectedroute в decode.
9. Обязательные adversarial tests: failed confirmation→remainingstep,
   repaired firstleg broken by secondRESET→jointfail; step reset/extra repair
   отвергаются; forbidden fallback/duplicate activeaddresses; residualopposite
   initial_overshoot noSET и запрет oppositeleg; omitted tap/oldpartials;
   matrix fit/freeze order, gain tamper, unsafe nativeR/ADC, protocol JSON
   reserialization allowed only against pinned Gitbytes; all300 arithmetic.

Документы и tests не меняют runtime. Formal hardware GO требует завершённого
actual nativeSIM и собственного полного raw audit PASS, без пересмотра
thresholds под результаты.


Запуск из корня клона (без доступа к плате):

```shell
python audit_fresh_conv1.py --repo . --session results/cifar_fresh_conv1/SESSION_UUID --output results/cifar_feedback_reviews/fresh_SESSION_audit
python -m unittest test_audit_fresh_conv1 -q
```

Только tests импортируют collector для создания явных FakeConnector fixtures.
Сам auditor использует стандартную библиотеку и pinned независимые replay helpers.
PASS означает целостность и согласованность измерений, а не точность весов.
При исчерпании quiet route PASS допустим лишь с acquisition_complete=false,
без summary/CIFAR подстановок. Transport/current/deadline interruptions
не интерпретируются как завершённый эксперимент.
