# Независимый аудит fresh300 без BL8

Отдельный протокол и отдельный аудитор; первоначальный audit_fresh_conv1.py
и его результаты не изменяются. Научная гипотеза — исключение BL8 из
конкретного маршрута может повысить сохранность записанных уровней. Это не
диагноз неисправности ячеек и не доказательство конкретной физической причины.
Изменяются 160 из 300 основных пар, поэтому это не строгая однофакторная
замена одной ячейки. Улучшение заранее не предполагается.

## Зафиксированный scope

Все 300 обученных коэффициентов измеряются заново, без imported partials.
44 тайла шириной до 7, 108 различных primary адресов, 530 максимально
разрешённых candidate-pair attempts. Все primary и fallback пары исключают
BL8. Candidate addresses внутри тайла попарно не пересекаются.
Максимум 1 686 292 native calls и 91 000 SET/RESET. HW gain=.25,
SIM gain=.03; эти источники не физически эквивалентны. Прежние native
converters, signs, current/ADC guards, joint quiet gate и cutoff 21:50 MSK.

RESET сохраняет индекс конечной 55-уровневой лестницы, включая failed
confirmation и joint repair; максимум две repair rounds. Никакого SET до
пяти подтверждений quiet обеих ног. Active leg задана знаком trained weight.
При противоположном исходном residual — noSET, сохранение фактических post
измерений; не программировать неподтверждённую противоположную ногу.
При исчерпании обоих candidates — явно incomplete, без старых или CPU вкладов.

Матрица C2 обучается по свежим synthetic masks и собственному measured H;
independent selection и rank/gain constraints, freeze до CIFAR. Каждая
фактическая partial sum подтверждается новыми native запросами. Все 75 taps
суммируются перед одним обученным BN[t] и цифровым LIF step. Полный scope:
4 фильтра ×10 patches ×5 repeats ×12t =2400 выходов. Это не полный
классификатор и не одновременный 32-input аналоговый MVM.

## Проверка

Аудитор не импортирует collector или native backend. Он самостоятельно
воспроизводит raw команды, native conversion arithmetic, guards, RESET/SET,
математическую калибровку, суммы и BN/LIF через закреплённые независимые helpers.
Exact runtime/config/route/evidence SHA связываются с Git source и manifest.
Все historical capacity origins проверяются по сохранённым SHA; их наличие
обязательно. Capacity screening не становится новой hardware qualification.

17 offline tests PASS: successful HW/SIM arithmetic, quiet failure,
retained RESET index и repair cap, noSET initial overshoot, запрещённые BL8
primary/fallback, подмены ledgers и CIFAR partials, exact300/budget checks.
Fixture generation использует явно FakeConnector; сам аудитор не вызывает I/O.
Actual native SIM raw audit PASS обязателен до аппаратного GO.

```shell
python audit_fresh_conv1_no_bl8.py --repo . --session results/cifar_fresh_conv1_no_bl8/SESSION_UUID --output results/cifar_feedback_reviews/no_bl8_SESSION_audit
python -m unittest tests.test_audit_fresh_conv1_no_bl8 -q
```

PASS целостности отделён от fidelity, повторяемости и точности trained outputs.
0 spike differences к собственному W_eff не доказывает точности trained weights.
Цвет карты охвата не должен означать пригодность или максимальность 256 ячеек.
