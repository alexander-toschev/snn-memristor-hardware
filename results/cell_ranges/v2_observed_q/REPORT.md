# Наблюдённые q-точки, прошедшие MVM-screen

**Только offline-анализ сохранённых данных; новых измерений и ослабления критериев нет.**

Все 256 ячеек: 10 имеют хотя бы одну passing-точку; только 3 — две или более passing SET-точки. Новая карта содержит 5 кандидатов.

Min/max описывают лишь крайние наблюдённые passing q, а не непрерывный рабочий диапазон. Baseline/postRESET не считаются SET-точками; несколько пересечённых целей одного checkpoint остаются одной точкой. Успешное состояние могло быть перезаписано.

Screen не ограничивал q общим окном .01–.05: высокие baseline q также могли пройти. Мы не меняем пороги и никого не переводим в candidate подбором окна. q здесь только из mode7 при nominal .3 В; q свежей mode9 карты при .1 В к диапазонам не добавляется.

## Все проходящие измеренные состояния

| WL,BL | checkpoint / роль | q mean | q SD (5 mode7 ответов) | ε / допуск / запас, ADC | ≥2 SET-точек | новая карта | поздний G pass |
|---|---|---:|---:|---|---|---|---|
| 0,1 | 6 / local_pre_reset | 0.13846720896807713 | 5.2000179607578576e-05 | -11 / 115.13 / 104.13 | False | candidate | False |
| 0,1 | 7 / post_reset | 0.13865489995618449 | 5.9392798334231957e-05 | -12.6 / 115.19 / 102.59 | False | candidate | False |
| 0,8 | 43 / set_crossing | 0.040177914819379233 | 1.0396668840033803e-05 | 12 / 31.06 / 19.06 | True | reject | False |
| 0,8 | 44 / set_crossing | 0.050413348811329528 | 1.8145238317869403e-05 | -24.6 / 41.51 / 16.91 | True | reject | False |
| 1,3 | 160 / local_pre_reset | 0.028914274305492153 | 1.3593930463761288e-05 | 1.8 / 23.91 / 22.11 | False | reject | False |
| 1,15 | 208 / local_pre_reset | 0.081193007939134115 | 3.5169148870289151e-05 | -3.8 / 75.25 / 71.45 | False | candidate | True |
| 1,15 | 209 / post_reset | 0.081409000142123891 | 1.6938667291205171e-05 | -10.4 / 75.3 / 64.9 | False | candidate | True |
| 3,3 | 417 / local_pre_reset | 0.022956601054914795 | 2.1041603576669541e-05 | 16.4 / 20.78 / 4.38 | False | candidate | False |
| 3,3 | 421 / terminal | 0.023144332634732354 | 1.335007675806668e-05 | 14 / 21.44 / 7.44 | False | candidate | False |
| 3,8 | 437 / set_crossing | 0.040052628159543267 | 8.4762617738267463e-06 | -0.2 / 34.95 / 34.75 | True | reject | False |
| 3,8 | 438 / set_crossing | 0.050093135126165644 | 2.1595376812694727e-05 | -38.2 / 46.09 / 7.89 | True | reject | False |
| 5,8 | 665 / set_crossing | 0.030103846911686716 | 1.3823798344125916e-05 | 27.6 / 27.9 / 0.3 | True | reject | True |
| 5,8 | 666 / set_crossing | 0.040064007439433925 | 1.0391091642662494e-05 | -11 / 40.01 / 29.01 | True | reject | True |
| 6,8 | 788 / local_pre_reset | 0.030285870606465582 | 1.3861893585565607e-05 | 0 / 36.3 / 36.3 | False | reject | False |
| 6,8 | 792 / set_crossing | 0.030219621166995586 | 7.410763942871608e-06 | -1.6 / 36.74 / 35.14 | False | reject | False |
| 7,1 | 881 / local_pre_reset | 0.20841466939931777 | 7.7018520094393971e-05 | -5.8 / 106.07 / 100.27 | False | candidate | False |
| 7,1 | 882 / post_reset | 0.20856906067173481 | 9.3894796042626923e-05 | -1.2 / 106.2 / 105 | False | candidate | False |
| 7,3 | 890 / local_pre_reset | 0.043891913927420527 | 1.6674511081156271e-05 | 18.2 / 20.69 / 2.49 | False | candidate | False |
| 7,3 | 896 / terminal | 0.04507390434984794 | 1.4322451543542943e-05 | 13.4 / 21.41 / 8.01 | False | candidate | False |

Три ячейки с двумя passing SET-точками: (0,8), (3,8), (5,8). Это приоритеты для отдельной проверки, не подтверждённые непрерывные диапазоны; все три отклонены свежей картой. У нижней SET-точки (5,8) ε=27.6 ADC при допуске27.9 — запас лишь0.3 ADC, пограничное наблюдение.

Повторяемость записи, число различимых уровней/битность, совместная пригодность пар и точность CIFAR этим не установлены. Поздний G сравнивает terminal5 с final3, а не подтверждает сохранность каждой исторической MVM-кривой.

## Все 256 адресов

| WL,BL | Категория | Passing / SET | Наблюдённые q | новая карта | поздний G pass |
|---|---|---|---|---|---|
| 0,0 | no_passing_observation | 0 / 0 | — | reject | False |
| 0,1 | passing_non_SET_observations_only | 2 / 0 | 0.13846720896807713, 0.13865489995618449 | candidate | False |
| 0,2 | no_passing_observation | 0 / 0 | — | reject | False |
| 0,3 | no_passing_observation | 0 / 0 | — | reject | False |
| 0,4 | no_passing_observation | 0 / 0 | — | reject | False |
| 0,5 | no_passing_observation | 0 / 0 | — | reject | True |
| 0,6 | no_passing_observation | 0 / 0 | — | reject | True |
| 0,7 | no_passing_observation | 0 / 0 | — | reject | True |
| 0,8 | two_or_more_passing_SET_states | 2 / 2 | 0.040177914819379233, 0.050413348811329528 | reject | False |
| 0,9 | no_passing_observation | 0 / 0 | — | reject | False |
| 0,10 | no_passing_observation | 0 / 0 | — | reject | True |
| 0,11 | no_passing_observation | 0 / 0 | — | reject | True |
| 0,12 | no_passing_observation | 0 / 0 | — | reject | False |
| 0,13 | no_passing_observation | 0 / 0 | — | reject | False |
| 0,14 | no_passing_observation | 0 / 0 | — | reject | False |
| 0,15 | no_passing_observation | 0 / 0 | — | reject | False |
| 0,16 | no_passing_observation | 0 / 0 | — | reject | False |
| 0,17 | no_passing_observation | 0 / 0 | — | reject | True |
| 0,18 | no_passing_observation | 0 / 0 | — | reject | True |
| 0,19 | no_passing_observation | 0 / 0 | — | reject | True |
| 0,20 | no_passing_observation | 0 / 0 | — | reject | False |
| 0,21 | no_passing_observation | 0 / 0 | — | reject | True |
| 0,22 | no_passing_observation | 0 / 0 | — | reject | True |
| 0,23 | no_passing_observation | 0 / 0 | — | reject | True |
| 0,24 | no_passing_observation | 0 / 0 | — | reject | False |
| 0,25 | no_passing_observation | 0 / 0 | — | reject | False |
| 0,26 | no_passing_observation | 0 / 0 | — | reject | True |
| 0,27 | no_passing_observation | 0 / 0 | — | reject | True |
| 0,28 | no_passing_observation | 0 / 0 | — | reject | True |
| 0,29 | no_passing_observation | 0 / 0 | — | reject | False |
| 0,30 | no_passing_observation | 0 / 0 | — | reject | False |
| 0,31 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,0 | no_passing_observation | 0 / 0 | — | reject | False |
| 1,1 | no_passing_observation | 0 / 0 | — | reject | False |
| 1,2 | no_passing_observation | 0 / 0 | — | reject | False |
| 1,3 | passing_non_SET_observations_only | 1 / 0 | 0.028914274305492153 | reject | False |
| 1,4 | no_passing_observation | 0 / 0 | — | reject | False |
| 1,5 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,6 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,7 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,8 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,9 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,10 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,11 | no_passing_observation | 0 / 0 | — | reject | False |
| 1,12 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,13 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,14 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,15 | passing_non_SET_observations_only | 2 / 0 | 0.081193007939134115, 0.081409000142123891 | candidate | True |
| 1,16 | no_passing_observation | 0 / 0 | — | reject | False |
| 1,17 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,18 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,19 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,20 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,21 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,22 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,23 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,24 | no_passing_observation | 0 / 0 | — | reject | False |
| 1,25 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,26 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,27 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,28 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,29 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,30 | no_passing_observation | 0 / 0 | — | reject | True |
| 1,31 | no_passing_observation | 0 / 0 | — | reject | True |
| 2,0 | no_passing_observation | 0 / 0 | — | reject | True |
| 2,1 | no_passing_observation | 0 / 0 | — | reject | True |
| 2,2 | no_passing_observation | 0 / 0 | — | reject | True |
| 2,3 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,4 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,5 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,6 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,7 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,8 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,9 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,10 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,11 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,12 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,13 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,14 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,15 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,16 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,17 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,18 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,19 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,20 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,21 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,22 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,23 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,24 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,25 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,26 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,27 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,28 | no_passing_observation | 0 / 0 | — | reject | True |
| 2,29 | no_passing_observation | 0 / 0 | — | reject | False |
| 2,30 | no_passing_observation | 0 / 0 | — | reject | True |
| 2,31 | no_passing_observation | 0 / 0 | — | reject | True |
| 3,0 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,1 | no_passing_observation | 0 / 0 | — | reject | True |
| 3,2 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,3 | passing_non_SET_observations_only | 2 / 0 | 0.022956601054914795, 0.023144332634732354 | candidate | False |
| 3,4 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,5 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,6 | no_passing_observation | 0 / 0 | — | reject | True |
| 3,7 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,8 | two_or_more_passing_SET_states | 2 / 2 | 0.040052628159543267, 0.050093135126165644 | reject | False |
| 3,9 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,10 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,11 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,12 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,13 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,14 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,15 | no_passing_observation | 0 / 0 | — | reject | True |
| 3,16 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,17 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,18 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,19 | no_passing_observation | 0 / 0 | — | reject | True |
| 3,20 | no_passing_observation | 0 / 0 | — | reject | True |
| 3,21 | no_passing_observation | 0 / 0 | — | reject | True |
| 3,22 | no_passing_observation | 0 / 0 | — | reject | True |
| 3,23 | no_passing_observation | 0 / 0 | — | reject | True |
| 3,24 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,25 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,26 | no_passing_observation | 0 / 0 | — | reject | False |
| 3,27 | no_passing_observation | 0 / 0 | — | reject | True |
| 3,28 | no_passing_observation | 0 / 0 | — | reject | True |
| 3,29 | no_passing_observation | 0 / 0 | — | reject | True |
| 3,30 | no_passing_observation | 0 / 0 | — | reject | True |
| 3,31 | no_passing_observation | 0 / 0 | — | reject | True |
| 4,0 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,1 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,2 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,3 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,4 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,5 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,6 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,7 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,8 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,9 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,10 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,11 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,12 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,13 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,14 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,15 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,16 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,17 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,18 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,19 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,20 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,21 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,22 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,23 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,24 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,25 | no_passing_observation | 0 / 0 | — | reject | False |
| 4,26 | no_passing_observation | 0 / 0 | — | reject | True |
| 4,27 | no_passing_observation | 0 / 0 | — | reject | True |
| 4,28 | no_passing_observation | 0 / 0 | — | reject | True |
| 4,29 | no_passing_observation | 0 / 0 | — | reject | True |
| 4,30 | no_passing_observation | 0 / 0 | — | reject | True |
| 4,31 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,0 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,1 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,2 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,3 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,4 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,5 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,6 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,7 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,8 | two_or_more_passing_SET_states | 2 / 2 | 0.030103846911686716, 0.040064007439433925 | reject | True |
| 5,9 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,10 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,11 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,12 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,13 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,14 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,15 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,16 | no_passing_observation | 0 / 0 | — | reject | False |
| 5,17 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,18 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,19 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,20 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,21 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,22 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,23 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,24 | no_passing_observation | 0 / 0 | — | reject | False |
| 5,25 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,26 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,27 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,28 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,29 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,30 | no_passing_observation | 0 / 0 | — | reject | True |
| 5,31 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,0 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,1 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,2 | no_passing_observation | 0 / 0 | — | reject | False |
| 6,3 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,4 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,5 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,6 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,7 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,8 | one_passing_SET_state | 2 / 1 | 0.030285870606465582, 0.030219621166995586 | reject | False |
| 6,9 | no_passing_observation | 0 / 0 | — | reject | False |
| 6,10 | no_passing_observation | 0 / 0 | — | reject | False |
| 6,11 | no_passing_observation | 0 / 0 | — | reject | False |
| 6,12 | no_passing_observation | 0 / 0 | — | reject | False |
| 6,13 | no_passing_observation | 0 / 0 | — | reject | False |
| 6,14 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,15 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,16 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,17 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,18 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,19 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,20 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,21 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,22 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,23 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,24 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,25 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,26 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,27 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,28 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,29 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,30 | no_passing_observation | 0 / 0 | — | reject | True |
| 6,31 | no_passing_observation | 0 / 0 | — | reject | True |
| 7,0 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,1 | passing_non_SET_observations_only | 2 / 0 | 0.20841466939931777, 0.20856906067173481 | candidate | False |
| 7,2 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,3 | passing_non_SET_observations_only | 2 / 0 | 0.043891913927420527, 0.04507390434984794 | candidate | False |
| 7,4 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,5 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,6 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,7 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,8 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,9 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,10 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,11 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,12 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,13 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,14 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,15 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,16 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,17 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,18 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,19 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,20 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,21 | no_passing_observation | 0 / 0 | — | reject | True |
| 7,22 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,23 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,24 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,25 | no_passing_observation | 0 / 0 | — | reject | True |
| 7,26 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,27 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,28 | no_passing_observation | 0 / 0 | — | reject | False |
| 7,29 | no_passing_observation | 0 / 0 | — | reject | True |
| 7,30 | no_passing_observation | 0 / 0 | — | reject | True |
| 7,31 | no_passing_observation | 0 / 0 | — | reject | True |

SHA256 всех семи входов закреплены в скрипте и проверены до анализа; числовые поля/ID/роли сохранены в ranges.json и cells.csv. Полный raw здесь повторно не аудирован и этим компактным набором не заменён.

Воспроизведение из корня репозитория (нужен новый выходной каталог):

```sh
python analyze_cell_ranges.py --repo . --output results/cell_ranges/v2_observed_q_recheck
```
