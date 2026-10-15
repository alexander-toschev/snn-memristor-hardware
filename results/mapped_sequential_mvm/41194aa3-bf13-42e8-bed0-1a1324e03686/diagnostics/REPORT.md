# Диагностика последовательного MVM

Источник: simulator; сессия `41194aa3-bf13-42e8-bed0-1a1324e03686`; acquisition `completed`.
Одиночные отклики: 192/256; разности: 10/47.
Статусы тактов: {'inconclusive': 139, 'pass': 5}; LIF с eligibility: {'inconclusive': 24}.
Отдельные численные статусы LIF: {'pass': 7, 'inconclusive': 17}.

Новые измерения и переоценка порогов не выполнялись. Compact-файлы сверены с manifest; это не повторный независимый аудит raw.

## Неподтверждённые коэффициенты

| WL-пара / индекс | Физические + / − | Причины |
|---|---|---|
| 0 / 0 | [0, 0] / [1, 0] | negative_cell WL1 BL0 — basis: signal=1 < 5×max(1,SD)=5 ADC; negative_cell WL1 BL0 — selection: signal=1 < 5×max(1,SD)=5 ADC; negative_cell WL1 BL0 — final: signal=1 < 5×max(1,SD)=5 ADC |
| 0 / 1 | [0, 1] / [1, 2] | basis: absDabs=3 < 5×max(1,SDbound)=5 ADC; selection: absDabs=3 < 5×max(1,SDbound)=5 ADC; final: absDabs=3 < 5×max(1,SDbound)=5 ADC |
| 0 / 3 | [0, 3] / [1, 4] | positive_cell WL0 BL3 — basis: signal=3 < 5×max(1,SD)=5 ADC; positive_cell WL0 BL3 — selection: signal=3 < 5×max(1,SD)=5 ADC; positive_cell WL0 BL3 — final: signal=3 < 5×max(1,SD)=5 ADC; negative_cell WL1 BL4 — basis: signal=3 < 5×max(1,SD)=5 ADC; negative_cell WL1 BL4 — selection: signal=3 < 5×max(1,SD)=5 ADC; negative_cell WL1 BL4 — final: signal=3 < 5×max(1,SD)=5 ADC; basis: absDabs=0 < 5×max(1,SDbound)=5 ADC; selection: absDabs=0 < 5×max(1,SDbound)=5 ADC; final: absDabs=0 < 5×max(1,SDbound)=5 ADC |
| 0 / 4 | [0, 4] / [1, 6] | basis: absDabs=2 < 5×max(1,SDbound)=5 ADC; selection: absDabs=2 < 5×max(1,SDbound)=5 ADC; final: absDabs=2 < 5×max(1,SDbound)=5 ADC |
| 0 / 5 | [0, 5] / [1, 7] | negative_cell WL1 BL7 — basis: signal=3 < 5×max(1,SD)=5 ADC; negative_cell WL1 BL7 — selection: signal=3 < 5×max(1,SD)=5 ADC; negative_cell WL1 BL7 — final: signal=3 < 5×max(1,SD)=5 ADC |
| 0 / 9 | [0, 9] / [1, 9] | negative_cell WL1 BL9 — basis: signal=2 < 5×max(1,SD)=5 ADC; negative_cell WL1 BL9 — selection: signal=2 < 5×max(1,SD)=5 ADC; negative_cell WL1 BL9 — final: signal=2 < 5×max(1,SD)=5 ADC |
| 0 / 10 | [0, 10] / [1, 12] | negative_cell WL1 BL12 — basis: signal=1 < 5×max(1,SD)=5 ADC; negative_cell WL1 BL12 — selection: signal=1 < 5×max(1,SD)=5 ADC; negative_cell WL1 BL12 — final: signal=1 < 5×max(1,SD)=5 ADC |
| 0 / 11 | [0, 11] / [1, 14] | positive_cell WL0 BL11 — basis: signal=2 < 5×max(1,SD)=5 ADC; positive_cell WL0 BL11 — selection: signal=2 < 5×max(1,SD)=5 ADC; positive_cell WL0 BL11 — final: signal=2 < 5×max(1,SD)=5 ADC |
| 0 / 12 | [0, 13] / [1, 31] | basis: absDabs=0 < 5×max(1,SDbound)=5 ADC; selection: absDabs=0 < 5×max(1,SDbound)=5 ADC; final: absDabs=0 < 5×max(1,SDbound)=5 ADC |
| 0 / 13 | [0, 31] / [1, 10] | basis: absDabs=3 < 5×max(1,SDbound)=5 ADC; selection: absDabs=3 < 5×max(1,SDbound)=5 ADC; final: absDabs=3 < 5×max(1,SDbound)=5 ADC |
| 1 / 0 | [2, 0] / [3, 2] | basis: absDabs=4 < 5×max(1,SDbound)=5 ADC; selection: absDabs=4 < 5×max(1,SDbound)=5 ADC; final: absDabs=4 < 5×max(1,SDbound)=5 ADC |
| 1 / 1 | [2, 5] / [3, 3] | basis: absDabs=0 < 5×max(1,SDbound)=5 ADC; selection: absDabs=0 < 5×max(1,SDbound)=5 ADC; final: absDabs=0 < 5×max(1,SDbound)=5 ADC |
| 1 / 2 | [2, 6] / [3, 5] | basis: absDabs=2 < 5×max(1,SDbound)=5 ADC; selection: absDabs=2 < 5×max(1,SDbound)=5 ADC; final: absDabs=2 < 5×max(1,SDbound)=5 ADC |
| 1 / 3 | [2, 9] / [3, 7] | positive_cell WL2 BL9 — basis: signal=3 < 5×max(1,SD)=5 ADC; positive_cell WL2 BL9 — selection: signal=3 < 5×max(1,SD)=5 ADC; positive_cell WL2 BL9 — final: signal=3 < 5×max(1,SD)=5 ADC |
| 1 / 5 | [2, 12] / [3, 10] | negative_cell WL3 BL10 — basis: signal=2 < 5×max(1,SD)=5 ADC; negative_cell WL3 BL10 — selection: signal=2 < 5×max(1,SD)=5 ADC; negative_cell WL3 BL10 — final: signal=2 < 5×max(1,SD)=5 ADC |
| 1 / 6 | [2, 14] / [3, 11] | positive_cell WL2 BL14 — basis: signal=1 < 5×max(1,SD)=5 ADC; positive_cell WL2 BL14 — selection: signal=1 < 5×max(1,SD)=5 ADC; positive_cell WL2 BL14 — final: signal=1 < 5×max(1,SD)=5 ADC; negative_cell WL3 BL11 — basis: signal=3 < 5×max(1,SD)=5 ADC; negative_cell WL3 BL11 — selection: signal=3 < 5×max(1,SD)=5 ADC; negative_cell WL3 BL11 — final: signal=3 < 5×max(1,SD)=5 ADC; basis: absDabs=2 < 5×max(1,SDbound)=5 ADC; selection: absDabs=2 < 5×max(1,SDbound)=5 ADC; final: absDabs=2 < 5×max(1,SDbound)=5 ADC |
| 1 / 7 | [2, 15] / [3, 15] | positive_cell WL2 BL15 — basis: signal=2 < 5×max(1,SD)=5 ADC; positive_cell WL2 BL15 — selection: signal=2 < 5×max(1,SD)=5 ADC; positive_cell WL2 BL15 — final: signal=2 < 5×max(1,SD)=5 ADC |
| 1 / 8 | [2, 31] / [3, 9] | basis: absDabs=0 < 5×max(1,SDbound)=5 ADC; selection: absDabs=0 < 5×max(1,SDbound)=5 ADC; final: absDabs=0 < 5×max(1,SDbound)=5 ADC |
| 2 / 0 | [4, 0] / [5, 0] | negative_cell WL5 BL0 — basis: signal=2 < 5×max(1,SD)=5 ADC; negative_cell WL5 BL0 — selection: signal=2 < 5×max(1,SD)=5 ADC; negative_cell WL5 BL0 — final: signal=2 < 5×max(1,SD)=5 ADC |
| 2 / 1 | [4, 1] / [5, 9] | negative_cell WL5 BL9 — basis: signal=1 < 5×max(1,SD)=5 ADC; negative_cell WL5 BL9 — selection: signal=1 < 5×max(1,SD)=5 ADC; negative_cell WL5 BL9 — final: signal=1 < 5×max(1,SD)=5 ADC |
| 2 / 2 | [4, 2] / [5, 7] | positive_cell WL4 BL2 — basis: signal=2 < 5×max(1,SD)=5 ADC; positive_cell WL4 BL2 — selection: signal=2 < 5×max(1,SD)=5 ADC; positive_cell WL4 BL2 — final: signal=2 < 5×max(1,SD)=5 ADC |
| 2 / 3 | [4, 4] / [5, 5] | positive_cell WL4 BL4 — basis: signal=2 < 5×max(1,SD)=5 ADC; positive_cell WL4 BL4 — selection: signal=2 < 5×max(1,SD)=5 ADC; positive_cell WL4 BL4 — final: signal=2 < 5×max(1,SD)=5 ADC; basis: absDabs=3 < 5×max(1,SDbound)=5 ADC; selection: absDabs=3 < 5×max(1,SDbound)=5 ADC; final: absDabs=3 < 5×max(1,SDbound)=5 ADC |
| 2 / 4 | [4, 6] / [5, 10] | basis: absDabs=4 < 5×max(1,SDbound)=5 ADC; selection: absDabs=4 < 5×max(1,SDbound)=5 ADC; final: absDabs=4 < 5×max(1,SDbound)=5 ADC |
| 2 / 5 | [4, 10] / [5, 1] | positive_cell WL4 BL10 — basis: signal=2 < 5×max(1,SD)=5 ADC; positive_cell WL4 BL10 — selection: signal=2 < 5×max(1,SD)=5 ADC; positive_cell WL4 BL10 — final: signal=2 < 5×max(1,SD)=5 ADC |
| 2 / 6 | [4, 11] / [5, 12] | positive_cell WL4 BL11 — basis: signal=1 < 5×max(1,SD)=5 ADC; positive_cell WL4 BL11 — selection: signal=1 < 5×max(1,SD)=5 ADC; positive_cell WL4 BL11 — final: signal=1 < 5×max(1,SD)=5 ADC; negative_cell WL5 BL12 — basis: signal=2 < 5×max(1,SD)=5 ADC; negative_cell WL5 BL12 — selection: signal=2 < 5×max(1,SD)=5 ADC; negative_cell WL5 BL12 — final: signal=2 < 5×max(1,SD)=5 ADC; basis: absDabs=1 < 5×max(1,SDbound)=5 ADC; selection: absDabs=1 < 5×max(1,SDbound)=5 ADC; final: absDabs=1 < 5×max(1,SDbound)=5 ADC |
| 2 / 7 | [4, 13] / [5, 4] | positive_cell WL4 BL13 — basis: signal=2 < 5×max(1,SD)=5 ADC; positive_cell WL4 BL13 — selection: signal=2 < 5×max(1,SD)=5 ADC; positive_cell WL4 BL13 — final: signal=2 < 5×max(1,SD)=5 ADC |
| 2 / 9 | [4, 15] / [5, 31] | basis: absDabs=1 < 5×max(1,SDbound)=5 ADC; selection: absDabs=1 < 5×max(1,SDbound)=5 ADC; final: absDabs=1 < 5×max(1,SDbound)=5 ADC |
| 2 / 10 | [4, 31] / [5, 14] | negative_cell WL5 BL14 — basis: signal=2 < 5×max(1,SD)=5 ADC; negative_cell WL5 BL14 — selection: signal=2 < 5×max(1,SD)=5 ADC; negative_cell WL5 BL14 — final: signal=2 < 5×max(1,SD)=5 ADC |
| 3 / 0 | [6, 0] / [7, 0] | basis: absDabs=2 < 5×max(1,SDbound)=5 ADC; selection: absDabs=2 < 5×max(1,SDbound)=5 ADC; final: absDabs=2 < 5×max(1,SDbound)=5 ADC |
| 3 / 2 | [6, 4] / [7, 5] | basis: absDabs=0 < 5×max(1,SDbound)=5 ADC; selection: absDabs=0 < 5×max(1,SDbound)=5 ADC; final: absDabs=0 < 5×max(1,SDbound)=5 ADC |
| 3 / 3 | [6, 5] / [7, 7] | negative_cell WL7 BL7 — basis: signal=1 < 5×max(1,SD)=5 ADC; negative_cell WL7 BL7 — selection: signal=1 < 5×max(1,SD)=5 ADC; negative_cell WL7 BL7 — final: signal=1 < 5×max(1,SD)=5 ADC |
| 3 / 4 | [6, 6] / [7, 3] | positive_cell WL6 BL6 — basis: signal=2 < 5×max(1,SD)=5 ADC; positive_cell WL6 BL6 — selection: signal=2 < 5×max(1,SD)=5 ADC; positive_cell WL6 BL6 — final: signal=2 < 5×max(1,SD)=5 ADC; negative_cell WL7 BL3 — basis: signal=1 < 5×max(1,SD)=5 ADC; negative_cell WL7 BL3 — selection: signal=1 < 5×max(1,SD)=5 ADC; negative_cell WL7 BL3 — final: signal=1 < 5×max(1,SD)=5 ADC; basis: absDabs=1 < 5×max(1,SDbound)=5 ADC; selection: absDabs=1 < 5×max(1,SDbound)=5 ADC; final: absDabs=1 < 5×max(1,SDbound)=5 ADC |
| 3 / 6 | [6, 8] / [7, 2] | negative_cell WL7 BL2 — basis: signal=3 < 5×max(1,SD)=5 ADC; negative_cell WL7 BL2 — selection: signal=3 < 5×max(1,SD)=5 ADC; negative_cell WL7 BL2 — final: signal=3 < 5×max(1,SD)=5 ADC |
| 3 / 8 | [6, 10] / [7, 10] | positive_cell WL6 BL10 — basis: signal=3 < 5×max(1,SD)=5 ADC; positive_cell WL6 BL10 — selection: signal=3 < 5×max(1,SD)=5 ADC; positive_cell WL6 BL10 — final: signal=3 < 5×max(1,SD)=5 ADC |
| 3 / 9 | [6, 12] / [7, 6] | positive_cell WL6 BL12 — basis: signal=3 < 5×max(1,SD)=5 ADC; positive_cell WL6 BL12 — selection: signal=3 < 5×max(1,SD)=5 ADC; positive_cell WL6 BL12 — final: signal=3 < 5×max(1,SD)=5 ADC; negative_cell WL7 BL6 — basis: signal=3 < 5×max(1,SD)=5 ADC; negative_cell WL7 BL6 — selection: signal=3 < 5×max(1,SD)=5 ADC; negative_cell WL7 BL6 — final: signal=3 < 5×max(1,SD)=5 ADC; basis: absDabs=0 < 5×max(1,SDbound)=5 ADC; selection: absDabs=0 < 5×max(1,SDbound)=5 ADC; final: absDabs=0 < 5×max(1,SDbound)=5 ADC |
| 3 / 10 | [6, 13] / [7, 31] | basis: absDabs=4 < 5×max(1,SDbound)=5 ADC; selection: absDabs=4 < 5×max(1,SDbound)=5 ADC; final: absDabs=4 < 5×max(1,SDbound)=5 ADC |
| 3 / 12 | [6, 31] / [7, 9] | basis: absDabs=4 < 5×max(1,SDbound)=5 ADC; selection: absDabs=4 < 5×max(1,SDbound)=5 ADC; final: absDabs=4 < 5×max(1,SDbound)=5 ADC |

## Провалившиеся такты

| Пара / шаблон / t | Причины |
|---|---|

## LIF: численный результат отдельно от допуска

| Пара / шаблон / знак | Численно | Итог | Ограничения |
|---|---|---|---|
| 0 / dispersed / 1 | pass | inconclusive | Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 0 / dispersed / -1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 0 / grouped / 1 | pass | inconclusive | Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 0 / grouped / -1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 0 / dense_all / 1 | pass | inconclusive | Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 0 / dense_all / -1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 1 / dispersed / 1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 1 / dispersed / -1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 1 / grouped / 1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 1 / grouped / -1 | pass | inconclusive | Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 1 / dense_all / 1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 1 / dense_all / -1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 2 / dispersed / 1 | pass | inconclusive | Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 2 / dispersed / -1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 2 / grouped / 1 | pass | inconclusive | Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 2 / grouped / -1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 2 / dense_all / 1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 2 / dense_all / -1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 3 / dispersed / 1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 3 / dispersed / -1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 3 / grouped / 1 | pass | inconclusive | Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 3 / grouped / -1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 3 / dense_all / 1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |
| 3 / dense_all / -1 | inconclusive | inconclusive | Эталон не содержит и спайк, и тихий такт; Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G; Не все row/signed такты подтвердили среднее и повторяемость |

## Предложение будущего минимального подтверждения

Ниже — post-hoc подмножество текущих разностей, а не новая квалификация. Перед следующим опытом маршруты фиксируются, затем заново измеряются basis/selection и final. Не менять масштаб по неудачным final-отсчётам.

| WL-пара | Старые индексы | Позиций / ячеек | Знаки + / − |
|---|---|---|---|
| 0/1 | [2, 6, 7, 8] | 4 / 8 | 2 / 2 |
| 2/3 | [4] | 1 / 2 | 1 / 0 |
| 4/5 | [8] | 1 / 2 | 1 / 0 |
| 6/7 | [1, 5, 7, 11] | 4 / 8 | 1 / 3 |

Новый опыт не запускается этим отчётом. Стабильные одиночные отсчёты не гарантируют плотные суммы: шум, общий baseline и компенсация знаков проверяются новыми измерениями каждого вклада. Никаких выводов о физической причине неисправности или готовности CIFAR.
