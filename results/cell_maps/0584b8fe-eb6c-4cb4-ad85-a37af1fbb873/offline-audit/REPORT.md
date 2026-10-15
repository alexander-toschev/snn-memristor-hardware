# Независимый аудит карты ячеек

Сессия `0584b8fe-eb6c-4cb4-ad85-a37af1fbb873`: сбор `completed`, аудит **passed** — 523814 проверок, 0 расхождений.

Проверены 4136 пар request/result: mode9 1536, mode10 2600, mode7 0; явных пропусков MVM 0. Пересчитаны native conversions, охват/порядок, DAC, ID, временные отметки, модельные guards, классификация всех 256 ячеек и CSV маска. Проверены result SHA и 64 исходников по локальным байтам и Git `34b8819bf3d501f337b4a0ad401204cb8c5397cd`; приватные файлы — согласованные SHA и локальные байты при наличии, без Git.

Классификация: candidate 5, reject 251, unmeasured 0. Маска пригодна к рассмотрению: **True**. Частичный failed-прогон не экспортирует рабочую маску.

| Причина | Число ячеек |
|---|---:|
| nonlinear_zero_adjusted_response | 99 |
| unstable_high_signal | 154 |
| weak_or_nonpositive_signal_high | 152 |
| weak_or_nonpositive_signal_low | 170 |

Причины пересекаются: одна ячейка может не пройти несколько критериев.

Название unstable_high_signal не означает, что все эти ячейки шумные: у 152 CV не определён из-за неположительного high-сигнала, только у 2 измеренный CV превышает порог. Оба SNR-порога прошли 86 ячейки; из них 81 исключены только по двухуровневой линейности.

## Предварительные кандидаты

| WL, BL | Сигнал .05 / .1 В, ADC | SNR low / high | CV high | ε / допуск, ADC | ΔG / допуск, % |
|---|---:|---:|---:|---:|---:|
| 0, 1 | 1160.000 / 2306.200 | 266.12 / 674.22 | 0.0015 | -13.800 / 115.310 | -0.117 / 5.000 |
| 1, 15 | 757.400 / 1504.000 | 499.41 / 303.85 | 0.0033 | -10.800 / 75.200 | -0.150 / 5.000 |
| 3, 3 | 208.000 / 426.800 | 104.00 / 426.80 | 0.0020 | +10.800 / 21.340 | -0.098 / 5.000 |
| 7, 1 | 1070.800 / 2126.600 | 343.81 / 511.28 | 0.0020 | -15.000 / 106.330 | +0.000 / 5.000 |
| 7, 3 | 213.200 / 432.200 | 163.52 / 199.36 | 0.0050 | +5.800 / 21.610 | -0.100 / 5.000 |

## Ранее использованные BL2/BL3

Здесь показаны новые одиночные измерения карты, а не повтор прежнего AB-теста. В таблице отсутствует проверка суммирования.

| WL, BL | Сигнал .05 / .1 В, ADC | ε / допуск, ADC | Причины |
|---|---:|---:|---|
| 0, 2 | 65.000 / 277.800 | +147.800 / 13.890 | nonlinear_zero_adjusted_response |
| 0, 3 | 120.400 / 283.600 | +42.800 / 14.180 | nonlinear_zero_adjusted_response |
| 1, 2 | 55.200 / 262.600 | +152.200 / 13.130 | nonlinear_zero_adjusted_response |
| 1, 3 | 173.200 / 373.000 | +26.600 / 18.650 | nonlinear_zero_adjusted_response |
| 2, 2 | 0.000 / 0.000 | +0.000 / 2.000 | weak_or_nonpositive_signal_low; weak_or_nonpositive_signal_high; unstable_high_signal |
| 2, 3 | 0.000 / 0.000 | +0.000 / 2.000 | weak_or_nonpositive_signal_low; weak_or_nonpositive_signal_high; unstable_high_signal |
| 3, 2 | 118.600 / 386.400 | +149.200 / 19.320 | nonlinear_zero_adjusted_response |
| 3, 3 | 208.000 / 426.800 | +10.800 / 21.340 | candidate |
| 4, 2 | 55.000 / 268.800 | +158.800 / 13.440 | nonlinear_zero_adjusted_response |
| 4, 3 | 0.000 / 0.000 | +0.000 / 2.000 | weak_or_nonpositive_signal_low; weak_or_nonpositive_signal_high; unstable_high_signal |
| 5, 2 | 0.000 / 0.000 | +0.000 / 2.000 | weak_or_nonpositive_signal_low; weak_or_nonpositive_signal_high; unstable_high_signal |
| 5, 3 | 0.000 / 0.000 | +0.000 / 2.000 | weak_or_nonpositive_signal_low; weak_or_nonpositive_signal_high; unstable_high_signal |
| 6, 2 | 0.000 / 0.000 | +0.000 / 2.000 | weak_or_nonpositive_signal_low; weak_or_nonpositive_signal_high; unstable_high_signal |
| 6, 3 | 155.000 / 343.600 | +33.600 / 17.180 | nonlinear_zero_adjusted_response |
| 7, 2 | 72.000 / 292.200 | +148.200 / 14.610 | nonlinear_zero_adjusted_response |
| 7, 3 | 213.200 / 432.200 | +5.800 / 21.610 | candidate |

Координаты WL/BL — логические адреса Connector; соответствие физическим ячейкам независимо не калибровалось. Candidate означает предварительный отклик текущего состояния при 0.05/0.1 В по критериям criteria.json. Reject не доказывает физическую неисправность. Нулевой ADC может быть отсечкой. Программируемость, выбранные пары, суммирование комбинаций и CIFAR accuracy не проверены; после записи весов карта требует обновления. Пороги инженерные, не валидированные научные нормы.

Аудитор не создаёт Connector и не вызывает аппаратуру. Исходные файлы сессии не изменяются. Воспроизведение после сохранения скрипта в корне репозитория:

```sh
python -B audit_cell_map.py --source-repo . --session results/cell_maps/0584b8fe-eb6c-4cb4-ad85-a37af1fbb873 --output-dir results/cell_maps/0584b8fe-eb6c-4cb4-ad85-a37af1fbb873
```
