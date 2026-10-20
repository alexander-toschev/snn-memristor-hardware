# Аппаратный диагональный MNIST128

Сеанс `a4327210-2cc4-4c19-bb7b-b8820d2e84e4`: `completed`; независимый raw replay **PASS**.
Gain WL0–7: `[20.0, 80.0, 400.0, 10.0, 20.0, 80.0, 400.0, 10.0]`. 107533 воздействий/2361 импульсов; training images=128.

|Фаза|Изображений HW|Спайки HW learning/control|Спайки SIM learning/control|
|---|---:|---:|---:|
|preprobe|16|52/87|81/74|
|train|128|372/683|446/584|
|postprobe|16|34/87|46/74|
|retentionprobe|16|34/87|46/74|

|Запись HW|Импульсов|Ожидаемый ΔG|Противоположный|Точно нулевой|
|---|---:|---:|---:|---:|
|RESET|195|103|75|17|
|SET|118|90|18|10|

Training обновлял 78 разных адресов; eligible128. Image-contributing: 232/256; без input events BL[0, 16, 24]. Checkpoints/preparation coverage не равно обучению всех256.
Retention post→retention: ненулевой ΔG learning/control 124/121; max|ΔG|=2.98641e-07/2.74384e-07S.

Метрики learning/control разделены; gain выбран только freshfit. Точно нулевой ΔG определяется точным равенством, знак ожидается по рецепту SET/RESET; это не оценка полезности обучения. Диагонали HW/SIM могут различаться. Уменьшение/рост числа спайков не является accuracy. Этот опыт не содержит classifier/readout и не full10k.

Знаки ΔG основаны на одиночных pre/post native-R отсчётах: противоположный знак не отделён от шума/дрейфа отдельной оценкой неопределённости. Control также имеет небольшие ненулевые изменения G; неизменными остались именно суммарные probe-спайки.

Воспроизведение анализа в новом каталоге отчётов (существующие отчёты не перезаписываются):

```sh
python -B tools/analyze_mnist_diagonal.py --session results/mnist_diagonal_20261020/a4327210-2cc4-4c19-bb7b-b8820d2e84e4 --output-dir reports/2026-10-20/mnist-diagonal-hardware-replay
```
