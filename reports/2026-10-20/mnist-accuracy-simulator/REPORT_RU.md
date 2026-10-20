# MNIST100: native SIM, замороженное состояние после STDP

Независимый replay **PASS**. Сессия `17ff1c2c-d2d7-4629-9034-db5d3c681bc5`
загрузила копию final array родителя `ee0db984-cd00-4052-95e6-069e71c720eb`.
Получены заново 128 official train и 100 official test: **56 336 физических
вызовов модели, 0 SET/RESET**, 62.70 с acquisition. Это native simulator,
не аппаратная accuracy.

| Readout | Основная strict accuracy | Conventional argmax | Silent | Tied | Coverage |
|---|---:|---:|---:|---:|---:|
| WL0..3 после STDP | **17/100 = 17%** | 21% | 10 | 0 | 90% |
| Парный контроль WL4..7 | 12/100 = 12% | 17% | 9 | 0 | 91% |

Silent и tied считаются abstain/ошибкой; из знаменателя 100 они не исключаются.
Wilson95 strict: learning 10.89–25.55%, control 7.00–19.81%.
Набор — заранее выбранный официальный test prefix0..99; это пилот,
не оценка на всех 10 000 и не доказательство причинного выигрыша STDP.
Банки содержат разные физические ячейки, контроль не рандомизирован.

Локальные readout фиксированы до test labels: train-only стандартизация,
ridge λ=1, one-hot10, bias без штрафа; 4×32 spike raster признаков.
Gain `[200,200,200,100]` повторён для контроля, взят из родителя без нового fit.
Train/test спайки: learning 356/302, control 584/453.
Во всех 256 ячейках parent→start и start→end G совпали в сохранённых native
SIM отсчётах. Это не предсказание аппаратной стабильности.

Полные scores, predictions, confusion и метрики: [ACCURACY.json](ACCURACY.json).
Проверки сырого потока и состояния: [AUDIT.json](AUDIT.json).
Readout и измеренные feature arrays сохранены рядом. Архив всей acquisition
с проверкой точных байтов и двумя secret scans:
`results/mnist_accuracy_20261020/archives/17ff1c2c-d2d7-4629-9034-db5d3c681bc5.tar.gz`
(7 293 884 байта).

Воспроизведение анализа из корня репозитория:

```console
python -B experiments/mnist_accuracy_20261020/analyze.py --session results/mnist_accuracy_20261020/17ff1c2c-d2d7-4629-9034-db5d3c681bc5 --output-dir reports/mnist-accuracy-simulator-replay
```
