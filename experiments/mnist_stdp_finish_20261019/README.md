# Завершение прерванного MNIST STDP — отдельная реконструкция readout

Исходная сессия `results/mnist_stdp_retraining_20261019/7a536bf8-0269-4edd-b454-a8ba2c33c219`
не имеет final manifest. Сохранены STDP30000/T100, train-признаки30000,
train-only scaler и пять завершённых эпох readout (295 updates). Оптимизатор
и конечный head не сохранены. Начальный status=failed/network_executed=false
в start_manifest — fail-safe маркер старта, а не заключение о выполненных этапах.

Новый маршрут не изменяет эту сессию и не запускает STDP повторно.
`INPUT_SNAPSHOT.json` закрепляет все29 файлов (4,644,084,840 байт).
Read-only qualification проверяет точный inventory/SHA, все30000 матриц,
clipping отдельно от поздней пластичности,235 train batches, новые W,
threshold/bias, официальные labels и train-only scaler. Это проверка сохранённых
свидетельств, не повторное выполнение BindsNET.

После независимого review и регистрации отдельный `experiment.py --run`:

1. Создаёт новый каталог `results/mnist_stdp_finish_20261019/<uuid>`.
2. Инициализирует head заново seed42 и повторяет все25 эпох/1475 updates.
   Это **fresh head replay**, не продолжение с epoch6. Momentum0, lr.001,
   weight_decay.0001, batch512, последние304 строки сохраняются.
3. До test проверяет исходные initial head, первые5 перестановок/RNG и RNG
   перед шестой эпохой побайтово. Train loss первых5 эпох сравнивается с
   заранее фиксированными atol1e-6/rtol1e-5. Несовпадение останавливает маршрут.
4. Фиксирует head/scaler/checkpoint; восстанавливает feature network именно
   из новых STDP-весов этой сессии. Исторический model.pt не загружается.
   learning=False на сети и слоях, conn.norm=None; W/bias/threshold должны
   совпасть с frozen_state всех235 train batches.
5. Извлекает test10000/T100 в79 batches (хвост16), один раз оценивает
   финальный head25, сохраняет logits/predictions/confusion. Сверяет неизменность
   исходной incomplete session и замороженных параметров после test.

Запуск только локально, с существующей изолированной `.venv-mnist-pilot`.
Raspberry используется только для физических MemriBoard операций.
Модель, checkpoint, encoder, BindsNET и семантика broken E/I hook остаются
закреплёнными; реконструкция не обещает исторические69.65%.

```powershell
& .venv-mnist-pilot/Scripts/python.exe -B -m unittest tests.test_mnist_stdp_finish -v
& .venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_stdp_finish_20261019/experiment.py --preflight
```

`--preflight` выполняет CPU-проверку сохранённых4.6GB/скейлера и проверку
окружения, но не строит/запускает нейронную сеть и не обучает head.
Source/scientific review выполнено, SPEC зарегистрирован условно. CPU input
qualification прошла; фактический runtime preflight пока не прошёл guard8GiB
доступной RAM. Регистрация не обходит этот guard: `--run` повторяет preflight
и не начнёт обучение при нехватке памяти. Решение о запуске принимает основной
агент после успешного preflight. Отдельный `tools/audit_mnist_stdp_finish.py`
связывает обе сессии; старый full auditor не ослабляется.

Оценка по сохранённым данным: head25 примерно3–4мин, test features около8мин
плюс проверка/диск/загрузка. Это оценка, не гарантия. Новый предел2ч и3GiB
выходных файлов. Существующий строгий helper-preflight требует18GiB свободного
диска и8GiB доступной RAM; GPU лимит сохраняется из первоначальной среды.
Traincounts4.42GB используются read-only по исходной ссылке, не копируются.
Для GitLFS публикации нужны lossless chunks≤512MiB с SHA исходного массива
и manifest реконструкции; оригинал в исходной сессии не удаляется/не изменяется.
