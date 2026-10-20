# CIFAR seed0: native SIM programming/readback hybrid

Все 4800 signed Conv1 коэффициентов прошли public MemriBoard Connector
programming/readback. Независимый raw replay **PASS**. После замены полного
Conv1 измеренными effective weights:

| Метрика, test1000 | Clean PyTorch | Native readback → hybrid Torch |
|---|---:|---:|
| Conventional FS | 72.5% | **16.5%** |
| Conventional TWD | 76.4% | **15.0%** |
| Strict FS (silent/tied — ошибка) | 71.9% | **16.0%** |
| Strict TWD | 76.0% | **14.6%** |

Это **не literal native MVM и не плата**: native simulator измерял запись
и чтение R, затем восстановленный Conv1 и остальная сеть исполнялись в Torch.
Нет подстановки ideal weights для пропущенных/плохих коэффициентов.

Предварительная characterization:1304 calls/256 pulses,8 фиксированных
ячеек,2 цикла. Общий диапазон L25.972–H153.990µS зафиксирован до test scores
по median RESET/SET. Все weights использовали один finite approximate
протокол, без retries/retuning после accuracy.

Основная сессия `868268a6-9277-4809-863f-86420d58ef87`:
**735855 calls,135651 pulses,48/48 tiles**,370.61с.
Final readback снят после всех записей каждого тайла.
Weight RMSE0.090365; RMSE/std(ideal)=1.6260; знак совпал у64.65%.
У95.52% legs finalG отличается от target более чем на1% mapping span.
Эти результаты показывают плохую точность данного bounded programming
рецепта; сам PASS replay подтверждает целостность исполнения, не качество.
Без дополнительных сравнений причины конкретных ошибок не изолированы.

Hybrid eval занял39.89с. Полные predictions/spikes и метрики:
[seed0/HYBRID.json](seed0/HYBRID.json);
[независимый replay](seed0/AUDIT.json).
Raw/arrays сохранены exact-byte архивом51 012 612B,
decoded raw792 446 766B; обе secret scans и byte verification PASS.
Characterization сохранена отдельно с собственным архивом/replay.

Seeds1/2/42 native programming **не запускались**: пользователь переключил
приоритет на расширенный MNIST. Их clean результаты сохранены отдельно.
Отрицательный seed0 не перезапускался; mapping не подстраивался по test.
