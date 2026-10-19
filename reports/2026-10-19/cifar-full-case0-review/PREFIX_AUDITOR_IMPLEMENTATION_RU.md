# Отдельный аудит прерванного CIFAR: границы допуска

Новый путь `tools/audit_cifar_interrupted_prefix.py` не изменяет семь frozen
исходников исходного CIFAR и не подставляет footer старому полному аудитору.
`tools/cifar_prefix_replay.py` — отдельная статическая производная независимого
replay. Сессия платы и её повреждённый хвост остаются неизменными.

## Что проверяется

1. SHA/размер всех исходных gzip-частей; полное декодирование, CRC, завершение
   gzip и JSONL. Принимается только непрерывная последовательность закрытых
   частей. Повреждённая часть целиком исключается из допуска, а не «ремонтируется».
2. Prefix заканчивается на настоящем `stage_complete` внутри проверенных
   частей. Повторяются command/result, native R/ADC, guard, программирование,
   basis/fit/selection/heldout, C, позиции и final_basis. Каждый stage имеет
   ровно один basis_complete и heldout_complete. EOF внутри stage недопустим.
3. Результат каждого stage — отдельные raw/C float32 массивы
   `[12,4,28,28]`, calibration SHA и source/session/raw provenance. Для smoke
   форма `[12,4,1,1]`; это явно не полный tensor и не вход классификатора.
4. Проверенные counts прерванного prefix — нижняя граница числа операций;
   неизвестный буферизованный/повреждённый хвост не оценён. Старый final_basis
   относится к окну собственного stage и не характеризует текущую плату.

## Два явных ограничения локального переноса

По умолчанию требуется наличие и совпадение всех локальных исходных байтов.
Опция `--private-source-receipt` разрешает только два заранее известных файла
MemriCORE с фиксированными SHA и размерами. Проверяются session identity,
полный root-observed receipt и SHA/согласованность наблюдаемого stdout.
Это **remote operator verification**, а не локальная проверка приватных байтов
или криптографически подписанная аттестация. Ограничения receipt сохраняются
в результате; другой отсутствующий файл не получает исключения.

Опция `--portable-condition` задаёт отдельную reviewed политику
`excluded-rank-condition-v1`. Разрешено отличие только диагностического
condition при совпадающих ranks, rank(X)<8 или rank(Y)<8 и отсутствии full-C
среди кандидатов на обеих платформах. Оба condition должны быть >10⁶ либо None.
Все остальные поля, C, selection и regularized condition проверяются с прежними
atol=rtol=10⁻¹⁰. Для X/Y сохраняются singular values и NumPy rank tolerance;
значение между tol/16 и 16tol требует review вместо допуска.
Результат отдельно сохраняет `strict_numeric_replay=FAIL_condition_only` и
decision equivalence. Это не PASS старого frozen full-аудитора.

## Проверки перед обработкой actual raw

Восемь offline-тестов прошли за18,54с: настоящий fake-Connector smoke/prefix,
неполный stage, пропущенный native result, пропущенный final_basis, дубликаты
completions, неверный range, CRC/truncation/gap, изменение bytes после inventory,
матрица/condition/rank-margin и private-receipt substitutions. Это модульные
проверки, не native SIM и не аппаратный опыт.

Проверка actual start_manifest отдельно подтвердила74 записанных source pins:
72 локально и2 по root-observed private receipt; записанные INI/protocol совпали.
Streaming actual replay запущен как диагностический анализ. Этот документ
не утверждает, что какой-либо actual stage уже прошёл семантический аудит,
не разрешает переиспользование данных и не даёт аппаратный launch GO.

Completed-stage продолжение имеет отдельный source registry после root review,
проверку final manifest/result inventory, finalized raw index и literal range.
Для аппаратного продолжения дополнительно нужны native SIM interruption/resume
qualification, независимый review и запуск основным агентом через MemriBoard.
