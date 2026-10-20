# Независимый review runtime provenance v2

**GO только для локального анализа сохранённых результатов.** Пользователь
отменил обязательность повторных ручных проверок двух private-драйверов и
operator receipts. Это не подтверждение их байтов через Git: новый результат
явно сообщает `private_runtime_not_independently_reverified`.

Старые отчёты, receipts, registry и grants не переписываются. Если текущий
tracked файл отличается от исторического pin, проверяются точные bytes
из заранее известного локального Git commit
`b7232c4367096d1c29eeb4d4c0c7e40645a99142`, по разрешённому пути и SHA.
Произвольный revision из отчёта не принимается; fetch/checkout/filters не
выполняются. Отсутствие необходимого Git object остаётся явной ошибкой.

Новые producer SHA и provenance v2 зарегистрированы отдельно. Во время
review исправлена неоднозначность: новый producer обязан иметь
`provenance_generation=v2`, а старый producer не может выдавать себя за v2.
Поэтому новая форма не попадает в сокращённую ветку сравнения исторических
метаданных. Отсутствующий private receipt допускается; явно переданный
старый receipt проверяется и сохраняется как прежнее наблюдение оператора.

Физические операции не менялись. Независимо сверены 11 закреплённых
physical/helper файлов, неизменность resumable Collector/entry/durable и
численного prefix replay, SHA обоих исторических execution grants.
Изменения contract относятся к допуску версий metadata; проверки raw,
INI, command/current/ADC, бюджета, layout, payload, portable-condition и
численного результата не ослаблены. STDP использует тот же provenance
модуль; его frozen numerical replay остаётся прежним.

Независимые focused tests: **7/7 PASS за 1,250 с**, включая обе отрицательные
проверки producer/generation, неверные historical SHA, ошибки Git, public
source/INI, сохранение старого конечного плана и реальные сохранённые
stage13/grant. На этом этапе тесты явно использовали candidate registration.

После gate-only регистрации выполнена отдельная фактическая проверка
**без mock**: все 8 current source SHA совпали; для сохранённой сессии
`ac92f059-fec1-4c08-99f5-325a46b7124e` source verification без receipt
завершилась с новым честным private-статусом, historical contract проверен
через Git, старый grant подтверждён. Повторного raw replay, classifier,
модели и аппаратных вызовов в этих проверках не было.

Новые hardware grants и stage registry не созданы. Это намеренное отсутствие
допуска продолжать старую раскладку после изменения пользователем плана
на stable-cell subset. Локальная проверка уже собранных 25 stages возможна;
будущий аппаратный план требует отдельного решения. Этот review не
объявляет stages19–24 принятыми, полный CIFAR завершённым или качество
распознавания измеренным.
