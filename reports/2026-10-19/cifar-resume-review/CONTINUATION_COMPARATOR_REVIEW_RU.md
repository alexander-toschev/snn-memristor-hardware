# Review сравнения исходного SIM и полного hardware continuation

**Инструмент одобрен для будущих завершённых данных. Фактическое полное
двустороннее сравнение пока не выполнено.** Точные SHA кода, тестов и инструкции
закреплены в [CONTINUATION_COMPARATOR_SOURCE_FREEZE.json](CONTINUATION_COMPARATOR_SOURCE_FREEZE.json).
Независимо выполнены9/9 тестов (4,769с), без native SIM, платы или модели.

Проверены разные схемы original fullSIM и hardware composite: они не маскируются
под одну legacy-сессию. Адаптер требует48 уникальных стадий,4800 правильно
размещённых коэффициентов, один case/checkpoint/protocol, все heldout/retention
метрики. Оба tensor payload побайтово собираются из48 stage-exports и сравниваются
по SHA с classifier payload. Missing/duplicate/nonfinite вклад отвергается;
CPU/нулевой подстановки или выбора «лучшей» попытки нет.

Source/manifest/registry/grant и explicit private receipts проверяются отдельными
reviewed read-only функциями. Два найденных пробела исправлены: session ID
сертификата привязан к report/path/start identity; численные diagnostics должны
покрывать все стадии в порядке и согласовываться с общим статусом. Для
condition-only policy проверяется domain, стабильность margin16 и неизменный
допуск остальных полей. Здесь проверяется согласованность сохранённых diagnostics,
а не повторяется SVD или семантический raw replay.

Дополнительно на фактических сохранённых данных проверены original fullSIM
classifier JSON/manifest и73 source bytes, а также аппаратный префикс0–12
с его sidecars/remote private receipt. Крупные raw не перечитывались.
Синтетический полный filesystem test подменяет sidecar binding явно; реальные
проверки выше отдельно подтверждают доступные положительные маршруты. Будущие
аппаратные стадии должны всё равно пройти свой независимый audit.

CPU controls сравниваются в фиксированном abs/rel1e-5, spikes — точно;
расхождение даёт `review_required`. Silent first-spike/TWD и численные ничьи
сохраняют механический argmax, но не считаются успешным распознаванием.
Проверка фактического старого SIM подтвердила `valid_correct=null` для silent
calibrated first-spike. TWD scores и нейросетевые выходы доверяются pinned
classifier report: этот адаптер не исполняет alpha/head/BN/LIF заново.

`consistent_reports` означает согласованность свидетельств, не точность платы.
Один case не даёт test accuracy. SIM gain.015 и HW.125 различаются; старый fullSIM
использовал непрерывный массив, HW composite объединяет старый префикс с новыми
Connector/programming/calibration-сессиями. Recovery SIM с новым массивом на
тайл не является новым full48 baseline. Эти ограничения сохранены в output.

Frozen acquisition, исходные аудиторы и данные не изменялись. Новый comparator
не выполняет аппаратных действий и не разрешает дополнительный эксперимент.
