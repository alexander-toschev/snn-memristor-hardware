# Заранее заданный резервный план readout

Зафиксирован во время input inventory сессии `1dfea7b2-229b-4396-8e2b-64fc8f4c1fe5`, до получения её accuracy. Применяется при необходимости доработки цифрового readout. Native MVM, LIF и сырые данные не изменяются.

Сохранённые train_counts:10000 изображений. Stratified split seed42:8000 fit /2000 validation. Статистики и маска вычисляются только поfit. Четыре кандидата: исходный SD-floor1e-6 безмаски; маска нулевой дисперсии и floor1e-6; та же маска и floor0.01; та же маска и floor0.1. Единицы:log1p spike counts. Исключённые колонки зануляются и приfit, и приeval.

SGD как в исходном протоколе, epoch checkpoints25/50/100. Итого12 кандидатов. Выбор: наибольшая strict validation accuracy, затем меньшая validation cross-entropy, затем меньше epochs, затем порядок scaler-кандидатов выше. Test labels/logits в выборе не используются.

Один refit выбранного рецепта на10000 с новым train-only scaler/mask; freeze head; одна финальная оценка на10000test. Исходный baseline сохраняется. Если baseline test уже наблюдался, новый результат не объявляется независимым confirmatory тестом. Не повторять test-оценки до желаемого процента. Цель>50% не гарантируется.
