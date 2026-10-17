# Офлайн-тесты

Запуск из корня с зависимостями из `requirements.txt`; CPU-классификатор дополнительно требует `requirements-classifier.txt`:

```sh
python tools/run_tests.py
python tools/run_tests.py tests.test_compare_fresh_traces tests.test_export_fresh_trace -v
python tools/run_tests.py --pattern test_analyze_*.py
```

Runner добавляет корень, локальную `.deps` и публичные Python-модули MemriBoard в путь импорта. Тесты acquisition используют FakeConnector и реальные converters, а не аппаратный или native simulator backend. Некоторые тесты требуют сохранённых исторических результатов; отсутствие таких данных должно быть явно сообщено как skip или ошибка, а не принято за успешный аудит.

Три старых test-модуля независимых аудиторов остаются в корне: `test_audit_cifar_feedback.py`, `test_audit_cifar_feedback_adaptive.py`, `test_audit_cifar_memristor.py`. Их пути/хеши упоминаются в frozen проверках. Runner включает их в полный запуск без изменения исходных байтов.

Остальные unit-тесты находятся здесь. При обращении к данным используйте корень репозитория, а не текущую папку теста. Новые тесты не должны вызывать `--run-hardware`, штатный simulator или изменять исторические results/weights/vendor.

Известная историческая несовместимость: `test_audit_remaining_reserves`
проверяет frozen SHA диспетчера B/C (`632c534f…`). В текущей ветке уже до
перестройки каталогов диспетчер имел SHA `78ab85d6…` из-за новых сценариев.
Поэтому `test_final_source_bytes_and_actual_parent_bindings` требует checkout
исходного B/C commit и может завершаться FAIL в современной ветке. Проверка
не отключена и frozen pins не обновлены ради зелёного результата.
