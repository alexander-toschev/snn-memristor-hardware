# Строгая типизация нового archive64-маршрута

Проверяются ВСЕ семь исполняемых новых `.py` (`experiment`, `native`, `inputs`,
`cache`, `lif`, `readout`, `schema`), а не заменяющие их stubs.
`strict=true`, запрет explicit Any и Any от отсутствующих импортов; никаких
`type: ignore` или `cast(Any)`. Конкретный публичный Connector сохранён.

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) '.tmp/typing-deps')
python -m mypy --python-executable .venv-mnist-pilot/Scripts/python.exe --config-file typing/mnist_archive64/mypy.ini
```

Mypy2.4 устанавливается локально, например `python -m pip install --target
.tmp/typing-deps -r typing/mnist64/requirements.txt` (интерпретатор проверки
должен совпадать с wheel). Рабочий запуск использовал bundledPython3.12.

`.pyi` описывают только старые untyped границы: публичные MemriBoard
Connector/converters/init, BindsNET LIFNodes, прежний raw Writer и существующие
read-only checkpoint/INI/path helpers. Их реализация не меняется. Connector.impact
намеренно возвращает `object`: Collector.call проверяет tuple/list, длину,
типы ID/ADC и конечное положительное R до использования. Checkpoint helper
сам проверяет SHA/форму/dtype/finite W. Это не новая аттестация private-драйверов.

Единственный narrow `cast(MemmapFactory, numpy.lib.format.open_memmap)` описывает
реально существующую untyped функцию NumPy2.0; runtime dtype/shape проверены
сразу после вызова. Это не cast native-ответа и не скрытие Any.
NumPy/Torch собственные опубликованные аннотации используются напрямую;
`follow_imports=silent` не запускает mass-refactor исторических модулей.

Unit tests подтверждают архивный LIF/рефрактерность/задержку, неизменность входного
тока, равенство групп4 и всех64, полную25BL команду, cache invalidation,
content-seeded encoder и отсутствие доступа к test labels до сохранения head.
