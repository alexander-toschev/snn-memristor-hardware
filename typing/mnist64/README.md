# Strict types for MNIST64

Scope: `experiments/mnist64_20261020/core.py` and its structural types. The
configuration uses mypy strict plus prohibitions on explicit, imported and
expression-level Any. It does not silence errors in this module or pretend
that the historical untyped experiments were type-checked.

```sh
python -m pip install -r typing/mnist64/requirements.txt
python -m mypy --config-file typing/mnist64/mypy.ini
python -B -m unittest tests.test_mnist64 -v
```

`Connector` is the concrete public `manager.board.Connector`, imported only
under TYPE_CHECKING. No runtime driver is imported by these annotations.
`Converters` and `RawWriter` are narrow structural contracts for the existing
module/writer objects. Events, commands, measurements, images, gain selection,
updates and summaries have explicit TypedDict shapes; callbacks and all
Collector state/methods are annotated.

The `.pyi` files describe only the used public boundary of existing untyped
modules: Connector.impact/open_port/close_port; inherited emit/read_r/vector/
checkpoint; native logging; fixture validation and pure LIF/fit helpers. They
are static-only, outside vendor and historical acquisition directories. Their
signatures were read from the actual source, not used to replace runtime code.

Connector documents `(resistance_ohms, request_id, adc)`. Runtime call still
checks tuple length, echo/ADC exact types, positive finite R and ADC range.
Inherited reads validate addresses, native converters/current/headroom and
maximum4BL. Fixture validators still check the exact schema and encoding.
Stubs describe successful validated returns, not an exemption from validation.

One narrow `cast(PulseDecision, payload)` bridges mypy's open-TypedDict
keyword-unpack limitation. Every known field is explicitly constructed and
metadata is restricted by `Unpack[PulseMetadata]`; it is not cast to Any and
is not an unchecked external response. An assertion narrows the already
validated non-fit gain. Small intermediate names preserve dictionary order,
values, raw commands and numerical operations.

This typing change follows the completed SIM run, both independent raw audits
and exact-byte archives. Executed source is retained in commit1ae4fa1; strict
replay of that historical run uses that source snapshot. No new SIM/hardware
run was performed solely for annotations.
