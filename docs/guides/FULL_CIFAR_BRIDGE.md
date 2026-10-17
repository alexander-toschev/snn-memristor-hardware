# Full Conv1 bridge preparation (software only, not enabled hardware import)

This software-preparation package was promoted from its reviewed staging copy.
It changes no frozen collector, dispatcher, classifier or acquisition auditor. No native
backend, hardware, drivers, checkpoint weights or CPU `W*x` are invoked here.
The later native-link extension reads pinned extracted weight bytes to verify
route identities, without using them to compute measured contributions.
See `FULL_CIFAR_NATIVE_LINK.md` for its completed historical300-tap raw linkage proof.

Implemented files:

- `full_cifar_plan.py`: lazy full-image coverage, exact4800 coefficients, strict
  CHW75 partitions into logical32/32/11 tiles. Each logical tile may contain
  any number of disjoint physical microtiles. Default64*3 physical microtiles;
  `--maximum-taps7` demonstrates arbitrary splitting without changing coverage.
- `full_cifar_export.py`: normalized ADC journal decoder and dense float32LE
  exporter. Requires frozen per-microtile2x2 C and fixed source gain before any
  CIFAR record. Explicit hardware gain1 / native-SIM gain0.03; synthetic tests
  use1 and are labeled. No fitted per-weight scale. No missing zero/CPU fill.
- `audit_full_cifar_export.py`: independent stdlib implementation; imports none
  of the above code. Separately validates partitions, order, freeze, indices,
  native ADC conversion, paired-zero correction, C/gain, complete physical keys,
  dense reconstruction, payload/source bindings and output assertions.
- `test_full_cifar_bridge.py`: ten synthetic software mutation/boundary tests.
- `validate_full_cifar_bridge_synthetic.py`: full1806336-key in-memory synthetic
  check with analytically known varying values by filter/logical tile. No ADC
  values are represented as measured on hardware. No native simulator is used.

## What is not implemented or attested

There is no full-image collector or JIT weight programming strategy here. The
software accepts a **proposed normalized ADC ledger**, not arbitrary existing
native raw sessions. The extension now independently verifies request/result,
ADC, TTFS vectors, task references, route and frozen C/gain for the explicitly
pinned historical300-tap acquisition. That proof does not cover a future
full-image collector. Fresh physical calibration acquisition and that collector's
programming/current/transport safety still require a reviewed protocol and
independent audit. Hashing an arbitrary raw manifest alone does not prove that
a ledger ADC came from its referenced task. These boundaries are
explicit in both export and audit reports:
`hardware_provenance_verified=false`, `hardware_import_enabled=false`.

Consequently the existing `full_cifar_classifier.py` remains disabled for measured
imports, and this draft auditor is NOT registered as a trusted hardware-export
auditor. A successful synthetic/normalized-ledger replay is not full hardware
integration, classification accuracy, a physical qualification or a launch.

## Proposed normalized ledger contract

Input JSON schema: `full-conv1-adc-ledger-v1`; explicit `source_kind` matching the
plan (`hardware`, `simulator`, or `synthetic_test`); bound `{path,sha256}` for the
plan and native raw manifest. All paths are repo-relative, reject escapes, drive
letters and UNC paths. `ledger` contains ordered parts with `path`, `sha256`,
`bytes`, `records`, plus concatenated SHA and total record count. Each UTF-8
JSONL line must finish with a newline. Duplicate/reordered/missing parts fail.

Every event has a globally consecutive `ordinal`, `micro_id` and bound raw
manifest SHA. A `calibration_frozen` event contains a hash over all its other
fields, exact route hash,2x2 C, fixed gain, `calibration_uses_cifar=false` and
`fitted_weight_scale_used=false`. A second freeze for the same physical microtile
is rejected. A future JIT reprogramming epoch needs a separately reviewed schema;
it cannot silently reuse or update these frozen entries.

Each `measured_partial` identifies `(case_id,repeat,micro_id,t,y,x)`, image SHA
and the frozen calibration SHA. Its nonempty `terms` list contains actual
positive/negative paired rows, each `[zero_before,signal,zero_after]` ADC triple,
and six distinct raw task reference strings. Multiple terms allow a physical
microtile's measured suboperations to be added; their task identity and TTFS
composition still require the future native normalizer/auditor above.

The pinned native conversion semantics are MemriBoard
`convert_adc_to_volt(1,14,5,adc)` from
`vendor/MemriBoard/manager/service/converters.py`, lines67-75:
`round(adc*5/16383,5)`. The current converter source SHA is
`28c6495c56afa225501dd6b0dea999215f08b73f30e016ff7ea34d6c5412701c`.
This stdlib replay uses that formula without importing the driver package.
Invalid, noninteger or saturated ADC>=16383 is rejected; hardware headroom and
current guards must additionally be verified by the native acquisition auditor.

For each paired term, subtract the mean of before/after zero voltages from each
signal row, apply frozen C, subtract corrected negative from positive, divide
by fixed gain. Sum terms then physical microtiles into their logical tile.
All75 taps are therefore summed before trained BN/LIF by the future consumer.
No learned weights are loaded by exporter/auditor.

Dense output axes: `[case,repeat,T12,F64,Y28,X28,logical_tile3]`, little-endian
float32. Repeats are separate; no averaging or state carryover between repeats
or images. The classifier must call downstream with a fresh LIF state for each
image/repeat, keep silent t10/t11, and preserve fractional average-pool outputs.
The draft binary export has no automatic adapter into the current tensor-only
classifier contract; adding that adapter must remain behind trusted audit review.

Audit float32 comparison allows absolute2e-7 plus relative1e-6 due to independent
arithmetic operation ordering. Raw ledger hashes, payload hash, exact coverage
and source/shape/stage assertions are checked separately.

## Reproduce offline

From repository root:

```powershell
python -m unittest tests.test_full_cifar_bridge -v
python validate_full_cifar_bridge_synthetic.py --output results/my_bridge_review/new_synthetic_proof.json
```

Generate a new full-image plan without acquisition:

```powershell
python full_cifar_plan.py --repo . --repeats 1 --maximum-taps 32 --output results/my_bridge_review/ten_image_plan.json
```

The plan contains no physical commands and reports native impacts/pulses as
unknown rather than inventing a programming policy. A future normalized input
can be processed offline with:

```powershell
python full_cifar_export.py --repo . --manifest path/to/normalized_manifest.json --output path/to/new_export
python audit_full_cifar_export.py --repo . --manifest path/to/normalized_manifest.json --export path/to/new_export --output path/to/new_audit.json
```

No such native full-image input currently exists. All outputs are created only
after complete replay, existing outputs are preserved. An incomplete journal
raises an error and never becomes a success or a partially filled dense tensor.

One image/one repeat has1806336 logical partials; ten images/five repeats have
90316800. More physical microtiles increase physical partial count. This is not
a native impact count: zero brackets, rows, suboperations, calibration and writes
require a real bounded collector design. Memory scales with dense output and
one-byte-per-physical-key coverage; the iteration schedule itself is lazy.

## Recorded software checks

Ten targeted tests passed in0.379seconds. The separate full in-memory synthetic
proof passed1806336 distinct partial keys/1806528events,192frozen calibrations,
all64filters/all28x28positions/allT12, with maximum float32 error
3.7050247175818285e-9 and zero backend calls (42.75seconds). Its machine-readable
result is `results/full_classifier_bridge_20261016/synthetic_full_coverage.json`. It checks both independent replay
engines and analytical axis values, but does not exercise a full-size on-disk
native normalized ledger, because no native full-image data exist. Chunked file
hash/order/tamper behavior is tested separately on explicit synthetic snippets.
