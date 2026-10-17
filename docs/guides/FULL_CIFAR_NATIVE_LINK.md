# Native linkage extension: verified historical300, full-image import still closed

All work is offline. No Connector, driver, native simulator or hardware call is
made. The existing CPU classifier and acquisition runtimes remain unchanged.

## Actual native data verified

`native_full_cifar_link.py --check-historical-300` uses a fixed, explicitly
reviewed root, not a user-supplied JSON claiming that an audit passed:

- Session `results/cifar_feedback_remaining_reserves/58ca618b-c37f-454e-be25-db21d1d25f60`.
- Manifest SHA `5fb5041cbdba148ff0a02182512a98ca0784fed074e2830d6fc06b230f933171`.
- Acquisition audit SHA `62a3f84f10ce55eb55be4930c605108a31440359d7e46c086d5f594a48ae162e`.
- Pinned acquisition auditor SHA `4f2f73eb96fd286aabde4e0513b7aac2ef96e57ef873ce36573a73a29a3b0525`.

The top manifest binds three ancestor sessions. Imported wrappers are followed
by exact file SHA/index/result equality to original native tile journals. The
original result bytes, native task IDs and session identities are preserved.
Only completed contributing microtiles are replayed; failed/unimported attempts
remain covered by the separately pinned historical acquisition audits.

Actual replay result in `results/full_classifier_bridge_20261016/actual_native_300_link.json`:

|Evidence|Result|
|---|---:|
|Distinct trained `(filter,tap)` pairs|300 =4*75|
|Contributing native microtiles|28|
|Real CIFAR images|10|
|Spatial positions per image|1, at(13,13)|
|Logical timesteps / repeats|12 /5|
|Reconstructed native partial points|16800|
|Fresh native CIFAR mode10 tasks|120110|
|Maximum difference versus saved trained-unit partials|0.0|
|New backend / hardware calls|0 /0|

This is an actual-hardware-record compatibility proof, not a new measurement,
current-state map, full64-filter image, classifier prediction or accuracy claim.
The earlier full1806336-key proof remains **synthetic software data**. Neither
proof is relabeled as the other.

## What the new parser checks

- Every ordered raw chunk name/SHA/byte count/line count and concatenated SHA;
  pairs can cross part boundaries. Requests/results are consecutive by task ID.
- Session/source identity and identical request metadata in the result. The
  native returned ID, raw ADC and `native_result` agree; errors/orphans fail.
- Native CIFAR command is mode10/id0, one native ADC sample,32 inputs with at
  most one0.1V input. Zero/DAC0 and0.1V/DAC82/native decoded0.1 match the source
  converter. Native voltage is `round(adc*5/16383,5)`; invalid/saturated ADC or
  output above the reviewed4V threshold is refused.
- Source converter, checkpoint and prepared TTFS artifact SHA are bound in all
  recorded manifests. Full real image bytes are independently hash-checked.
  TTFS is reconstructed from uint8 pixels; CHW tap→pixel and physical WL/BL
  command are checked, not taken from saved `active_taps` declarations.
- Each native row executes zero-before, the exact ordered active one-hot
  signals (or a fresh `empty_zero_signal`), then zero-after. Missing or repeated
  terms cannot be turned into a CPU zero. A measured nonzero empty response is
  retained as an observed offset/noise contribution.
- Frozen route and C/gain files match their raw freeze events before the first
  CIFAR task. C has zero affine offset, calibration does not use CIFAR or trained
  targets, decoder gain is the fixed hardware1. Fitted alpha is never used.
- Positive/negative zero-corrected row sums, C-corrected rows and signed/gain
  contributions are reconstructed, then compared with every saved partial's
  numerical value, raw ADC and task references.
- Trained weight bytes and CHW slots are checked against the pinned Conv1 NPY,
  without computing `W*x`. This identifies the intended coefficient; it does
  not imply that the physical conductance programmed that coefficient accurately.

Programming/current/native-R safety and matrix-fitting/selection mathematics
are covered by the pinned historical acquisition audits; this extension does
not silently claim to recompute those whole audits. It verifies their identities
and raw linkage at the measurement boundary.

## Full-image scope gate and normalized-record adapter

`native_full_cifar_adapter.py` provides:

1. `verify_native_block_scope`: requires exact planned tap partition and actual
   enumerated native keys for every case/repeat/T12/y28/x28. It rejects declared
   counts in place of identities, duplicates, omitted positions and spot13,13.
2. `verify_all_blocks`: requires all planned physical microtiles, all64 filters
   and all4800 taps. No missing tile can be supplied by an old result.
3. `normalized_record`: converts a linked native row into the normalized ADC
   format using exact repo-relative session/folder/task references, image and
   calibration identity. It carries measured empty-zero responses unchanged.
4. `compare_normalized`: an independent raw reconstruction must match normalized
   ADC/reference/metadata exactly before its dense-export arithmetic is trusted.

Scope results are structural, not security credentials. There is no registered
full hardware collector/export auditor, and `require_enabled_hardware_source`
always refuses enablement. An arbitrary `passed`, `source=hardware` or coverage
boolean cannot unlock the classifier. The existing classifier's own guard also
remains unchanged.

The historical adapter may infer(13,13) only because its fixed reviewed source
contract and fixture pin identify that position. Generic full-image parsing
requires explicit `output_y` and `output_x` on every native CIFAR request.
Some historical14-tap microtiles cross logical32-tap boundaries; the scope gate
does not pretend they are valid32/32/11 full-image tiles. Future microtile plans
must match an explicitly reviewed physical schedule and exact partition.

## Minimal integration proposal, not enabled today

1. The future bounded full-image collector must preserve this native request/
   result schema, explicit spatial coordinates and a route hash per microtile;
   freeze C and fixed source gain before collecting its responses. Its source,
   input, command budget, programming safety and decoder audit need review.
2. A source-bound full-session adapter streams each committed raw journal through
   `NativeLink`; the scope gate requires all expected actual keys. The normalizer
   writes the resulting ADC/task-reference records and their ordered hashes.
3. A separate reviewed full-export auditor reruns the native reconstruction,
   exactly compares normalized records, then independently reconstructs the
   dense payload using `audit_full_cifar_export.py`. Register that **executable
   auditor source** only after review. The importer must rerun the trusted audit
   or otherwise verify its reproducible source-bound result, not trust a JSON
   success flag supplied beside a payload.
4. Add a small verified float32LE→tensor adapter after that audit. For each case
   and repeat independently, sum all logical tiles before trained BN/LIF and
   call `downstream` with fresh state. Keep allT12 steps, including10/11, and
   fractional post-pool currents. Compare with the separate CPU reference.

The missing full-image collector and its safety/source audit are explicit gaps.
No uncontrolled configuration or input declaration can substitute for them.

## Reproduce

From repository root (all tests use explicit synthetic fixtures):

```powershell
python -m unittest tests.test_full_cifar_bridge tests.test_native_full_cifar_link tests.test_native_full_cifar_adapter -v
```

From repository root, read the already saved actual native data:

```powershell
python native_full_cifar_link.py --repo . --check-historical-300 --output results/my_bridge_review/new_actual_native_300_review.json
```

Do not overwrite the preserved proof. The full actual replay is already PASS;
22 targeted tests initially passed in0.441seconds. Promotion adds one regression
test rejecting Windows backslash parent traversal; the current suite has23 tests.
Their scope includes wrong ADC/native
result, wrong DAC/id/voltage, TTFS/tap/BL/image changes, missing or duplicate
pairs, changed/late freeze, missing spatial metadata, missing/duplicate full-image
keys, altered normalized ADC/reference and attempted self-attestation.

## Promoted files and preserved evidence

Code: `full_cifar_plan.py`, `full_cifar_export.py`, `audit_full_cifar_export.py`,
`native_full_cifar_link.py`, `native_full_cifar_adapter.py`.
Tests: `test_full_cifar_bridge.py`, `test_native_full_cifar_link.py`,
`test_native_full_cifar_adapter.py`, `validate_full_cifar_bridge_synthetic.py`.
Documentation: `FULL_CIFAR_BRIDGE.md` and `FULL_CIFAR_NATIVE_LINK.md`; the main
README was not replaced. Evidence: `synthetic_full_coverage.json`,
`actual_native_300_link.json`, `ten_image_plan.json` and the original
`GOAL_REVIEW.md` were copied byte-for-byte to
`results/full_classifier_bridge_20261016/`. No runtime/dispatcher changes were made.

The original review records the adapter hash before promotion hardening.
Only its lexical path guard and the corresponding regression test changed:
Windows `..\\` traversal is now rejected in origin references as well as `/../`.
The native parser, planner, exporter and independent numerical auditor bytes
are unchanged. Preserved actual/synthetic proofs still bind their original,
unchanged parser/arithmetic hashes; they were not rewritten or expensively rerun
for this path-only hardening. The promotion note records exact old/new hashes.
