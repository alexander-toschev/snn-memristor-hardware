# Independent B/C auditor draft

The implementation is `audit_cifar_feedback_remaining_reserves.py`; tests are `test_audit_remaining_reserves.py`. Only SHA-pinned independent A/adaptive audit helpers are imported. No collector, driver, converter package or backend is imported/executed.

## Verified saved synthetic journals

- `replay_success_v2/audit.json`: PASS323048 checks,0 errors;12938 calls,20 writes,1200 fresh partials.
- `replay_capacity_v2/audit.json`: PASS196954 checks,0 errors;7789 calls,229 writes; complete B screening fails capacity before SET, then C completes600 partials. No full300 summary.
-13 offline tests PASS, including route/preference and partial-contribution mutation, forbidden partials after failed B, unknown/duplicate/reordered candidates, ancestor file tampering, cross-source parent, exact280+20 mathematical coverage, missing block, failed-A new-only accounting and actual-session fail-closed gate.

These are explicitly FakeConnector synthetic fixtures, not native SIM or hardware runs. No parent provenance is asserted for synthetic fixtures. The in-memory280+20 composition in a unit test is a mathematical coverage test only and is never saved as acquired data.

## Independent coverage

Raw chunks and complete request/result ledgers are replayed for both new blocks. This includes native ADC conversion, latest-mode7 native-R current guards, RESET confirmations, exact candidate pools, simultaneous preferred reservation before ordinary fill, deterministic route, pair feedback, all writes/skips, fixed-gain fidelity, fresh C fit/selection/freeze,600 actual partials per block, final anchors and readback. B precedes C in recorded monotonic time. The bounds are independently derived from immutable real CIFAR fixture and routing: B21421/2794,C18957/2466,total40378/5260.

The session path implements parent manifest/audit/hash binding, recursive verified42 ancestry through the pinned A auditor, exact exclusion B11+C9, retained280 origins, separate ancestor accounting and full300/2400 sum75Р Р†РІР‚В РІР‚в„ўtrainedBNР Р†РІР‚В РІР‚в„ўLIF only when both new blocks complete. Completed-A and baseline-plus-failed-A are distinct fixed choices; a failed A contributes only its new attempts once. Any scientific capacity failure prevents a full summary and preserves the completed other block without historical fallback.

## Remaining freeze requirements / honest limitations

`FINAL_PINS` and `FINAL_PROTOCOL` are intentionally empty. Real-session auditing fails before reading acquisition data until the final B/C runtime/config/dispatcher and actual source-specific parent bindings are reviewed. No hardware parent UUID/hash was invented. Source snapshot for synthetic fixtures is89764591Р Р†Р вЂљР’В¦; actual final runtime may change only through an explicit reviewed freeze.

Interrupted/transport-error raw is currently unsupported by the inherited complete-ledger replayer and produces audit failure with an explicit error. It is never reported as a consistency PASS or a complete acquisition. The auditor never modifies input journals. Scientific route-capacity failures with complete screening are fully supported and independently replayed. Parent historical raw is verified through pinned completed audit and all file hashes, rather than expensively replayed again.

Actual session-level ancestry/composition code remains unexercised end-to-end until real frozen B/C evidence exists. Synthetic block replay and independent unit tests do not substitute for that future check. Quality failure remains separate from technical completeness.

## Reproduce after promotion

From public repository root, run the unit-test command below. Eleven self-contained tests use only tracked code/weight artifacts and temporary local files. Five additional tests explicitly skip if the optional saved synthetic journals or historical A session are absent. Skips are not passes; the current16/16 result used all original evidence. The synthetic replay command is run from the enclosing workspace root, because the saved fixture directory remains outside the public checkout. No source/raw files were copied into the public repository by this promotion.

```text
python -m unittest test_audit_remaining_reserves -v
python .tmp/hardware-public/audit_cifar_feedback_remaining_reserves.py --repo . --session .tmp/reserve_remaining_draft/synthetic_fixtures/success --synthetic-fixture --synthetic-plan .tmp/reserve_remaining_draft/configs/cifar_feedback_remaining_reserves.json --output .tmp/reserve_remaining_audit_draft/replay_success_NEW
```

For the failure fixture replace `success` with `capacity_failure` and choose a new output folder. Outputs may never be inside the source session. Existing reports are not overwritten.

## Deliberately disabled real-session CLI

`FINAL_PINS={}` and `FINAL_PROTOCOL=None` remain unchanged. Calling the ordinary session audit cannot produce PASS until reviewed final pins are supplied. This tool has no backend-start operation. Its synthetic pins describe the saved89764591РІР‚В¦ draft fixtures, not the later runtime snapshot. The separate A runtime/config/dispatcher/auditor are untouched.

## Manifest commitment hardening

The manifest must bind both completed_microtiles.json and summary.json for a completed acquisition. Whenever a partial checkpoint, summary or capacity_failures.json exists, its root artifact must also be present in result_sha256; all listed bytes are hash-checked. Manifest declarations independently equal40378 impacts,5260 pulses,5260 decisions,20 new coefficients and280 imported coefficients. Three additional mutation tests remove these commitments or change each declaration. The current suite has16 tests; the earlier13-test and synthetic replay results above remain historical evidence, not an assertion of a new backend run.
