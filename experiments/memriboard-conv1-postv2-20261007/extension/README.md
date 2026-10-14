# Post-v2 Conv1-all-64 BoardSimulator extension — DRAFT, NOT APPROVED

All new source and any future outputs live **only** under this `extension/` directory. The original v2 pilot, its frozen protocol and its failed postprocessing are untouched. This is a simulator-only proposal based on the verified v2 hashes; it is **not** a physical-board validation, test-set evaluation, hybrid inference experiment, or training/STDP job. No simulator measurements have been run for this extension.

## Parent audit / decision boundary

Do **not** create an approval file, invoke `--execute`, freeze, or launch remotely before the parent reviews the remote environment, runtime/storage cost and protocol. `PROTOCOL.draft.json` is intentionally mutable. The runner supports read-only `--verify` and synthetic tests, with simulator import deferred until authorized execution. It uses only `torch.load(..., map_location='cpu')`, local training-file patches and BoardSimulator; it has no CUDA scheduling or trainer calls. Parent must arrange a window with no concurrent STDP/training load before approval; the JSON confirmation is an operator assertion, not a live process monitor.

Dry checks from repository root (no BoardSimulator MVM, no array writes):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s .tmp/openclaw-spikes/memriboard-conv1-postv2-20261007/extension -p test_dry.py -v
PYTHONDONTWRITEBYTECODE=1 python3 .tmp/openclaw-spikes/memriboard-conv1-postv2-20261007/extension/runner.py --verify
```

The parent, **after** review, would hash the exact three source files shown by `--verify` and supply a separate JSON file under `extension/` with keys `decision: "APPROVED_BY_PARENT"`, `stage`, `filters`, `source_sha256` (the complete `--verify` mapping), `checkpoint_sha256` (draft protocol value) and `no_concurrent_training_confirmed: true`. The approval is an explicit local guard, **not** a cryptographic signature or automatic freeze; the parent owns provenance and execution. Later invocation requires `--execute --stage <pilot|small|medium|remaining> --approval <parent-file>`. Do not use this command until authorized. Approval hash and source/input digests are checked before creating any arrays. Distinct stage approvals are required; preceding stage manifest and all completed filter hashes must verify. The code never creates approvals itself.

## Fixed proposal / safety

Checkpoint `twd1_sigma0p0_seed0/best_fs.pt` must have shape `[64,3,5,5]` and its exact v2 SHA-256. Every filter has 75 channel-major taps; two positive-conductance columns encode signed differential weights in three 32×8 tiles. Three independently seeded fresh arrays per tile/filter: `27001 + 100000*filter + 1000*replicate + 101*tile` (576 distinct seeds). Exact v2 training-file-only stratified split hash; eight fixed train and eight held-out validation images, four locations and four spike times per image. Stage order is filter 0 (pilot), 1–3, 4–15, 16–63. Stage32 across three replicas must pass before programming tiles 1–2 for that filter. Any readback, deterministic MVM or validation gate failure stops; no post-hoc retuning/retry. Sign match is descriptive, not a pass criterion.

Each **pre-call** safety check independently computes positive-column `sum(decoded_voltage / resistance)` from both HardCore-observed and actual simulated-device resistance views, rejecting invalid views or current **at or above 0.015 A**. Requested and DAC-decoded inference voltages must each be at most **0.3 V**, never clipped post-result. `drive` checks before `HardCore.dot_product` and the subclass checks again before `BoardSimulator.mode_mvm`. Pass the scaled voltage vector into `dot_product`, not the original binary spike vector. ADC saturation/zero-nonzero-drive aborts; programming pulse voltages are logged separately and are not claimed to respect the inference cap. No hardware connector is imported.

Per-filter outputs would be in `extension/results/filter_NN/`: preflight, initial/final array hashes, raw cell writes/readbacks, independent pre-call safety lines, raw patches, metrics, gates, per-replicate train-fitted affine and validation summaries, and a SHA-256 completion marker for every artifact. Filter-level per-run budget: 1,800 s, 14,500 MVM calls, 20 s/cell alarm; first write projects 64 cells within 900 s. A stage manifest is written only on full success. The code refuses an incomplete filter directory on restart rather than reusing a partially programmed array. Completed filters are skipped only if all output hashes and passed gates match; source changes invalidate the marker. An interrupted/failed filter requires *new parent-reviewed protocol and fresh arrays*, not deleting markers or replaying the same seeds.

## Parent cost/pitfall checklist

- Bounds, **not measured runtimes**: at most 576 arrays, 28,800 target cell writes, 928,000 allowed MVM calls and 32 filter-hours at the per-filter ceiling. Planned fixed schedule is 7,020 raw patch records and 14,040 MVM calls/filter (plus 28,080 pre-call guard records/filter), ~898,560 MVMs for 64 filters. Storage includes serialized arrays and verbose per-pulse histories: audit remote disk, CPU and scheduling before authorizing even the pilot.
- This maps **only Conv1**, and repeats sparse fixed patches. A good gate does not imply classification accuracy, generalization, full-network inference or hardware behavior. Train-fitted surrogate is evaluated unchanged on held-out validation, with no training-update noise σ reinterpreted as inference noise.
- The frozen v2 one-filter runner failed after measurement on nested `array` key aggregation; this runner consumes its own **flat** records and rejects duplicate/missing tiles and wrong split counts before reporting. Still inspect a synthetic dry preflight and first pilot artifacts before expansion. The simulator may have model-specific clipping/nonlinearities even when the independent predrive guard passes.
- No authority to run, freeze, or tune gates is delegated to this source draft. Parent owns the final protocol, remote audit/cost and go/no-go decision.
