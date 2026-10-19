# MNIST full recovery: independent archival contract review

The user's subsequent clarification requests repeating the original STDP
training. This document remains the readout/evaluation contract; loading only
the old weights is superseded as the main experiment. See
`../mnist-stdp-training-review/ARCHIVE_CONTRACT.md`. Evaluation must use the
newly trained weights when that reviewed route is executed.

19 October 2026. Preparation and source review only; no feature extraction,
readout training, test evaluation or hardware operation was executed for this
review. Frozen128, CIFAR and STDP acquisition sources remain unchanged.

## Evidence and fixed protocol

The archived `experiments/mnist_baseline_20261019/evidence/evaluation/live.log`
explicitly records collection totals30,000 train and10,000 test, followed by
`readout_probe_skipped` and `readout_linear`. This is stronger evidence for the
selected result than the generic60,000 default in its runner. Its saved summary
reports69.65%; the saved classifier head, scaler and exact runtime RNG are absent.

| Step | Archived behavior to preserve |
|---|---|
| Data | First train IDs0–29,999 and test IDs0–9,999, original IDX order; test is a separate split, not a holdout carved from train |
| Features | Frozen saved64×1×5×5 Conv1; T100; deterministic content-hash Poisson seed123, scale0.0015, boost1; archived synchronous Input→Conv1 delay and known broken post-window hook |
| Feature batching |128, shuffleFalse, drop_lastFalse;235 train batches ending with48,79 test batches ending with16; reset state between batches |
| Feature dimensions |36,864 raw spike-count columns, flatten C,H,W; discard the unused ten WTA-histogram columns |
| Transform | Float32 `log1p(counts)`; mean and sample standard deviation over train rows only; SD floor1e-6 |
| Test transform | Exactly the saved train mean/SD, with no test-derived adaptation |
| Head | `torch.nn.Linear(36864,10)`, default initialization and bias=True |
| Optimizer | SGD lr0.001, weight_decay0.0001; default momentum0, dampening0, nesterovFalse |
| Loss | CrossEntropyLoss, mean reduction; integer labels0–9 |
| Training |25 complete epochs, batch512, shuffleTrue, num_workers0, drop_lastFalse;59 batches/epoch, final batch304, total1,475 optimizer updates |
| Selection | Use final epoch25; no best-epoch selection, early stopping, validation search or test-dependent tuning |
| Evaluation | One final test pass, batch512, shuffleFalse, drop_lastFalse;20 batches, final batch272; argmax class, exact correct/10,000 |

Source locations under `reference_source/`: `evaluation.py:75–132` defines head,
optimizer, epochs and evaluation; `evaluation.py:285–303` defines feature batches;
`evaluation.py:330–341` selects the exact counts/log/z-score/Linear path;
`counts_readout.py:501–515` fixes prefix order and no-drop batching, and its
post-batch reset follows count collection. The generic `best_readout` summary is
selection among available reported readout methods, not best-epoch training.
Here the log explicitly skipped alternative probes.

## Declared reconstruction differences and implementation requirements

The full route must retain the independently reviewed128-pilot freeze:
`learning=False`, zero nu, no_grad, `conn.norm is None`, exact checkpoint copy,
and per-batch invariance of all connection weights/biases and thresholds. This
is explicitly stronger than the archived zero-nu-only collector. Preserve the
undefined `connection` exception; do not repair E/I or normalization silently.

Choose new fixed head/shuffle RNG seeds before execution, e.g.42, because the
historical state is unavailable. Separate construction RNG, head initialization
and training-order generators; save actual states or each epoch's permutation.
Explicit permutations may differ from the old DataLoader's global-RNG sequence:
record that as a reconstruction difference rather than claiming bitwise replay.

For bounded RAM, keep counts as fresh disk-backed arrays and transform each
readout batch. A stable streaming sample-variance calculation (Chan/Welford) can
accumulate in float64 over float32 log-counts, then cast saved mu/SD to float32.
This preserves the statistical definition but changes floating-point reduction
order/precision relative to the original float32 whole-matrix mean/std. Freeze
the chosen method before test and disclose it. Do not use a cancellation-prone
E[x²]−E[x]² implementation or silently change the SD floor. Constant train
features use the1e-6 floor; test values must not affect the scaler.

Fixed128 batch success does not test48- or16-sample tails. The full collector
must assert exact contiguous IDs, true batch sizes, valid monitor shape and
complete output coverage, including train48→test128 transition. Do not pad,
drop, overwrite prior rows or reuse the final batch's neuron state. An
incomplete memmap is not a complete feature artifact: track written rows,
flush at checkpoints and require complete coverage plus SHA before readout.

Save source/dependency and checkpoint/data hashes, effective config, all IDs and
labels, count arrays, per-batch input/count hashes and freeze evidence, scaler,
initial/final head, optimizer/RNG state, per-epoch training diagnostics, final
test logits/predictions and exact correct count. Failures remain explicit; no
test-driven rerun with altered hyperparameters belongs to this fixed result.

## Scale and bounded interpretation

Float32 counts consume5,898,240,000 bytes (about5.49GiB), before headers/labels and
reports. Saving a second normalized full matrix would double that storage and
is unnecessary. The head has368,650 parameters. Feature collection requires
314 batches; head training1,475 updates and final test20 batches.

The completed128-pilot manifest at
`results/mnist_feature_pilot_20261019/3c12404c-b409-4b9a-8a97-7eadea6f9c07/manifest.json`
reports8.544s and5.035s for its two128-example passes. Applying those two timings
to314 batches gives roughly26–45minutes for feature collection alone. This is
a planning sensitivity estimate from one repeated image batch, not an ETA or
confidence interval. It excludes scaler/head training, test evaluation, larger
file hashing and storage overhead, and assumes comparable machine load. The
pilot's monitor is CPU bool (471,859,200 bytes/batch), and its float32 reduction
temporary is1,887,436,800 bytes; low CUDA allocation does not eliminate host RAM
requirements. Do not infer full-run memory or timing from GPU peak alone.

No source-review approval implies69.65% accuracy or hardware performance. Any
future real memristor test must use public `manager.board.Connector`, native
converters and INI driver selection exclusively; no GPIO/SPI/serial/FPGA,
internal driver, `.interface` or dynamic-driver bypass is permitted.

Final implementation review is pending the author's new route. This contract
fixes the intended reconstruction and documents its limits before execution.
