# Full MNIST reconstruction — draft, execution gate closed

**Superseded as the primary experiment:** the user clarified that MNIST must
repeat parent STDP training, then evaluate its newly learned weights. This
unexecuted frozen-weight draft is retained only as a possible reviewed readout
component. Its launch gate remains closed.

This recovers the missing readout for the selected April26 MNIST archive.
It is a **new reconstruction**, not a guarantee of the archived69.65% result.
No new STDP and no hardware/native MemriBoard experiments are included.
The successful128 pilot, CIFAR and STDP source files remain unchanged.

## Frozen contract

| Stage | Exact scope |
|---|---|
| Feature train | First30,000 official train IDs0…29,999,235 batches128, last48 |
| Feature test | All10,000 official test IDs0…9,999,79 batches128, last16 |
| Frozen extractor | Checkpoint SHA43f7d623…,64×1×5×5, T100,36,864 counts |
| Encoder | Archived deterministic content hash seed123,rate_scale.0015,boost1 |
| Reset | Before/after each batch; no sample padding or dropped tails |
| Normalization | CPU float32 log1p; train-only mean/sample SD, correction1,floor1e-6 |
| Readout | Linear(36,864,10,bias=True), default torch initialization |
| Training | SGD lr.001,weight_decay.0001,momentum0,CE mean,25epochs |
| Train batching |512,shuffleTrue,num_workers0,drop_lastFalse;59/epoch,last304 |
| Selection | Final epoch25 only;1,475 updates; no validation/test selection |
| Evaluation | Test once, batches512,last272; save every logit/prediction/confusion |

Feature/head/shuffle seeds are separately fixed42. Exact historical RNG state
is missing. The selected log used `readout_probe_skipped`; only the archived
counts_zscore+Linear route is reconstructed, with CUDA digital head training.
The dataset bytes are the SHA-pinned locally preserved MNIST copy; the original
run did not attest its dataset/source/environment bytes.

Train counts are extracted first. The scaler is fitted only to those counts.
Readout training and saved final head finish before test examples/labels are
loaded for extraction. Preflight hashes compressed test files for provenance
without using test examples for fitting. Test is evaluated once, after final
epoch; no hyperparameter retry or best-test-epoch choice exists in this entrypoint.

## Declared reconstruction differences

The archived builder initially normalizes random weights; checkpoint loading
overwrites them and thresholds. Its undefined `connection` post-window hook
remains broken/caught, so its later normalization and E/I code do not execute.
This is not repaired. Default synchronous `net.run` retains the one-step delay.
Learning is explicitly disabled on every layer/network, unlike archived
zero-nu-only collection. Every connection weight/bias and every threshold must
remain byte-identical before/after every batch; `norm=None` is required.

The scaler uses bounded-memory Chan combination in float64 over CPU float32
log1p values, then saves float32 mean/SD. It implements the same statistical
formula but **does not claim bitwise equivalence to archived float32 full-array
reductions**. A dedicated seed42 shuffle generator gives a reproducible new
order; each actual epoch order and generator state is saved. Initial/final
head parameters are saved as pickle-free NumPy arrays.

The original successful128 pilot is imported as a SHA-pinned helper, without
editing or invoking its main. Its exact environment/source/IDX/checkpoint
checks remain effective. The first train batch must match both recorded input
and output SHA from pilot UUID3c12404c-b409-4b9a-8a97-7eadea6f9c07. This checks
route continuity, not full-data repeatability or accuracy.

## Resources and artifacts

Counts alone:5,898,240,000 bytes (5.49GiB), stored as train/test float32.npy
memmaps. No full normalized copy is saved. Epoch orders add6MB; labels, IDs,
scaler, head, events and predictions are small. Scoped Git LFS rules include
all result.npy files; existing rules cover event JSONL. Reserve18GiB disk for
results, Git LFS duplication and overhead; abort below2GiB free. Result budget
is8GiB. Wall budget4hours is checked between bounded batches/epochs; it does
not asynchronously interrupt an in-flight CUDA call.

Host RAM preflight requires8GiB available. Monitor storage lives on CPU;
recording~0.47GB bool plus float32 reduction~1.89GB, stack/copy temporaries and
memmap pages require host memory. GPU preflight requires4GiB free using the
same isolated `.venv-mnist-pilot`; no silent device/batch fallback. CUDA is
synchronized after net.run before CPU Monitor access. CPU and CUDA memory are
reported separately. Dataset extraction always reads binary IDX with CRC,
exact header/length checks; checkpoint uses weights_only and exact SHA.

Measured128 pilot batches took8.544s then5.035s. Multiplying by314 batches gives
roughly26–45minutes for features alone; this is a planning sensitivity range,
not an end-to-end ETA. Scaler, training, hashing, OS memory pressure and disk
I/O are additional. Full run timing is not yet measured.

Every run gets a fresh UUID directory. Events include each batch's input/count
SHA, IDs, activity, state hashes, timing and memory; split completion is explicit.
Incomplete/failed collections retain artifacts marked failed, never accuracy.
Final manifests hash all result files. No auto-resume or reuse of partially
filled memmaps is permitted.

```text
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_full_recovery_20261019/experiment.py --preflight
```

Only after independent source/scientific review and root registration:

```text
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_full_recovery_20261019/experiment.py --run
```

This route is entirely software. Any later physical memristor measurement must
use public MemriBoard Connector, its native converters and INI driver selection.
