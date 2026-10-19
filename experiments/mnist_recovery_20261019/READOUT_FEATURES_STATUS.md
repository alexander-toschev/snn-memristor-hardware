# Saved-feature recovery check, 19 October 2026

**Blocked for the selected 69.65% archive: no attributable spike-count features
were found.** No training, feature extraction, pickle execution, native SIM or
hardware access was performed in this check. No runnable readout command is
invented when its inputs do not exist.

The archived evaluation run is `20260426T111422Z_138b06a7f9b4`; its original
directory contains only `model.pt` (9,813 bytes), `label_map.pt` (296,695),
`live.log`, `status.json` and `summary.json`. The parent STDP run contains
checkpoints and logs but no feature matrices. The publication archive manifest
contains 11 artifacts, none a feature matrix, readout head or scaler.

An additional search of the original `snn-mnist` repository included ignored
files and `.npy`, `.npz`, `.h5`, `.hdf5`, `.pkl`, `.pickle`, `.dat`, `.memmap`,
`.bin`, count/feature/readout-named `.pt` files, excluding environment/Git
internals. Other memmaps exist, so a claim that no memmaps exist is incorrect.
However, no `.memmap` had either expected complete feature matrix size:

| Matrix | Float32 counts only, 36,864 columns | Historical counts+WTA, 36,874 columns |
|---|---:|---:|
| First 30,000 train | 4,423,680,000 bytes | 4,424,880,000 bytes |
| First 10,000 test | 1,474,560,000 bytes | 1,474,960,000 bytes |

The root `Xtr.memmap` and `Xte.memmap` have 655,560,000 and 131,112,000 bytes;
their int64 labels have 40,000 and 8,000 bytes (5,000/1,000 samples). They were
modified on 21 June, and are incompatible with this April run's shape/split.
Numerous further memmaps belong to June sweep folders. They must not be used as
the chosen baseline without independent checkpoint, encoder and split provenance.
This search does not prove there is no differently named backup on another host.

The reference `evaluation.py:274` supports optional `SNN_COUNTS_MEMMAP`; default
is in-memory. The exact run did not record that environment variable or a feature
manifest, so whether its original counts ever persisted cannot be established.
Its runner saves Conv1, label map and summary; it does not save the Linear head
or train normalization. `label_map.pt` cannot recover those parameters.

## Minimal next step without new STDP or board access

1. Establish an isolated, repository-local environment and pin actual package
   versions. Torch alone is insufficient: compatible BindsNET and torchvision
   are required for a reference extractor. The currently available historical
   Git snapshot has no environment lock attested to the run.
2. SHA-verify the archived checkpoint and datasets before loading. Attempt
   restricted `torch.load(..., weights_only=True, map_location='cpu')`; validate
   a finite 64x1x5x5 tensor and expected configuration. If restricted loading
   rejects metadata, inspect it and explicitly handle the specific trusted
   structure; do not silently retry unrestricted pickle. Preserve original
   checkpoint bytes and record extracted tensor SHA separately.
3. Freeze reference extractor semantics explicitly. The archived post-window
   normalization hook refers to undefined `connection` (builder has `conn`),
   catches the error and thereby skips its E/I branch. Initial normalization
   exists. Silently fixing this changes the experiment. Source SHA was not
   attested to the April run, so either variant is a new reconstruction.
4. First collect a small fixed train-only pilot (e.g. first 128 samples).
   Use T100, deterministic Poisson base seed123, rate scale0.0015, boost1,
   batch128 and original sample order. Disable STDP and explicitly verify
   Conv1 tensor SHA before/after each pilot batch. State-reset and threshold
   behavior also require verification; nu=0 alone is not proof of frozen weights.
   Record elapsed time, peak memory, nonfinite checks and repeatability.
5. After the pilot, collect first30,000 train and first10,000 test counts to
   fresh repository-local arrays with sample IDs, labels, source/dependency
   hashes, checkpoint hash and encoder configuration. Never overwrite archive.
6. Restore only the readout: log1p(counts), train-only mean/sample-std with floor
   1e-6; Linear(36,864,10) with bias, SGD lr0.001, weight_decay0.0001,
   CrossEntropyLoss,25 epochs,batch512,shuffle train,drop_last=False. Choose and
   record a new fixed head/shuffle seed (e.g.42) before test. Historical exact
   RNG state is missing. Save head, mean/std, RNG, predictions and labels.
   Evaluate test once; do not tune on test or promise 69.65%.

## Runtime and storage

No measured benchmark exists for this reconstructed extractor on the current
machine, so runtime is **unknown**. The archived CUDA run lasted21m41s overall
(11:14:22–11:36:03 UTC), including other setup/calibration; that is neither a
CPU estimate nor an isolated readout benchmark. Pilot seconds/sample must be
measured before estimating40,000 samples.

Float32 counts require5,898,240,000 bytes (about5.49GiB); preserving the unused
10 WTA columns adds1,600,000 bytes. Labels and IDs are small. Keeping normalized
float32 matrices adds another5.49GiB; computing normalization per training batch
avoids that disk copy. A 128-sample T100 monitor of36,864 float32 neurons alone
is1,887,436,800 bytes (about1.76GiB), before network state and temporary arrays.
The historical pipeline's counts/log/z-score copies can require more than16GiB.
The pilot must check available memory; changing batch size may change historical
behavior and must be recorded rather than disguised as exact reproduction.

The linear head has368,650 parameters (~1.41MiB float32); mean/std together
~0.28MiB. Training has59 batches/epoch,1,475 updates total. Forward plus weight
gradient matrix multiplication is approximately1.11e12 floating operations
across25 epochs (not a time estimate). All head training remains digital.

The existing safe command remains archive inspection only:

```text
python experiments/mnist_recovery_20261019/experiment.py --inspect
```

This check leaves every frozen CIFAR source and historical artifact unchanged.
