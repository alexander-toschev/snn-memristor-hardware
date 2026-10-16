# Full CPU classifier and future measured Conv1 boundary

This is the complete **ten-class subset of CIFAR100**, not a 100-way classifier.
Output indices 0..9 map to fine labels `[0,1,2,3,5,8,13,14,17,19]`.
The code performs no acquisition and imports no MemriBoard modules.
Full hardware classification has not been run or integrated. The hardware
import CLI deliberately raises `NotImplementedError` until a reviewed
collector-specific full-export auditor exists. Self-declared hardware flags,
coverage masks and an arbitrary JSON saying "passed" cannot enable it.

## Actual checkpoint architecture

`weights/best_fs.pt` is loaded on CPU with `weights_only=True`; every state key
and shape is validated by `load_state_dict(strict=True)`. The pinned top-level
`weights/cfg.json` controls the reference constructor.

1. uint8 RGB CHW /255, no mean/std normalization. TTFS: T=12,
   `floor((1-x)*11)`, pixels below 0.1 silent.
2. Conv1: 3 -> 64, 5x5, no padding/bias, output 64x28x28.
3. Trained BN1[t], LIF, average pool2 -> 64x14x14.
4. Conv2: 64 -> 128, 3x3 padding1; BN2[t], LIF, average pool2 -> 128x7x7.
5. Conv3: 128 -> 256, 3x3 padding1; BN3[t], LIF; no pooling.
6. Flatten12544 -> FC512 (no bias), trained BN_fc[t], LIF. Dropout is
   disabled in evaluation.
7. FC512 -> 10 with bias, output LIF. Report first-spike, membrane-sum and
   trained TWD `sum_t softplus(alpha[t])*spikes[t]` decoders separately.

All LIF steps use `v=.5*v+current`, spike at `v>=1`, hard reset to0;
negative membrane values are preserved. BN uses stored running statistics and
eps=1e-5. The late t10 and t11 TTFS inputs are zero, but trained BN and all LIF
states still advance. Output membrane reports are **after reset**.
First-spike ties (including all-silent outputs) choose the first index, matching
the original forward helper; reports expose silence/tie count.

`full_cifar_reference/{temporal_csnn,lif,encoders}.py` are byte-identical snapshots
of the corresponding local `src/models/` files at preparation. They are pinned
before import. This proves consistency with the available source and checkpoint;
the original historical training commit is not established by these snapshots.
BN eps/LIF semantics are source/config properties, not serialized learned weights.
The older training repository's `evaluate.py` reads a nested `model` config and
uses non-strict loading; this checkpoint's architecture fields are top-level, so
that launcher is not reused here.

## Reproduce CPU validation

Use a CPU Python environment with PyTorch; no torchvision or network download is
needed. The ten genuine full CIFAR100 test images are already hash-bound in
`weights/cifar100_conv1_fixture.json`, with original dataset index and fine label.
Paths below are relative to this repository. The Python environment is selected
by the researcher; no machine-specific paths are embedded in the module.

Optional isolated CPU environment (tested PyTorch2.14.1):

```powershell
python -m venv .venv-classifier
.venv-classifier/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-classifier.txt
.venv-classifier/Scripts/python -m unittest test_full_cifar_classifier -v
```

On Linux use `.venv-classifier/bin/python`. Dependency installation is a user
setup step, not performed by the classifier. PyTorch2.14.1 was already available
in the preparation environment; the installation command itself was not tested
as a clean reinstall here.

```powershell
python -m unittest test_full_cifar_classifier -v
python full_cifar_classifier.py --cpu-reference --case-index 0 --output results/my_cpu_check/case0.json
0..9 | ForEach-Object { python full_cifar_classifier.py --cpu-reference --case-index $_ --output "results/my_cpu_check_all/case$_.json" }
```

Outputs are not overwritten. CPU reports say `source=cpu_reference_only` and
`hardware_acquisition_performed=false`. Ten selected examples are diagnostic
fixtures; their correct-count is not accuracy on the complete test set.

## Prepared measured-input contract (disabled pending acquisition/export audit)

The eventual dense CPU-readable tensor must contain **all** pre-BN Conv1
partials in checkpoint units for one specified full image. Shape is
`[T=12, filter=64, y=28, x=28, logical_tile=3]`, float32. Logical tiles partition
CHW-flattened75 taps as `[0,32),[32,64),[64,75)`; the final tile has11 active
coefficients. Physical microtiles may subdivide these logical tiles, but the
future exporter must audit their complete disjoint contribution first.

The payload is a safely loaded tensor-only `.pt` dictionary with `partials` and
same-shaped boolean `coverage`. Every coverage entry must be true, including
measured zeros. No missing filter, position, time or tile is filled by zero or
CPU `W*x`. The three partials are summed **before one trained BN[t] and one LIF
step**. Physical command order must not add leak steps.

The manifest declares schema `full-conv1-measured-v1`, source `hardware`, stage
`pre_bn`, units `checkpoint_conv1`, shape, exact tap ranges, image/checkpoint
SHA, `bn_applied=false`, `cpu_replacement_used=false`, and calibration frozen
before acquisition. Payload, raw acquisition manifest and independent full-export
audit each require repo-relative paths and SHA. These structural fields are
necessary, not sufficient evidence of measurement. No trusted exporter auditor
is registered yet; current four-filter/ten-patch sessions cannot satisfy this
contract. Future integration requires raw-to-export validation, frozen decoder
provenance and an explicit code review to enable this boundary.

After validation, only Conv1 is measurement-derived; BN/LIF, pooling, Conv2/3,
FC layers and output decoders are digital trained operations. The separate CPU
reference remains available for comparison, never as fallback.

## Coverage and cost, without launching hardware

All64 filters contain4800 trained coefficients, used at784 spatial positions.
There are602112 pre-BN Conv1 values and1806336 logical tile partials per image
at T12. Five repeats require9031680 logical partials per image, or90316800
for ten images. These are **not native impact counts**: paired row reads, fresh
zero brackets, one-hot decomposition, physical microtiles, calibration, resets
and SET attempts add protocol-dependent I/O. No new physical budget or launch
is authorized by this estimate. Existing four-filter spot checks do not cover
these full tensors and cannot produce a measured classifier prediction.

## Validation scope

Nine tests compare all ten complete real-image CPU outputs with the unchanged
original forward/encoder, check CHW and spatial slices independently, enforce
late BN/LIF and negative/reset semantics, and reject incomplete/nonfinite/
misordered/unattested measured inputs. Float32 tiled accumulation may differ
from direct convolution by roundoff; tests require <=2e-6 Conv1 max error,
<=1e-5 output membrane/TWD error and exactly equal output spikes. Reports retain
the actual errors and predictions, including misclassifications.
