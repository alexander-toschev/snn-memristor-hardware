# Independent MNIST frozen-feature pilot review

19 October 2026. This is a source review, not a model launch, runtime benchmark,
accuracy measurement or approval of hardware execution. The proposed first-128
train-only pilot is a new reconstruction. The saved 69.65% result lacks its
trained Linear head, train normalization, exact RNG state and an attested runtime
source/dependency snapshot.

## Archived semantics to preserve explicitly

- `reference_source/csnn_mnist_net.py:313–321` initially normalizes random Conv1
  weights when configured. Loading the checkpoint at lines156–170 subsequently
  copies saved `W` and threshold over that initialization. Do not normalize or
  clamp the loaded weights again in the pilot.
- The post-window hook calls the original complete `net.run` before its own
  body (lines247–249). With `w_norm_enable=True`, its undefined `connection.w`
  at line253 raises; the broad exception at line299 suppresses the error. This
  skips post-window normalization and the later E/I, adaptive-threshold and WTA
  branches. The builder defines `conn`, not `connection`. Preserve the archived
  source unchanged and do not inject a `connection` module global. Initial
  normalization is not disabled by this bug. Even a repaired hook would be a
  post-window hook, not timestep-level E/I.
- Evaluation summary selects T100, deterministic Poisson, rate scale0.0015,
  boost1, 64 filters, 5×5 kernels and no padding. `evaluation.py:285` uses
  batch128, dataset order and `shuffle=False` in the collector. The pilot uses
  train IDs0–127 only; no test images or head fitting are needed.
- `encoders.py:149–198` uses a content-derived integer hash with base seed123
  when deterministic=True. It is not a call to `torch.manual_seed(123)`.
  Pixels normalize to [0,1], with probability `1-exp(-pixel*0.0015)` per step.
  Boost1 keeps this encoder branch; the collector's boost branch instead uses
  `torch.rand`. Record actual input-spike SHA and counts, not only a seed label.
- Counts are monitor spikes summed over T and flattened in C,H,W order.
  `evaluation.py:330` keeps only the first N=36,864 count columns for the chosen
  Linear pipeline. The ten WTA histogram columns are not classifier inputs.
  Original collection resets network/monitor state after each batch. A fresh
  reset before each pilot repeat should be explicit and checked.

All `reference_source` paths above are under
`experiments/mnist_baseline_20261019/`. Their manifest and source hashes must be
verified before ordinary imports. The source snapshot's relation to the exact
69.65% execution remains unproven; preserving it does not remove that limitation.

## Frozen-state contract for the new pilot

The official BindsNET0.3.3 source archive at commit
`1b9786152db6fc67a38d2ddb3090ac434bca239f` was inspected locally in
`.tmp/mnist-pilot-source/inspect/` without executing a model:

- `Network.train(False)` sets `learning=False`; `run` passes it to connection
  updates. Assert this state on the network and all layers. Use explicit
  `torch.no_grad()` and zero learning rates as additional checks.
- Connection `update(learning=False)` bypasses the learning rule. Historical
  zero rates alone do not do so: a rule may still clamp weights. In this source,
  `weight_decay=1.0` stores zero and the guarded multiplication is skipped;
  claiming it necessarily erases weights would be incorrect.
- `Network.run` invokes connection normalization after every window even in
  evaluation mode. Assert every connection's `norm is None`; do not silently
  change it to obtain a passing result.
- Keep the default synchronous `net.run` input semantics (`one_step=False`).
  Conv1 sees the prior Input spike state, so replacing it with a same-step
  convolution would change the extracted features.
- Record exact tensor hashes and `torch.equal` invariants for all connection
  weights and all thresholds before/after every repeat. Check baseline equals
  the restricted-loaded checkpoint tensor, not merely a postprocessed copy.
  Save the effective config, package/runtime/source hashes, device, input IDs,
  labels, counts, finiteness, silence, elapsed time and peak memory.
- Two runs of the same128 inputs after state reset should have identical input
  spikes and count tensors. A mismatch is reported and investigated; do not
  switch batch size, encoder, thresholds or device silently. Silent counts are
  an uninformative feature pilot, not successful recognition.

Explicit `learning=False` is stronger than the archived collector's zero-nu-only
freeze. It is an intentional documented reconstruction choice, not evidence of
exact historical reproduction. Historical readout/head reconstruction remains
separate; no accuracy follows from passing frozen-feature invariants.

Implementation review is pending the author's final files. No128-image or full
model run, training, native simulator or hardware call was made for this review.

## Initial implementation review

The first draft correctly uses restricted checkpoint loading, train-only pinned
IDX files, ordinary imports of unchanged archived modules, fixed128×T100,
explicit learning=False, repeated state reset and exact input/count/weight hashes.
It does not train a head or expose test evaluation.

A blocking lifecycle issue was reported to the author: its run uses
`torch.inference_mode()` but resets BindsNET state after leaving that context.
Official `Input.forward` assigns `self.s=x`; `LIFNodes.forward` replaces `self.v`
and `self.s`. These can therefore be inference tensors, and reset modifies them
in place. A one-element tensor-only check on the existing CPU Torch runtime
confirmed that this raises `RuntimeError: Inplace update to inference tensor
outside InferenceMode is not allowed`. No network was executed. Use
`torch.no_grad()` for this stateful path, or consistently include its entire
mutable state lifecycle in the inference context; the former matches the
archived collector's approach.

Other requested hardening before approval: verify imported BindsNET source bytes
against the selected official archive; assert layer learning flags and exact
connection inventory; bind loaded threshold and bias baselines; explicitly
classify all-silent features as uninformative. These are pending checks, not
claims that the final implementation contains these defects.

## Corrected implementation verdict

Reviewed `experiments/mnist_feature_pilot_20261019/experiment.py` SHA256
`ddb138fb9696a1c8301252d1e6675c582e781412e5be4d12082d6ea72660a00f`
(before root approval registration). The author resolved the lifecycle issue
with `no_grad`, verified all48 official BindsNET Python source files, asserted
all layer learning flags and exact connection inventory, and bound checkpoint
weight/threshold and zero bias baselines. Full state hashes are rechecked before
and after each pass; all-silent counts have an explicit uninformative flag.

A second reporting defect was found and fixed: the completion result initially
overwrote the detailed `passes` list with integer2. It now returns
`completed_passes=2`, preserving per-pass input/count/weight SHA, spike counts,
time and memory measurements in the final manifest.

Independent source-only tests `tests.test_mnist_feature_pilot`: **4/4 pass**.
They validate source pins, train-only prefix/shape handling, corrupt input
rejection and the closed launch gate; they do not establish model execution or
CUDA correctness. The author separately reports a successful CUDA dependency
preflight, not a network run. Environment lock/provenance completion and root
registration remain separate steps.

No remaining source-review blocker for this bounded first128-train, twice-reset
feature pilot. This verdict does not approve larger extraction, head training,
test evaluation or hardware. The pilot must still demonstrate actual invariant
and repeatability success; runtime, memory sufficiency, non-silence and accuracy
are not inferred from these source-only tests.
