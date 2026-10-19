# MNIST: parent STDP training contract and reconstruction limits

19 October 2026. Source review and protocol preparation only. The user's latest
instruction requests MNIST STDP training, so recovering a head from the existing
frozen checkpoint alone does not fulfill that instruction. The separately
prepared frozen-weight full-recovery route is not a replacement for this stage.

## What the archived parent actually records

Parent run `20260425T113337Z_5b89f013f42b` trained30,000 images and saved weights;
its summary has no readout result or label map. The next day's selected run
`20260426T111422Z_138b06a7f9b4` loaded that parent's
`model_after_train.pt`, skipped STDP, and reported69.65% after fitting a digital
counts/z-score/Linear readout. The saved parent and selected Conv1 tensors match;
their checkpoint containers need not have identical bytes.

Primary local evidence is under
`experiments/mnist_baseline_20261019/evidence/{training,evaluation}/` and
`experiments/mnist_baseline_20261019/reference_source/`. The latter source
snapshot is not attested to the actual historical executions. Original initial
weights/RNG, dependency lock, trained readout and normalization are missing;
exact accuracy reproduction cannot be promised.

| Parameter | Parent/source behavior |
|---|---|
| Samples | First30,000 MNIST train examples in dataset order, one pass, no shuffle |
| Online batch | **One image**, not the128-image evaluation batch |
| Window | T100, synchronous BindsNET run, reset neuron/trace/monitor state after each image |
| Budget |30,000 windows ×100 timesteps =3,000,000 online learning timesteps |
| Geometry |64×1×5×5 Conv1, stride1, padding0;64×24×24 LIF outputs |
| Input | Deterministic content-hash Poisson, base_seed123, rate_scale0.0015, boost1, TBNCHW |
| Plasticity | Native BindsNET `WeightDependentPostPre`, learning=True |
| Rates | `nu_minus=+0.001` is pre-event depression rate; `nu_plus=+0.0001` is post-event potentiation rate |
| Bounds | Native rule clamps weights to[0,1] after each update |
| Neuron | Threshold0.38, rest/reset0, membrane tau150, refractory2, dt1 from archived builder/defaults |
| Traces | Native Input/LIF traces enabled; source defaults tc_trace20, nonadditive trace reset |
| Initialization | Constructor's random draws, then builder uniform[0,0.3], then per-filter absolute-sum normalization to78.4 |
| Competition | EI requested22.5/120, but archived post-window hook fails before its EI branch |

The CSNN training loop is `agent_mvp/experiment_runner.py:152–249`. It calls
`net.run` for each `ds[i]`, reads spike monitor activity and resets state. The
FC branch's `snn_mnist_net.set_seed` must not be mistakenly attributed to this
CSNN branch: neither its runner nor `build_csnn` calls `manual_seed`. A configured
seed42 alone therefore does not recover the original initial RNG state. A new
fixed seed42 can be selected and saved explicitly as a reconstruction choice.

## Do not silently repair these semantics

The archived normalization hook references undefined `connection.w`; the
builder variable is `conn`. Its catch-all exception skips post-window
normalization, adaptive threshold and EI/WTA code. Initial normalization before
training does execute. The hook wraps an entire100-step run, not each timestep.
Changing its name or moving inhibition into the timestep loop is a new algorithm.

Initial normalization is particularly consequential: a filter has25 weights,
but target sum78.4 implies mean3.136, above the native upper bound1. Thus many
initial weights are clipped on the first native learning-rule update, independent
of STDP correlation. A large initial→final difference alone is not evidence of
learning. Do not silently clamp or renormalize before the first run to hide this.
Save the raw initialized W and first-step/first-window W evidence separately.

The inspected official BindsNET0.3.3 Conv2d rule computes pre depression from
post trace × current pre spike and post potentiation from current post spike ×
pre trace, with factors(W−Wmin) and(Wmax−W). Spatial contributions are summed;
the batch1 reduction is squeeze. Positive pre/post rates are intentional; do not
negate the configured `nu_minus`. The archived constructor passes
`weight_decay=1.0`; in this selected library it stores factor0 and skips the
guarded decay multiplication, then still clamps. It does not automatically erase
the weights. Keep the version pinned and disclose that semantics.

## Proposed bounded execution structure for review

1. Prepare a separate software-STDP route using unchanged archived builder and
   encoder plus the reviewed isolated environment. Assert learning=True, exact
   native rule/rates/bounds, normNone and unchanged threshold/bias before launch.
   Set and save a new explicit construction RNG seed/state; save initialized W,
   effective config, source/runtime/data hashes and intended sample IDs.
2. First use a separately reviewed small online-training pilot, e.g.128 sequential
   images with batch1/T100. The completed frozen128 feature pilot used
   learning=False and cannot qualify the online update path. Save per-window
   input/spike counts, pre/post W hashes, finite/bound checks and cumulative
   coverage. Preserve lack of changes or silence as outcomes; do not tune rates.
3. A first-window native connection-weight monitor can save100×1,600 float32
   values (640,000 bytes) to separate initial clipping from later plasticity,
   without changing learning equations. Its buffer and sampling semantics must
   be reviewed. Do not retain all3million weight states by default: that alone
   would be19.2GB and is outside this proposed bounded evidence scope.
4. After online-pilot validation, full training starts from the same explicitly
   fixed fresh initialization and processes exactly IDs0–29,999. Do not silently
   continue from the pilot and repeat its first128 samples. If resuming is later
   supported, its checkpoint, RNG, trace reset and next-ID contract need review.
   Save checkpoints/progress periodically, stop on nonfinite weights or errors,
   and never promote partial coverage to completed30k training. Exact wall/storage
   budgets remain to be agreed with the implementation author before approval.
5. Freeze the newly trained weights and use them, rather than the old archived
   W, for train30k/test10k feature extraction and the fixed readout contract in
   `../mnist-full-recovery-review/ARCHIVE_CONTRACT.md`: train-only sample-SD
   normalization, Linear, SGD25epochs and one final test. Save the new checkpoint,
   head/scaler and test outputs. No test-based selection of STDP settings.

Historical parent wall time was2h30m46s on its former environment. That is not a
current ETA. The recent128-feature timings are not an STDP benchmark: training
uses batch1 and executes weight updates, while the pilot batched128 frozen
examples. A new bounded online pilot is needed for a meaningful estimate.

This is BindsNET software training, not measured physical STDP or a memristor
accuracy result. Every future real hardware operation must use public
`manager.board.Connector`, native converters and driver selection via INI.
GPIO/SPI/serial/FPGA, direct/internal driver, `.interface` and dynamic-driver
workarounds are forbidden. If MemriBoard cannot express an operation, report
the limitation instead of bypassing it.

## Author contract review

Reviewed `experiments/mnist_stdp_retraining_20261019/PROTOCOL.md` at SHA256
`f728353ee4eb8208b7a705c1c380c3b0795abdcddfb8c03e007f6a906f9d979f`.
It preserves the above semantics and proposes separate closed128 smoke/full
gates, all end-of-image weights (192MB for30k) and a first-window weight trace.
The suggested8h wall limit is explicitly pending smoke measurement before
registration. No scientific blocker to implementation was found; this is not
an execution approval or final source review.

The selected native `Monitor` accepts connections and records after the native
rule update. Thus its first-window W trace should be labeled post-update[t].
In a fresh zero-state network with zero bias and synchronous input, Conv1 sees
zero preceding Input.s at t0, so no target spike/trace contributes to plasticity
at that step. First post-update W should equal `clamp(initialW,0,1)`; recording
and checking this provides a useful separation of initial clipping from later
weight changes. CPU monitor copies require CUDA synchronization before reading.
