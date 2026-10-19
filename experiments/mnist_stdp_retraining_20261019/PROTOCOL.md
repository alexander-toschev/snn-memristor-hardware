# MNIST STDP retraining and evaluation — proposed contract, not launched

Repeat the parent training run `20260425T113337Z_5b89f013f42b`, then evaluate
**the newly learned weights**. The selected archived69.65% was a later readout
evaluation of that parent. Neither loading its old weights nor training only
a new Linear head satisfies this experiment.

This is GPU/BindsNET model STDP, not physical memristor STDP. All later physical
operations must use public MemriBoard Connector, native converters and INI
driver selection. No board/nativeSIM access belongs to this software route.
No existing frozen CIFAR/STDP/MNIST128 source is modified.

## Exact archived evidence

The SHA-pinned sources are under
`experiments/mnist_baseline_20261019/reference_source/`:

- `agent_mvp/experiment_runner.py`: CSNN branch builds a fresh model, accesses
  `ds[i]` for i0…29,999, runs each image for100 timesteps and resets afterward.
  There is no DataLoader/shuffle/minibatch during STDP. `resume_checkpoint`
  would skip training and therefore must be None for this experiment.
- Baseline-root `evidence/training/summary.json` (outside reference_source,
  SHA182dc87cbcb9ab1a497b6c6290f9abea0602e657a30081faf79df62e79677e71)
  records positive `nu_minus=0.001` and
  `nu_plus=0.0001`. Use these parent overrides, **not** defaults copied from the
  later evaluation checkpoint (whose default nu_minus is negative).
- `csnn_mnist_net.py` defines CSCfg defaults not expanded in summary. Preserve
  these exact defaults through a verified ordinary import and save the full
  effective config. No hidden hyperparameter correction.

## Proposed fixed training protocol

| Item | Value |
|---|---|
| Dataset | Verified preserved MNIST train IDX, IDs0…29,999, one pass |
| Batch/time budget | Batch1;30,000 windows×T100=3,000,000 model timesteps |
| Architecture | Input1×28×28;Conv1=64×1×5×5,stride1,pad0;64×24×24 LIF |
| Initialization | Fresh archived builder; no loading old pretrained weights |
| Input | ToTensor-equivalent float32/255, deterministic Poisson rate.0015,base_seed123,boost1,TBNCHW |
| LIF | threshold.38,rest/reset0,refrac2,tc_decay150,dt1 |
| Traces | Both input/output enabled; default tc_trace20,nonadditive,trace_scale1 |
| Learning rule | Native BindsNET WeightDependentPostPre, learning=True |
| Rates/bounds | pre-depression+.001;post-potentiation+.0001;wmin0,wmax1 |
| Execution | Archived synchronous one_step=False;reset state after each image |
| Seed | Explicit new seed42 for Python/NumPy/Torch/CUDA construction |

The archived CSNN training branch does not actually invoke set_seed despite
having cfg.seed42. Its initial RNG state and dependency versions were not
attested. Explicit seed42 and the separately pinned BindsNET0.3.3/PyTorch2.4.1
environment therefore define a **new reconstruction**, not bitwise replay.
Use the isolated `.venv-mnist-pilot` without modifying it.

The update implemented by native WeightDependentPostPre is, before clipping:

```
ΔW = -0.001 × sum_spatial(post_trace × pre_spike) × (W - 0)
     +0.0001 × sum_spatial(post_spike × pre_trace) × (1 - W)
W ← clamp(W + ΔW, 0, 1)
```

Trace timing/order is delegated to the pinned native rule and Network.run;
the new entrypoint must not reimplement or reorder learning updates.
The archived weight_decay argument1.0 becomes an internal zero multiplier that
the native rule skips; it must not be described as erasing weights each step.

## Initialization and broken E/I hook must remain visible

The builder samples uniform[0,.3], then normalizes each filter's25 weights to
L1 sum78.4. Mean initial weight is3.136, above wmax1. The first native update
clamps weights into[0,1], independently of correlation-driven plasticity.
Do not pre-clamp, normalize to1, or silently change78.4. Save initial weights
before any step and native per-step weights for the first sample (100×1600
float32 =640kB), separating initial clipping from later STDP changes. Native
Monitor records after each rule update. Fresh synchronous zero state means
postupdate[0] must equal clamp(initialW,0,1), verified exactly; final recorded
W must equal post-window W, exposing unexpected normalization.

With `w_norm_enable=True` and `ei_enable=True`, the post-window hook tries
undefined `connection.w` and catches the exception. Its later normalization,
E/I22.5/120, WTA and adaptation operations are skipped. The inhibitory layer
exists but no E/I connections are installed. The hook follows a complete
net.run, not each timestep. Preserve this archived bug, explicitly describe
effective behavior, and never inject a global `connection` alias.

## Evidence and gates

Proposed simple entrypoint `experiment.py` should expose:

1. `--preflight`: hashes/config/environment/budgets only; no network run.
2. `--smoke-128`: separate reviewed train-only prefix128 from a fresh seed42
   initialization, T100/batch1; verify learning flags/rates, finite bounded
   weights after updates, actual changes beyond initial clipping, threshold
   invariance, reset/order coverage. Save all128 end-of-image weights and
   first-sample per-step weights. No test/readout or promotion to full training.
3. `--run`: separate reviewed full30k fresh training; final checkpoint saved
   before freezing, then new-weight feature extraction and readout evaluation.

Both run gates must remain closed until independent source/scientific review.
A model smoke is needed because the prior128 pilot had learning disabled and
does not qualify native STDP updates. Full training must not continue from
the smoke's weights or reuse old pretrained weights.

For each training image save ID,input-spike SHA/count,output-spike count,
weight SHA/min/max/change statistics, elapsed time and checkpoint trajectory.
Store every end-of-image W in float32.npy (30,000×1600×4=192MB) plus first-image
per-step weights, initial/final weights, thresholds, config, source/environment
hashes and bounded event records. A zero-learning result is reported, not fixed
by changing rates midway. CUDA synchronize must precede CPU Monitor reads.
No fallback from failed monitor reads to last-step spike counts.

## Evaluation of newly learned weights

After30,000 images, freeze the exact newly trained W and all thresholds; do
not reinitialize, normalize, load archived pretrained W or run more STDP.
Use the validated frozen extractor semantics, train30k/test10k in batches128
(last48/16), T100, full36,864 counts. New trained W receives its own SHA; it
must not be checked against the old archived weight hash/pilot counts.

Fit log1p train-only mean/sample SD(correction1,floor1e-6); bounded float64
Chan statistics over float32log1p, cast savedfloat32, are a declared numerical
reconstruction difference from full-array float32 reductions. Linear+bias,
SGDlr.001/wd.0001,CEmean,25epochs,batch512,shuffleTrue,seed42,1,475updates.
Fix finalepoch25 before test; evaluate10k test once. Save scaler,head,shuffle
orders,RNG,predictions/logits/confusion and all artifactSHA. No test tuning.

The unexecuted `mnist_full_recovery_20261019` draft may supply reviewed scaler/
readout functions, but its old-checkpoint loader and firstbatch-old-pilot SHA
must never serve as the new-trained-weight extractor. A distinct bridge must
accept only the just-created training checkpoint with verified provenance.

## Resources and qualification

Training historical elapsed time was2h30m46s on an unattested environment;
current batch1/nativeSTDP performance is unknown until the new128 smoke.
The earlier frozen batch128 timing does not estimate training. Suggested full
wall budget8h, checked between bounded windows/stages; no asynchronous cutoff
inside a CUDA step. Final budget needs measured smoke before registration.

End-of-image training weights~192MB; full counts5.90GB; total target<8GiB raw
artifacts and18GiB free disk including LFS duplication. Retain8GiB free host
RAM for feature Monitor/aggregation and4GiB GPU free at preflight. All binary
artifacts use Git LFS and repository-relative paths. No new full model run,
STDP smoke, head training or hardware operation has been executed here.
