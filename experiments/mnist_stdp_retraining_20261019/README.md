# Fresh MNIST STDP → readout of new weights

The implementation follows [PROTOCOL.md](PROTOCOL.md). Only the fixed128 smoke
is registered after independent scientific review. The full30k/evaluation gate
remains closed. No SNN/model was executed during
implementation. This is BindsNET/CUDA learning, not a physical board run.

The archived parent config supplies positive pre-depression.001 and
post-potentiation.0001; training starts from fresh seed42 initialization,
never from the old pretrained weights. Native WeightDependentPostPre performs
updates. The first image's post-update weight monitor verifies pure clipping
at step0 separately from subsequent plasticity and checks that final-window
normalization did not change the last recorded weights. Every image logs
input/activity/weight hashes and saves its entire1600-coefficient weight state.

The128 smoke trains on IDs0…127 with batch1/T100, saves a new checkpoint, and
stops without readout/test. Full training independently restarts seed42 from
ID0 and processes30,000 images. It then freezes exactly those newly learned
weights for30k/10k features and25epoch digital readout. It never imports the
old frozen-feature collector or its old-weight/firstbatch acceptance checks.
Only bounded scaler/readout functions from the retained recovery draft are
reused; that helper source is SHA-pinned. The separate successful128 pilot
provides unchanged environment/source checking and state-hash utilities.

All source/checkpoint/data paths are relative to the repository. `.npy`,`.pt`
and event artifacts are covered by Git LFS. No output overwrites existing
archives or runs. Failed results remain explicitly failed and cannot supply
accuracy. Training timeout30minutes(smoke)/8hours(full) is checked between
windows; timing from the smoke will qualify full-budget expectations.

Preflight constructs no network. The shared source check hashes both train and
test compressed IDX files; this is provenance access, not extraction or test
evaluation. The smoke itself consumes only train IDs0…127. Saved
`construction_rng_*` states are captured after fresh model construction;
explicit seed42 and the recorded initial weights define this reconstruction,
not the unattested historical initialization.

```text
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_stdp_retraining_20261019/experiment.py --preflight
```

After separate review/registration, root can launch the fixed smoke:

```text
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_stdp_retraining_20261019/experiment.py --smoke-128
```

Full training has an independent gate and must not launch before smoke review:

```text
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_stdp_retraining_20261019/experiment.py --run
```

No automatic fallback, changed rates, fixed E/I bug or retraining after seeing
test is implemented. All later real hardware operations remain MemriBoard-only.
