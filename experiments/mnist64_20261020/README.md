# MNIST64: resident 64 inputs → 4 LIF

Run from repository root:

```sh
python -B experiments/mnist64_20261020/experiment.py --dry-run --preflight-source simulator
python -B experiments/mnist64_20261020/experiment.py --run-simulator
```

The same entry performs STDP128, freezes the physical weights, collects new128 readout-training and100 official test responses, fits NumPy ridge and prints accuracy. Progress distinguishes STDP from collection of test responses. Two UUID sessions are saved in `results/mnist64_20261020`: training and read-only accuracy. Simulator accuracy loads a copy of the training session's **final** array; the original remains immutable. Seed20261019 matches the preceding diagonal experiment.

Prepared hardware command, **only after the user reviews this simulator result and asks to continue**:

```sh
python -B experiments/mnist64_20261020/experiment.py --run-hardware
```

Every physical action uses public MemriBoard Connector, native converters and INI. There are no direct-driver calls or private-driver attestation gates. Current/voltage/native-response checks, finite budgets, exclusive hardware lock and public close remain. Hardware pulse duration is unknown; simulator uses100modelsteps (.0001s submitted).

Mapping: feature0..31 uses BL0..31/WL0..3; feature32..63 uses BL0..31/WL4..7. Each logical neuron adds both measured contributions **before one** LIF update. All256 cells are allocated/prepared/checkpointed, but blank image borders naturally contribute no events: training128 activates53/64 features (212cells). Updated-cell coverage is reported separately. There is **no separate control bank**.

Fresh independent label-free fit32 selects four gains from the fixed grid. No eligible gain stops the experiment without training/test. No test-driven adaptation. STDP is software eligibility controlling measured conductance writes, not proof of intrinsic device timing-dependent plasticity. After each image at most one update per logical neuron is allowed. Preparation≤2048pulses, training≤512, combined≤2560. Exact fixture caps: training57004calls; accuracy47908calls/zero writes; combined104912calls. Input voltage0.05V, at most4BL per public command. Start/end checks quantify read-only drift.

Readout is4×32spike raster→10classes, fixed ridgeλ1, train-only standardization, unpenalized bias. Train labels are loaded for fitting; the readout file is saved before test labels are opened. Primary strict accuracy keeps all100 examples in the denominator and treats silent/tied examples as incorrect abstentions; conventional argmax, coverage, Wilson interval and complete predictions/confusion are separate. This fixed-prefix100 test is a pilot, not full10000 accuracy.

Independent raw replay (repeat for each of the two UUIDs):

```sh
python -B experiments/mnist64_20261020/analyze.py --session results/mnist64_20261020/TRAIN_UUID --output-dir reports/2026-10-20/mnist64-simulator/train
python -B experiments/mnist64_20261020/analyze.py --session results/mnist64_20261020/ACCURACY_UUID --output-dir reports/2026-10-20/mnist64-simulator/accuracy
```

Replay independently recomputes64-dimensional traces, gain selection, native request ownership, both-tile sums, mapped updates, frozen-parent binding and ridge predictions. It never opens a Connector. Four focused tests cover mappings/caps, sum-before-LIF, raw mutation rejection, upper-plane write addresses and test-label ordering.
