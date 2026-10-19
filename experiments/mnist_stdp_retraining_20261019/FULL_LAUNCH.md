# Separate full-launch qualification

`run_full.py` is a new launcher; the executed smoke acquisition `experiment.py`
SHA7f2a35d9… is unchanged. It imports that verified module normally and calls
its public `execute(..., smoke=False, ...)` after its own independent full-run
gate. It never modifies the old gate, injects code or monkeypatches behavior.

The new FULL_SPEC binds the frozen training/readout specification, exact
successful smoke manifest and independent actual-data review SHA. Before any
network construction, the wrapper verifies every qualifying smoke artifact,
the helper source, parent config, environment and data. The independent actual
smoke review and full source-path review passed; this separate fixed full-run
gate is now registered. Root coordinates the launch after commit/preflight.

Fresh seed42 training starts at image0, not from smoke/archived weights.
The tested native STDP routine trains30,000 batch1/T100 windows, saves its new
checkpoint, then performs the separate full feature/readout path. That full
path has not yet been executed; qualification must distinguish its source/
synthetic tests from actual smoke evidence. No historical69.65% promise.

The full route uses8h wall budget,8GiB output limit,18GiB initial disk reserve;
8GiB available host RAM and4GiB GPU free are preflight requirements. These
limits are unchanged from the frozen helper. Runtime extrapolation and
limitations are in the saved smoke author/independent reports.

Unlike the smoke's old manifest, `planned` contains preflight's1,475 head
updates,314 feature batches and5.90GB counts. `confirmed_execution` is derived
only from completed events. If interrupted mid-epoch, the confirmed head count
means updates at the last completed epoch, not an invented exact partial count.
Successful final `head_updates` comes from the actual completed training code.

```text
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_stdp_retraining_20261019/run_full.py --preflight
```

Only after actual-data review, full-path source review and registration:

```text
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_stdp_retraining_20261019/run_full.py --run
```

The old `experiment.py --run` remains closed; this separate launcher is the
reviewable full-run entrypoint. Only root coordinates model launches.
