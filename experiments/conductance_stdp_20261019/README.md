# Conductance STDP pilot implementation

Implementation of [PROTOCOL.md](PROTOCOL.md), independently reviewed for native
SIM qualification. The earlier design review is in [REVIEW.md](REVIEW.md).
The simulator gate contains the reviewed protocol and pins core/raw source bytes;
the hardware gate remains empty pending review of actual native SIM evidence.
Do not overlap this pilot with an active hardware CIFAR session.

Only public `manager.board.Connector.open_port/impact/close_port` performs I/O.
Standard INI selects the driver; native converters are unchanged. Digital LIF
and all-event STDP eligibility run after each fresh bracket; updates occur only
at training-window boundaries. This is algorithmic STDP, not a demonstrated
intrinsic pulse-timing law or classification experiment.

This rule applies to every real read, diagnostic, SET, RESET, MVM and STDP
operation. Direct GPIO/SPI/serial calls, physical-driver imports, dynamic module
loading, `Connector.interface` access and custom hardware bypasses are forbidden.
The Connector alone selects and calls its driver. If its public interface cannot
perform a required operation, report that limitation instead of bypassing it.

Repository-root dry-run (no Connector opened):

```text
python experiments/conductance_stdp_20261019/experiment.py --dry-run --preflight-source simulator
```

The root agent organizes native SIM qualification using:

```text
python experiments/conductance_stdp_20261019/experiment.py --run-simulator
```

Hardware requires separate root execution after native SIM review and closure
of other Connector sessions:

```text
python experiments/conductance_stdp_20261019/experiment.py --dry-run --preflight-source hardware
python experiments/conductance_stdp_20261019/experiment.py --run-hardware
```

Outputs are new UUID directories under `results/conductance_stdp_20261019/`:
source/INI/protocol/environment manifests, initial SIM array hash, complete raw
request/response gzip (including invalid native responses and driver output),
checkpoint/LIF/eligibility observations, pulse decisions and current skips.
Partial failures retain partial observations and raw index, marked failed.
All paths are repository-local; no previous session is overwritten.

Complete schedule invariants:232 cell checkpoints,28 windows,896 bank LIF steps,
4,776 mode10 calls. With U training attempts, A admitted preparation pulses and
B admitted training pulses: total impacts=6,456+A+2U+B <=6,712;
total pulses=A+B <=128. Each training attempt has paired control R read followed
by fresh training-cell R; skipped preparation decisions still have checkpoints.
The retention interval is60 measured seconds without board I/O.

HW pulse duration is unknown (submitted width0); SIM uses0.0001,100 modeled
steps. Fixed gain100/volt may silence SIM or saturate HW. Both are descriptive
outcomes; no automatic gain, amplitude or threshold adjustment. Identity
calibration only; initial states are measured separately, not assumed equal.

Raw cap64MiB expanded, gzip level1,2GiB initial free-disk requirement and1GiB
reserve. Raw requests and results have wall-clock/monotonic timestamps and
sequential task IDs. Protocol SHA binds parameters; source hashes bind schedule.
All raw is retained even when a valid measurement gives no useful learning.

Offline tests (no driver):

```text
python -B -m unittest tests.test_conductance_stdp -v
```

They check analytic trace/eligibility cases, negative LIF inputs, hard reset,
exact worst-case budget, control matching, skip behavior, raw preservation,
failed open/close, current guard, and cleanup after logging failure.
