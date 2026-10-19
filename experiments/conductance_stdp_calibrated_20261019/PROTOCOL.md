# Conductance STDP v2: independent fit-only digital gain

Prepared19 October2026 after the original `conductance_stdp_20261019` native
SIM completed with no post spikes or training pulses. This is a new version
motivated by those results, not a reinterpretation of the old run. Old sources
and results remain unchanged. Both native execution gates start closed; root
performs native SIM only after review/commit, and hardware only after the active
CIFAR experiment and successful independent review of new native evidence.

Every hardware operation uses public `manager.board.Connector`, its native
converters and driver selected through INI. Reads, diagnostics, SET/RESET and
MVM all obey this rule. Direct driver, `.interface`, GPIO/SPI/serial/FPGA,
dynamic-driver loading or a custom bypass is forbidden. If the public interface
cannot express something, report the limitation.

## Unchanged physical protocol

The new collector inherits the SHA-pinned v1 public-Connector I/O functions:
`call`, `read_r`, `pulse`, `vector`, `bracket`, `checkpoint`, `all_cells`. It does
not replace the driver, converter, current guard or native reply validation.
Ordinary Python imports and source SHA verification are used; no runtime module
injection or monkeypatching belongs to acquisition.

- BL[6,8,9,14], trained WL5 and no-learning control WL7:8 cells.
- MVM amplitude0.05V, at most one active BL. Each response is a fresh
  zero/signal/zero bracket, with fresh nativeR before every active signal.
- SET1.7V/sign1, RESET1.4V/sign0. No additional writes or amplitude search.
  Preparation is eight decisions per cell: BL6/8 fourSET then fourRESET;
  BL9/14 fourRESET then fourSET. Alternate bank order by decision+BL index.
- Before every write, fresh nativeR and GUI guard: RESET≤40mA, SET≤2mA.
  Before MVM, current≤5mA and330×current<3.3V. NativeR/ADC validity and measured
  MVM ADC≤4V remain unchanged. Invalid measurements stop globally; a blocked
  write is explicitly skipped, not fabricated as a pulse or success.
- Training writes only WL5; each attempted update has a paired control guardR
  before the immediately adjacent trained guardR. The control receives the
  same preparation but no training pulse. This matches explicit guard reads,
  not the treatment pulse's internal readback dose.
- Native SIM width0.0001s=100model steps; hardware submits0 and physical pulse
  duration is unknown. These are not equivalent certified pulse durations.
- Per-cell checkpoint:3 nativeR reads and3 guarded zero/signal/zero MVM
  brackets,15 impacts. End-to-end raw responses/commands/errors are retained.

## Changed digital protocol, fixed before any new acquisition

Replace the four old pretest windows with two **fit** windows plus two pretest
windows. Total windows remain28:2fit+2pretest+16train+4posttest+4retention.
Each has16 timesteps and one pre event per BL. Every timestep measures both
banks, alternating WL order by timestep parity. State resets per window.

| Phase | Even window event times for BL[6,8,9,14] | Odd window times |
|---|---|---|
| Fit |[1,5,9,13]|[13,9,5,1]|
| Train |[3,10,11,2]|[2,11,10,3]|
| Pre/post/retention probe |[1,4,9,12]|[9,12,1,4]|

Train order is intentionally different from v1; the close pairs reverse
relative pre timing. This is software event timing, not control of physical
pre/post pulse overlap inside a memristor. No target labels or CIFAR/MNIST
weights enter this protocol.

After collecting both fit windows, replay **only WL5 fit measurements** for
each fixed digital gain[10,20,40,80,100,200,400,800,1600] V⁻¹. All nine candidates
and their32-step trajectories are saved. WL7 fit measurements are retained
but excluded from selection. The lower candidates were preregistered before
this run because earlier hardware responses were larger than native SIM;
the grid must not be extended after seeing new fit or heldout results.

For every candidate, LIF is `u=0.5*u+gain*centered_voltage`; a post spike occurs
at u≥1 and hard-resets to0. Negative inputs are retained. An eligible gain
produces1–3 posts in **each** of the two fit windows, and no post at any
zero-input timestep. Among eligible gains choose the minimum tuple
`(abs(total_fit_posts-4), gain)`. This unique gain is frozen before pretest and
training and used identically for both banks throughout the rest of the run.

If no candidate qualifies, close the Connector and preserve the complete
prefix with status `uninformative_calibration`, exit3. No pretest or training
occurs, and no grid expansion/retry occurs. Preparation may already have
changed up to64 cell states by pulses; this outcome does not imply no writes.
Eligibility is not guaranteed on either device, especially after CIFAR writes.

Fit measurement rows have no fabricated post/eligibility values. Their
`fit_measurement_step` records retain raw bracket references; candidate digital
replays are separately marked calibration evidence. The26 subsequent windows
have832 actual `lif_step` records. Candidate evaluation does not add physical
impacts or replace any later measured contribution with a basis lookup.

## Logical STDP and outcomes

As in v1, traces decay byexp(−1/4) each logical timestep. Eligibility increments
are `pre_trace_i*post -1.05*post_trace*pre_i`, then current pre/post events enter
their traces. Zero-lag pairing is excluded; all retained prior events contribute.
After each training window, each WL5 cell receives at most one attempted update
when |eligibility|≥0.25: positiveSET, negativeRESET. No fullRESET is inserted
between these updates. All decisions, measured skips and native replies remain
separate evidence. Pretest/posttest/retention never choose or revise the gain.

Each training window is followed by all-cell checkpoints; posttest and a60s
retention interval are followed by the same measurements. Preparation checkpoints
follow every decision. NativeR changes can be linked to individual admitted
pulses; training MVM changes are per-window aggregates, after up to four writes.
They cannot establish an isolated per-pulse MVM effect or remove drift causality.

Report separately: complete physical coverage; gain-fit eligibility; heldout
posts/silence and one-post-per-input saturation; causal/anticausal eligibility;
attempted/admitted/skipped learning pulses; per-cell nativeR/MVM changes and
control differences. Calibration does not prove stable conductances, effective
learning or network accuracy. Even an eligible fit can lead to silent/saturated
training or probes; preserve those outcomes without retuning. This is
algorithmic boundary-updated conductance STDP, not intrinsic physicalΔt-STDP.

## Exact bounds and evidence

Let A≤64 be admitted preparation pulses, U≤64 training attempts, B≤U admitted
training pulses. A complete run has:

- **6456+A+2U+B impacts ≤6712**; admitted pulses A+B≤128.
-4776 mode10 impacts,232 cell checkpoints,28 windows.
-64 fresh fit measurement steps and832 non-fit LIF steps.
-288 offline candidate steps (nine gains×two windows×16), no extraI/O.
- Control learning pulse count0; same frozen gain on both banks.

No-eligible fit prefix has1472+A impacts≤1536,912 mode10,80 checkpoints,
two fit windows and no evaluated LIF/train windows. Errors may stop earlier;
those are failed/partial sessions, not valid uninformative-fit outcomes.

Raw writer is the unchanged pinned v1 writer:64MiB expanded cap,2GiB initial
free-space check,1GiB periodic reserve, gzip plus indexed SHA. Outputs use a
fresh repository-relative UUID directory. Source/helper/INI hashes, config,
runtime versions, raw trace, observations and all result hashes are preserved.
An independent auditor must reconstruct candidate gain selection from fresh
fit rawADC, verify gain freezing/order and both banks, replay STDP decisions and
signs, and reconcile every native call/budget. Fake test trajectories qualify
code only; informative native SIM evidence must come from actual driver data.

```text
python experiments/conductance_stdp_calibrated_20261019/experiment.py --dry-run --preflight-source simulator
python experiments/conductance_stdp_calibrated_20261019/experiment.py --run-simulator
python experiments/conductance_stdp_calibrated_20261019/experiment.py --run-hardware
```

Root registered the reviewed canonical protocol for **native SIM only** after
scientific review and7/7 offline tests. Hardware registration remains empty.
Root launches native SIM after source freeze, independent auditor preparation
and commit; registration is not evidence that acquisition occurred. No native
simulator or hardware run was executed while preparing this source.
