# MemriBoard experiment - simulator first, REV3

Configuration: `experiments/unn_rev3_20261014/protocol.json`.
Backend: native Connector with `cb_type=simulator`. Execution is not approved.
Every simulator, analysis program or hardware launch requires the user's explicit
permission. Editing this configuration does not grant permission.

## 1. Establish a reproducible simulator baseline

Use the compatible MemriBoard installation identified in the configuration, with
its normal Connector and BoardSimulator. Keep the board constants in
`simulator_settings.ini`; do not replace Application through `sys.modules`.
An existing simulation-array serial must be explicitly supplied. Never use the
user's existing `.cb` as scratch storage for programming; later write experiments
need separately approved, newly created arrays and new result directories.

Initial readback uses `(WL,BL)=(0,2),(2,6),(2,4)` so the same small address set
can later be compared with hardware. First read once per cell, review, then
collect 20 reads per unchanged state. This measures read repeatability only.

The simulator characterization specification uses q targets
`[0.01,0.02,0.03,0.04,0.05]`, 5 independent programming cycles per cell/target,
and 5 post-write reads per cycle. q means `330/R`, not siemens or WR-divider
weight. Preserve the historical 0.3-3.0 V, 0.01 V-step, 125 us staircase only as
a simulator reference. The REV3 write acquisition runner is not implemented;
this is not a claim that changing JSON updates the old frozen scripts.

Keep untouched-model results distinct from any later source fixes or fitted
models. Model defects are listed in `HARDWARE_REQUIREMENTS.md`.

## 2. Hardware bring-up, only as a separately approved stage

Switching to real hardware is an explicit decision, not an automatic fallback
when the simulator fails to load. Check the entire native driver path, active
board users, selected chip, timing, conversion and safe idle behavior first.
Changing `board_type` alone does not switch `cb_type`.

No whole-array scan: WL=1 is excluded. The three initial cells were recommended
by the operator, not newly measured here. Capture raw ADC, returned request ID,
latency, timestamp and any error before conversion. Do not instantiate HardCore
because its constructor scans all cells.

Readback comes before switching. A later switching approval must specify a
single cell, polarity, voltage, pulse duration and budget. Start with one pulse
and review its readback. Do not execute two unconditional 3 V RESET pulses.

## 3. Calibrate a physical programming window

The candidate resistance targets 2000, 1000 and 500 ohm are proposals within
the operator's reported range. They are not approved or proven reachable.
Determine SET/RESET direction, timing, compliance and a reproducible reset
endpoint before cycling. Record unsuccessful attempts and stop at the budget.

Proposed pilot: 3 cells x 3 targets x 5 independent cycles, with 5 reads per
cycle. This is 45 programming attempts plus reset operations and verification,
not 45 pulses. A held-out block of another 5 cycles per cell/target is reserved
for prediction validation, not tuning. Freeze the cell/target order before each
block. No automatic move from pilot to held-out acquisition.

Report bias, signed-error SD and RMSE per cell/target in siemens, alongside raw
resistance and ADC. Retain repeat-read SD separately. Freeze an achievable
conductance window from pilot data before collecting held-out results.

## 4. Differential mapping after characterization

For a qualified window `[G_lo,G_hi]` in siemens:

`G_plus/minus = (G_hi+G_lo)/2 +/- (G_hi-G_lo)*w/(2*maxabs(w))`

`w_hat = (G_plus-G_minus)*maxabs(w)/(G_hi-G_lo)`

Qualify every participating cell. The candidate WL pairs `(0,2),(3,4),(5,6)`
avoid WL=1 but are not evidence that those columns are usable. Retain the frozen
weight hash and filter order. Do not assume two independently selectable physical
arrays: physical inventory and addressing still need verification.

MVM is a separate qualification step. At low resistance, drive only a bounded
subset of active rows, check current and ADC headroom, then sum partial outputs
digitally. Report this read/latency cost. Existing Pearson/RMSE thresholds can be
shown as historical comparators; any new acceptance criteria must be frozen
before held-out validation. Do not optimize them using test-set outcomes.

## Deliverables and implementation boundary

REV3 supplies configuration, offline conversion/statistics and a thin native
simulator connection layer. It does not replace the old full-filter runner yet.
No new results have been produced. Every later run must save raw records,
configuration/source hashes, actual backend, array identity, pulse history,
approval scope and failures in a new session directory.
