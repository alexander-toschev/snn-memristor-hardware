# Two-crossbar plan — four simulator phases

Entry point: `experiment.py`. The user explicitly approved all four phases using
MemriBoard's simulator backend. This is a software rehearsal of the original
four-phase plan, not a real-hardware experiment. Only local files are required.

## Preparation and initial readback

Verify hashes of `weights/conv1_weights.npy` and `weights/best_fs.pt`. Freeze settings,
source hashes and seed 20261014 before measurements. Create two new 32-BL x 8-WL
arrays A/B inside the session directory and connect native `manager.board.Connector`
with `cb_type=simulator`, `board_type=rp5_fpga_python`, `task_format=SI`.
Read (WL,BL)=(0,2),(2,6),(2,4) once, then 20 times. WL=1 is excluded throughout.

## Phase 1 — write precision

Three initial cells x q targets [.01,.02,.03,.04,.05] x 5 programming cycles.
Each attempt: read, bounded RESET staircase until q<=.005, bounded SET until the
first crossing of target q, then five readbacks. Both staircases: .3 through 3.0 V,
.01 V step, 125 us, at most 271 pulses each. No unbounded retries.
q=330/R is an MVM gain; G=1/R is siemens. Report signed bias, sample SD (ddof=1)
and RMSE per cell/target across cycles. Repeat reads are not independent writes.
Count failed reset/programming attempts explicitly. Exclude failed resets from
independent-cycle error SD; retain their raw measurements and attempt records.

## Phase 2 — 14 stratified filters, K=1

Frozen order: 63,41,32,30,38,43,60,47,2,9,46,12,0,1. Each filter has 75 taps.
Use tiles [0:32], [32:64], [64:75] on arrays A,B,A respectively, WL pair (0,2).
Reuse the same arrays; preserve each tile's readback/MVM before overwriting it.
Differential mapping q+/-=.03 +/- .02*w/maxabs(w); inverse w_hat=(q+-q-)*maxabs/.04.
Use the same bounded reset/programming procedure, then three reads per cell.
This historical simulator window is not a qualified physical conductance window.

For every filter, compare readback weights and native MVM outputs with the ideal
weights. Inputs: zero, all 75 basis vectors and 16 fixed-seed sparse vectors with
8 active taps. MVM input amplitude .1 V, unused rows zero; account for actual DAC
rounding. Sum the three tile outputs digitally. Compare MVM both against ideal
weights and against readback reconstruction. Raw ADC is the MVM measurement;
Connector's single-cell resistance return is inapplicable to summed current.
Historical comparator: Pearson >=.9 and RMSE<=.6*SD(expected), not a newly proven
acceptance threshold. Record failures and continue this simulator survey.

## Phase 3 — all 64 Conv1 filters

Apply the same procedure independently to filters 0..63, K=1. Save every filter,
including failures. Earlier phase-2 observations are not substituted for this run.

## Phase 4 — mean-of-K

Preselected difficult filter 41: three separate reprogramming attempts.
Successful RESET is recorded; statistical independence is not established.
Evaluate prefix means K=1,2,3 of reconstructed weights and measured MVM outputs.
Report actual RMSE/Pearson and failed resets; do not assume variance reduction or
recovery. This small demonstration does not estimate a population-level K law.

## Deliverables and limits

Every session saves protocol, source/weight hashes, raw tasks/results including
pulse amplitudes, widths, IDs, timestamps and latency, per-attempt/filter CSV/JSON,
both arrays, summary and manifest. Invalid ADC or transport errors stop the run;
scientific target misses remain results. No existing results or weights are overwritten.

The unmodified model has known issues (signed SET compliance and stale initial
last_resistance); see HARDWARE_REQUIREMENTS.md. The only vendored source patch
makes INI loading relative to its module. Synthetic MVM results do not establish
CIFAR accuracy or agreement with physical hardware. No training or hardware I/O.
