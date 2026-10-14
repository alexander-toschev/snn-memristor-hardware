# Two-crossbar real-hardware experiment plan (512 cells = 256 differential weights at K=1)

Author: snn-experimenter · 2026-10-13 · Status: PROPOSAL (hardware not yet arrived) · REV 2 — sequential-reprogram strategy added per Alex's idea

## Context and constraint
NNGU will provide **2 physical crossbars of 32×8 = 256 cells each** (512 cells total).
Differential encoding costs 2 cells/weight → **256 weights max at K=1** per *simultaneous* mapping.
One Conv1 filter = 75 taps → 3 tiles (32+32+11) × 2 columns (G+/G−) = 6 columns = 150 cells,
so the 2 chips hold **~3 filters simultaneously** (3 × 150 = 450 cells).

**Key realization (REV 2):** the crossbars are re-programmable, so we are NOT limited to a
permanent 2–3 filters. We can **sequentially reprogram** the same 2 chips through many
filters, saving each cycle's readback to file before the next erase/reprogram. This turns a
2-filter demo into a full "N of M filters validated on real hardware" result.

## The one scientific question this experiment answers
**Does our MemriBoard simulator accurately predict real memristor write-noise and
differential-mapping fidelity?** Everything we concluded in simulation (write-variance
forcing mean-of-K, the structural "concentrated-signal" hard filters, the −7.2 pp full-Conv1
hybrid) rests on that simulator being faithful. Sequential multi-filter validation turns this
from a 2-point check into a sweep over the whole difficulty range.

## Endurance analysis — is sequential reprogramming feasible? (Alex's Q1)
- The MemriBoard code has **no numeric write-cycle limit**. `endurance` exists only as a
  *measurement ticket* (`docs/README.md`: "standard endurance test for RRAM cells") — i.e. it
  measures cycles-to-failure, it does not impose one. The `write_weight` loop is bounded by
  `WRITE_ATTEMPTS=10` verify sweeps (voltage 0.3→3.0 V, step 0.01 V, pulse 125 µs), which is a
  *time* bound, not an endurance bound.
- Real 1T1R ZrO₂ / HfOx / TaOx ReRAM endurance is typically **10⁵–10⁷ cycles**. Our sequential
  plan costs ~1 program cycle per cell per filter. 64 filters → ≤64 cycles/cell; 14 filters →
  ≤14 cycles/cell. **Endurance is not a constraint** — the actual limit is wall-clock time.

## Time estimate (Alex's Q3)
- Per filter (K=1, one programming cycle + readback + save): ~5–10 min (write-verify dominates;
  worst case 10 sweeps × ~270 pulses, but breaks early; serial round-trip overhead on the COM
  port). **14-filter stratified sample ≈ 1–3 h; full 64-filter Conv1 ≈ 5–11 h.**
- Add Phase 1 characterization (~30–60 min) + bring-up.

## Expanded scope (Alex's Q2)
Ordered by value, each a progressive superset of the previous:
1. **Phase 1 — write-precision characterization** (the fundamental unknown): a conductance
   sweep on a few cells → real σ_write, cycle-to-cycle variance, read reproducibility. Decide
   K=1 vs K>1. Reconcile the writable range (docstring 0.07–0.33 vs our 0.01–0.05 mapping).
2. **Phase 2 — differential mapping, sequential, stratified sample (14 filters):** cycle the 2
   chips through our already-characterized 14-filter stratified sample (easy → hard:
   filter_63 … filter_41 … filter_12), one filter at a time, K=1. Save each cycle's
   programming history + readback + RMSE/Pearson vs ideal before erasing. This validates the
   simulator across the FULL difficulty range, not just 2 points.
3. **Phase 3 — full Conv1 (64 filters, optional):** if endurance/time permit, extend to all 64
   filters. ~5–11 h, feasible as an overnight run.
4. **Phase 4 — mean-of-K on real hardware (optional):** on the hard filters that fail K=1
   (e.g. 41), re-program with K=2/K=3 and show the variance reduction recovers fidelity —
   validating the mean-of-K method physically.

## Per-cycle protocol (same frozen-protocol discipline as simulation)
For each filter: erase (RESET) → program differential G+/G− (the same 0.03±0.02·w/maxabs
mapping, window re-derived from the real writable range) → read back → compute RMSE/Pearson vs
the ideal weight → append to a results file → proceed to the next filter. Record all
programming/readback histories + hashes. Honest scope: real-hardware result, K=1 first.

## Deliverables (much stronger than a 2-filter demo)
1. Real σ_write vs the simulator's ~0.0074 (Phase 1).
2. Real-vs-predicted fidelity across the 14-filter stratified difficulty range (Phase 2), or
   all 64 (Phase 3) — "N of M filters, spanning easy→hard, validated on real hardware."
3. Physical validation of the mean-of-K variance-reduction (Phase 4, optional).


## Update 2026-10-14 — driver confirmed

Alex confirmed (relayed from Grigory Zharkov, NNGU hardware contact): the real-hardware
driver will be **`rp5_fpga_python`** — a Raspberry Pi 5 + FPGA Python driver — not a direct
serial/COM link to `HardCore` as previously assumed in this plan and in the `.tmp/` dry-run
scripts (`seq_reprogram_dryrun.py`, `rev2_common.py`, `fresh_control.py`).

Alex is waiting on SSH access to the hardware host; `rp5_fpga_python` docs/repo not yet in
hand. Open items once driver access arrives:
1. Confirm whether `rp5_fpga_python` exposes a `BoardSimulator`-compatible interface (same
   `connect()`/`HardCore` surface) or needs a new adapter layer.
2. Confirm timing/batching behavior vs. the assumed serial write-verify staircase (R4/R5 in
   `MEMRISTOR_HARDWARE_REQUIREMENTS.md`) — FPGA-mediated access may batch commands differently
   than a direct COM link.
3. Re-validate Phase 1 (write-precision characterization) is unaffected by the driver swap
   before committing to Phase 2's 14-filter stratified run.

Scope remains **inference-only**: programming already-trained CSNN/CIFAR-100:20 Conv1 weights
onto the two crossbars, not on-chip training.
