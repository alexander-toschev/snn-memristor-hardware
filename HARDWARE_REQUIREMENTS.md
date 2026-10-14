# MemriBoard configuration audit - REV3, 2026-10-14

This replaces the previous requirements interpretation. Evidence here is a static
inspection of local source, existing rehearsal results, the operator's UNN PDF,
and configuration/source excerpts read from Pi before the user's stop request.
No new simulator or hardware experiment was run. Hardware calibration is pending.

## Interface and backend

The experiment must use the existing MemriBoard Connector. `cb_type=simulator`
selects BoardSimulator; `cb_type=real` selects the registered hardware driver.
`board_type=rp5_fpga_python` does not override a simulator chip selection.
The current REV3 configuration selects the simulator.

## Board parameters

| Parameter | Reference value | Interpretation |
|---|---:|---|
| DAC | 12 bit, 5 V | Nominal code-to-voltage reference |
| ADC | 14 bit, 5 V | Raw ADC must be retained |
| `res_load` | 3000 ohm | Single-cell read divider |
| `res_switches` | 10 ohm | Series switch resistance |
| `gain` | 11 | Single-cell read amplifier |
| `sum_gain` | 330 ohm | MVM transimpedance |
| `vol_read` | 0.3 V | Nominal read pulse |
| `read_time`, `blank_time`, `blank_time_between` | 1200, 5, 55000 | Template labels these in microseconds; actual FPGA use unverified |
| `soft_cc` | 0.002 A | Configuration value, not proof of hardware compliance |

Keep these reference values instead of changing amplifier gains to obtain nicer
results. Their presence in the Pi settings has been observed; their actual
calibration and timing behavior have not been verified.

## Correct weight and conductance units

`HardCore._calculate_weight_value` has two different definitions:

- WR divider: `w_wr = res_load/(res_load + R)`.
- MVM: `q = sum_gain/R = 330/R`. The historical experiment calls `set_mvm()`.

Physical conductance is `G_S = 1/R = q/330`. Calling q "conductance in siemens"
or applying the WR inverse formula to q is incorrect.

| MVM q | R (ohm) | G (microsiemens) |
|---:|---:|---:|
| 0.01 | 33000 | 30.303 |
| 0.03 | 11000 | 90.909 |
| 0.05 | 6600 | 151.515 |
| 0.07 | 4714.286 | 212.121 |
| 0.33 | 1000 | 1000 |

The old 57-297 kohm / 3.4-17.5 microsiemens window was calculated with the WR
formula and does not describe the MVM experiment.

The operator reports useful physical cells around 400-2000 ohm: 500-2500
microsiemens, or MVM q=0.165-0.825. This is a reported cell range, not a measured
programmable window. Keep it separate from the historical simulator window.

## Model limits and mismatch

`MemristorModel` compares `v_inp/current` directly with numeric limits 100 and
100000. These are **ohms**, despite the stale Russian comments saying kohm.
The old document's 100 kohm-100 Mohm limits were wrong by a factor of 1000.
Clamps alone do not establish the reachable analog range.

At T=300 K, the linear branch has `A*area_filament = 1e8*1.7e-12 = 170e-6 S`.
At state=1 this gives about 5882 ohm and MVM q=0.0561. It explains why the
historical rehearsal plateaus around 0.056 instead of reaching q=0.1 or 0.33.
The model is not calibrated to the operator's 400-2000 ohm cells. Do not tune
its parameters to make a filter pass and call that hardware validation.

## Noise and statistics

`noise_rate=0.03` scales a Gaussian perturbation of the internal state at each
eligible switching integration step. It is neither a measured 3% weight error
nor an independent Gaussian draw applied once per programmed weight. The state
interval and near-zero-state branch further affect its distribution.

Use separate measurements:

1. Repeat reads of one unchanged cell/state: read repeatability, sample SD.
2. Independent programming cycles at one cell and one target: signed errors,
   bias, sample SD (`ddof=1`), RMSE and failures.
3. Different cells: device variation, reported separately.

The old Phase 1 computes `std(abs(error))` across different targets and cells;
that is not write sigma. Mean-of-K independent writes and K repeated reads
answer different questions. Do not pool different targets to estimate sigma.
Cycle-to-cycle SD still includes read noise unless explicitly decomposed.
Fit on pilot cycles; compare predictions on new held-out cycles. Three cells
are a pilot, not evidence for the entire array.

## Source defects requiring separate code repair and validation

The reviewed historical simulator source has these issues:

- Negative SET current is compared to positive `soft_cc` without `abs`; the
  intended compliance condition is not reached for positive resistance.
- `HardCore.apply_voltage` passes `res_load` where ADC conversion expects
  `res_switches`. Later explicit reads do not repair every caller's result.
- `HardCore.__init__` and `set_mvm()` scan the whole array, including WL=1.
- Array creation changes state/barrier after initializing `last_resistance`.
- MVM ADC code generation is not an observed saturation model; overrange raw
  codes must not be interpreted as valid measurements.

These are source defects, not settings to hide by changing sigma. Historical
artifacts remain frozen. Any repaired-model run needs a new identity and approval.

## Current and quantization

At 400 ohm, 32 rows at 0.3 V draw an idealized 24 mA, above the historical 15 mA
limit; transimpedance 330 ohm would demand 7.92 V on a nominal 5 V ADC. This
prevents copying the old full-column drive into the low-resistance board.

REV3 leaves hardware MVM disabled. Its provisional review settings are a 5 mA
software ceiling, at most 8 active rows per partial read, and 80% ADC headroom.
These do not establish a hardware-safe limit. A measured per-cell lower resistance
bound and confirmed board limit must pass the current calculation before every
drive; subdivide further or reject if the bound fails.

With a 5 V/4095 DAC scale, code 246 is 0.300366 V, although `d2v` rounds it to
0.300 V. A strict 0.3 V input ceiling uses code 245 (0.299145 V). The inspected
driver's fixed read code is 246; record this distinction and confirm its read
calibration rather than silently rewriting the driver.

No endurance rating, physical array count, or experiment duration has been
established for this setup. Count actual pulses and measure latency; do not equate
one filter with one physical switching pulse or promise a 1-3 hour run.
