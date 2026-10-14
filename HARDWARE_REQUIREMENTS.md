# MemriBoard → Physical Memristor: Hardware Requirements

**Port target:** TWD + Conv1 memristor-mapping approach from the MemriBoard *simulator* to a real memristor chip/board.

**Status:** Knowledge-synthesis deliverable. Read-only with respect to all experiment artifacts and manuscripts; no experiments launched, no manuscripts edited.

**Date:** 2026-10-12 · **Author:** snn-experimenter agent

---

## 0. Scope and the one-line verdict

This document translates *measured MemriBoard-simulator findings* into *physical-device requirements* for an inference-only deployment: weights are programmed once (offline, write-verify) and then only read out for MVM. Every number is tagged **MEASURED** (observed in our BoardSimulator runs), **DERIVED** (arithmetic on measured quantities), or **LITERATURE** (industrial/literature value, cited).

**Headline conclusions (details + citations below):**

1. **The bottleneck is write noise, not read noise, not ADC/DAC, not area.** Measured repeat-READ noise is ~1e-18 (negligible); the 12-bit DAC / 14-bit ADC are already over-specified for the current failure mode; the full 13.6M-cell array is a *sub-mm² to ~1 mm²* die (density is not the problem).
2. **K=1 requires σ_write ≤ ~0.0033 absolute conductance units** (≈8% of the 0.04 mapping span, ≈3.6 effective bits across the span) — a ~√5 ≈ 2.24× reduction from the measured single-shot σ ≈ 0.0074. *This assumes the noise is variance-dominated; a measured systematic per-tap bias (46–76% of the gate bound) must be handled by write-verify/calibration regardless of K.*
3. **The mapping window (0.01–0.05) contradicts the simulator's own docstring (0.07–0.33) and its VTEAM resistance clamps.** This must be resolved on hardware before any port; it is not answerable from simulation alone.
4. **Because the write noise is ~absolute (magnitude-independent), widening the mapping span is a free ~6.5× lever** — but only if the physical writable window is real.

---

## 1. What the MemriBoard simulator actually is (re-derived from source)

Read directly from the simulator source (`memristor_hw/MemriBoard/`), so the requirements below are anchored to *what was simulated*, not to a generic ReRAM model.

### 1.1 Device model — VTEAM with a state-blended nonlinearity (LITERATURE + MEASURED-source)

`simulator/memristor.py`, `MemristorModel.apply_voltage`:

```
current = (state · j_lin  +  (1−state) · j_nonlin) · area_filament
j_lin    = A · v                         # A = DR·1e8·exp(1−300/T) → 1e8 at T=300 K
j_nonlin = B · v · exp(−(q/(kB·T))·(φ_e − 0.5·0.07·√|v|))   # B = 8e22, φ_e = 1 eV
```

- `state` ∈ [0,1] drifts under `reset` (v ≥ +0.4 V) and `set` (v ≤ −0.4 V) with Boltzmann-barrier kinetics; the model docstring attributes it to Kipelkin et al., *Frontiers in Neuroscience* 18:1456386 (2024), doi:10.3389/fnins.2024.1456386, and Kipelkin et al., *Mathematics* 11(5):1268 (2023).
- `area_filament = 1.7e-12 m²`; resistance clamps `resistance_min = 100 kΩ`, `resistance_max = 100 MΩ`.
- **The I–V is not perfectly linear**: it is a convex blend of an ohmic term (`j_lin`) and an exponential-in-√V term (`j_nonlin`). At read voltage 0.3 V the nonlinear term is material. Any real device must be linear-enough over [0, 0.3 V] *or* its nonlinearity must be calibrated out — see §3.4.

### 1.2 Board signal chain (MEASURED from `simulator_settings.ini`)

| Parameter | Value | Meaning |
|---|---|---|
| `dac_bit` / `vol_ref_dac` | 12 bit / 5 V | write + input-voltage DAC, LSB = 5/4095 = **1.22 mV** |
| `adc_bit` / `vol_ref_adc` | 14 bit / 5 V | readout ADC, LSB = 5/16383 = **0.305 mV** |
| `res_load` | 3000 Ω | sense resistor (defines the weight scale) |
| `res_switches` | 10 Ω | switch-series resistance |
| `vol_read` | 0.3 V | read voltage |
| `gain` | 11 | per-cell read amplifier |
| `sum_gain` | 330 | column current-sum → voltage |
| `soft_cc` | 2 mA | set-compliance current limit |

### 1.3 Programming loop (MEASURED from `MemriNeurons/cores.py`, `HardCore`)

- Array = **32 rows × 8 columns** (256 cells), `MAX_VOL_INFERENCE = 0.3 V`, `MAX_SUM_CURRENT = 15 mA`.
- Write: `WRITE_ATTEMPTS = 10`; per attempt a staircase `V_START=0.3 → V_RESET=V_SET=3.0 V` in `V_STEP=0.01 V` (270 steps), pulse `T_US=125 µs`, break-on-crossing. Worst case ≈ 2,700 pulses/cell; write-verify cuts this sharply.
- Read = Ohm's-law readout `current = v_read/(R + res_load)`; **weight** is the normalized ratio `weight = res_load/(res_load + R)` (`converters.r2w`). In the MVM path, `weight = sum_gain/R` (`cores._calculate_weight_value`).

### 1.4 The mapping (MEASURED from frozen protocols)

- Signed weight `w` → differential pair `G± = 0.03 ± 0.02 · w/maxabs_w`; effective weight = `(G+ − G−) · maxabs_w/0.04`.
- So the **conductance window is 0.01–0.05** (common mode 0.03, span 0.04) in normalized-weight units.
- **Docstring mismatch (flagged):** `write_weight` docstring says "In the simulator, weights range from 0.07 to 0.33" — a *different, non-overlapping* window than the 0.01–0.05 the frozen mapping programs. See §3.3.

---

## 2. Measured facts vs assumptions/extrapolations

Each requirement is tagged. **MEASURED** = observed in BoardSimulator runs (frozen protocols; see workspace `memory/2026-10-11.md`, `memory/2026-10-09.md`, `memory/2026-10-08.md`). **DERIVED** = arithmetic/statistical consequence of measured quantities. **LITERATURE** = from cited industrial/peer-reviewed sources (and therefore a *target*, not something we reproduced).

| # | Requirement | Value | Tag | Basis |
|---|---|---|---|---|
| R1 | Inference input voltage ceiling | ≤ 0.3 V | **MEASURED** | `MAX_VOL_INFERENCE=0.3 V`; enforced in `predrive`/`dot_product` |
| R2 | Column-sum current hard limit | < 15 mA (strict) | **MEASURED** | `MAX_SUM_CURRENT=15 mA`; `predrive` rejects ≥15 mA |
| R3 | Observed peak column current | ~1.03 mA (margin ~13.97 mA) | **MEASURED** | mean-of-K=5 verify run, 2026-10-11 |
| R4 | Write staircase | 0.3→3.0 V, 0.01 V step, V_SET=V_RESET=3 V, 125 µs | **MEASURED** | `HardCore` constants + frozen protocol |
| R5 | Write attempts / sweeps | ≤ 10 write-verify sweeps | **MEASURED** | `WRITE_ATTEMPTS=10` |
| R6 | DAC resolution | 12 bit, 5 V ref (LSB 1.22 mV) | **MEASURED** | `simulator_settings.ini` |
| R7 | ADC resolution | 14 bit, 5 V ref (LSB 0.305 mV) | **MEASURED** | `simulator_settings.ini` |
| R8 | Single-shot differential write error σ | ≈ 0.005–0.01 absolute (≈0.0074 typ.) | **MEASURED** | write-noise decomposition, 2026-10-11 |
| R9 | Noise is ~absolute (not ∝weight) | magnitude-independent across filters | **MEASURED** | identical σ across different `maxabs_w` |
| R10 | Per-tap normalized weight error | 0.06–0.26 (of maxabs_w) | **MEASURED** | σ/S with span 0.04; matches R8 |
| R11 | Repeat-READ noise (fixed state) | ~1e-18 (negligible) | **MEASURED** | variance is WRITE, not read |
| R12 | K=5 redundancy needed for hard filters | mean-of-5 averaging | **MEASURED** | 4 hard filters pass only at K=5 |
| R13 | σ_write for K=1 (variance-dominated) | ≤ ~0.0033 (≈8% of span, ≈3.6 bits) | **DERIVED** | R8/√5; §3.2 |
| R14 | Systematic per-tap write bias | 46–76% of gate bound | **MEASURED** | write-noise decomposition (χ²≫df) |
| R15 | Conductance range to cover 0.01–0.05 | G ≈ 3.4–17.5 µS (R 57–297 kΩ) | **DERIVED** | `r2w` + res_load=3 kΩ; §3.3 |
| R16 | Conductance range for docstring 0.07–0.33 | G ≈ 25–164 µS (R 6–40 kΩ) | **DERIVED** | same mapping; §3.3 |
| R17 | Model resistance clamp | R ∈ [100 kΩ, 100 MΩ] | **MEASURED** | `resistance_min`/`_max` in `memristor.py` |
| R18 | ON/OFF ratio requirement | ≥ span/σ_write distinguishable levels | **DERIVED** | §3.3 (differential, analog MLC) |
| R19 | MVM linearity | must be near-ohmic over [0, 0.3 V] or calibrated | **DERIVED** | VTEAM nonlinearity; §3.4 |
| R20 | Array granularity | 32×8 = 256 cells | **MEASURED** | `ROW_NUM×COL_NUM` |
| R21 | Full-scale cell count | ~13.6M differential cells ≈ 53,125 arrays | **DERIVED** | 6.8M weights × 2; §3.5 |
| R22 | Array area | ~0.03–0.3 mm² (device+periphery) | **LITERATURE** | 22–28 nm 1T1R/1D1R density; §3.5 |
| R23 | Endurance | ≥ ~2.7k pulses/cell (one-time programming) | **DERIVED** | worst-case staircase; §3.6 |
| R24 | Retention / temperature | 10 yr @ 75–85 °C (ReRAM); PCM drift caveat | **LITERATURE** | §3.6, §4 |
| R25 | Training vs inference noise separation | `memristor_sigma` = in-silico only | **MEASURED** | §3.7 |

---

## 3. Per-parameter requirements

### 3.1 Voltage and current headroom (R1–R4)

- **Inference ≤ 0.3 V** is the hard safety ceiling and is already respected by the simulator (`predrive` and `dot_product` both clamp/reject >0.3 V inputs). A real front-end DAC must be able to source any row voltage in [0, 0.3 V] with adequate resolution (see §3.5 below).
- **Column current < 15 mA** is the destructive-failure bound; our peak was **1.03 mA**, a **13.97 mA (≈14×) margin**. This is a *comfortable* margin, but it is a property of the *simulated* device (µS-level conductances, `soft_cc=2 mA`). On a real device the margin collapses if (a) the LRS is much lower-resistance than 100 kΩ, or (b) `soft_cc` is raised to write faster. **Requirement:** the deployed device must keep the *summed* column current under the board's 15 mA rail at the mapped conductance window, i.e. `Σᵢ vᵢ·Gᵢ(0.3 V) < 15 mA` for the worst-case column (all rows at the window top, 0.3 V).
  - Check: 32 rows × 0.3 V × G_top. At the *used* window top G=17.5 µS → 32×0.3×17.5e-6 = **0.168 mA**. At the *docstring* window top G=164 µS → 32×0.3×164e-6 = **1.57 mA**. Both under 15 mA; only a low-R LRS (kΩ-scale) would threaten the rail. **LITERATURE note:** production HfOx/TaOx LRS is typically 1–10 kΩ (not 100 kΩ), so a naive direct port with a kΩ LRS would produce 32×0.3×(1/1e3)=9.6 mA — *still* under 15 mA, but ~10× closer to the rail and `soft_cc`-limited. The 100 kΩ LRS in the simulator is unusually *high* and is a model-specific choice to be re-checked (§3.3, §5).

### 3.2 Write precision required for K=1 (the quantitative question)

**Setup.** Effective signed weight `w_eff = (G+ − G−)·maxabs_w/S`, S = 0.04. Single-shot differential-conductance std `σ_diff ≈ 0.0074` (R8). Per-tap error normalized by maxabs_w is `σ_diff/S` (R10). Averaging K *independent* writes reduces *variance* by K (std by √K); the measured systematic bias (R14) is common to all attempts and is **not** reduced by averaging.

**Variance-dominated answer.** To make a single write (K=1) reproduce the mean-of-5 fidelity, the single-shot std must equal the current mean-of-5 std:

```
σ_write(K=1) = σ_diff / √5 ≈ 0.0074 / 2.236 ≈ 0.0033   (absolute conductance units)
```

- As a fraction of the mapping span: **0.0033/0.04 ≈ 8.3%** (current single-shot: 18.5%).
- As a fraction of common mode 0.03: **11%** (current: 24.7%).
- As distinguishable levels across the span: S/σ = 0.04/0.0033 ≈ **12 levels ≈ 3.6 bits** (current single-shot: ~5.4 levels ≈ 2.4 bits; K=5: ~12 levels ≈ 3.6 bits).
- Equivalently: a **5× variance reduction**, or **2.24× σ reduction**, or **~1.2 bits** of extra write precision.

**Two independent levers** (do not conflate them):

1. **Device-level write precision** (the literal answer to the question): `σ_write ≤ ~3.3e-3` absolute, i.e. ~3.6 effective bits across the 0.04 span. For a *relative*-precision device this translates to **σ_write ≤ ~8% of the 0.04 span**, or ~3.3e-3/0.03 = 11% of common-mode conductance. **LITERATURE:** HfOx/TaOx ReRAM with write-verify routinely reaches 2–4 bits/cell and conductance programming σ of a few %; our target is at the *easy* end of demonstrated MLC capability, but we must specify it *per-device and per-board* (the simulator's µS window is not a standard device).
2. **Algorithmic span widening** (free, but conditional): because the noise is **~absolute** (R9), mapping into a larger span shrinks the *weight-domain* error ∝1/span. Going from span 0.04 to the docstring's 0.07–0.33 (span 0.26) gives a **6.5×** error reduction — *more* than the √5 ≈ 2.24× needed for K=1. This is the single cheapest path to K=1, **but** it is gated on resolving the window/clamp mismatch (§3.3): if 0.07–0.33 is not actually writable, the lever does not exist.

**Caveat that must not be hidden.** The question assumes variance-dominated noise. Our decomposition shows **systematic per-tap bias at 46–76% of the gate bound** (R14). Averaging does not touch bias; a σ_write reduction alone will not make K=1 sufficient if the bias stays. **K=1 therefore additionally requires write-verify (or per-device calibration) to drive the bias to ~0**, not just a variance reduction. The bias is the reason the honest target is "σ_write ≤ 0.0033 **and** bias ≈ 0 via closed-loop write-verify," not "σ_write ≤ 0.0033 alone."

### 3.3 Conductance range, ON/OFF, and the window/docstring/clamp mismatch (R15–R18)

**Weight ↔ conductance map.** `weight = res_load/(res_load + R)` with res_load = 3000 Ω, so `G = 1/R = (1/res_load)·weight/(1−weight)`. This is monotonic but **nonlinear in weight** (it saturates: weight→1 as G→∞). Concrete conversions:

| weight | R | G |
|---|---|---|
| 0.01 (used window, low end) | 297 kΩ | 3.4 µS |
| 0.03 (common mode) | 97 kΩ | 10.3 µS |
| 0.05 (used window, high end) | 57 kΩ | 17.5 µS |
| 0.07 (docstring, low end) | 39.9 kΩ | 25.1 µS |
| 0.33 (docstring, high end) | 6.09 kΩ | 164 µS |

**Three-way mismatch (this is the central hardware-blocking issue):**

1. The frozen mapping programs **0.01–0.05** (G 3.4–17.5 µS).
2. The `write_weight` docstring claims the *physical writable* range is **0.07–0.33** (G 25–164 µS) — non-overlapping with (1).
3. The VTEAM model's own clamps are **R ∈ [100 kΩ, 100 MΩ]** = **G ∈ [10 nS, 10 µS]**. This clamps the high-conductance end at 10 µS, so **both** windows' upper halves are unreachable in the *model* as written (0.05→17.5 µS and everything in 0.07–0.33 exceed the 10 µS clamp). The used window's top (0.05) and the *entire* docstring window sit above the model's `resistance_min` clamp.

**Requirement the hardware must satisfy** (regardless of which window we ultimately use): the device's continuously-programmable conductance range must **cover the chosen window with the required number of distinguishable levels**, i.e.

```
(G_max − G_min) / σ_write  ≥  S/σ_write  =  12 levels (K=1 at span 0.04)
                                       or  35 levels (if span widened to 0.26)
```

- **Conductance range:** used window needs G ∈ [3.4, 17.5] µS (span 14.1 µS); docstring window needs G ∈ [25, 164] µS (span 139 µS). Note the two spans differ by ~10×, which is why the docstring window gives the 6.5× precision lever *if* it is real.
- **ON/OFF ratio:** for *differential analog* encoding the requirement is **not** a binary ON/OFF ratio but **analog dynamic range** = (number of distinguishable conductance levels). A conventional ReRAM ON/OFF of 10–100 (LITERATURE: HfOx/TaOx filamentary devices typically 10–10²; some 1T1R stacks 10³) is *more than enough* for 12–35 levels, *provided* the intermediate states are stable. The relevant metric is not ON/OFF but **MLC linearity and level stability**, which is what our σ_write captures.
- **Linearity:** see §3.4.

### 3.4 MVM linearity requirement (R19)

The simulator's *safety check* (`predrive`, `dot_product`) assumes Ohmic columns (`I = v/R`, i.e. weight = `sum_gain/R`), but the *actual* MVM (`send_mode_mvm_to_crossbar` → `apply_voltage`) uses the full VTEAM `I = state·j_lin + (1−state)·j_nonlin`, where `j_nonlin ∝ v·exp(−c·√v)` (R1.1). These two disagree by construction; the simulator effectively *trusts* the Ohmic view for safety and computes the dot product through the nonlinear device.

**Requirement for a real chip:** either
- **(a)** the device's read I–V over [0, 0.3 V] is linear to within the weight-fidelity gate, or
- **(b)** the nonlinearity is characterized once and **compensated** (a per-device I–V lookup or a `j_nonlin` calibration), and the compensation is re-validated across temperature.

At 0.3 V the VTEAM exponential term is non-negligible, so (a) is unlikely to hold for a real filamentary device; **(b) is the realistic requirement** and is not currently modeled in the readout path beyond the Ohmic assumption. This is an open, hardware-only question (§5).

### 3.5 ADC/DAC resolution (R6–R7, R20)

**Current spec (MEASURED):** 12-bit DAC (1.22 mV LSB), 14-bit ADC (0.305 mV LSB).

**Derived effective bits at the 0.3 V inference ceiling:**

- **DAC:** the input uses only [0, 0.3 V] of the 5 V DAC reference → **0.3/1.22 mV ≈ 246 levels ≈ 7.9 effective bits** of input-voltage resolution. If the SNN input is 8-bit-quantized spikes, this is *sufficient but not generous*; a 5 V-referenced 12-bit DAC wastes 94% of its range. **Requirement:** source resolution ≥ input quantization (8 bits) over [0, 0.3 V]; a lower reference (e.g. 0.6–1 V) or a higher-bit DAC recovers margin at no safety cost.
- **ADC:** sized for the 15 mA full-scale (15 mA × sum_gain 330 = 4.95 V ≈ the 5 V ref). At the measured 1.03 mA peak, the ADC uses only ~0.34 V ≈ **6.9% of range → ~14 − 3.85 ≈ 10 effective bits**. **The ADC is over-specified for the current write-noise-limited regime** — the 14-bit readout is not the bottleneck (read noise is ~1e-18, R11).

**Bottom line (DERIVED):** ADC/DAC are **not** binding constraints at the current operating point. They become binding only after σ_write is reduced toward the K=1 target, at which point the readout must not re-introduce quantization noise above the write floor. The current 14-bit ADC comfortably clears that. **LITERATURE note:** in analog CIM macros, ADC precision is the dominant *energy* cost, so we should not over-buy it — 8–10 effective bits is the common design point (see e.g. 4 Mb CIM macros using 4–8-bit ADCs; marked LITERATURE-extrapolated).

### 3.6 Scale, granularity, and density (R20–R22)

- **Granularity (MEASURED):** one MemriBoard array = 32×8 = 256 cells; the dot product sums 32 rows into a column.
- **Scale (DERIVED):** 6.8M-parameter wide-BNTT network → 2 cells/weight (differential) → **~13.6M cells**. At 256 cells/array that is **13.6M/256 = 53,125 arrays** (the pre-stated "53,152" corresponds to ~13.6M at 256 cells/array; the difference is rounding in the 6.8M figure — state both, use ~53.1k).
- **Density (LITERATURE):**
  - 1T1R ReRAM bit cell ≈ 8–10 F² with a minimum transistor; at F = 28 nm → ~6.3e-3 µm²/cell → 13.6M cells ≈ **0.086 mm²** of array (device-only).
  - 1D1R/4F² crossbar cell ≈ 4 F²; at F = 22 nm → ~1.9e-3 µm²/cell → 13.6M cells ≈ **0.026 mm²**.
  - With periphery (decoders, sense amps, ADC/DAC) typically 2–4× the array, a realistic **total die is ~0.1–0.3 mm²**. Reference points (LITERATURE, from ReRAM Wikipedia entry, fetched 2026-10-12): Weebit/Leti **1 Mb** array in **28 nm** FDSOI; SanDisk **32 Gb** ReRAM at **24 nm**; Micron/Sony **16 Gb** at **27 nm**; a Chinese group's **64 Mb 1T1R** at **130 nm**. Our 13.6M-cell (≈13.6 Mb) requirement sits comfortably inside mature, demonstrated densities.

**Conclusion:** **density/area is not the blocker.** A 13.6M-cell differential array is a small chip by demonstrated ReRAM standards. The hard problems are precision (§3.2), the window/clamp definition (§3.3), and reliability (§3.6).

### 3.7 Endurance, retention, temperature (R23–R24) — and training vs inference

**Programming endurance (DERIVED):** worst-case staircase is 270 steps × ≤10 sweeps = **≤2,700 pulses/cell**; write-verify typically needs far fewer. Because deployment is *one-time offline programming*, endurance is **amortized** — the device must simply survive ~10³–10⁴ pulses once. **LITERATURE:** HfOx (ITRI) has demonstrated **1e10 cycles** (Lee et al., IEDM 2008, doi:10.1109/IEDM.2008.4796677; IEDM 2010 <0.3 ns switching + 1e10 endurance); TaOx (Panasonic) has shown **1e12** in the lab but ships products rated **1e5** (ReRAM Wikipedia). Our one-shot ~1e3–1e4 pulse budget is **4–8 orders of magnitude inside** even conservative ratings. Endurance only becomes a real requirement if we move to *online* learning/fine-tuning (frequent re-writes), which is **out of scope for this inference port**.

**Retention (LITERATURE):** production ReRAM targets **10 years @ 75–85 °C** (e.g. the 64 Mb 1T1R demo reports 10 yr @ 75 °C extrapolated). The deployed weights must hold their conductance for the deployment lifetime. **This is a genuine requirement, not a given:** filamentary ReRAM retention is temperature-dependent and our VTEAM model has explicit `exp(−q·φ/(kB·T))` temperature terms, so conductance drifts with T. **Requirement:** retention ≥ deployment lifetime at the board's operating temperature, *or* periodic re-programming/refresh.

**Temperature stability (DERIVED from the model + LITERATURE):** the read conductance is T-sensitive. Options: (a) characterize G(T) and compensate; (b) keep the device isothermal; (c) accept drift if it stays inside the fidelity gate. **Not answerable from simulation** — the simulator runs at fixed T=300 K (§5).

**Training vs inference — explicit separation (MEASURED):**

- **TRAINING-time noise (`memristor_sigma`)** perturbs *optimizer weight updates* in silico (Paper1's `memristor_sigma=0.1` on Adam). It is a **software-injected robustness prior**, not a physical process, and it is **irrelevant to this inference port**. It would only become a hardware requirement if we did *on-chip training*, where the physical write noise *is* the `memristor_sigma`.
- **INFERENCE-time noise (this document)** = readback + MVM. Its components are, in order of measured impact: **write noise σ≈0.0074 (dominant)** ≫ ADC/DAC quantization ≫ repeat-read noise ~1e-18 (negligible). The requirements in §3.2–§3.6 target exactly this inference path.

---

## 4. Candidate-technology comparison (honest feasibility at our scale)

| Technology | Endurance | Retention | Analog MLC | Fit for us | Feasibility note |
|---|---|---|---|---|---|
| **HfOx ReRAM (1T1R)** | 1e6–1e10 | 10 yr @ 85 °C | 2–4 bits/cell (write-verify) | **Best overall** | CMOS/BEOL-compatible, foundry-available (TSMC/ITRI lineage, Weebit/Leti 28 nm); the standard choice for analog CIM. σ_write of a few % achievable with write-verify → meets our ~8% span target with margin. |
| **TaOx ReRAM** | up to 1e12 (lab), 1e5 (product) | 10 yr @ 85 °C | 2–4 bits/cell | **Good** | HP migrated TiOx→TaOx for stability; Panasonic shipped 4 Mb. Similar class to HfOx. Slightly better endurance headroom; needs Pt/Ir high-WF electrode (cost). |
| **Pt/TiO₂ (HP memristor)** | ~1e9–1e10 (reported) | years (forming-dependent) | moderate | **Marginal** | The canonical memristor (Strukov et al., *Nature* 453:80, 2008, doi:10.1038/nature06932), and the lineage of our VTEAM model, but Pt electrode cost + electroforming variability + the TiOx→TaOx industry migration argue against it. Only preferred if the board's µS high-resistance regime specifically requires it. |
| **PCM (GST)** | 1e6–1e8 | drift-limited (resistance rises over time) | up to 4 bits but **drift-dominated** | **Poor for us** | Conductance **drift** is a fundamental, time/temperature-dependent error that directly defeats a one-time-programmed inference deployment unless aggressively compensated (Burr et al., IEEE JETCAS 6(2):146, 2016; Ambrogio et al., *Nature* 558:60, 2018). Endurance ~1e8 (BAE) is fine, but drift is disqualifying for our *static* weight fidelity gate. |

**Feasibility verdict (honest):** at 13.6M cells / 32×8 granularity with a ~3.6-bit write-precision target, **HfOx or TaOx 1T1R ReRAM is the realistic path**; Pt/TiO₂ is defensible only to preserve continuity with the VTEAM model; **PCM is not recommended** because conductance drift is precisely the error class our fidelity gate penalizes.

---

## 5. Open questions we cannot answer from simulation alone

1. **Window truth.** Is the *physical* writable conductance window 0.01–0.05, 0.07–0.33, or neither? The docstring and the frozen mapping disagree, and the VTEAM `resistance_min=100 kΩ` clamp contradicts both. **Resolution requires a device datasheet / IV + programming characterization.**
2. **Is σ_write truly variance-dominated?** The measured bias (46–76% of bound) is systematic and *not* reduced by K. We need a single-device write-verify characterization separating per-attempt variance from per-tap bias to know whether the K=1 σ_write target (0.0033) is sufficient *or* whether write-verify must also converge bias.
3. **Does the "absolute noise" property hold on silicon?** Our leverage argument (span-widening = 6.5×) depends on noise being magnitude-independent. On a real filamentary device, noise is often *lognormal in resistance* (relative), which would invalidate the span-widening lever and change the σ_write target.
4. **MVM linearity.** The simulator's Ohmic safety check disagrees with its own nonlinear VTEAM MVM. A real device's `j_nonlin` over [0, 0.3 V] is unmeasured; compensation (a) vs. linear-enough (b) is undecidable without hardware.
5. **Temperature coefficient.** G(T) is unknown; the model has T=300 K fixed. Retention and drift under the board's real thermal profile are unmeasured.
6. **The 100 kΩ LRS.** This is unusually high for ReRAM (production LRS is kΩ-scale). A kΩ LRS changes the current-margin calculation (§3.1) by ~10× and interacts with `soft_cc`. Which regime the real board expects is unknown.
7. **DAC range waste.** Whether the 0.3 V ceiling is a device limit or a simulator convention — if the device tolerates higher read voltages, the DAC dynamic-range waste (7.9 effective bits) can be recovered by raising `vol_read`/lowering the DAC reference.
8. **Periphery energy.** The simulator models resolution but not ADC/DAC energy or area; the real analog-CIM energy/area tradeoff (ADC-dominated) is unquantified here.
9. **What exactly `maxabs_w` normalization becomes at 6.8M scale.** The mapping is per-filter `maxabs_w`; whether a single global normalization (vs per-tile) is needed at full scale, and its interaction with the write window, is unestablished.

---

## 6. Minimum hardware characterization campaign to unblock the port

Ordered, smallest-first, each closing one open question (§5):

1. **Single-device IV + programming sweep** (one wafer of candidate HfOx/TaOx 1T1R): measure writable conductance window, σ_write (single-shot), per-tap bias after write-verify → answers Q1, Q2, Q3, Q6.
2. **Linearity scan** at read voltages 0–0.3 V and 25–85 °C → answers Q4, Q5.
3. **Retention bake** at deployment temperature for the mapped levels → confirms R24.
4. **Small 32×8 array MVM** with the same gate to reproduce the 1.03 mA / 13.97 mA margin on silicon → validates R2–R3 at hardware.

---

### Sources (cited inline)

- Kipelkin, I. M., et al. "Memristor-based model of neuronal excitability and synaptic potentiation." *Frontiers in Neuroscience* 18:1456386 (2024). doi:10.3389/fnins.2024.1456386 — VTEAM device model (also in `MemriBoard/simulator/memristor.py` docstring).
- Kipelkin, I., et al. "Mathematical and experimental model of neuronal oscillator based on memristor-based nonlinearity." *Mathematics* 11(5):1268 (2023). doi:10.3390/math11051268.
- Strukov, D. B., Snider, G. S., Stewart, D. R., Williams, R. S. "The missing memristor found." *Nature* 453:80–83 (2008). doi:10.1038/nature06932 — Pt/TiO₂ memristor.
- Lee, H. Y., et al. "Low power and high speed bipolar switching with a thin reactive Ti buffer layer in robust HfO2 based RRAM." *IEDM* 2008. doi:10.1109/IEDM.2008.4796677 — Ti/HfO₂ (ITRI), <30 µA, <10 ns; IEDM 2010 adds <0.3 ns + 1e10 endurance.
- Resistive random-access memory — Wikipedia (fetched 2026-10-12), for: Panasonic TaOx (1e12 lab / 1e5 product), 64 Mb 1T1R @130 nm (1e7 cycles, 10 yr @75 °C), SanDisk 32 Gb @24 nm, Micron/Sony 16 Gb @27 nm, Weebit/Leti 1 Mb @28 nm FDSOI, 4F² 1D1R, sneak-path/selector discussion. https://en.wikipedia.org/wiki/Resistive_random-access_memory
- Phase-change memory — Wikipedia (fetched 2026-10-12), for: BAE 1e8 write-cycle endurance and conductance-drift limitations. https://en.wikipedia.org/wiki/Phase-change_memory
- Burr, G. W., et al. "Recent progress in phase-change memory technology." *IEEE JETCAS* 6(2):146–162 (2016). doi:10.1109/JETCAS.2016.2547718 — PCM endurance/drift (LITERATURE-extrapolated).
- Ambrogio, S., et al. "Equivalent-accuracy accelerated neural-network training using analogue memory." *Nature* 558:60–67 (2018). doi:10.1038/s41586-018-0180-5 — PCM (and analogue-memory) drift compensation (LITERATURE-extrapolated).

> **Note on sourcing:** `web_search` was unavailable in this environment; web grounding was done via `web_fetch` of the two Wikipedia articles above plus the cited primary literature. Numbers tagged **LITERATURE** that I could not re-verify live (PCM endurance/drift, CIM ADC design points, MLC σ_write of "a few %") are marked **LITERATURE-extrapolated** and should be re-checked against the cited papers before being used as hard specs.

---

### Provenance note

Measured facts in §2 are taken from the frozen MemriBoard protocols and their results; the authoritative log is in workspace `memory/2026-10-08.md`, `memory/2026-10-09.md`, `memory/2026-10-11.md`, and the frozen artifact hashes recorded there. No experiment artifacts or manuscripts were modified by this deliverable.
