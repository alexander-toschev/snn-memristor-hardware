# Driver interface contract for `rp5_fpga_python`

**Purpose:** this document specifies the exact `conn` object interface that `HardCore`
(`MemriNeurons/cores.py`) expects from its transport layer. Today that role is filled by
`BoardSimulator` (`simulator/src.py`), a pure-software model with no real I/O (it persists
crossbar state to a pickle file). For the real two-crossbar hardware experiment, the
real-hardware driver is `rp5_fpga_python` (Raspberry Pi 5 + FPGA, Python). **If `rp5_fpga_python`
implements this same interface, the existing test/protocol code (`seq_reprogram_dryrun.py`,
`rev2_common.py`, `fresh_control.py`, `drift_test.py`) runs against real hardware with zero
changes** — only the object passed into `HardCore(conn)` changes from `BoardSimulator()` to the
real driver instance.

Source of truth: `/mnt/d/projects/memristor-aware-twd-snn/memristor_hw/MemriBoard/`
(`simulator/src.py`, `MemriNeurons/cores.py`, `manager/service/converters.py`,
`simulator/simulator_settings.ini`).

---

## 1. How `HardCore` uses `conn`

`HardCore.__init__(self, conn)` stores `conn` and calls exactly three methods on it, always
returning a 2-tuple `(adc_or_res_value, task_id)`:

| HardCore caller | conn method called | When |
|---|---|---|
| `apply_voltage(vol, rev, bl, wl)` | `conn.mode_7(v_dac, tms, tus, rev, task_id, wl, bl)` | every write/erase pulse (program loop in `write_weight`) |
| `measure_resistance(bl, wl)` / `read_one_weight` | `conn.mode_7(0, 0, 0, 0, 0, wl, bl)` | every read (v_dac=0 means "read only, no write") |
| `multiplication(...)` (wr-mode single-cell MVM, not used in our protocol) | `conn.mode_9(v_dac, task_id, wl, bl)` | only in `_contour == 'wr'` single-cell multiply path |
| `dot_product(x, wl, ...)` | `conn.mode_mvm(v_dac_list, tms, tus, rtms, rums, wl, task_id)` | every MVM column drive — this is what our Phase 1/2 protocol actually exercises (`set_mvm()` mode) |

**Our protocol only ever calls `HardCore.set_mvm()`**, then uses `apply_voltage` (erase/program,
via mode_7) and `dot_product` (readout, via mode_mvm). `mode_9` / `multiplication` are NOT used
by our dry-run/rehearsal code and can be stubbed if the real driver doesn't support wr-mode.

---

## 2. Exact method signatures required

```python
class YourDriver:
    def connect(self, crossbar_serial: str) -> int:
        """Return 1 on success, 0 on failure. crossbar_serial identifies which of the
        2 physical crossbars to address (we use 2 logical arrays, 'array_A'/'array_B' style
        serial strings in our dry-run code — real hardware may use a fixed board id /
        channel-select instead; this needs to be clarified with the driver author)."""

    def mode_7(self, v_dac: int, tms: int, tus: int, rev: int, task_id: int,
               wl: int, bl: int) -> tuple[int, int]:
        """Single-cell write-or-read pulse.
        - v_dac: DAC code (0..4095 for the simulator's 12-bit DAC) for the pulse voltage.
          v_dac == 0 means "read only" (no write pulse applied).
        - tms, tus: pulse duration in milliseconds / microseconds (combined: tms*1000+tus
          microseconds total, per the simulator's send_mode_7_to_crossbar 'duration' kwarg).
        - rev: polarity/direction flag. rev > 0 (nonzero/"reset" branch in the simulator's d2v
          sign convention) vs rev == 0. IMPORTANT: confirm the exact reset(RESET)/set(SET)
          polarity convention with the driver author — this is safety-critical (wrong polarity
          could damage cells or fail to erase).
        - task_id: opaque request id, echoed back unchanged in the returned tuple.
        - wl, bl: word-line (column, 0..7) and bit-line (row, 0..31) cell address.
        - Returns (adc_value, task_id): adc_value is the raw ADC readback. Our code converts
          this to resistance via converters.convert_adc_to_res (gain, res_load, vol_read,
          adc_bit, vol_ref_adc, res_switches) — see simulator_settings.ini for the expected
          board constants (dac_bit=12, adc_bit=14, res_load=3000 ohm, vol_ref_adc=5V, gain=11,
          vol_read=0.3V). CONFIRM these constants match the real board, or supply the real
          board's equivalents.

    def mode_9(self, v_dac: int, task_id: int, wl: int, bl: int) -> tuple[int, int]:
        """Single-cell multiply (wr-mode). NOT used by our current protocol scripts.
        Can be a stub (e.g. raise NotImplementedError) unless Phase 4 extensions need it."""

    def mode_mvm(self, vols: list[int], tms: int, tus: int, rtms: int, rums: int,
                 wl: int, task_id: int) -> tuple[int, int]:
        """Matrix-vector-multiply column drive — THIS IS THE CORE OPERATION our Phase 1/2
        protocol uses for readout (HardCore.dot_product -> conn.mode_mvm).
        - vols: list of up to ROW_NUM=32 DAC codes, one per row, each in [0, max_dac_value]
          where max_dac_value corresponds to MAX_VOL_INFERENCE=0.3V (the hard safety ceiling;
          HardCore already clamps/zeros values above this before calling mode_mvm — the driver
          does not need to re-check this, but MUST NOT exceed 0.3V on any row itself).
        - tms, tus, rtms, rums: drive/read timing parameters (write duration ms/us, read
          settle ms/us) — in the simulator these are accepted but unused (BoardSimulator.mode_mvm
          ignores them). On real hardware these may matter; confirm with the driver author
          what resolution/semantics are expected, or whether they're vestigial.
        - wl: target column (0..7).
        - task_id: opaque, echoed back.
        - Returns (adc_value, task_id): raw ADC sum-current readback for that column. Our code
          converts this to a voltage via converters.convert_adc_to_volt(gain=1, adc_bit,
          vol_ref_adc, adc_value) then scales by scale_w. CONFIRM sum_gain=330 and the ADC
          mapping match the real board (simulator_settings.ini: adc_bit=14, vol_ref_adc=5,
          sum_gain=330)."""
```

---

## 3. Safety-critical constants that must be confirmed against the real board

These are **assumed by `HardCore` and our protocol code** (hardcoded as Python class constants
or read from `simulator_settings.ini`) and must be matched or overridden for real hardware:

| Constant | Simulator value | Where used | Why it matters |
|---|---|---|---|
| `MAX_VOL_INFERENCE` | 0.3 V | `HardCore`, read-side safety clamp | Hard ceiling enforced in software before every `dot_product`/`apply_voltage(vol<=0.3)` read call. If the real driver's DAC scale differs, this clamp will be silently wrong. |
| `MAX_SUM_CURRENT` | 15 mA | `HardCore.dot_product` predrive check | Destructive-failure bound on summed column current — zeros the drive vector if predicted current would exceed this. MUST match the real board's actual current limit. |
| `dac_bit` / `vol_ref_dac` | 12 bit / 5 V | DAC code <-> voltage conversion (`v2d`/`d2v`) | Wrong values here corrupt every voltage sent to the array. |
| `adc_bit` / `vol_ref_adc` | 14 bit / 5 V | ADC code <-> voltage/resistance conversion | Same risk on the readout side. |
| `res_load`, `res_switches`, `gain`, `sum_gain`, `vol_read`, `soft_cc` | 3000Ω, 10Ω, 11.0, 330, 0.3V, 2mA | Resistance/weight conversion (`converters.py`) | Determine the weight<->resistance mapping; wrong values silently skew every measured weight. |
| `WRITE_ATTEMPTS`, `V_START`, `V_RESET`, `V_SET`, `V_STEP`, `T_US` | 10, 0.3V, 3V, 3V, 0.01V, 125µs | `HardCore.write_weight` program loop | Defines the write-verify staircase. The real driver's pulse timing/granularity may differ — needs confirmation (see open question in plan doc).

**Action for the driver author:** confirm whether `rp5_fpga_python` uses the same
board-constant conventions (readable from an equivalent `simulator_settings.ini`-style config,
or hardcoded differently), or whether our code needs a parallel `real_board_settings.ini` with
re-measured values (Phase 1 of the hardware plan exists specifically to re-measure several of
these: write-noise σ, the writable conductance window, systematic write bias).

---

## 4. Open questions for the driver author (Grigory Zharkov / NNGU)

1. Does `rp5_fpga_python` expose these exact three methods (`connect`, `mode_7`, `mode_mvm`) with
   matching signatures, or a different API that needs an adapter/shim class in front of it?
2. What does `connect(crossbar_serial)` need as its argument on real hardware — a board/channel
   id, an IP/host:port for the RPi5, a device path, something else? Our dry-run code currently
   passes a filesystem path string (since it is just a pickle file key for the simulator).
3. `rev` polarity convention in `mode_7` — which value means RESET (erase to low-conductance)
   vs SET (program to target conductance)? This is safety-critical; wrong polarity could
   either fail silently or stress cells unnecessarily.
4. Real timing requirements: do `tms`/`tus` (mode_7) and `tms,tus,rtms,rums` (mode_mvm) map to
   literal pulse-width/settle-time parameters the FPGA needs, or are these vestigial
   simulator-only parameters we can hardcode/ignore?
5. Real board constants (Section 3 table) — need the real counterparts of `dac_bit`,
   `vol_ref_dac`, `adc_bit`, `vol_ref_adc`, `res_load`, `res_switches`, `gain`, `sum_gain`,
   `vol_read`, `soft_cc`, `MAX_SUM_CURRENT`.
6. Does `mode_mvm` on real hardware return a single scalar ADC code per call (one column), same
   as the simulator, or does it support batched/multi-column reads that our per-column loop
   should use instead for efficiency?
7. Any rate limits / cooldown needed between pulses that the simulator doesn't model (thermal,
   command queue depth on the RPi5 side, etc.)?

---

## 5. What stays unchanged regardless of the answers above

- Scope is **inference-only**: we program already-trained CSNN/CIFAR-100:20 Conv1 filter
  weights (precomputed, frozen `weights_sha`) onto the crossbar; no on-chip training.
- Protocol shape (erase -> program differential G+/G- -> readback -> RMSE/Pearson vs ideal ->
  log -> next filter) does not change; only the `conn` object underneath `HardCore` changes.
- The 14-filter stratified sample list and gates (`pearson_min=0.8`, etc.) are already frozen
  in `rev2_common.py` / `seq_reprogram_dryrun.py` and do not need to change for the hardware
  swap itself — only if Phase 1 characterization reveals the real write-noise/bias profile
  requires re-tuning the gates or K (mean-of-K).
