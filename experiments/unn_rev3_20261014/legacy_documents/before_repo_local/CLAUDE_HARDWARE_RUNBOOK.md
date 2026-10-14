# Operator runbook - REV3, 2026-10-14

> Acquisition entry point: `experiment.py` at the repository root. Its top settings block overrides the reference JSON for the current read session; the effective protocol is saved with the results. The old nested entry point is disabled. RUN_APPROVED is false; no run is authorized.

**Current user instruction: do not launch anything without explicit permission.**
Configuration/code edits are authorized. No simulator, analysis program, driver,
GUI experiment, hardware read, write or erase is authorized by this document.
Do not treat a saved Boolean or an earlier login approval as permission to run.

## Current state

The experiment uses the existing MemriBoard Connector with simulator backend.
`cb_type` selects the backend; `board_type` selects the real-driver family.
Read `DRIVER_INTERFACE.md` before interpreting either setting.

The earlier Pi reconnaissance found the registered `rp5_fpga_python` driver,
the `RPI_modes_FPGAed` class and board settings matching the simulator reference.
That observation did not establish which chip backend was selected in the GUI.
The user subsequently specified the simulator as the current backend.
No hardware driver was instantiated and no measurements were acquired here.

## Before any approved simulator session

Use `experiments/unn_rev3_20261014/protocol.json`. Keep the native Connector,
normal MemriBoard dependencies and one explicitly selected compatible checkout.
No fake manager/Application modules. A simulation array file must be explicitly
selected; loading a pickle requires a trusted local file. No automatic hardware
fallback, array creation, whole-array scan or write operation.

The proposed first stage reads each of the three configured cells once. The
repeatability stage is a separate choice of 20 reads per cell. Raw records go to
a new directory and are not allowed to overwrite historical results.
Native Connector resistance has already undergone its internal clamping;
scientific analysis must retain raw ADC and convert it without silent repair.

## Before any separately approved hardware session

1. Review the complete installed driver and confirm mode_7 zero-write behavior,
   idle state, timing and the exact installed version. The constructor opens
   SPI/GPIO and already counts as hardware I/O.
2. Ensure no other application/session is controlling the same board.
3. Confirm the selected real chip and physical array addressing. Do not assume
   there are two connected arrays because the simulation uses array_A/array_B.
4. Review the exact approved command scope with the operator. Start from the
   known cells `(WL,BL)=(0,2),(2,6),(2,4)` and only the agreed read operation.
5. Never address WL=1. It is a column in the software address convention;
   references calling it a row do not change the excluded numeric WL value.

HardCore cannot be used unchanged: its constructor and `set_mvm()` scan all
cells. The new simulator session deliberately does not construct HardCore.

## Switching and MVM

Both are disabled and not implemented in the new acquisition session. Review
the configuration's proposals only after initial characterization. Missing timing,
polarity, compliance, current limit or budget values are deliberate blockers,
not defaults to guess. The operator specifies a 3.3 V absolute ceiling;
3.0 V is a proposed lower operational cap, not permission to apply it.

The PDF describes a switching pulse followed by a fixed nominal 0.3 V read
pulse. Such a sweep is a state/readback experiment, not a measurement of the
instantaneous I-V curve during the switching pulse.

Save every pulse, raw ADC, ID and latency before proceeding. Keep failed cells
and unsuccessful programming attempts in the results. Stop on invalid readings
or a budget violation. Do not power down the Pi or contact anyone unless the
user explicitly asks; hardware shutdown requires operator coordination.

