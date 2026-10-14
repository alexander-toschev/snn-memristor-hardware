# SNN Memristor Experiment

Current configuration: `experiments/unn_rev3_20261014/protocol.json`.
The selected backend is **simulator**, through MemriBoard's existing
`manager.board.Connector`. No simulation or hardware run has been authorized.

`cb_type` selects `simulator` or `real`; `board_type` identifies the hardware
driver. A configured `rp5_fpga_python` alone does not mean hardware is active.
In the GUI, `Manager.use_chip()` obtains `cb_type` from the selected chip's
database record. Do not infer it from `settings.ini` alone.

## Current revision

- `TWO_CROSSBAR_PLAN.md`: simulator-first experiment and later hardware stages.
- `HARDWARE_REQUIREMENTS.md`: corrected units, mapping, noise and source audit.
- `DRIVER_INTERFACE.md`: native Connector contract and version differences.
- `CLAUDE_HARDWARE_RUNBOOK.md`: operator approval and hardware prerequisites.
- `experiments/unn_rev3_20261014/`: explicit configuration and simplified Python.

REV3 has configuration, a native simulator readback experiment and offline
analysis. It does not yet implement a write-characterization or filter-mapping
acquisition runner. The historical scripts do not consume the new configuration.
Neither an import nor a configuration edit should start an experiment.

Python entry points in `experiments/unn_rev3_20261014/`:

- `experiment.py`: explicit first-check or repeat-read acquisition, simulator only.
- `connector_session.py`: the existing Connector, zero-amplitude mode-7 tasks.
- `config.py`: configuration, board units and allowed cell addresses.
- `measurements.py`: raw-ADC conversion and signed-error statistics.
- `analyze_readback.py`: existing JSONL to a new JSON report, no device access.

No commands were run to test these entry points. Launching requires a separate
user approval; the saved execution and read-baseline flags remain false and
`crossbar_serial` remains unset. Runtime dependencies and behavior are untested.

## Source versions

The historical rehearsal used the modified MemriBoard tree at
`D:/projects/memristor-aware-twd-snn/memristor_hw/MemriBoard`; source hashes are
recorded in the REV3 configuration. This tree has the modern Connector interface,
BoardSimulator and MemriNeurons. The upstream submodule checked out in this
repository is older and does not provide that complete API. Do not silently
substitute one for the other. REV3 must be pointed at the compatible installation.

Previous experiments and results remain unchanged for reproducibility. Superseded
top-level documents are retained in `experiments/unn_rev3_20261014/legacy_documents/`;
their hardware settings and conclusions are not the current protocol.

## Third-party code

`memristor_hw/MemriBoard/` is a Git submodule of
<https://github.com/neurocomputer/MemriBoard> (GPLv3).
See `THIRD_PARTY_NOTICE.md`. No third-party driver or simulator source was modified
as part of this revision.
