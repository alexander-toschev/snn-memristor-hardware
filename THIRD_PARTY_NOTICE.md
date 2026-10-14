# Third-party code: MemriBoard

`memristor_hw/MemriBoard/` in this repository is a copy of the MemriBoard
memristor-crossbar simulator and hardware-control stack, originally from:

- Upstream: https://github.com/neurocomputer/MemriBoard
- License: **GNU GPLv3** (see `memristor_hw/MemriBoard/LICENSE`)

It is included here (minus `venv/`, `__pycache__/`, and local simulator state
files) because our test/protocol code (`experiments/`) imports it directly
(`from simulator.src import BoardSimulator, create_crossbar_array`,
`from MemriNeurons.cores import HardCore`) and the rest of this repository is
meaningless without it — see `DRIVER_INTERFACE.md` for how our code uses it.

This repository as a whole inherits GPLv3 terms with respect to the
MemriBoard-derived portions; our own experiment/plan code is otherwise under
no additional restriction beyond that inherited scope. If you only need the
interface contract (not the simulator implementation), see
`DRIVER_INTERFACE.md`, which documents the relevant surface without requiring
you to read the MemriBoard source.
