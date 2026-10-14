# Third-party code: MemriBoard

`memristor_hw/MemriBoard/` in this repository is a **git submodule** pointing to the
upstream MemriBoard memristor-crossbar simulator and hardware-control stack:

- Upstream: https://github.com/neurocomputer/MemriBoard
- License: **GNU GPLv3** (see the LICENSE file inside the submodule)

It is referenced as a submodule (not copied in) because our test/protocol code
(`experiments/`) imports it directly (`from simulator.src import BoardSimulator,
create_crossbar_array`, `from MemriNeurons.cores import HardCore`) and the rest of this
repository is meaningless without it — see `DRIVER_INTERFACE.md` for how our code uses it.

## Fetching the submodule

A plain `git clone` of this repo leaves `memristor_hw/MemriBoard/` as an empty directory.
To fetch it:

```bash
git clone --recurse-submodules https://github.com/alexander-toschev/snn-memristor-hardware.git
# or, if already cloned without --recurse-submodules:
git submodule update --init --recursive
```

This repository as a whole inherits GPLv3 terms with respect to the MemriBoard-derived
portions pulled in via the submodule; our own experiment/plan code is otherwise under no
additional restriction beyond that inherited scope. If you only need the interface contract
(not the simulator implementation), see `DRIVER_INTERFACE.md`, which documents the relevant
surface without requiring you to fetch the submodule.
