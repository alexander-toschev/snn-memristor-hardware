# Third-party code

The current experiment uses a local source snapshot in `vendor/MemriBoard/`.
Upstream: https://github.com/neurocomputer/MemriBoard (GNU GPLv3).
The full upstream LICENSE is retained. Source commit and per-file hashes are
recorded in `vendor/manifest.json`. The compatible MemriNeurons source snapshot
is retained alongside it for provenance; the current experiment does not import it.

One local source patch in `simulator/src.py` resolves simulator_settings.ini next
to the source module instead of relative to the working directory. The numerical
model, converters and native Connector are unchanged.

The older `memristor_hw/MemriBoard/` submodule was removed from the working tree;
its gitlink remains recoverable from commit 067c4b7 and its local Git objects are
retained. No submodule initialization or neighbouring repository is needed.
Local Python package versions are pinned in requirements.txt.
