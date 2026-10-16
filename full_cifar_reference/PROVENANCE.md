# Source snapshots

These three Python files were copied byte-for-byte from the local training
repository `memristor-aware-twd-snn`, without modifying the source:

|Snapshot|Original repo-relative source|SHA256|
|---|---|---|
|temporal_csnn.py|src/models/temporal_csnn.py|73ee710edc86666ba6fceab46644724793834d6b57e1f121d63098976dac9d89|
|lif.py|src/models/lif.py|61762a092c7c7ea1b21823ec01cda9fa24355ca2f6d3f1f65e535ec12fdda91d|
|encoders.py|src/models/encoders.py|16a532134eaf78055db355f531214daa2af888a7fe6f30ba905f0c01db96f431|

Source repository HEAD read from `.git/HEAD` and its loose master ref during
preparation was `9a5c052bd86d64ea046d2b28618d770aa62bb3bb`. This is recorded
context, not a claim that working-tree bytes were checked against that Git blob
or that this was the original training commit. Exact copied byte digests above
are the reference identity; `full_cifar_classifier.py` checks them before import.
`__init__.py` is a new package marker, not an original training source file.

No driver modules are imported. The snapshot is used to compare a separately
implemented functional CPU downstream with the available original forward and
encoder. Loading the learned checkpoint remains `weights_only=True`, CPU-only,
and strict state-dictionary key/shape matching.
