# Official writable-cell waveform survey

Public source: neurocomputer/MemriDocs, commit
`e29ce4f06262a2bf96b15535422bc1ecc06679a2`, `writable_cells/`.
The JSON is copied byte-for-byte and its hash is pinned in the adapter.

```powershell
python -B experiments/official_writable_cells_20261114/experiment.py --run-simulator --smoke
python -B experiments/official_writable_cells_20261114/experiment.py --run-simulator
```

Authorized hardware, from the Raspberry checkout and its environment:

```bash
python -B experiments/official_writable_cells_20261114/experiment.py --run-hardware --pilot2 --publish
python -B experiments/official_writable_cells_20261114/experiment.py --run-hardware --publish
```

`--pilot2` uses WL0/BL8 and WL3/BL8. The full survey visits all256 once, with the
historical44 first. `--priority-only` limits to44. These are separate explicit
commands; the script never automatically starts a second hardware run.

Each cell gets SET, RESET, then IV through native MemriBoard waveform generation
and public Connector mode7. SET is a0–1.7–0V triangle repeatedtwice with100us
requested duration. RESET is0–2.2–0V repeatedtwice with1ms requested duration.
IV applies reverse/direct1.2V triangles threetimes with130us from JSON (the DOCX
screenshot says100us). Physical duration remains unknown with the installed
hardware driver; no timing fidelity is claimed.

The legacy GUI importer incorrectly keeps the ADC inequality when converting
its threshold to resistance. This experiment explicitly uses DOCX stop criteria
R<=296ohm for SET and R>=2998ohm for RESET instead. Original ADC thresholds9801
and5400 are retained in provenance and are not treated as ohms. Termination uses
fresh native resistance values. The test is therefore named
`official-writable-docx-stops-json-waveforms-v1`, not a bit-exact legacy import.

At most620 native waveform tasks/cell, plus fresh guards and initial readings.
Caps:620decisions and1300public calls per selectedcell;2hours for256cells,
30minutes for smaller runs; raw2GiB plus2GiB free reserve. Existing guards remain:
2mA reverse/SET,40mA direct/RESET, native ADC/id/resistance validation. Every skipped
current-guard pulse is recorded and marks that cell's protocol as modified.
Zero-voltage waveform tasks are reads, not counted as programming pulses.

The map shows the observed DOCX switching criterion Rmin>100ohm andRmax/Rmin>2.
It also records current-GUI ratio>=2 separately. Extrema use only actual waveform
responses, excluding additional current-guard readings. Invalid resistance/ADC
marks a cell measurement error; malformed response/id stops the whole run.
All request/response raw and native stdout are retained, including failures.
The port and writer close before terminal manifest and optional publication.

Green means observed resistive switching only. It does not establish precise
conductance programming, signed/group MVM repeatability, or MNIST accuracy.
Inspect guard-modified flags and unknown physical duration before interpreting
this as a reproduction of the official test. MNIST and grouped MVM are separate.
