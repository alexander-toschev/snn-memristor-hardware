# REV 2 — two-crossbar general rehearsal (BoardSimulator)

Headless BoardSimulator rehearsal of `docs/two-crossbar-hardware-plan.md` (REV 2):
modeling EXACTLY 2 reused 32×8 crossbars through the full CIFAR-100 Conv1 pipeline,
before running on real NNGU hardware.

Located under the MemriBoard repo to follow its convention of root-level example/experiment
scripts (`snn_example.py`, `xor_example.py`) — this is a multi-phase experiment, so it lives in
a descriptive subdirectory instead of the root.

## Scripts
- `rev2_common.py` — shared headless BoardSimulator adapter (imports `manager/service/converters.py`
  via importlib + FakeApplication stub), frozen protocol constants, split digest, 14-filter list.
- `phase1_sigma_write.py` — σ_write characterization (conductance sweep + cycle-to-cycle).
- `phase2_sequential.py` — sequential differential mapping of all 14 filters through 2 reused
  arrays (erase → program G+/G− → readback → RMSE/Pearson → save → next filter).
- `phase3_hybrid.py` — hybrid forward pass (14 mapped filters + 50 clean) vs clean FS.

## Run
From this directory (`python3` with numpy; phase3 also needs torch + the SNN repo on `sys.path`):
```bash
python3 phase1_sigma_write.py
python3 phase2_sequential.py
python3 phase3_hybrid.py
```

## Key results (2026-10-13 rehearsal)
- Phase 1: σ_write ≈ 0.002 (0.01–0.05 window); docstring range 0.07–0.33 NOT writable
  (saturates ~0.056). Cycle-to-cycle σ ≈ 0.0017.
- Phase 2: all 14 filters PASS the full-75 gate (erase-before-write; 2 arrays reused,
  `create_crossbar_array` called exactly 2×).
- Phase 3: hybrid 14-of-64 → Δ −2.7 pp (clean 0.7250 → hybrid 0.6980), 18.7% prediction change.

## Scope
Simulator rehearsal only — the real hardware will differ (VTEAM has no endurance/aging terms).
Before the real run, Phase 1 must re-measure the actual writable conductance range and drift.
