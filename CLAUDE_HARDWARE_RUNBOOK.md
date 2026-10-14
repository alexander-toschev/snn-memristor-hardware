# Runbook — local simulator, four phases

The user approved all four phases of TWO_CROSSBAR_PLAN.md with the simulator.
This supersedes the earlier no-launch instruction only for simulator work.

1. Open `experiment.py`; check `BACKEND="simulator"` and the four phase calls.
2. Verify local weights against `weights/manifest.json`. The full checkpoint is
   stored for provenance; the experiment loads only `.npy` with allow_pickle=False.
3. Install the pinned runtime dependencies inside `.deps/` if absent (Python 3.12):
   `python -m pip install --target .deps -r requirements.txt`.
4. Run `python -m unittest experiments.unn_rev3_20261014.test_suite -v`.
5. Check the native SI/DAC interface with
   `python -m experiments.unn_rev3_20261014.check_connector --run-simulator`.
6. Run `python experiment.py --run-simulator`. It creates a new session in
   `results/simulator/`, new A/B arrays, then baseline and phases 1,2,3,4 in order.
7. Review `manifest.json` for execution status, `summary.json` and phase JSON/CSV
   for scientific outcomes, and `raw_tasks.jsonl` for every attempted measurement.
8. Audit the completed session with
   `python -m experiments.unn_rev3_20261014.verify_run results/simulator/SESSION`.
   `audit.json` checks completeness and integrity; `metrics.json` summarizes
   scientific outcomes separately. Do not audit a session that is still running.
9. For Git storage, losslessly pack a passing audited trace with
   `python -m experiments.unn_rev3_20261014.trace_storage results/simulator/SESSION`.
   The packer verifies reconstructed bytes before writing `trace_storage.json`.
   The verifier can also read the ordered gzip parts when the original JSONL
   is absent. Keep the original until the compressed copy is verified and saved.
10. Report programming failures and unsuccessful accuracy comparisons as results.
   Never relabel a completed execution as successful scientific validation.

Native backend selection is cb_type=simulator; board_type is only the driver name.
Every operation uses Connector.impact. WL=1 excluded. No HardCore constructor,
whole-array scan, real driver, automatic real fallback, or implicit reuse of arrays.
Only two fresh arrays are reused across filters; each tile is measured before erase.

Single-cell results are converted from raw ADC without native clamping. MVM uses
raw ADC->Vout, allows zero for zero inputs, and creates a new voltage list for
EVERY call because the native Connector mutates voltage lists during conversion.

Hardware remains a separate future stage. Its inventory, installed driver path,
active users, timing, polarity, current limits and safe idle state must first be
reviewed. Never infer two physical chips from simulated A/B arrays. No Raspberry Pi
control, shutdown or contact with the hardware owner is part of this run.
