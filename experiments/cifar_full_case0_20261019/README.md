# Full measured Conv1, fixed CIFAR case 0

The new collector uses only the public MemriBoard Connector and native converters.
Historical collectors and trusted-import gates remain unchanged. The protocol must
be registered after source review; native SIM qualification precedes hardware.
Completed smoke is not full-classifier qualification.

Scope: CIFAR100 test index 9, case 0 of the saved 10-class subset; 64 Conv1
filters, 4,800 weights, 784 positions and T12. Four filters occupy WL pairs
0/1, 2/3, 4/5, 6/7. Sixteen groups each have 32/32/11-tap stages: 48 states.
The first two stages use 256 cells; the last uses 88, with unused inputs zero.
Up to four inputs are active per physical read. Larger TTFS masks are split;
every contribution is acquired fresh. All splits and tiles sum before one
trained BN/LIF update per timestep. No cache, CPU substitute or missing-zero
fill. Downstream BN/LIF and network layers are digital. One image gives a
prediction, not dataset accuracy. Original CPU reference is wrong on this fixed
image: true fine/subset class0, all three CPU decoders predict class1. The case
is retained without choosing a more favorable example. See the
[CPU bridge validation](../../reports/2026-10-19/cifar-full-case0-review/cpu-bridge-case0.json).

Run from repository root, in the foreground without detach:

```text
python experiments/cifar_full_case0_20261019/experiment.py --dry-run --preflight-source simulator --scope smoke
python experiments/cifar_full_case0_20261019/experiment.py --run-simulator --scope smoke
python experiments/cifar_full_case0_20261019/experiment.py --run-simulator --scope full
python experiments/cifar_full_case0_20261019/experiment.py --dry-run --preflight-source hardware --scope full
python experiments/cifar_full_case0_20261019/experiment.py --run-hardware --scope full
```

Issue commands one at a time. Review each completed result before the next
step; this list is not an automatic launch chain. `smoke` programs the first
four filters across all three tiles (300 weights), then measures only position
(0,0) for T12. `full` programs all64 filters and measures all784 positions.
Both scopes perform SET/RESET and change the selected driver's array state.
Smoke cannot supply the missing filters or positions of a full classifier.

Dry-run verifies pinned assets, helpers, INI and native source without opening a
Connector. Hardware uses repository-local INI and the Raspberry native fingerprint.
Launches use standard Connector driver selection. Repository setup supplies local
`.deps`; classification also needs PyTorch in the environment or `.deps-classifier`.
Each launch writes a new UUID directory under `results/cifar_full_case0_20261019/`.
The standalone classifier performs the independent raw audit before consuming
either measured branch. For a completed full SIM or hardware session, replace
`SESSION_ID` with the emitted directory name and choose a new output filename:

```text
python experiments/cifar_full_case0_20261019/classifier.py --cpu-reference --output reports/cpu-case0-new.json
python experiments/cifar_full_case0_20261019/classifier.py --session results/cifar_full_case0_20261019/SESSION_ID --output reports/measured-case0-new.json
```

The second command reports raw, calibrated and CPU predictions together. It
requires a reviewed auditor hash, unchanged acquisition source/result hashes
and a complete full trace; smoke, aborted and incomplete traces cannot be
promoted. Register source hashes before acquisition and preserve that checkout.
Native SIM collection alone is not the full integration check: its complete
trace must pass this audit and classifier path before the corresponding hardware
route is qualified. The historical import gate remains unchanged.

For hardware reports, the machine running classification also needs the exact
Pi-only native source files recorded in the acquisition manifest. If absent
locally, obtain verified matching files or classify on Pi with CPU dependencies
available; do not bypass the source check. Keep classifier reports outside the
immutable acquisition directory.

The frozen numerical contract is `protocol.json`. Read amplitude is 0.05 V;
global scale is 0.125 V per checkpoint weight on hardware and 0.015 on SIM.
Both differential legs receive bounded RESET; SET uses only the signed leg.
Each ramp is `[0.30, 0.60, 0.90, 1.20, 1.50, 2.00, 2.50, 3.00]` V, at most
eight decisions per leg. Fresh native R precedes each pulse; sign0 uses 0.04 A
and sign1 unchanged board soft compliance 0.002 A. Current-limited attempts are
skipped and logged. RESET stops after an admitted pulse gives legacy q=330/R
<=0.005. SET stops within max(0.00061 V, 5% of target), at overshoot, or at ramp
limit. Approximate weights remain outcomes, without per-weight correction.
Hardware pulse duration is unknown; SIM width 0.000125 encodes 125 modeled steps
and does not establish physical pulse duration.

Every vector has fresh mode7 reads for all active cells and eight sequential
mode10 WL reads. Native-R current, GUI output and observed ADC headroom checks
remain active. Post-write basis checks worst-four-input headroom before acquisition.
Each stage has initial zero/singleton basis, 64 fit masks, 32 selection masks,
frozen C, 32 independent heldout masks, inference, then final zero/singleton basis.
Each repeats three times; fit/selection/heldout have before/after zero brackets.
Seeded masks are disjoint. Identity, diagonal and full ridge C candidates are
chosen only on selection observations. Full C requires unregularized normalized
design condition <=1e6 and infinity-norm gain <=10. Regularized Gram condition
is reported separately. C targets measured singleton sums, never trained weights
or image labels. Both raw and corrected branches remain. Audit reports heldout
residuals, immediate/post-all/final effective weights, retention and zero drift.

Inference uses eight before and eight after zero reads per spatial position,
shared across T12; elapsed duration and drift are reported. New SIM/HW results
must establish this policy's behavior. Calibration availability is not accuracy.

| Component | Calls |
|---|---:|
| Inference mode10: 4,531,456 signal +602,112 zero | 5,133,568 |
| Inference fresh mode7 guards | 6,742,400 |
| Calibration mode10 including zero brackets | 502,272 |
| Calibration fresh mode7 guards | 466,608 |
| Programming, guards, up to nine feedback brackets per weight | <=1,612,800 |
| Total | <=14,457,648 |

Hard ceilings: 15,000,000 calls and 115,200 nonzero pulses. The
[hardware survey audit](../../reports/2026-10-19/survey256-hardware-review/AUDIT.json)
records mean latencies of 0.003013 s for mode10 and 0.002384 s for mode7.
Applying those measured means to inference plus calibration gives
`(5,133,568 + 502,272)*0.003013 + (6,742,400 + 466,608)*0.002384`, or
**about 9.49 hours of connector calls before programming and overhead**.
Inference alone accounts for about 8.76 hours. This is a planning estimate,
not a guaranteed minimum or ETA: writing, compression, Python work, retention
checks, pauses and latency changes add uncertainty. Do not promise completion
within a shorter session from a single timing sample.

Requests, responses and driver stdout/stderr are lossless gzip level1 chunks
<=64 MiB expanded and <=128 MiB compressed with SHA256/count manifests. Expanded
raw is capped at 32 GiB; periodic disk checks preserve 8 GiB. Full preflight
requires 72 GiB free, allowing another raw-sized copy for LFS. Results gzip uses
Git LFS. Terminal shows progress instead of millions of driver lines. Invalid
native responses are preserved before validation; partial runs retain an index
and abort manifest but cannot be classified.

`audit_native.py` streams raw independently and reconstructs both complete
`[12,64,28,28,3]` float32 tensors (7,225,344 bytes each). It rechecks guards,
coefficient identity, pulse budgets, native commands, C masks/fitting, positions
and final basis. `reconstruct.py` is the earlier structural test helper, not an
alternative hardware trust route. Historical source pins remain unchanged.

Offline regression checks (fake Connector only):

```text
python -B -m unittest tests.test_full_case0_draft tests.test_full_case0_native tests.test_cifar_case0_classifier -v
```
