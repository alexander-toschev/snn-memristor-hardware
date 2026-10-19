# Independent review: full CIFAR case0 draft

Status: **source review complete; registered for native SIM qualification**.
The findings below were resolved in the frozen collector/auditor. No Connector,
native simulator or hardware launch was performed by this reviewer. Historical
runtime is untouched; native SIM and full replay remain execution requirements.

## Confirmed independently

- Integer TTFS scheduling matches float32 `floor((1-pixel/255)*11)` with
  `pixel/255>=0.1` for **all256 uint8 values**.
- All2352 `(tile,y,x)` schedules for fixture case0 match the independent
  reconstruction schedule. No active BL repeats within a logical tile/window.
  There are52,675 active logical tap contributions before16 filter groups.
- Geometry is64 filters×75 taps =4800 coefficients,16 groups×3 stages;
  four differential WL pairs, tile sizes32/32/11. First two stages cover256
  cells; third88 cells with unused BL commands zero.
- Collector contributions come from fresh mode10 commands. No CPU convolution
  substitution or singleton lookup was found in the actual inference path.
- Reconstruction keeps `[12,64,28,28,3]` partials. The existing digital consumer
  explicitly sums the final three-tile axis before trained BN[t]/one LIF step.
  Its trusted-import gate is still closed; structural reconstruction alone
  does not enable classification.
- Active revision changes the common input to0.05V, gains to0.125 hardware /
  0.015 simulator, and eight programming levels to
  `[0.3,0.6,0.9,1.2,1.5,2.0,2.5,3.0]`. Final consistent files and tests are
  still required before their freeze.

## Findings sent to author and resolved before freeze

1. **Independent calibration replay:** counting192 fit /96 selection /96 heldout
   observations is insufficient. Verify exact deterministic masks, repeat/index
   identities and disjointness; otherwise replayed fit inputs can masquerade as
   heldout. Author is adding this check.
2. **Programming replay:** verify ascending steps once per leg/polarity, finite
   per-coefficient budgets, and full skipped-write proof from immediately
   preceding native R. A global pulse ceiling alone is insufficient.
3. **Fresh vector guards:** matching phase/mask/filter/tap metadata does not rule
   out reuse of an earlier feedback guard with the same metadata. Require reads
   after the preceding completed vector and the correct full address coverage.
4. **Invalid native responses must survive:** `Collector.call` originally logged
   responses only after tuple-length validation, and serialized raw NaN/Inf
   directly into an `allow_nan=False` writer. Those responses would disappear
   behind an abort message. Log JSON-safe original response before validation;
   test malformed tuples and nonfinite fields.
5. **Pre-open input validation / checkpoint binding:** reject malformed images,
   nonfinite or wrong-shaped64×75 weights before opening Connector. Bind case0
   and the actual Conv1 weights to the verified checkpoint; emitting a constant
   checkpoint SHA does not prove which array was passed to `run`.
6. **Finish downstream provenance:** full audited raw export must be explicitly
   consumed by the new downstream route. Never bypass the historical import
   gate or present the draft reconstruction's declared hashes as independent
   hardware attestation.

## Execution checks after source freeze

- Run focused offline collector/auditor/trace tests and mutation cases; check
  budget arithmetic including fresh R, zero brackets and feedback (not just
  programming pulses).
- Verify shared zero-before/after acrossT12 is explicitly reported; quantify
  zero drift, heldout additivity and final-basis retention without hidden refit.
- Confirm phase and stage order, all4800 coefficients, all784 positions,
  complete raw request/response chain and rejection of any missing contribution.
- Freeze source/protocol hashes before native SIM. A complete fake trace is
  software evidence only, not native SIM or board validation.

## Standalone digital bridge, completed offline

`experiments/cifar_full_case0_20261019/classifier.py` now provides a separate
consumer. It preserves the historical classifier's closed import gate and source.
The new route requires a registered independent auditor hash, registered protocol,
completed **full** acquisition, unchanged source and result hashes, successful
stream replay and complete finite raw/calibrated tensors. Smoke data cannot be
classified. The frozen auditor is registered after final review; registration
itself does not claim successful native SIM or hardware qualification.

The collector's pinned `conv1_weights.npy` was independently compared with the
loaded checkpoint Conv1: exact equality. The complete three-tile axis is summed
before the existing pinned BN[t]/LIF/downstream implementation.

Six focused offline bridge tests passed, including an explicitly synthetic
complete auditor-output boundary test: raw CPU-matching partials retain zero
error, deliberately zeroed calibrated partials retain unit relative error and
are never replaced by CPU data. This does not replace full native trace replay.
Actual CPU case0 validation is saved in
`cpu-bridge-case0.json`: membrane, spike and TWD outputs match the checkpoint's
own forward exactly; tiled-versus-direct Conv1 maximum difference is
`5.960464477539062e-7`. No Connector or native driver was called.

**Case0 is already misclassified by the CPU checkpoint:** original test index9,
true fine/subset class0; all three CPU decoders predict class1. This is a useful
physical-versus-digital diagnostic image, not evidence of dataset accuracy.

Commands from the repository root:

```console
python experiments/cifar_full_case0_20261019/classifier.py --cpu-reference --output reports/cpu-case0-new.json
python experiments/cifar_full_case0_20261019/classifier.py --session results/cifar_full_case0_20261019/SESSION_ID --output reports/measured-case0-new.json
```

The second command deliberately rejects unregistered auditors and incomplete
sessions. Repository-local `.deps-classifier` and `.deps` supply CPU dependencies.

Final source freeze, after registration and six passing bridge tests:

- Auditor SHA256: `d3cd8bf2cf11512f81aabff0300c5fde391dad240fd9008834150e698910e717`.
- Classifier SHA256: `64ba6e85cda19367f93a0c8294f77b6c9b6f5f26cbf40d188685bc47430053e5`.
- Protocol canonical SHA256: `2cb1d57689508725f7155bf7b1d0d17fec9d4f6b65782171e2b85481bc9aa558`.

Registration affects the new route only. Successful full import emits
`registered_classifier_import_verified=true`; the auditor's legacy standalone
import flags do not grant import authority by themselves. Do not change frozen
Python sources after acquisition and then claim their fingerprints still match.
