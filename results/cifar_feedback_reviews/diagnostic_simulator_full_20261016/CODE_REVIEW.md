# Independent technical review: diagnostic full CIFAR

Review date: 2026-10-16. Reviewer: goal_control. Verdict: **PASS for a full native MemriBoard simulator acquisition**, not an apparatus qualification or an accuracy claim. No backend was run by this reviewer.

Reviewed source snapshot:

- `cifar_feedback_diagnostic.py`: `d4bcc3c00d9fc37552af3c2f3b205fa9bb795ce057cafedd93c864ee33069e62`
- `configs/cifar_feedback_diagnostic.json`: `c798a58227d25cc5ca7fe8a6f8b3472efb28fe90dde8759d86840e8b42e92246`
- `experiment.py`: `252332367886083b2609d222361fe17846ccc33011bc16d36610f941b4391124`

The decoder accepts C, two actual measured row sums and a positive fixed source gain. It does not accept weights or input bits. Fitted alpha, including negative values, remains diagnostic. Effective weights are a separate explanatory reference. Fresh C2 calibration is frozen before CIFAR acquisition; no CIFAR response fits it. Fixed gains are hardware 1 and simulator 0.03 V/weight; these different scales do not establish physical equivalence between the backends.

This is a distinct fresh collection of all 28 tiles and all 300 coefficients, without importing a historical pilot. The existing aggregator checks exact 600 partials per tile, unique tile/partial identities, and sums all 75 taps before trained BN and one digital LIF step. Complete output is 200 trajectories / 2400 time points. Missing measured contributions are rejected rather than filled from a CPU reference.

The new mode deliberately retains failed physical fidelity, unresolved/wrong signs, poor matrix selection quality and negative fitted alpha. These failures do not alone prevent diagnostic acquisition of later tiles. This does not modify the old pilot continuation gate and does not turn a failed weight assignment into a qualified one. Inherited transport, ADC, current, finite-resistance, budget and route-availability stops remain in force. Full limits are 580210 Connector calls and 74940 nonzero programming attempts; no automatic retry expansion is added.

Chunk review: the original Discovery stream and subsequent parts explicitly use UTF-8 with LF. Each JSON line is preserved, each part is bounded at 32 MiB, and the ordered index records per-part and concatenated SHA, bytes and record counts. Request/result pairs may cross a boundary, so the auditor must concatenate parts in index order. The finally path closes the stream and writes the index; a power loss is not claimed to produce a completed manifest. The manifest hashes all saved parts. The dedicated test covers Unicode, boundary-crossing request/result pairs and idempotent close.

Dispatch requires an explicit simulator/hardware/dry-run mode and rejects stage/pilot-session and legacy phase options. Tests reviewed include negative-alpha quality failure with 600 newly measured partials, full 300/2400 coverage, missing partial rejection, insufficient-route abort and no default I/O. Root/programmer reported 29 runtime regression tests passing and the independent auditor reported 26 tests passing; this review did not duplicate those runs.

Coverage remains an explicitly limited 118-address candidate envelope, not proof that the other 138 addresses are dead or that 118 cells can reproduce weights. The output separates RESET-probed, eligible, selected, SET-attempted and represented addresses. Maximum justified use and exclusion evidence still require interpretation of actual results.

Full hardware collection requires a separate review of the resulting full SIM audit and the actual hardware feedback pilot raw/audit when available. Technical acquisition completion, physical weight fidelity, Conv/BN/LIF agreement and whole-classifier accuracy remain distinct claims. No hardware authorization is issued by this SIM review.
