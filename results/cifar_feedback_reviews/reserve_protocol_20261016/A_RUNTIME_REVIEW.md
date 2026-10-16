# Reserve A: independent runtime review

2026-10-16. Goal controller; offline source/data review only. No native simulator or hardware invocation by this reviewer.

**GO for one native SIM execution of the frozen A protocol. This is not hardware GO or a claim of weight quality/maximum coverage.** Five focused offline tests passed as reported by the programmer; reviewer read their assertions and independently checked the frozen source hashes and budget arithmetic. No duplicate native run is authorized by this report.

## Snapshot

| File | SHA256 |
|---|---|
| cifar_feedback_reserve.py | a4e3a822ede1460c905435000912dc91b851b26c8d0608006ceba12708d3f7c9 |
| configs/cifar_feedback_reserve.json | d042e27faf56a0334f7669689255ea45ad1d43cface8a3fc93ffdf42f4415b9c |
| test_cifar_feedback_reserve.py | faca038ea8cf63a6e09d3353426271843eb227a912ae942a5c9c679262779948 |
| experiment.py | 718cb130a40c4e17b20041c503bb7f8557fd1d1f96abccdfe5546be0ee4e7a37 |

## Reviewed invariants

- Exactly filter3,parent4,taps52–64 are replaced. Parent identity, source, audit and manifest hashes are pinned; every parent result hash is rechecked. The historical42 ancestry is reverified against its actual immutable files, not merely copied metadata. Exact300 coverage is checked before exclusion and on new aggregation; both former hardware12+1 contributions are excluded, leaving287 taps.
- All33 candidates receive the inherited bounded RESET and independent quiet confirmation. No SET occurs before the route gate. Required reserve WL7/15 is the negative leg of tap64; other12 negative legs follow original quiet order. Positive WL6/2 and6/11 have priority only if freshly quiet; no duplicates or historical eligibility substitution.
- One sweep per candidate/pair, no adaptive retry, reversal or amplitude increase. The inherited pinned Discovery checks every nonzero command against latest valid native mode7 R, current limit, total impact/write/decision caps. Pair guards are refreshed after earlier candidate RESETs. Mode10 is zero/one-hot at0.1V; invalid responses and observed headroom abort. Raw request/result/error records remain conserved and chunk finalization is idempotent.
- Independent arithmetic using only fixture/routing JSON:33×244 RESET calls +13×811 pair calls +5728 measurement-tail calls =24323. Nonzero ceiling33×55+13×109=3232. Tail is156 mode7 +280 anchors +822 matrix probes +4470 CIFAR calls. This does not predict actual elapsed time or actual admitted writes.
- All13 assignments finish before fresh readback/anchors/C2 calibration. Matrix is frozen and its hash recorded before600 newly measured CIFAR partials. Decoder accepts only actual two-channel sums, C and fixed positive source gain; fitted alpha and trained weights do not replace measured contributions. All75 taps are summed before trained BN[t] and one digital LIF step. Final independent anchors/readback retain drift and quality failures.
- An unsuccessful route or transport leaves failed/partial new evidence, never copies excluded old13 contributions into apparent success. Baseline remains unchanged. Success records287 imported and13 fresh; cumulative totals add parent cumulative exactly once. Reused historical data are not described as current array state.
- Dispatch requires explicit mode, excludes conflicting legacy/stage options and has no default I/O. Native source is selected through public Connector; no private interface/driver call is introduced.

## Conditions and limits

The independent reserve auditor must validate real native SIM output before a hardware GO. The simulator may fail the reserve quiet gate: that is preserved evidence, not permission to relax the gate or invent a successful route. A transport/guard failure is not an accuracy-only failure. Failed weight fidelity remains diagnostic and does not silently alter the protocol.

A is only the WL7 capacity check and two WL6 unused addresses. The separately proposed finite B/C package covers the remaining known quiet-unused and historically supported outside-pool addresses. Neither A nor B/C proves global optimality over every state and mode of all256 cells. Historical zero-only cells are not labelled dead.
