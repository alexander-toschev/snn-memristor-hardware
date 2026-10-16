# Adaptive recovery: design review, not execution approval

The parent reports failed full hardware session c0f8f6d3-9087-42c7-a586-1656b8d96f5e: original f0_t0/t1/t2 completed, f0_t3 stopped before programming because 13/17 positive and16/17 negative candidates passed quiet screening versus14 required. This report has not yet been independently checked against local raw. No repeated run is proposed.

Approved design direction: preserve quiet and current/ADC guards. For every original chunk, permit at most2 microtiles. Each attempt screens the complete original pool in its frozen order. Choose the largest available k=min(pending coefficients, positive eligible, negative eligible), take the firstk pending coefficients and firstk eligible cells on each side, freeze routing, then program and measure. Zero capacity stops. Any pending coefficients after the second attempt stop; no silent dropping, CPU filling or third screening until success.

Reuse of42 coefficients is conditional on immutable raw/audit proof of the first3 completed original tiles and an explicit new parent import schema. The failed fourth tile supplies no convolution contribution, but all its commands count in the combined history. New data cover the remaining258 coefficients. Source ledgers must assign every (filter,tap) exactly once among all300. Different acquisition times are disclosed; old states are not claimed to persist on the board.

Each microtile gets fresh post-all-write anchors and C2 calibration before fresh CIFAR inputs. Fixed gain remains unchanged, fitted alpha remains diagnostic. All75 coefficients are summed before trained BN and one LIF step. The new code must not mutate old runtime/results or weaken v1 import/audit checks.

Independent budget calculation from saved routing/fixture, using only stdlib JSON reads and enumerating all contiguous one-cut partitions, confirms:

| Scope | Maximum impacts | Maximum nonzero writes | Maximum microtiles |
|---|---:|---:|---:|
| Fresh all300 |868466|117180|56|
| New258 after immutable42 |756104|101382|50|
| Original f0_t3 alone |37374|5266|2|

Per-attempt candidate screen is244 impacts per candidate, with the original2(N+3) candidates even for a small remainder. Per coefficient:811 pair-programming calls,12 mode7 readback calls. Anchors contribute20(n+1) per microtile; synthetic C2 masks and actual fixture bit patterns were counted explicitly. All prior parent attempts, including the failed route, remain additional historical totals rather than being discarded from accounting.

N=1 matrix identifiability: noisy observed X can have rank2, but Hmask targets Y have rank1. A full ridge correction therefore has rank1 and must fail the existing minimum-singular-value constraint. Request an explicit adversarial noisy-X/rank1-Y regression and clear N1 identity/diagonal restriction in the new adapter; do not change pinned historical mathematics. A full matrix is not identified by input noise.

The existing compare_cifar_diagnostic.py/tests received read-only PASS for their declared full-fresh-v1 scope: strict hashes/audit binding, exact2400 outputs, no CPU/fitted-alpha flag, recomputed aligned metrics and repeatSD. Their fixed28tiles/16800partials/no-parent requirements intentionally exclude adaptive sessions; a separately reviewed comparison adapter/schema will be needed.

Status: design/budget review PASS; runtime/test review pending implementation, then native SIM and independent audit before any new hardware decision. No backend was called by this reviewer.
