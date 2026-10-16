# Adaptive runtime review — 16 October 2026

Read-only review completed for cifar_feedback_adaptive.py, config, experiment.py dispatch and seven focused tests. **Final native-SIM gate: PASS.** No blocking source finding. Programmer reports7/7 focused offline tests PASS; reviewer independently matched all four frozen hashes below. Hardware approval remains separate after measured SIM audit.

- runtime: `a46eb7b94891d1adf48d10f126227e10ecfa696aba61a98f293199a8dcb9f2ba`
- config: `c788a91a1dc51e4a9f6070354c9597249a45b4770decb07847030f683ef09663`
- dispatcher: `83fd05070e09de9f296d143803be44c0ac07f0366fdffee2a6edd62f4b3f6e71`
- test: `fb609ea238928499494ce9908af028b0ae70aa0bd44284e559ba9ae75c12affb`

The collector screens the complete original candidate pool on each of at most two microtile attempts, chooses first-k pending taps and first-k quiet candidates, and preserves all native quiet/current/ADC constraints. Zero capacity aborts immediately. A residual after two attempts aborts with the unmeasured taps disclosed. There is no third sweep, hidden retry or silent coefficient drop. Impact/write/decision budgets are reduced by cumulative attempts and enforced by inherited per-command guards.

The previously audited42 taps are optionally imported only from the exact source-specific parent with pinned manifest and independent-audit SHA. All parent result files are rehashed, and each prefix tile must contain the exact14 trained taps and complete600-point partial ledger. Failed parent f0_t3 contributes no inference values; its attempts remain in cumulative history. The full parent state is not claimed to persist. Source-specific SIM restart state is explicitly disclosed.

Microtile routing, fresh post-write anchors, calibration and fixed gain are frozen before CIFAR. Matrix selection now explicitly checks reference design rank: a noisy rank-two measured design cannot identify a full matrix from rank-one singleton targets. N=1 uses eligible identity/diagonal candidates. The old pinned helper remains unchanged.

The new aggregator rejects duplicate microtile identities, wrong parent/tap assignment, changed trained weights, incomplete microtiles and missing/duplicated taps. Each filter must contain every original coefficient exactly once. Existing strict partial-ledger validation then requires600 real partials per contributing tile and sums all75 coefficients before trained BN and one digital LIF step. It never fills a missing contribution with a CPU result or zero.

Completed-microtile checkpoint replacement is atomic; raw and tile files retain original acquisition identities. Exceptions preserve counters, failed tile data, chunk index and manifest. Old input/session files are read-only. Hardware lock and public native Connector selection are unchanged. Dispatch requires an explicit backend/dry-run and permits resume only with the adaptive scenario.

Independent fixture-based budget review confirmed868466/117180/56 for fresh all300 and756104/101382/50 for new258 after imported42 (impacts/writes/microtiles). Historical parent attempts are additional explicitly counted history.

The checkpoint embeds full tile objects. Prior full SIM tile JSON files totaled44.45MB; adaptive splitting adds overhead. Check actual completed checkpoint size before publication against GitHub's file-size limit; this is a publication check, not a demonstrated acquisition blocker. Raw logs retain32MiB parts.

No backend or hardware action was performed by this reviewer. Existing v1 comparator remains deliberately strict; adaptive results require the separately reviewed comparison/audit schema rather than weakening old checks.
