# Independent B/C auditor review

2026-10-16. Offline review only; no acquisition/backend/Git action, no changes to A files.

Reviewed public draft: auditor `b2ceaafb692257df1993e9f90a37b0a6bf18038ca1dcc8ea28f85831b4daaf35`, tests `93bd2b78711ac8c71b412c130600c4bb25e1828e268f7d23b0829b19248e49cc`, draft document `4eb5a3d741c7021fd18c8347a2ba73907dcdac736bc83faa51d40b481682aab2`. Thirteen tests passed as reported; source assertions reviewed, no duplicate full synthetic/native replay performed.

## Final verification of the two fixes

Both findings below are **closed** in auditor `0b1093add45acf1d007f0dea0ef055c61c6d2e62404825bc1b1ed9046e452448`, tests `d038d64d6c0ad1c6f457cc304df6efd30f6bd5382c2fdcc78cfcc7be75cfaaf5`, document `d7da76a496a8234bbc3715a2f9fd545b6f5c03d8b30f76d09ca1fa04ac1e61e0`. Actual hashes were verified.

`verify_manifest_contract` now requires completed summary/checkpoint commitments and commitments for any existing root summary/checkpoint/capacity-failure artifacts. It checks all five manifest budget/coefficient declarations. `verify_manifest` invokes this after hashing listed files. The reviewer independently ran the three new focused regression tests: **3 PASS,0 failures,0 skips**; these exercise two completed commitments, three partial artifacts and five declaration mutations. The author reports the expanded full16-test suite PASS; it was not redundantly rerun here.

## Verdict

**GO for prebinding finalization: no remaining issue from this review.** Actual-session validation remains intentionally disabled by empty FINAL_PINS/FINAL_PROTOCOL. This report is not a native SIM or hardware GO. The fixes did not require changes to the separately frozen A protocol or its existing audit.

## Historical findings, now closed

1. `verify_manifest` requires manifest commitments for protocol, INI, start and imported data, but omits `completed_microtiles.json` and `summary.json`. Require both for a completed acquisition, and require an entry whenever a partial checkpoint or `capacity_failures.json` exists. `verify_files` currently checks every listed hash and all tile files, but not these unlisted root artifacts. Later independent arithmetic does check contents; the missing check concerns the promised manifest binding, not fabrication of an unverified mathematical result. Add regressions removing each required commitment.
2. Check manifest declarations against the independently fixed contract: maximum impacts40378, pulses5260, decisions5260, new20 and imported280 coefficients. The draft checks the plan and actual measured counters, while its manifest declarations may disagree unnoticed. Add a mutation test for each declaration. This is report/provenance consistency, not an uncontrolled physical-budget bug.

Both items were sent to the independent-auditor author. Neither requires modifying collectors or the pinned A helper.

## Checked algorithm and scope

The auditor imports only SHA-pinned independent audit helpers. It replays request/result raw, conversions, native-R current guards and bounded RESET/SET, fresh quiet confirmation, exact candidate order and pools, preferred-slot reservations before ordinary fill, all actual matrix/anchors/CIFAR/final-read commands. It requires consumed raw events, exact partial ledgers and fixed-gain fidelity, without importing collector arithmetic or native backend packages.

B has29 candidates/11 coefficients; C27/9. Budgets are independently derived from immutable real routing/fixture and checked against21421/2794 and18957/2466. Aggregate limit40378/5260 and B-before-C monotonic order are checked. Extra block/retry directories are rejected. All seven remaining known reserves have deterministic raw-derived dispositions.

Scientific capacity failures require complete screening, no SET/pairs/partials/frozen route, and no full summary. B capacity failure permits exactly the scheduled C; incomplete C does not create full300. Interrupted/transport-error ledgers fail closed under the inherited replay and are explicitly documented as unsupported partial-audit cases, not consistency PASS. This conservative limitation is acceptable provided it remains explicit.

Parent manifest and audit hashes, source/session/status/counters, every bound result file and pinned auditor identity are verified. Completed-A and baseline-plus-failed-A are distinct branches. The A helper verifies recursive42 ancestry, old source snapshots and historical warnings. Immediate wrapper origins are matched to their actual files and result bodies. B11+C9 exclusion leaves exact280; failure cannot quietly restore excluded old blocks. On complete new B/C, independent sum75→trained BN→LIF reconstructs2400 outputs. Failed-A adds only new attempts once to baseline cumulative history.

Synthetic tests demonstrate raw mutation rejection, candidate/route corruption, failed-B fabricated partial rejection, cross-source/tampered ancestry and exact coverage arithmetic. They do not establish end-to-end actual parent binding for a future hardware session. Explicit synthetic markers and disabled actual entrypoint prevent that claim.

## Remaining goal conditions

- Obtain and independently audit the one bounded physical A result; currently only native SIM A exists. Preserve a failed A as evidence, without relabelling baseline or using automatic fallback.
- Fix the two auditor checks, bind B/C to the actual reviewed source-specific A or baseline+failed-A evidence, freeze final source/config/dispatcher/auditor hashes and exercise actual import branches.
- Run bounded native SIM B/C and audit before hardware review. Then perform the reviewed finite physical B/C package; no new candidates or retries are added after observing accuracy.
- Keep baseline full300/2400 result and its quality failure distinct from new composition success. Report all ten predetermined reserve dispositions, actual contribution/SET/quality, unique contributing-cell union and tile capacity. Do not assume historical union counts add arithmetically after replacement.
- Complete comparison/report publication with source provenance and remaining limitations. Accuracy failure itself is a valid experimental outcome, but technical or unassessed reserve gaps must not be silently labelled completed.
