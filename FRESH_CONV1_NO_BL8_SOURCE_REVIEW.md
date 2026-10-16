# No-BL8 source review — 16 October 2026, 20:20 MSK

**GO for one native simulator run after a clean source commit. This is not hardware GO.** The reviewer inspected the new collector, configuration, dispatcher and focused tests, verified inherited source hashes, and independently executed only the offline input/provenance loader and route/budget count. No native backend was opened.

## Frozen snapshot

| File | SHA256 |
|---|---|
| fresh_conv1_no_bl8.py | df4843f0de95933d6be7bd7cb6d3e318815f0b7053dacf321594ffa347c682bb |
| configs/fresh_conv1_no_bl8.json | 8b458cbe1048f8e18155f1edc82b2240cae889df588bfcdbb2a4bc388e62087a |
| weights/fresh_conv1_no_bl8_route.json | 5c80e32a13097f5fd0c998a4d416811ce7cb2edb00ddfbe5e298d688b5662abb |
| experiment.py | 78ab85d62914b37dd090c480e27fd46a13ffd21f8d0ee043f1622eff3bb7a012 |
| test_fresh_conv1_no_bl8.py | 18634e1ab46a4f67f6d882b6121006af145a111c75fb8af642b798464a52835f |

The programmer reports 24 offline tests passing after the metadata freeze: nine new and fifteen inherited tests. The reviewer read the tests, including actual FakeConnector collection of seven coefficients and 600 partials, exclusion before impact, default no-I/O, explicit dispatch, deadline rejection and transport-failure cleanup. This review did not rerun native simulation or duplicate those tests.

## Verified invariants

- All four filters retain taps0..74 with their trained weights. Logical tap8 remains in all four filters; physical BL8 appears in neither leg of any primary or fallback candidate. Runtime mode7 and mode10 checks additionally reject BL8 before inherited I/O.
- The deterministic filtered route has 44 tiles, 300 coefficients, 530 bounded candidate-pair attempts and 108 distinct primary addresses. These are planned addresses, not accurate-weight qualifications.
- Independent arithmetic: `530*2336 + 300*811 + 204912 = 1686292` impacts; `530*110 + 300*109 = 91000` nonzero attempts. No extra retry is added. The inherited shared budget checks occur before each impact.
- Frozen original `fresh_conv1.py`, planner, original config and original route match their expected SHA256. The new loader calls the original loader, retaining historical evidence/manifest validation. It imports input plans only, never prior measured contributions.
- Current/native-R, headroom, fresh joint quiet gates, bounded ramp/repair logic and global stops are inherited unchanged. The new launcher preserves cutoff checks before initialization/open and injects its clock into the inherited per-impact guard. Connector cleanup and incomplete raw preservation remain in the failure path.
- Each tile gets fresh calibration after its writes. Linear C is frozen before actual CIFAR responses; measured two-row sums use fixed gains0.25 for hardware and0.03 for simulator. There is no posthoc fitted-gain decoder, CPU weight substitution or input-pattern lookup. All75 contributions are summed before trained BN and one digital LIF step per logical timestep.
- Default invocation opens nothing. A distinct explicit `experiment.py --test-cifar-fresh-conv1-no-bl8` route leaves the original scenario intact. `review_pending=False` was applied only after source review; it is not a substitute for the operational hardware gate below.

## Conditions still required before hardware

The actual new native simulator session must finish and pass its independent raw/provenance/arithmetic audit. Quality failure remains visible and is not silently relaxed. The Raspberry must use the same published frozen files and pass its own dry-run/preflight. Only the root operator may start one bounded hardware run, no later than21:05MSK, targeting completion by21:30, with the enforced21:50 cutoff and already scheduled22:00 shutdown retained. A failed or incomplete run does not authorize a repetition.

This variant tests whether avoiding physical BL8 improves weight fidelity under the unchanged protocol. It does not diagnose those cells as dead, prove parallel MVM, or establish a full hardware classifier. Source-only planning/review files contain no new acquisition payload; publication of other blocked raw results is a separate unresolved approval matter.
