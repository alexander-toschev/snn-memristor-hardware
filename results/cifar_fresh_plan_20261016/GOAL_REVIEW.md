# Fresh300 source gate — 16 October

**GO for one native-simulator run after the primary agent commits a clean reviewed snapshot. The final promoted suite passed15 tests in3.144s, confirmed by the implementation agent against the hashes below. Hardware GO is not granted by this document.** Hardware requires the actual native-SIM result and independent raw audit, plus time remaining before the fixed deadline.

Reviewed root files implement a separate fresh acquisition of all300 trained coefficients, four filters0–3 with75 taps each,44 microtiles, ten fixed input patches, T12 and five repetitions. No historical partial responses are imported. Historical data only determine the frozen candidate route. The complete actual output must contain200 trajectories/2400 time points.

## Closed integration checks

- One shared global counter is passed across all44 tile collectors. `impact` checks total calls and nonzero pulses before the next command and updates actual attempt deltas in `finally`, including a failed native call. Tile/quiet-attempt bounds remain additional limits. A current refusal stops the entire run; it cannot silently continue consuming a different budget.
- Deadline is checked before backend construction, before `open_port`, and before every impact using UTC plus a per-collector monotonic limit. The cutoff is16 October18:50UTC/21:50MSK. `finally` closes Connector, retains failed-tile/raw data, records completed tiles and counters and releases the lock. An incomplete route never reaches successful full aggregation.
- Default invocation opens no backend. The public `experiment.py --test-cifar-fresh-conv1` entry requires an explicit mutually exclusive mode and rejects conflicting legacy/stage/resume options. All paths are repository-relative resources; the source/config/route/evidence hashes enter the manifest. Private native sources remain subject to the existing source preflight.
- Source review permits `review_pending=false` only for these promoted reviewed bytes. Config must exactly equal the reviewed specification; routing is recomputed from pinned original weights and evidence and compared with the saved route. Fresh inputs remain bound to the inherited immutable checkpoint/fixture pins.

## Independent route and budget check

JSON-only enumeration, without invoking the collector, found300 unique `(filter,tap)` pairs, eleven microtiles per filter and537 finite candidate-pair attempts. Within every microtile all primary/fallback addresses are mutually disjoint. A fallback cannot overwrite another target's reserved primary or fallback.

The independent count is **1,702,644 maximum impacts /91,770 nonzero pulses**:

- RESET leg:4 initial calls +55×4 write/feedback +56×16 confirmation =1120.
- Candidate pair:2×1120 +3×32 joint gates =2336.
- Whole acquisition:537×2336 +300×811 SET/confirmation +204912 measurement tail.
- Pulses:537×110 RESET +300×109 SET.

Failed five-read quiet confirmation continues remaining RESET levels, never restarts the55-level sequence. Joint repair is limited to two rounds and must consume a new level on each failing leg. There are at most two predeclared candidate pairs per tap. SET uses only the planned sign(weight) leg, no reversal; initial overshoot is explicitly recorded without SET. Finite unsuccessful programming remains a scientific result; exhaustion of both quiet routes is an incomplete acquisition.

## Scientific integrity

HW fixed gain0.25 and SIM0.03 are declared before new data. A changed gain/route is a new protocol, not proof of a single isolated causal improvement over the older gain1 result. Historical endpoint screening is not a continuous programmable-range guarantee. Neither114 primary addresses nor a successful route is a count of accurately programmed weights.

All tile writes precede fresh anchors and independent calibration fit/selection. Linear C has zero affine offset and is frozen before CIFAR; inherited rank/noise/gain controls remain active. Actual contributions come only from measured physical row sums through C divided by the fixed gain. Fitted alpha is diagnostic, not a weight correction. All75 taps sum before trained BN[t] and one digital LIF step per logical timestep. Missing/duplicate600-point tile ledgers are rejected by aggregation. There is no full hardware classifier, CPU filling, per-weight digital target/response scaling or reuse of old measured contributions.

The previous JIT experiment demonstrated matched C9 improvement166%→37/40%, with unqualified targets and three quiet failures. This source gate does not assert that the new protocol has corrected those failures. Accuracy, repeatability and current-state coverage require the new data and independent analysis.

## Reviewed hashes

| File | SHA256 |
|---|---|
| `fresh_conv1.py` | `031150dbb5cfe285e5c3f8edc18757978f145ecb0b8528126ca4957371d60cfb` |
| `fresh_conv1_plan.py` | `465d9520acea8aecbb175e29aa18719b1787d17b6e4a54ba8c5577648a877aab` |
| `configs/fresh_conv1.json` | `5b47aff5d26245f8ab8aa742473c5b097d7722fe929d7c8eeaa558c935260922` |
| `weights/fresh_conv1_route.json` | `f3bc1279ec53ef76f364c2cb02ecbc0b32b2f1329bba869cf308e0edf91377f9` |
| `weights/fresh_conv1_capacity.json` | `f1c109655557e8186d7356250c80194facd7f2e5ab209fdacfe5ab0cb38c2d72` |
| `experiment.py` | `2e57a87e6d7197ff9c8ac13624b9e112c53c8a1a4345db5099ce76f5a927043a` |
| `test_fresh_conv1.py` | `6336060dba79b3caf6ca0af8777177aa23f853400b012acadb3c45e21b636c1f` |

The earlier draft suite had14 passing tests. Final promoted suite:15 tests passed in3.144s, including dispatcher/deadline coverage. The implementation agent confirmed this final snapshot rather than reusing the draft result. This reviewer performed source/route/budget checks and made no backend calls. Updated documentation SHA256 is `dd458853ea2151d7bbbbd3fa7f99b60e43cf61aa0ef55bbac835e6015324b7c6`.

## Time gate

The1.7M maximum is not an ETA. Prior observed rates are only planning evidence. Use the new native-SIM actual counts/timing and the hardware comparison to decide whether the one physical run can finish with margin. The primary agent's planning target is hardware start by20:15 and acquisition finish by21:30; hard cutoff remains21:50, with shutdown22:00 already scheduled. Do not begin late on the assumption that early stops must occur, extend the timer, or weaken checks to produce a completed label.
