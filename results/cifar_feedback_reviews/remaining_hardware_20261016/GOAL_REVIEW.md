# Final goal review — 16 October 2026

**Verdict: GO for completion of the bounded physical CIFAR Conv1 experiment.** The measurement, independent verification, native-simulator comparison, and finite reserve-coverage requirements are satisfied. Final delivery remains conditional on the primary agent publishing the final reports after the secret scan and synchronizing the user's local repository. This is not a claim that the board accurately reproduces the trained model.

Reviewed hardware session: `58ca618b-c37f-454e-be25-db21d1d25f60`, source commit `418c4ad4ca1551d7733ccc4718ae85aab55937fa`, with a clean recorded tracked tree. The independent audit reports **669,378 checks, zero errors**. The comparison binds this session, its manifest, summary and audit to the corresponding completed native-simulator session `99631af1-c73e-42ed-abcd-20d4e580f691` (604,836 checks, zero errors).

## What was completed

- All four filters 0–3 contain their complete 75-tap trained target vectors: 300 unique `(filter, tap)` contributions, without dropped or duplicated coefficients. There are 2,400 full outputs: four filters × ten real patches × twelve logical timesteps × five repetitions.
- Final composition is 20 newly measured B/C coefficients, 13 retained measured A coefficients and 267 other hash-bound measured coefficients. Imported contributions remain historical measurements, not new measurements of the current final board state. The recursive ancestry retains the original 42 coefficients and accounts for failed attempts separately.
- All physical responses use the public MemriBoard Connector and native converters. Physical one-hot responses are measured afresh within each contributing acquisition, bracket-zero corrected and digitally summed. No target-weight dot product or lookup replaces an actual measured contribution.
- Each contributing tile's linear calibration matrix is frozen before its CIFAR responses. Hardware decoding uses the fixed gain 1 V per weight unit; the simulator uses its declared 0.03 V per weight unit. Fitted alpha is diagnostic only. All 75 taps are summed before trained BN[t] and one digital LIF update per logical timestep.
- B/C used 26,370 impacts and 2,163 programming pulses, with zero current skips, within the frozen limits of 40,378 impacts and 5,260 pulses. Both blocks completed. The cumulative ancestry ledger contains 420,685 impacts and 34,376 programming pulses, including noncontributing historical attempts exactly once.

## Finite maximum-use requirement

All ten previously identified useful reserves now have an outcome under the frozen A/B/C protocol: nine were freshly quiet-qualified, assigned and measured in actual CIFAR contributions; one was boundedly rejected. The final contributing-history union is **123 RESET-probed addresses, 114 ever quiet-qualified, 114 assigned and 104 receiving SET attempts**. The set of quiet-qualified but never assigned addresses is empty. Relative to A, B/C adds six assigned addresses and displaces none; relative to the adaptive baseline, A/B/C adds nine.

WL3/BL31 exhausted its 55 RESET attempts and failed the quiet-mean criterion: 0.317832 V versus a 0.000620 V limit, with sample SD 0.000510363 V. It was not assigned. This is a negative qualification of this bounded protocol and state, not proof of a dead device. Other quiet addresses fill the route, preserving every trained tap.

This closes the finite set of known, justified reserves, rather than merely maximizing a fixed pool counter. It does not prove an absolute optimum over every possible operation on all 256 cells. Earlier unsupported singleton responses and nonadditive joint responses do not justify silently treating remaining addresses as programmable linear weights. The union is across contributing acquisitions; it is neither 114 simultaneous accurate weights nor a map of the final instantaneous board state.

## Scientific result and limits

The physical run is complete, but weight fidelity fails the retained engineering criteria. Across all 300 target coefficients, raw target-gain NRMSE is approximately 1.34082 before CIFAR and 1.34211 afterwards. This failure has not been relabeled as a calibration success.

| Comparison, 2,400 points | Conv sum RMSE | Spike mismatches |
|---|---:|---:|
| Hardware versus trained reference | 0.265705291 | 45 |
| Hardware versus native simulator | 0.286371603 | 85 |
| Hardware versus its independently measured effective weights | 0.001567752 | 0 |
| Native simulator versus trained reference | 0.051685602 | 50 |

Agreement with effective weights supports the measured operation under this protocol; it does not establish accurate programming of the trained weights. The hardware-versus-trained BN current RMSE is 1.013585527; LIF pre-reset/post-reset RMSE is 1.317945922/1.287263881. Spike agreement alone would conceal these analog discrepancies. The experiment covers four Conv1 filters on ten patches, not full-network CIFAR100 classification accuracy. Sequential one-hot physical operations and temporal tiling are not parallel analog multiplication of all four complete filters. Physical pulse duration remains unknown; simulator model steps are not a measured hardware duration.

## Review evidence and delivery condition

This reviewer inspected actual compact artifacts, recomputed coefficient and address-set completeness, checked artifact hashes and comparator bindings, and used the frozen independent raw auditor's completed replay. No hardware, simulator or duplicate full raw audit was invoked for this review.

- HW manifest SHA256: `5fb5041cbdba148ff0a02182512a98ca0784fed074e2830d6fc06b230f933171`.
- HW audit SHA256: `62a3f84f10ce55eb55be4930c605108a31440359d7e46c086d5f594a48ae162e`.
- HW summary SHA256: `a57cec7992ec1418e9ec88417914108ec60a38023c64007cd5b248ac13df0202`.
- Independent auditor SHA256: `4f2f73eb96fd286aabde4e0513b7aac2ef96e57ef873ce36573a73a29a3b0525`.
- Final comparison SHA256: `8dbc84618533c99f43e69e61adf0110e98262eff9c927ecded6e12024788f788`.

Detailed requirement closure is in [COMPLETION_AUDIT.md](COMPLETION_AUDIT.md). No further physical run is required merely to convert a reported fidelity failure into a pass. Before declaring delivery complete, the primary agent must confirm final secret-checked publication and the local repository/report links. This review does not independently assert those final transport actions have already happened.
