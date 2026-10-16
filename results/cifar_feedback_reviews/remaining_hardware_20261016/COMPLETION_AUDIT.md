# Completion audit — physical CIFAR preparation and experiment

Scope: the user's 16 October objective and the frozen [finite A/B/C scope](../reserve_protocol_20261016/FINAL_SCOPE.md). Reviewed final HW session `58ca618b-c37f-454e-be25-db21d1d25f60` and native SIM session `99631af1-c73e-42ed-abcd-20d4e580f691`. **Measurement requirements are complete; final publication and local delivery require the primary agent's confirmation.** Accuracy failures remain scientific results and are not silently converted into success.

## Requirement matrix

| Requirement | Actual evidence | Status |
|---|---|---|
| Four CIFAR100 Conv1 filters, all coefficients | Exact unique ledger 0–3 × taps 0–74, 300 coefficients | Complete |
| Ten real patches, T12, five repetitions | 200 trajectories of 12 steps, 2,400 outputs; prepared fixture and trained artifacts are hash-bound | Complete |
| Physical trained-weight programming and measured responses | Every tap has a physical assignment and actual measured CIFAR contributions; misses/exhaustion remain recorded | Complete acquisition; fidelity failed |
| No software replacement of missing hardware terms | Actual branch consumes measured ADC responses and frozen C/gain; target W dot products are reference-only | Complete |
| Calibration separated from CIFAR | Fresh post-write anchors and synthetic calibration/selection precede frozen matrix and CIFAR; linear C, zero affine offset | Complete |
| Fixed physical units | HW gain 1; SIM gain 0.03; fitted alpha diagnostic only, no per-coefficient target/response decoding | Complete |
| LIF semantics | Sum all 75 taps before trained BN[t]; one digital leak/reset step per logical t, not per physical microcommand | Complete |
| Native MemriBoard integration | Reviewed and pinned public Connector/converters; INI-selected native backend; frozen guard logic and raw tasks | Complete |
| Bounded execution and errors | B/C 26,370/40,378 impacts, 2,163/5,260 pulses, zero current skips; both blocks complete; raw audit zero errors | Complete |
| Native SIM comparison | Corresponding completed native SIM, independent audit 604,836/0; final 2,400-point comparison | Complete |
| Provenance and immutable imports | Hardware source 418c4ad… clean; 20 fresh +280 imported, including A13; complete recursive original ancestry and exact hashes | Complete |
| Maximal justified use of known reserves | Ten of ten dispositions; nine actual assignments and one bounded negative screen; no ever-quiet unused address in the contributing union | Complete within declared operation/scope |
| Preserve quality failures | Mean/noise/sign and Conv/BN/LIF metrics retained; no 20% threshold relaxation or claim of trained-model equivalence | Complete |
| Reproducible repository path | Acquisition source was committed, checked out on Pi and launched through experiment.py; configs, weights, raw, audit and report references use repository paths | Complete for recorded execution |
| Final secret scan, publication and local user delivery | Raw/results available in review checkout; final review/report publication and local sync are performed by the primary agent | Pending delivery confirmation |

## Exact composition and accounting

The final result retains **267 original contributions +13 measured A replacements +20 measured B/C replacements =300**. A's old corresponding 13 contributions and B/C's old corresponding 20 contributions are removed from the mathematical sum, not counted twice. Each filter still has 75 coefficients. Imports retain their own measurement times and matrices; they are not presented as new measurements after later programming.

B contains filter 2 taps 0–10, eleven fresh coefficients; C contains filter 1 taps 0–8, nine fresh coefficients. Both blocks supply 600 actual partial points, for 1,200 fresh partials. There is no capacity-failure fallback to old B/C values. All final 2,400 full outputs are available.

The A parent ledger was 394,315 impacts and 32,213 pulses; adding B/C's 26,370 and 2,163 gives **420,685 impacts and 34,376 pulses** across the recorded ancestry. This ledger includes failed/unimported attempts and must not be described as only the commands contributing numerically to the final 300 coefficients.

## All ten known reserve dispositions

Addresses below are `(WL, BL)`. Assignment means a measured differential leg, not independently proven target-weight accuracy. A zero SET count is valid when the chosen feedback direction programs the other leg; no artificial SET is added to inflate coverage.

| Stage | Address | Assigned filter/tap | Actual outcome | SET attempts in that assignment |
|---|---|---|---|---:|
| A | (6,2) | 3/52 | Fresh quiet, assigned, CIFAR measured; target fidelity failed | 109 |
| A | (6,11) | 3/53 | Fresh quiet, assigned, CIFAR measured; target fidelity failed | 109 |
| A | (7,15) | 3/64 | Fresh quiet, assigned, CIFAR measured | 0 |
| B | (4,7) | 2/0 | Fresh quiet, assigned, CIFAR measured | 0 |
| B | (5,6) | 2/0 | Fresh quiet, assigned, CIFAR measured | 61 |
| B | (5,15) | 2/1 | Fresh quiet, assigned, CIFAR measured | 84 |
| C | (2,3) | 1/0 | Fresh quiet, assigned, CIFAR measured | 109 |
| C | (3,12) | 1/0 | Fresh quiet, assigned, CIFAR measured | 0 |
| C | (3,13) | 1/1 | Fresh quiet, assigned, CIFAR measured | 0 |
| C | (3,31) | None | Not freshly quiet after 55 RESET attempts; fixed fallback preserves tap coverage | 0 |

For (3,31), independent confirmations give mean 0.317832 V and sample SD 0.000510363 V; the absolute-mean and SD limits are each 0.000620 V. Its mean fails. No retry, larger voltage budget or threshold relaxation was added after seeing this result.

Contributing-history address union: **123 screened, 114 ever quiet-qualified, 114 assigned, 104 SET-attempted**. Set comparison gives zero quiet-qualified-but-unused addresses. A/B/C adds nine actual assignments over the 105-address adaptive baseline; B/C alone adds six over A's 108 and removes none. These are unions across different states and times. They do not mean all 114 coexist as accurate weights. The remaining 133 array addresses are outside this finite qualified operator scope, not classified as dead. Earlier zero singleton/nonadditive joint responses do not supply a valid linear weight assignment. No absolute global optimum across arbitrary drivers, encodings or all 256 devices is claimed.

## Source, matrix and provenance checks

Final source and input pins were checked against actual files and recorded manifest bindings:

| Artifact | SHA256 |
|---|---|
| B/C runtime | `4403648b91c9708724ad282f12a5d4c1cc0e37b9f75fc7c49568cdb2656e7724` |
| Frozen reserve plan | `e7ed4f737268b2555d7d02bb4d97f84ff52484e77f323896eae5dafff4ab456a` |
| B/C config | `ebab6a06cdad535d9d12c74917b474f517e8f394feb6f01c78dd81665dc93437` |
| experiment.py | `632c534f825750948aa089914aaae107e6b138b73a64181284caf7cf9a59e54a` |
| Independent B/C auditor | `4f2f73eb96fd286aabde4e0513b7aac2ef96e57ef873ce36573a73a29a3b0525` |
| HW manifest | `5fb5041cbdba148ff0a02182512a98ca0784fed074e2830d6fc06b230f933171` |
| HW independent audit | `62a3f84f10ce55eb55be4930c605108a31440359d7e46c086d5f594a48ae162e` |
| HW completed microtiles | `9ef8ceb148d8b354c1fd9250547ce401a9ab9b72899f34cf2dd2aab491a1adfe` |
| HW summary | `a57cec7992ec1418e9ec88417914108ec60a38023c64007cd5b248ac13df0202` |
| Final comparator | `8d851e8721e4febf4a3a6d2b00bb699e7701119aad21dd2344a291ca20c5996a` |
| Final comparison JSON | `8dbc84618533c99f43e69e61adf0110e98262eff9c927ecded6e12024788f788` |

The final hardware audit reports 669,378 checks, zero errors and no dirty tracked paths. The comparison's manifest, audit, summary, auditor and comparator references match the actual files. New raw-part indexes and tile/route/calibration artifacts are bound by the manifest. This reviewer did compact recomputation and binding checks, not a second full raw replay; the cited independent audit performed that replay.

B's route freezes at impact count 1,313 and calibration at 9,533; C's corresponding counts are 2,125 and 7,969, before their CIFAR measurements. B selects diagonal C approximately `[1.8, 1.0011993643]`; C selects `[1.0015285670, 1.0022380451]`, both with independent selection pass. B's first diagonal entry is materially different from one and is not described as evidence of physical crosstalk. Selection pass does not certify trained-weight fidelity. Both use fixed HW decoder gain 1.

Historical native-SIM ancestry includes the explicitly disclosed narrow nonexecuted-validator provenance exception from the adaptive run. It was not erased or converted into a clean-launch claim. The final B/C hardware launch itself is clean. Independent parent/raw verification and immutable hashes preserve the imported lineage.

## Interpretation and completion decision

The detailed comparisons are in [REPORT.md](../remaining_hardware_vs_simulator_20261016/REPORT.md). Hardware/trained Conv RMSE is 0.265705291 with 45/2,400 spike mismatches; hardware/native-SIM Conv RMSE is 0.286371603 with 85 mismatches. Hardware versus its independently measured effective weights has Conv RMSE 0.001567752 and zero spike mismatches. This separates a working measured operation from poor reproduction of trained weights.

All-300 raw target-gain NRMSE is about 1.34082 before CIFAR and 1.34211 after it. The fixed 20% engineering criterion remains failed. No success claim depends on concealing that failure, replacing measurements, rescaling by fitted alpha or calling an unexcited LIF trace a general accuracy result. The scope is four Conv1 filters on ten patches, not full CIFAR classification or simultaneous residency of four complete 75-tap filters.

**GO for completion of the agreed physical experiment and its finite maximum-use investigation.** No outstanding measurement requirement was found, and poor fidelity alone does not require another run under this task. The primary agent must still confirm the final secret scan, publication of these reports and local delivery before calling the whole user-facing delivery complete. New work on improved weight programming would be a distinct follow-up, not a hidden prerequisite added to this completed data collection.
