# B/C native SIM: final hardware gate

2026-10-16. **GO for one bounded hardware B/C execution after Raspberry preflight confirms the frozen source, INI and exact hardware-A parent.** No retry, additional address, larger budget or threshold relaxation is authorized by this review. Overall goal is not complete; physical B/C evidence and its audit/comparison remain outstanding.

Reviewer used offline reads, hash checks and compact coverage arithmetic. The independent raw audit below was already run by the primary agent; no duplicate raw replay, native SIM or hardware call was made by this reviewer.

## Verified evidence

Native SIM session `99631af1-c73e-42ed-abcd-20d4e580f691`, clean tracked source `7eb10233f70953235e0453031b3d8d4625c85bcd`. Manifest SHA `e5d7e2e6ce090100c400d928c440e4aac0ee51ae85e1b086e457bd218fa0c9ea`.

Independent audit **PASS604836 checks/0 errors**, SHA `008a0bd61d5988ceae2af0b9da17301a2a55e49cca797270d451b5364305d3cb`. Its manifest binding and frozen auditor SHA `4f2f73eb96fd286aabde4e0513b7aac2ef96e57ef873ce36573a73a29a3b0525` match local bytes. No capacity failures. The current source inventory matches runtime4403648b…, plan e7ed4f73…, config ebab6a06… and dispatcher632c534f… listed fully in `reserve_protocol_20261016/BC_FINAL_RUNTIME_REVIEW.md`.

Independently checked manifest commitments for summary, completed/imported wrappers, protocol/start records, each new tile and its frozen route/matrix/calibration/anchors/raw-part index. Wrappers contain exactly300 unique `(filter,tap)`,75 per filter, with280 historical and20 fresh;200 trajectories/2400 outputs. Both B11 and C9 produced600 actual partials. The retained A13 and historical42 lineage remain imported evidence, not fresh board state.

## Actual SIM outcome

| Block | Calls | Nonzero writes | Fresh weights / partials | Raw-target NRMSE | UncertaintyNRMS |
|---|---:|---:|---:|---:|---:|
| B |13241|1187|11 /600|.41393443|.16376618|
| C |10541|818|9 /600|.31069630|.20309892|
| Total |23782|2005|20 /1200|separate block criteria|separate block criteria|

Zero current skips; totals below40378 calls/5260 writes/decisions. Parent ancestry is counted once:596184 cumulative calls/38898 writes. Both matrices are identity with selection-pass; gain is the predeclared SIM0.03. Route/calibration freezes precede subsequent programming/CIFAR tasks respectively: B task2985/8706, C2429/6208. Each new tile retains final independent readback and anchors.

All seven preferred addresses were freshly quiet, assigned and actually measured: WL4/7,WL5/6,WL5/15,WL2/3,WL3/12,WL3/13,WL3/31. SET was not forced on every leg; WL4/7,WL3/12 andWL3/13 correctly received no SET when residual-selected polarity used the other leg.

Both blocks fail weight-fidelity criteria. There are no resolved wrong signs; B slot5 and C slot1 remain significant-sign unresolved. Before/after NRMSE is equal in this SIM result. These are retained diagnostic results, not grounds to change gain, quiet limits or write budget. Simulator/physical states and pulse durations are not assumed equivalent; hardware retains fixed gain1 and unknown physical pulse duration.

## Hardware conditions

Only the already reviewed B and C pools/routes are allowed:29 and27 candidates, one bounded RESET sweep per candidate and one residual-selected SET sweep per assignment. Every candidate must obtain fresh quiet evidence. Preferred cells reserve their slots before fallback; nonquiet preference does not veto other valid capacity. Actual native-R/current/ADC/transport guards remain unchanged.

Scientific B capacity failure is saved before SET and permits the once-scheduled independent C; other technical/safety errors stop all. Missing fresh contributions are never replaced with old B/C values or software weights. A partial package cannot claim a new full300 result.

After physical B/C, independently audit raw, compare hardware with native SIM/trained/own-effective weights, and report all ten finite-reserve dispositions together with A. Recompute actual assigned union and displaced addresses; do not infer accurate storage or global optimality from probe/SET counts. No new posthoc reserve search is part of this final package.
