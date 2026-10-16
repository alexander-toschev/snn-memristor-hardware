# Reserve A hardware: goal-control review

2026-10-16. Offline read/hash/compact arithmetic review; no backend call or duplicate raw replay. **A acquisition and the three designated reserve dispositions are complete. Overall finite reserve goal remains active: seven B/C addresses still require the frozen follow-up.** This is not a weight-quality success or B/C hardware GO.

## Evidence and completeness

Hardware session `e17464e9-a601-4b7e-adf5-8a1d56de41d2`, source `082452f542500c7f048a4cecc98319e73edbd736`, tracked clean. Independent audit PASS **406552 checks/0 errors**. Actual audit SHA `eeeb47b7a4f5cd98fec4fc99fc749b7d9af13ffe529f09e0aa1213d921d095a9`; manifest `83afba83e474cbd453fb82c5785123f9751eca32605cb33cb9a725af8f8cf9fe`; pinned auditor `372f48a05abc652cda0f2c9d5891660a9c23429f53d18ea84749c587f5e10459`.

Reviewed A runtime/config hashes match the manifest and local bytes. Summary, completed wrappers, new tile and frozen route/matrix/calibration hashes were independently checked. Compact wrappers contain exact300 unique `(filter,tap)`,75 per filter;200 trajectories/2400 outputs. Only13 taps52–64 are fresh, with600 actual partials;287 retain audited historical provenance. The former hardware12+1 chunks were replaced, not duplicated.

New15911 calls/1408 nonzero writes/1408 decisions/0 skips are below24323/3232. All-ancestry totals394315 calls/32213 writes include earlier failed attempts once. Quiet screening found15 positive and14 negative candidates, including the new reserve; thirteen assigned pairs were measured after all programming, in one microtile. Native one-hot reads remain sequential, not a parallel32-input MVM.

## Actual reserve dispositions

| Cell | Fresh quiet | New assignment | SET on this cell | Result |
|---|---|---|---:|---|
| WL6/BL2 | yes | positive tap52 |109|SET exhausted; target+.07804853, raw measured pair weight−.000308|
| WL6/BL11 | yes | positive tap53 |109|SET exhausted; target+.02827612, raw measured pair weight0|
| WL7/BL15 | yes, after13 RESET writes | negative tap64 |0|Pair hit; target+.06067321, raw measured pair weight+.057132|

All three were actually assigned and measured, rather than merely probed. The two quiet WL6 cells did not reproduce their desired pair weights. The quiet/assignment gate is therefore not a programmability or accuracy claim. WL7/15's zero SET count is expected: the residual selected the positive leg; a write on the reserve solely for a counter would be inappropriate.

The contributing union is119 RESET-probed,111 ever quiet,108 assigned,101 SET-attempted. Relative to the prior baseline, assigned105→108 adds exactly WL6/2,WL6/11,WL7/15; no former assigned address is displaced. These are unions across contributing historical and fresh chunks, not simultaneous array capacity or108 accurately stored weights.

## Quality and comparison

Fresh raw-target NRMSE before/after CIFAR is1.42267742/1.41639036; uncertaintyNRMS.01161001/.01436901. Wrong resolved sign remains slot3 (tap55); significant unresolved signs remain slots0,1,11 (taps52,53,63). Fresh fidelity fails. Selected C is diagonal `[1.0011794674,1.0024630506]`, selection-pass=true; that small measurement correction does not repair programming failure.

The comparison report at `results/cifar_feedback_reviews/reserve_hardware_vs_simulator_20261016/comparison.json` has SHA `ebd30df77c6814f8b87fc77fe0b8652a6354e3204e2689cf437c6aa5313ed639` and matching hardware audit/manifest bindings. Corrected-effective weight metrics in that report differ slightly from raw-target fidelity above; they must not be conflated.

| Comparison of complete derived outputs | Conv RMSE | BN RMSE | Spike differences/2400 |
|---|---:|---:|---:|
| HW vs trained |.26539069|1.0152302|45|
| HW vs own measured effective weights |.0015471237|.004844813|0|
| HW vs native SIM |.28723731|1.0766082|90|

Agreement with own effective weights supports the measured operator's consistency, not equivalence to trained weights or classifier accuracy. Imported287 data remain historical; five repeats are not assumed independent. Physical pulse duration remains unknown.

## Finite remaining goal

Three of ten predetermined reserves now have fresh hardware dispositions. B/C covers exactly WL4/7,WL5/6,WL5/15,WL2/3,WL3/12,WL3/13,WL3/31 with B11+C9, maximum40378 calls/5260 writes. Final source-specific actual parent bindings, dispatcher and independent auditor must be reviewed and frozen; then native SIM plus raw audit precedes B/C hardware review.

On B/C completion, retain full300/2400 measured composition, independent comparison and quality limitations; report all ten dispositions and recalculate actual unique usage after replacement. No new reserve search or retries are added because weights are inaccurate. A failure of scientific capacity and an unassessed technical stop remain distinct, with no silent old-contribution fallback. Overall completion is not asserted by this A-only review.
