# Fresh300 — hardware execution gate, 16 October

**GO for one bounded fresh300 hardware Conv1 run from the frozen reviewed source.** This gate follows the completed native-SIM experiment and independent raw audit. It does not approve automatic retries, additional addresses outside the frozen route, threshold changes, or a full hardware classifier.

The primary agent confirmed Raspberry source `a3800da9afd98928d3741101e0d96da26ab513c2`, clean tracked state, exact reviewed source/config pins and successful hardware dry-run with `hardware_access=false`,44 tiles/300 coefficients, gain0.25, maximum1,702,644 calls/91,770 pulses and cutoff18:50UTC. The source/launcher review remains [GOAL_REVIEW.md](../../cifar_fresh_plan_20261016/GOAL_REVIEW.md).

## Actual native-SIM evidence

Session `66dde3a2-2506-455e-9061-52e1865e8427` ran from the same clean source and completed44 tiles,300 unique trained `(filter,tap)` entries and200 measured trajectories/2400 time outputs. Every coefficient was newly measured; no ancestor contributions were imported. Actual acquisition:344,177 impacts,17,000 programming pulses, zero current skips, zero fallback attempts. Wall time from manifest:319.528459 seconds.

Independent audit completed **8,583,084 checks with zero errors**. This reviewer verified the actual audit's manifest binding and auditor hash, compared current source/config/route/evidence pins with the manifest, and independently enumerated exact300 coverage, trajectory completeness and tile quality. No duplicate raw audit or backend run was performed by this reviewer.

| Evidence | SHA256 |
|---|---|
| SIM manifest | `b12cf3a5387455a354e59def482a7409592a12fb2cfa866517f1ceeec73aad25` |
| SIM summary | `d0b36c8c0ab845cddd284e8cfffafbc2bfa065622a06ca73e971a282703c5f84` |
| Independent audit | `c34c38bf75bce6c66ecfb1446c4028f6fa2e5bbf5ae701af17e90571546d5c18` |
| Independent auditor source | `bfab9b2044388c11a6362e94471b7872fab134451e8c09187eda59c9e8caa5bb` |

## Quality and interpretive limits

Only2/44 SIM tiles meet the retained fidelity criteria. All300 raw target-gain NRMSE is0.327844496 before and after CIFAR; no resolved wrong signs,23 unresolved significant signs. Against the trained reference, Conv sum RMSE is0.062143894 with55/2400 spike mismatches. These failures remain explicit. Native SIM uses its fixed gain0.03; the planned hardware gain is0.25, so equal raw signal levels or accuracy are not assumed.

The purpose of the physical run is to obtain a complete new measured result under the predeclared improved programming/routing protocol, not to treat SIM arithmetic success as proof of weight accuracy. Historic endpoint capacities and114 planned primary addresses do not establish fresh programmable intervals or114 qualified weights. Fresh hardware raw/independent audit and trained/effective-weight/SIM comparisons remain required after acquisition. The second goal's improvement/repeatability claim is still open pending those results.

## Bounds and time

The frozen global maximum is1,702,644 impacts and91,770 programming pulses. Shared pre-impact counters, finite RESET confirmation/repair/candidate limits, no SET reversal, native valid-R/current/ADC guards and fail-closed route exhaustion remain unchanged. Fresh C is acquired after tile writes and frozen before its CIFAR responses; only measured row sums enter the actual output. All75 taps precede trained BN[t] and one digital LIF step.

At approximately19:39MSK there is about2h11m before the **hard21:50MSK acquisition cutoff**. SIM actual counts and prior hardware observations make one timely run reasonable; they do not guarantee a duration. The1.7M worst-case bound is not an ETA. Start promptly after the confirmed preflight and aim to finish collection by21:30, leaving room to preserve data and close Connector. Do not infer that a late start is safe merely because earlier runs stopped early.

The runtime checks UTC/monotonic time before impacts and checks the cutoff before backend construction/open. Python cannot guarantee interruption of an already blocked native call; the primary agent monitors foreground execution. At deadline retain partial results as incomplete, close Connector and preserve the already scheduled22:00MSK poweroff. Publication of separate SIM results/reviews must not change the hardware run's frozen executable snapshot.
