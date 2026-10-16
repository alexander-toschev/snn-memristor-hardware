# JIT hardware goal review and next Conv1 requirements

Reviewed session `ceedcbde-8443-4518-ae4e-8d901f2b3ddd`, source `f164bf0b42677962c63f95aec88d40ec5f221b61`. Independent audit: **419,753 checks, zero errors**; 16,589 impacts, 1,295 programming pulses, zero current skips. Eighteen pair decisions completed: fifteen SET blocks and three explicitly recorded joint-quiet failures. No CIFAR was measured in this protocol.

**Point1 verdict: an observed improvement on the matched C9 targets is demonstrated; reliable accurate programming of all nine targets is not demonstrated.** This is not a new all300 result and does not replace the historical approximately134% all300 NRMSE.

## Matched evidence

The comparison uses the same nine trained targets, same physical route and fixed HW gain1, without a fitted alpha or calibration matrix. Historical C9 raw NRMSE was1.660439874 before CIFAR and1.659568548 afterwards. JIT cycle-final NRMSE is **0.370541115 and0.401505995**. Both improve markedly on that historical measurement, but remain above the unchanged engineering criterion. The comparison is not a randomized causal control: changed acquisition order, elapsed state and two fixed-order cycles cannot establish which physical mechanism caused the improvement.

The major previous wrong-sign tap5 changed from approximately−0.20021 to+0.047614/+0.048708 for trained target+0.0505213. Neither JIT cycle ends with a resolved wrong sign. Nevertheless significant signs remain unresolved at taps0/7 in cycle1 and0/2/7 in cycle2. Noise/quantization uncertainty NRMS is0.010038/0.009995; low read noise does not repair systematic target error.

| Target | Actual observation | Interpretation |
|---|---|---|
| tap0,+0.0224526; positive WL2/BL3 | Both cycles exhaust109 SET attempts; final response0 | Quiet/readable does not establish programmability at this target |
| tap6,+0.105269; positive WL2/BL12 | Both cycles exhaust109 SET attempts; final+0.074224/+0.073062 | Target not reached within the frozen budget; do not increase budget implicitly |
| tap7,+0.0422894; positive WL2/BL14 | Joint quiet means0.000918/0.001100V exceed0.000620V; SET withheld in both cycles | Guard outcome retained, not a missing coefficient to fill with CPU |
| tap2,−0.0225791; negative WL3/BL2 | Cycle1 programmed; cycle2 joint quiet mean0.001346V exceeds0.000620V and SET withheld | Cycle-dependent reset qualification; final responses−0.022768/−0.000858 |

Cycle-end weight-vector RMS difference is0.007370435, normalized to target RMS0.144863943. Tap2 alone changes by0.02191 weight units. Thus a blanket repeatability claim is not justified. Gate failures are based on each physical leg, not a small difference between large responses. No gate or tolerance should be relaxed merely to conceal these outcomes. Failed quiet/SET results are state- and protocol-specific, not proof of dead devices.

The scientist's detailed disturbance map can resolve immediate/retention/later-write behavior. This review independently checked the compact cycle/gate/program results and recomputed the matched vector difference; it relies on the completed independent raw replay rather than duplicating it.

## Conditions for the next fresh Conv1 protocol

This section is a review checklist, **not authorization for an additional run or an expansion of scope**. The primary agent will review a separately frozen proposal.

1. Preserve all300 trained coefficients and all75 taps per filter. Every replacement route, capacity rule and treatment of a quiet/target failure must be declared before that acquisition. A reported diagnostic no-SET state may be measured honestly; it must not masquerade as the target weight. No dropped target, CPU fill or unreported historical reuse. Known unresolved taps require an explicit disposition in the proposal, not an assumption that JIT cured them.
2. Keep raw target-gain fidelity separate from corrected effective weights. Freeze physical gain/encoding before the new observations; no post-hoc alpha, absolute-value sign repair or per-weight target/measured scaling. Weight-aware routing, if introduced, must be declared and pinned and must still produce new physical responses.
3. After the actual final writes for each contributing state, acquire fresh independent anchors and calibration fit/selection data. Freeze C before that tile's CIFAR responses. Do not select C, mapping, scale or an apparent best programming cycle on the final CIFAR observations. If later writes intervene, distinguish the changed state explicitly and obtain applicable calibration; old C is not automatically valid.
4. Collect actual responses for four complete filters, the declared real patches/T12/repetitions. Preserve an exact unique coefficient/partial ledger, including empty logical timesteps. Sum all75 taps before trained BN[t] and one digital LIF step. Repeated physical microcommands do not add leak/reset steps.
5. Precompute one finite write/read budget and explicit cutoff, including failed gates, readback, calibration, retention, repeats and raw finalization. Retain native valid-R/current/ADC guards and public Connector. Before physical execution require the new native-SIM result and independent audit; a failed accuracy metric may remain a diagnostic result, but transport/provenance/safety failure cannot be relabeled as success.
6. Preserve raw and compare new Conv/BN/membrane/spikes against trained reference, independently measured effective weights and the matching native simulator. Report both completeness and fidelity. No full hardware classifier under the present scope.

Hard cutoff remains **21:50MSK on16 October**, with scheduled poweroff22:00. Leave time for saving and Connector closure. Do not expand retries or bypass checks because of the deadline.

## Goal status after this result

- Programming: matched-target improvement observed; exact/repeatable all-nine programming still unqualified.
- Map: new observations support eighteen addresses and nine pair trajectories; the scientist is preparing the evidence-labelled map. They do not newly qualify all114 historical assigned cells.
- Conv1/BN/LIF repeat: pending a reviewed new acquisition; JIT contains none of these outputs.
- Full classifier software: the CPU-reference evidence exists separately; measured full-input bridge maturity must remain explicit. No full hardware network has been authorized or run by this review.

Evidence hashes: manifest `361f0f8759941e481281f2847eb9823fc4bb5db3c4fbcbc402007897c2e227aa`; summary `a216f3f9501a9127a561f4819771f155d670ccb12bee0ba8da3fccb72c6082d3`; independent audit `46532e358ea4d640bd69d58f342e60e6710325e166cdce83f42cd975a6b32112`. No runtime or backend changes were made in this review.
