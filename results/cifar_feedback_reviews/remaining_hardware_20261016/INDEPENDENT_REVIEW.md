# Independent final hardware B/C review

Session: 58ca618b-c37f-454e-be25-db21d1d25f60. Recorded acquisition source commit: 418c4ad4ca1551d7733ccc4718ae85aab55937fa, with clean tracked files. This is the acquisition revision, distinct from the later data-publication commit.

## Evidence and verdict

The frozen independent auditor 4f2f73eb96fd286aabde4e0513b7aac2ef96e57ef873ce36573a73a29a3b0525 reports PASS: 669378 checks, zero errors. Its manifest commitment is 5fb5041cbdba148ff0a02182512a98ca0784fed074e2830d6fc06b230f933171. This review independently verified all 35 bound result files (245479041 bytes), including raw chunks, and the manifest/audit/summary bindings used by the comparator. It also checked the four final public runtime/plan/config/dispatcher hashes and recomputed compact trajectory comparisons. It did not repeat the numerical raw replay or perform any hardware operation.

No inconsistency was found in the final comparator or its report. The finite A/B/C reserve experiment has a complete measured outcome, including an explicit negative screening result. Accuracy remains failed; completion must not be presented as accurate trained-weight realization or full CIFAR classification.

## Completeness, bounds and measured routes

Fresh B/C: 26370 impacts, 2163 programming writes/decisions, zero skips, below the fixed 40378/5260 limits. B contributes 11 coefficients and 600 partials; C contributes 9 and 600. Exact composition is 20 fresh B/C + 13 earlier A + 267 original coefficients, with 300 unique logical weights and 2400 time outputs. The cumulative command history is 420685 impacts/34376 writes, including previous attempts once.

B fresh quiet counts are 12 positive/15 negative; it used 14068 impacts/1126 writes. C counts are 11/12; it used 12302/1037. Both selected diagonal C, frozen before actual CIFAR acquisition, and fixed hardware gain 1.0. The native-R/current, ADC, RESET/SET step and command-budget checks are covered by the bound raw audit. Nominal write amplitudes remain 0.3-3.0 V, mode10 input 0.1 V; hardware pulse duration is unknown. No calibrated physical pulse-width claim follows from simulator duration settings.

## Ten predetermined reserves

| Stage | WL/BL | Fresh screening and use | SET pulses on this address |
|---|---|---|---:|
| A | 6/2 | Quiet, assigned/measured at tap52 | 109 |
| A | 6/11 | Quiet, assigned/measured at tap53 | 109 |
| A | 7/15 | Quiet, assigned/measured at tap64 | 0 |
| B | 4/7 | Quiet, assigned/measured at tap0 | 0 |
| B | 5/6 | Quiet, assigned/measured at tap0 | 61 |
| B | 5/15 | Quiet, assigned/measured at tap1 | 84 |
| C | 2/3 | Quiet, assigned/measured at tap0 | 109 |
| C | 3/12 | Quiet, assigned/measured at tap0 | 0 |
| C | 3/13 | Quiet, assigned/measured at tap1 | 0 |
| C | 3/31 | Not quiet after the bounded RESET sweep; not assigned | 0 |

Nine of ten were assigned and measured in their respective stages; five received SET. Zero SET is consistent with residual-selected polarity, not missing data. Earlier A results remain historical, not new B/C measurements.

WL3/BL31 consumed its 55 RESET decisions. The five final centered MVM values have mean 0.317832 V and sample SD 0.000510363 V, against a 0.00062 V quiet limit. The mean, rather than the SD, fails by a large margin. C used ordinary fresh-quiet WL3/BL2 at tap2 under the prespecified fallback rule. This is a bounded RESET eligibility miss, not evidence of a dead cell or a hardware root cause. There were no unassessed reserves or automatic retries in this finite package.

## Accuracy and interpretation

B physical NRMSE: 0.259203 before CIFAR / 0.259807 after; uncertainty NRMS 0.007514 / 0.008733. No significant wrong or unresolved signs, but the 20% mean-error condition fails. Taps 2, 6, 8 and 10 exhausted the 109-step SET ramp.

C physical NRMSE: 1.660440 / 1.659569; uncertainty NRMS 0.009919 / 0.011112. Tap5 has a resolved wrong sign and tap0 remains unresolved; taps0,5,6 exhausted SET. These failures were retained rather than digitally replaced or normalized away.

Direct saved-trajectory recalculation agrees with the comparator:

| Comparison | Conv RMSE | Spike differences /2400 |
|---|---:|---:|
| Hardware vs native SIM | 0.286371603329 | 85 |
| Hardware vs trained CPU reference | 0.265705290646 | 45 |
| Hardware vs own effective-weight reference | 0.001567751760 | 0 |

Zero spike differences against the effective reference do not make the analog Conv exact. Native SIM is not an exact model of the board. The experiment concerns four Conv1 filters and ten fixed real CIFAR patches followed by trained BN and digital LIF, not full-network classifier accuracy.

The final contributing-history union contains 123 probed, 114 ever-quiet/assigned and 104 SET-attempted addresses. These are unions across completed retained and new measurements, not simultaneous capacity, a final conductance map, or 114 accurately qualified cells. No absolute optimum over all 256 cells or future states is established.

## Reproduction contract

At the recorded source revision with pinned local dependencies/private driver bytes and the repository-relative configuration and immutable parent data:

```text
python experiment.py --cifar-feedback-remaining-reserves --dry-run
python experiment.py --cifar-feedback-remaining-reserves --run-hardware
python audit_cifar_feedback_remaining_reserves.py --session results/cifar_feedback_remaining_reserves/58ca618b-c37f-454e-be25-db21d1d25f60 --output results/cifar_feedback_reviews/remaining_hardware_REAUDIT
```

The second command is a new bounded writing hardware experiment and requires deliberate operator authorization; it was not executed by this review. For saved-data reproducibility only the final offline audit is needed, with a new output folder. The finite completed-A bindings in configs/cifar_feedback_remaining_reserves.json prevent automatic parent substitution. Results may vary with subsequent physical states.
