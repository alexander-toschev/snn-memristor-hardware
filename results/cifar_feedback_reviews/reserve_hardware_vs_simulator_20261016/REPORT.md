# Reserve A: offline evidence comparison

13 new + 287 historical; no classifier accuracy, no fresh300 claim; consistency audit is not weight fidelity.

Full comparison available: True.

| Source | New coefficients | Historical coefficients | Derived outputs | New calls | New writes | All ancestry calls |
|---|---:|---:|---:|---:|---:|---:|
| simulator | 13 | 287 | 2400 | 13852 | 1245 | 572402 |
| hardware | 13 | 287 | 2400 | 15911 | 1408 | 394315 |

| Comparison | Conv RMSE | BN RMSE | Pre-LIF RMSE | Post-LIF RMSE | Spike differences /2400 |
|---|---:|---:|---:|---:|---:|
| hardware_vs_simulator | 0.28723731 | 1.0766082 | 1.3735107 | 1.3152559 | 90 |
| simulator_vs_trained | 0.053349726 | 0.13375585 | 0.16420924 | 0.19463678 | 55 |
| hardware_vs_trained | 0.26539069 | 1.0152302 | 1.3088915 | 1.2783196 | 45 |
| simulator_vs_own_effective_weights | 3.7816051e-17 | 1.082942e-16 | 1.2920216e-16 | 1.1754634e-16 | 0 |
| hardware_vs_own_effective_weights | 0.0015471237 | 0.004844813 | 0.0056175681 | 0.0055774612 | 0 |

| Source | Reserve | Fresh disposition | Assigned taps | SET pulses |
|---|---|---|---|---:|
| simulator | [6, 2] | freshly_quiet_assigned_and_measured | [52] | 71 |
| simulator | [6, 11] | freshly_quiet_assigned_and_measured | [53] | 34 |
| simulator | [7, 15] | freshly_quiet_assigned_and_measured | [64] | 0 |
| hardware | [6, 2] | freshly_quiet_assigned_and_measured | [52] | 109 |
| hardware | [6, 11] | freshly_quiet_assigned_and_measured | [53] | 109 |
| hardware | [7, 15] | freshly_quiet_assigned_and_measured | [64] | 0 |

comparison.json separates fresh13/old287/all300 weight metrics, fresh600 partials before BN/LIF, current skips, gains, immutable ancestry, audit/manifest/result hashes and contributing address unions/displacements. Assigned is not accurately programmed. Finite reserves do not prove absolute array optimality.

Audit PASS is trusted through a frozen auditor hash and bound artifacts consumed here; this report does not replay raw commands. Sequential one-hot sums are not simultaneous analog MVM. Five repeats are not assumed independent; HW and SIM have different states and fixed gains. No runtime code is imported.
