# Final finite-reserve evidence: B/C after A

20 fresh B/C +13 prior A +267 original; acquisition, weight fidelity and finite reserve coverage are separate.

Complete HW/SIM comparison: True.

| Source | Fresh B/C | Prior A | Original | Covered | Outputs | Assigned union | New impacts |
|---|---:|---:|---:|---:|---:|---:|---:|
| simulator | 20 | 13 | 267 | 300 | 2400 | 104 | 23782 |
| hardware | 20 | 13 | 267 | 300 | 2400 | 114 | 26370 |

| Source | Stage | Cell | Fresh disposition | SET pulses |
|---|---|---|---|---:|
| simulator | A | [6, 2] | freshly_quiet_assigned_and_measured | 71 |
| simulator | A | [6, 11] | freshly_quiet_assigned_and_measured | 34 |
| simulator | A | [7, 15] | freshly_quiet_assigned_and_measured | 0 |
| simulator | B | [4, 7] | freshly_quiet_assigned_and_measured | 0 |
| simulator | B | [5, 6] | freshly_quiet_assigned_and_measured | 70 |
| simulator | B | [5, 15] | freshly_quiet_assigned_and_measured | 38 |
| simulator | C | [2, 3] | freshly_quiet_assigned_and_measured | 48 |
| simulator | C | [3, 12] | freshly_quiet_assigned_and_measured | 0 |
| simulator | C | [3, 13] | freshly_quiet_assigned_and_measured | 0 |
| simulator | C | [3, 31] | freshly_quiet_assigned_and_measured | 48 |
| hardware | A | [6, 2] | freshly_quiet_assigned_and_measured | 109 |
| hardware | A | [6, 11] | freshly_quiet_assigned_and_measured | 109 |
| hardware | A | [7, 15] | freshly_quiet_assigned_and_measured | 0 |
| hardware | B | [4, 7] | freshly_quiet_assigned_and_measured | 0 |
| hardware | B | [5, 6] | freshly_quiet_assigned_and_measured | 61 |
| hardware | B | [5, 15] | freshly_quiet_assigned_and_measured | 84 |
| hardware | C | [2, 3] | freshly_quiet_assigned_and_measured | 109 |
| hardware | C | [3, 12] | freshly_quiet_assigned_and_measured | 0 |
| hardware | C | [3, 13] | freshly_quiet_assigned_and_measured | 0 |
| hardware | C | [3, 31] | not_freshly_quiet | 0 |

| Comparison | Conv RMSE | Spike differences /2400 |
|---|---:|---:|
| hardware_vs_simulator | 0.286371603 | 85 |
| simulator_vs_trained | 0.0516856022 | 50 |
| hardware_vs_trained | 0.265705291 | 45 |
| simulator_vs_own_effective_weights | 3.29696448e-17 | 0 |
| hardware_vs_own_effective_weights | 0.00156775176 | 0 |

See comparison.json for provenance, fixed gains, bound audits, ancestry, per-filter BN/LIF, fresh block metrics and replaced-cell accounting. Assigned/SET/accurate are distinct. No classification accuracy or absolute optimum is claimed. Maps show only contributing unions across time; no final conductance is inferred.
