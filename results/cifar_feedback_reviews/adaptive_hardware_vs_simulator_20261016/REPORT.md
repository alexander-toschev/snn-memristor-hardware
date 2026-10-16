# Adaptive CIFAR: hardware and native simulator

Exactly 300 trained coefficients and 2400 aligned outputs per session: 4 filters, 10 real patches, 12 steps, 5 repeats. Both sessions have a bound independent consistency PASS. Physical quality failure does not hide measured outputs.

| Comparison | Conv RMSE | BN RMSE | Membrane before reset RMSE | Membrane after reset RMSE | Spike mismatches /2400 |
|---|---:|---:|---:|---:|---:|
| HW - SIM | 0.28914462 | 1.0803787 | 1.3777423 | 1.3189361 | 85 |
| SIM - Wtrained | 0.053545263 | 0.13431471 | 0.16565953 | 0.1951299 | 55 |
| HW - Wtrained | 0.26647667 | 1.0171138 | 1.3092925 | 1.279226 | 50 |
| SIM - own W_eff | 3.7884955e-17 | 1.1165083e-16 | 1.3216992e-16 | 1.1951854e-16 | 0 |
| HW - own W_eff | 0.0015990958 | 0.0050217453 | 0.0059146702 | 0.0058595691 | 0 |

Per-filter metrics, bias/MAE/max error, reference RMS and repeat sample SD are in comparison.json. Zero reference RMS yields NRMSE=null. Five repeats are not claimed independent.

The independently acquired microtile partitions may differ. Identical inputs do not imply matched conductance states, time histories or noise. Hardware gain=1 and simulator gain=0.03 are fixed provisional scales. Fitted alpha is never used for decoding. Agreement with own W_eff does not establish trained-weight accuracy.

| Source | Completed microtiles | Imported coefficients | New impacts | All historical parent impacts | Cumulative impacts | Conv repeat RMS sample SD |
|---|---:|---:|---:|---:|---:|---:|
| simulator | 28 | 42 | 258031 | 300519 | 558550 | 0 |
| hardware | 29 | 42 | 327247 | 51157 | 378404 | 0.0012546582 |

Imported prefixes remain historical. All parent attempts, including its unimported failed tile, remain in cumulative counts. They are not mislabeled as new or as represented measurements. No raw ledger is rewritten or CPU value inserted.

This comparison checks bound audit/manifest/summary hashes and exact output/partition coverage; it does not replay full raw a second time. These Conv1/BN/LIF outputs are not CIFAR100 classification accuracy.
