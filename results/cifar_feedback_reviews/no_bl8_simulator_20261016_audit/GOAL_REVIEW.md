# Hardware gate — no-BL8 fresh Conv1, 16 October 2026, 20:30 MSK

**GO for one bounded hardware execution of the frozen no-BL8 protocol through experiment.py and the native public MemriBoard Connector.** This permits collecting a diagnostic result; it does not claim accurate weights, parallel MVM or full classification. No automatic retry or budget extension is permitted.

The reviewer inspected the actual completed native SIM manifest, summary and independent audit. No backend or duplicate raw replay was run by this reviewer.

- Session: `28aa7248-72db-428a-b0eb-92b1712a25af`; source commit `41739f3d58fcf61ed3b99a8566154b8e79695078`, tracked clean at launch, no provenance exception.
- Acquisition: all300 fresh coefficients,44 microtiles,108 selected addresses,2400 time outputs. Physical BL8 excluded; logical tap8 retained. No imported measurements.
- Actual calls341999; nonzero attempts16622; current skips0; no acquisition or close error. Frozen limits remain1686292 calls and91000 nonzero attempts.
- Independent audit: **PASS,8526935 checks,zero errors**. Its consistency result is distinct from the summary's `physical_weight_quality_pass=false`.

| Evidence | Verified SHA256 |
|---|---|
| manifest.json | 5ec0bc4789a87b67ae8b357c3205402eec4a625a5b7407e1c702a1f321aeccaa |
| summary.json | fca6b514d18812b23209b809ed0fb162540590a3b6e26f0256fcb345f4f657e5 |
| audit.json | b3cf7ab0c78efd8d68c9ca971c0fd2a6aafe6bfb6f31ec5d750c9e0bd28a616f |
| audit_fresh_conv1_no_bl8.py | 07d2dd0faecff90360df9c9c1771746b8d01045fb10119fa626263a3f201e9ca |

The manifest's runtime/config/route/dispatcher hashes match the source review: runtime `df4843f0de95933d6be7bd7cb6d3e318815f0b7053dacf321594ffa347c682bb`, config `8b458cbe1048f8e18155f1edc82b2240cae889df588bfcdbb2a4bc388e62087a`, route `5c80e32a13097f5fd0c998a4d416811ce7cb2edb00ddfbe5e298d688b5662abb`, dispatcher `78ab85d62914b37dd090c480e27fd46a13ffd21f8d0ee043f1622eff3bb7a012`.

The root operator reports Raspberry source synchronization and successful no-I/O hardware preflight; a later offline analyzer-only commit must leave these acquisition hashes unchanged. Hardware must use those exact files, unchanged current/native-R/headroom/joint-quiet gates, fixed hardware gain0.25, and independently frozen C before actual inputs. A fresh hardware audit and quality comparison are required afterwards. The SIM's gain0.03 and deterministic results are not physical equivalence evidence.

Latest start21:05MSK; target finish21:30; enforced acquisition cutoff21:50 and already scheduled poweroff22:00 remain in force. The conservative maximum call budget is not an ETA. If time or technical conditions fail, preserve an incomplete outcome and close the Connector rather than relax the protocol. The separate full-image hardware classifier remains prohibited and unprepared.
