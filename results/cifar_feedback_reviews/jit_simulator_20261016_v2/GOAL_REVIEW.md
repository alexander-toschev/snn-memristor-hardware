# JIT C9 — independent goal-controller review

**GO for one bounded hardware diagnostic**, conditional on the primary agent confirming the published, unchanged source/config pins and a successful Raspberry hardware dry-run. This is not approval for a full hardware classifier, a CIFAR acquisition or an automatic retry.

Reviewed native SIM session `6ebac641-2151-42ba-8b1e-48de43f282e1`, launched from `37fe3252f4ae5383b1315011663aaf389bf6c70e` with a clean recorded tracked tree. Independent v2 audit: **391,300 checks, zero errors**, complete 18 pair decisions, 15,455 impacts and 1,208 programming pulses, no current skips and no quiet-gate failures. Manifest SHA256 `3394ab8e3fb8ea4479b0f09964ee7f4740250f2cbd94be060a6d6803dbc70e54`; v2 audit SHA256 `3c0448cdfbcb6128445de42da31ef5ef212f340fc3e5aca2b0cd80a03cd445cc`.

## Scope and checks

- Two fixed-order cycles of the nine C9 trained targets, exactly 18 existing physical addresses; no remapping or omitted target. Limit 29,934 impacts and 3,942 programming decisions/pulses; per decision 1,663/219, checked before calls.
- Both legs undergo bounded RESET. Five independent paired brackets then check each leg's absolute mean and sample SD against two native ADC units. A small differential of two large responses cannot pass. Early individual RESET confirmation does not replace this joint gate.
- SET is allowed only after the fresh joint gate; one inherited bounded residual-directed sweep, without reversal or retry. A quiet failure retains measurements and skips SET. Current refusal, invalid R/ADC, transport error, deadline or budget stops the entire run.
- Native mode7 resistance refresh, existing current/headroom guards, public Connector/converters and INI selection remain unchanged. Sentinel WL3/BL8 is explicitly an observed address, also used by tap5, not an assumed invariant reference.
- One-second monotonic retention and fresh measurements of all nine pairs after each decision expose changes after subsequent writes. Two cycles are not randomized causal controls or evidence of long-term retention.
- No CIFAR inputs, matrix fitting, BN/LIF or software-weight replacement occur. Fixed gain is HW=1 and SIM=0.03; their physical scales differ explicitly.
- UTC and monotonic checks stop acquisition at **16 October 18:50 UTC / 21:50 MSK**, including before backend opening. `finally` closes Connector and preserves partial data. The primary agent must leave time for saving; Python cannot interrupt an already blocked native call. The scheduled Raspberry poweroff at22:00MSK remains in force.

## Auditor correction and scientific limits

The original failed audit remains preserved. Its three errors were an audit-contract mismatch: requiring two hardware-only private drivers in a simulator manifest, and requiring a reserialized protocol copy to have the same byte formatting as the source config. V2 still verifies the exact frozen config Git-blob SHA, then exact parsed JSON equality. Hardware still requires all private pins; SIM omits only the two hardware-only entries from the pinned dependency set. Raw data, runtime, thresholds and arithmetic were not changed.

SIM cycle-final target-gain NRMSE is **0.265499 / 0.301336**, with uncertainty NRMS **0.203099** in both cycles. Fidelity remains failed, despite all quiet gates passing; the unchanged ADC floor is not reduced by averaging. No resolved wrong signs were observed, but some significant signs remain unresolved. `programming_repeatability_qualified` is false. These results verify the acquisition path and disclose finite-resolution limitations; they do not establish an improvement on the hardware's previous approximately134% all-weight NRMSE. A bounded physical diagnostic remains scientifically useful because it tests the hardware-specific reset/interference hypothesis at its separately fixed gain, without asserting SIM fidelity success.

## Verified frozen hashes

| File | SHA256 |
|---|---|
| `cifar_jit_programming.py` | `cd1abd674b032516a78f5f3f9fdce1397375ca766b586770fa926d8509099480` |
| `configs/cifar_jit_programming.json` | `150c5939b3b9f38224159a16982825646705fb4ef5a1fc20c4f18f3b77eef9a5` |
| `experiment.py` | `ddf3d1642dfbba35ecdebdde8d3539840cc126f4a41af17502c4e345f897ff83` |
| Corrected independent auditor | `266faf7a06f2dac6dcaa2d2418426d3b6de2ba3de7ea23cdee22f3cb8399fac4` |

This review used source/config reading, actual compact SIM results, hash bindings and the completed independent raw audit. The reviewer made no backend calls and did not duplicate the raw replay. After hardware collection, a new independent audit and matched-target repeatability/disturbance analysis are required before any claim of improved writing or subsequent Conv1 protocol.
