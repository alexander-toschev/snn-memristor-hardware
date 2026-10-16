# Full-image CIFAR readiness, 16 October 2026

The full CPU classifier and software bridge are prepared at their documented validation boundary. A full-image hardware classifier is **not ready for launch** and is outside today's authorized hardware scope. Ten original-forward CPU comparisons, synthetic full-tensor coverage and native linkage of the historical four-filter experiment are different proofs; none is a full hardware classification result. The trusted full hardware import remains disabled.

The immediate hardware work remains four Conv1 filters, 75 coefficients each, ten prepared patches and T12, followed by trained BN and digital LIF. The separate no-BL8 variant is pending new native simulator validation and hardware review. Excluding a physical address does not establish successful programming, and 108 planned addresses do not mean 108 accurate cells.

## Reproducible read-count method

This is an offline planning calculation, not a new measurement or raw-data export. Its repository-relative inputs are:

- `weights/cifar100_conv1_fixture.json`, SHA256 `8cb9ec0fe30242328527073bb076f8019295d73a748343f58fc2a8671bf7c154`.
- `results/full_classifier_bridge_20261016/ten_image_plan.json`, SHA256 `968f4151ed2836b255ce90637d8349cdc4b2adc3ccdc113bae6e5a7ae1eaea4f`.
- `full_cifar_plan.py`, SHA256 `ef74b4b475611d262250f5bc2cd8dbe642d4df934b829193ce4459a62ca119be`.

For each of the ten real uint8 CHW images, verify its recorded 3072-byte image hash. A pixel v emits at `((255-v)*11)//255` when v>=26, otherwise it emits no event. For each output position y,x in 0..27, the 75-tap CHW receptive field uses image offset `(tap//25)*1024+(y+(tap%25)//5)*32+x+tap%5`. Partition taps within the logical boundaries [0,32),[32,64),[64,75), then count active inputs k for each physical microtile and timestep.

The current sequential two-leg operation needs `4+2*max(1,k)` native mode10 calls: independent before/after zeros on both legs and actual one-hot signal measurements. Empty timesteps still require independent zero signals. Sum over T12, all positions and 64 filters, separately for each image/repeat. An equivalent check is `calls=4*partials+2*active_coefficient_events+2*empty_partials`. No basis lookup or CPU replacement is included.

| Physical partition | Microtiles across 64 filters | Partials/image/repeat | Calls/image/repeat, range across ten images | Calls for all ten, one repeat |
|---|---:|---:|---:|---:|
| 32/32/11, physical feasibility unqualified | 192 | 1,806,336 | 16,190,208–17,178,240 | 167,855,232 |
| Maximum seven taps, preserving logical boundaries | 768 | 7,225,344 | 46,910,336–47,873,408 | 472,914,048 |

The seven-tap partition contains nine seven-tap and three four-tap microtiles per filter. Full Conv1 contains 602,112 logical output values per image/repeat. Physical partials are not native command counts.

At a hypothetical sustained 234–293 calls/s, using rates observed in short earlier operations as a planning reference, the seven-tap reads alone correspond to **44.47–56.83 hours per image** or **448.34–561.39 hours for ten images**. Five repeats require 2,364,570,240 reads. The larger 32/32/11 partition still corresponds to 159.13–199.26 hours for ten images. These are throughput extrapolations, **not ETAs or evidence that those partitions work physically**. Logging, interruptions, retention checks and transport can change throughput substantially.

## Additional programming and calibration

A full Conv1 plan must physically represent 4,800 coefficients. Extrapolating the bounded per-pair protocol solely for planning:

- One candidate pair per coefficient: at most `4800*(2336+811)=15,105,600` programming/confirmation calls and `4800*(110+109)=1,051,200` nonzero attempts.
- Two candidates per coefficient: at most `4800*(2*2336+811)=26,318,400` such calls and `4800*(220+109)=1,579,200` nonzero attempts.
- Fresh mode7 readbacks, anchors and independent C fit/selection for the 768 seven-tap-partition microtiles add 491,520 calls per full programming/calibration pass.

These are extrapolated upper bounds, not an approved protocol or expected wear. A reviewed 4,800-coefficient physical route does not yet exist. Programming once per tile and then measuring all images avoids multiplying programming by image count, but requires state and calibration validity over long dwell times. Short sessions and a one-second retention check do not establish that validity. Reprogramming per image instead adds repeated programming costs.

## Remaining readiness conditions

Before any authorized full-image hardware launch, prepare a physical route and finite acquisition schedule, a full spatial collector with task provenance, total read/write/storage/retention budgets, interruption and recovery behavior, independent full-export validation, and the enabled trusted tensor bridge. The saved plan's older gain1 contract must not silently inherit the current four-filter gain0.25. This estimate covers hardware Conv1 plus a digital downstream network; putting further layers on the board requires additional work.

The operator separately reports that the current parallel-input test still fails model selection: 118/128 held-out density-two cases and 51/128 density-four cases passed, with all LIF results inconclusive. This is **operator-reported status, not an independent raw audit by this document's reviewer**. It supplies no justified parallel shortcut to these sequential counts. The no-BL8 study concerns weight fidelity and does not establish a parallel MVM fix.

Today's software-preparation deliverable is complete within the stated CPU/bridge boundary. Today's weight-fidelity goal is still under investigation. A quick full hardware classification tomorrow must not be promised from the current sequential implementation.
