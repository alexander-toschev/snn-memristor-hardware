# Diagnostic reuse of completed calibration A

The original A/B verdict is preserved. This route does not repeat A and never
claims quality qualification. It uses no CIFAR labels or classifier outputs.

Stock native SIM at .05 V has a maximum singleton response of about 9 ADC:
8.5 microampere × 330 ohm = .002805 V, encoded by its 14-bit / 5 V ADC.
The previous 10 ADC lower-span requirement therefore cannot qualify this SIM.
New diagnostic resolution is `H − R − 3 SE − 2 ADC > 0` in both A cycles;
2 ADC conservatively covers quantization of two separately zero-centered
endpoints. With zero repeat noise, a 3 ADC span passes and 2 ADC does not.
Completeness and admitted SET/RESET requirements remain. A-only selection:
one WL pair, at most eight common BL, maximum count then minimum worst
original quality score then lexicographic tie break. Old flags remain visible.

Fresh B measures both signs, zero recipe and checkerboard, immediate and after
60 seconds. Both signed orientations must exceed `3 SE + 2 ADC`; missing,
invalid or unresolved observations block C. Original B quality evaluation is
saved separately. C uses the original disjoint fit/selection/heldout schedule,
freezes its matrix before heldout and retains its original quality outcome.
No reranking, threshold retries or identity substitution occur. Overall
`qualified` remains false even if C meets its own criteria.

Offline reselect, without Connector:

```sh
python -B experiments/stable_cell_diagnostic_20261020/experiment.py --reselect --session results/stable_cell_calibration_20261020/beb56c75-496b-4ac6-a088-5ba77b306d13 --output results/stable_cell_diagnostic_20261020/simulator_candidate.json
```

Bounded native SIM B/C continuation, when requested by the operator:

```sh
python -B experiments/stable_cell_diagnostic_20261020/experiment.py --run-simulator --candidate results/stable_cell_diagnostic_20261020/simulator_candidate.json
```

This copies the original final simulator array into a new session, keeps an
immutable initial snapshot and performs new physical-model impacts through
public MemriBoard Connector. The original session bytes stay untouched.
For hardware, reselect its own completed session to a different candidate and
use `--run-hardware`; do not replace a successful original B/C map with this
route unnecessarily. INI, native converters, current/voltage/ADC guards and
close-on-failure are inherited unchanged. Actual hardware pulse duration is
unknown (submitted 0); SIM receives .000125 s / 125 model steps.

For eight BL and one WL pair: B ≤6656 impacts / 768 pulses; C uses its
explicit deterministic mask count, capped by the computed candidate budget.
Four waits total at most 240 seconds; soft acquisition deadline 1800 seconds.
All 256 saved A cell entries, original verdict, new diagnostic reasons,
raw B/C commands/results and final map are retained in a new UUID directory.
This is an observations-based reanalysis, not a complete replay of old raw.
No continuation or hardware acquisition is launched by reselect.
