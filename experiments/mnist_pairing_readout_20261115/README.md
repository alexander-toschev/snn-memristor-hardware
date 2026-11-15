# Read-only pairing before expanded MNIST

All physical I/O uses the public MemriBoard Connector, converters and INI.
There are no SET/RESET commands in this study. Historical intervals come from
the 163 cells meeting the official switching criterion on 14 November; they
are not a guarantee that a target conductance can be programmed today.

## Protocol fixed before observations

- For each candidate: 3 repeats at increments 0.025 / 0.05 / 0.075 V above a
  global 0.025 V background on all 32 BL. Maximum requested voltage is 0.1 V.
- Each bracket freshly reads all 32 resistances on its WL, checks summed current
  <=5 mA and predicted output <3.3 V, then takes 4 before / 4 signal / 4 after
  mode10 samples. Measured output >4 V or invalid native ID/ADC aborts.
- Unsafe predicted current skips that bracket before signal application and is
  recorded. Missing safe basis is not fabricated. Other native errors abort.
- Full basis: 163×3×3×44 = 64548 calls. Heldout: at most 105 patterns ×3 repeats
  ×2 legs ×44 =27720 calls. Planned total <=92268; hard cap100000 calls/1200s;
  full raw cap512 MiB and reserve2 GiB. Simulator smoke uses only6 pinned cells,
  including BL30; its map is explicitly marked as smoke.

For each global amplitude, exact matching within each BL maximizes the number
of differential pairs first. A candidate edge needs a positive intersection of
historical conductance intervals and complete safe basis readings. Secondary
score is the sum of `log1p(expected_span_ADC/noise_penalty)`:

```text
expected_span_ADC = .6 * common_G_span * native_decoded_delta * sum_gain / nominal_ADC_step
nominal_ADC_step = INI vol_ref_adc / (2**adc_bit - 1)
noise_penalty = max(1, hypot(SD_plus, SD_minus))
                + max_abs_drift_plus + max_abs_drift_minus
                + linearity_residual_plus + linearity_residual_minus
```

Linearity residual uses a basis-only least-squares line through zero across
the three native-decoded increments; each leg gets a fixed extra5 ADC penalty
if fewer than3 safe amplitudes are available. Expected span is only a selection
proxy; it never replaces measured MVM. Actual per-sample voltage conversion
always uses the native converter; no ADC×rounded-convert(1) shortcut.
Current programmed weight amplitude is not an exclusion threshold.

Scores are rounded to12 decimals. Matching ties choose lexicographically
smallest WL pairs; global amplitude ties choose lower requested voltage after
pair count and summed score. Historical cardinality ceiling is148 cells.
Reserve cells are reported as unmatched, not declared defective.

Grouping greedily merges same-WL pairs with positive shared windows, preferring
largest common window, then WL/BL coordinates. At most31 merges keep transfer
readout <=74518 calls/load including up to256 prewrite guard calls. It is an
explicit heuristic, not a global optimal-grouping claim. If no multi-input
group exists, retain pairs and flag that limitation rather than drop cells.

The map is saved and hashed BEFORE3 independent heldout repeats over all
nonempty group masks. Predictions sum measured singleton responses, not ideal
weights. Drift, signed SD, individual-leg and signed residuals use the existing
5 ADC /10% criteria. Heldout failures never refit amplitude/pairing or disappear.
Every later weight load must undergo fresh programming readback and full MVM
validation; passing this current-state study is not MNIST qualification.

## Entry points

```text
python -B experiments/mnist_pairing_readout_20261115/experiment.py --dry-run
python -u -B tools/run_foreground_logged.py --log results/foreground_logs/pairing-SIM-smoke-UNIQUE.log experiments/mnist_pairing_readout_20261115/experiment.py --run-simulator --smoke
python -u -B tools/run_foreground_logged.py --log results/foreground_logs/pairing-HW-UNIQUE.log experiments/mnist_pairing_readout_20261115/experiment.py --run-hardware --publish
```

Only root launches hardware after review and Git publication. The lock is
`results/read_all/.hardware.lock`. All terminal outcomes close the port and
finalize raw/manifest/publication. Native foreground log is preserved separately.
After root pins the hardware-selected map, matched full SIM must use that same
map and amplitudes. The previous66.53% result uses a different historical map
and is not a matched comparison for this study.
