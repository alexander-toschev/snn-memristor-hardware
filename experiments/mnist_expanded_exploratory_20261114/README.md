# MNIST 64: exploratory transfer on 148 cells

This is an explicit exploratory branch of the measured-operator experiment.
It keeps programming misses, unresolved signals and the original strict MVM
failure flags. It does not replace a measured response with an ideal weight.

The map has 148 physical cells,74 differential pairs and 22 loads for 1600 learned
coefficients. This is the maximum nonoverlapping same-BL pairing among the 163
observed switching cells:15 cells have no partner in their BL. Pairing is
lexicographic by WL, without MNIST labels.31 two-input blocks and 12 single-input
blocks require 73920 read calls/load; fresh prewrite all 32 current checks add 256.

Every profile uses the same positive background 0.025 V and fixed increment 0.05 V.
Each block's shared measured conductance interval determines one affine scale
and offset for both legs, mapping the source range to 20–80% of that interval.
The scale is removed digitally after measured partial MVM. The absolute write
tolerance is 0.02 × source conductance span × block scale. Targets and errors
are retained separately from the measured inference operator.

## Run order

From the repository root, in the configured environment:

```text
python -B experiments/mnist_expanded_exploratory_20261114/experiment.py --run-simulator --expanded --exploratory-measured-lut --pilot --acquire-only --map experiments/mnist_expanded_exploratory_20261114/map_smoke6.json
python -B experiments/mnist_expanded_exploratory_20261114/experiment.py --run-hardware --expanded --exploratory-measured-lut --pilot --acquire-only --publish
python -B experiments/mnist_expanded_exploratory_20261114/experiment.py --run-hardware --expanded --exploratory-measured-lut --acquire-only --publish
python -B experiments/mnist_expanded_exploratory_20261114/experiment.py --run-simulator --expanded --exploratory-measured-lut --acquire-only --publish
```

The first command is a 6-cell simulator smoke including BL 30. The hardware pilot
uses the full 148-cell map for one load. No command automatically starts a new
hardware run. Full acquisitions require a terminal completed manifest before
classification; a pilot never yields a whole-network accuracy.

## Two fixed baseline policies, one acquisition

For every binary input pattern,3 independent basis repetitions freeze the
operator;5 further repetitions measure validation errors. Each leg contains
4 samples before the signal,4 at the signal and 4 after. The native ADC-to-voltage
converter is applied to every sample before averaging.

- `before_only`: signal mean minus preceding background mean.
- `bracket_centered`: signal mean minus the mean of preceding and following
  background means. This correction already existed in the earlier strict
  experiment; the new work makes its comparison explicit.

The same raw records support both policies. Background subtraction does not
remove the recorded drift or certify stability. Policies, map, weights,
digital LIF, scaler and classifier are fixed before test evaluation. There is
no selection of a winning policy by MNIST test labels.

On the local CPU host, after transferring and verifying the full session:

```text
python -B experiments/mnist_expanded_exploratory_20261114/experiment.py --evaluate-session UUID --baseline-method before_only --publish
python -B experiments/mnist_expanded_exploratory_20261114/experiment.py --evaluate-session UUID --baseline-method bracket_centered --publish
```

Each evaluation gets a separate result UUID and reports 10000-test strict/argmax
accuracy. Repeat for the hardware and simulator acquisitions using the same map.
The simulator starts from a new synthetic array; its initial state is not the
physical board's state. No new STDP training is performed in this transfer.

## Hardware and interpretation limits

Only public MemriBoard Connector, converters and INI are used. The scoped
programmer supports selected BL 0–31 without weakening the historical BL<25
collector. Voltage/current/ADC/ID guards and port cleanup remain mandatory.
Prewrite checks use the larger of fresh and target conductance plus all 32
background cells. Every MVM leg gets a fresh actual-current check afterward.

Raw logs are bounded at 8 GiB with 2 GiB free reserve; runtime at 8 hours. Physical
programming pulse duration is unknown. Strict MVM failure can continue only
under the explicit exploratory flag; invalid communication/current saturation
still aborts. The report must retain strict failure counts and programming
misses alongside accuracy.

Inference is hybrid: frozen measured partial MVM operators, digital composition,
CPU LIF and frozen CPU classifier. It is not a live hardware request for each
test image and not an analog implementation of LIF.
