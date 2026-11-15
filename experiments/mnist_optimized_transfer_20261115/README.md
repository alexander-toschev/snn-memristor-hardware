# Transfer on the pinned optimized map

This separate entrypoint reuses the existing bounded adaptive programming,
native measured partial-MVM LUT, 3 basis /5 independent validation repeats,
whole-load readback and fixed CPU LIF/head. Historical experiment source files
remain unchanged. The new reader/parser permits the selected increment 0.075 V
above 0.025 V (maximum 0.1 V), with the same fresh current and ADC guards.

An explicit `--map` from the reviewed read-only pairing session is mandatory.
Smoke maps are accepted only for a simulator pilot. Full hardware requires the
full diagnostic map, and uses the first normal weight load as the mandatory
post-write readback/MVM check. No redundant programming pilot is necessary.
Strict repeatability failures and unresolved amplitudes remain recorded under
the explicit exploratory flag; invalid communication/current/ADC guards abort.
Historical G windows do not promise target attainment. No ideal-weight fallback.

```text
python -u -B tools/run_foreground_logged.py --log results/foreground_logs/optimized-HW-UNIQUE.log experiments/mnist_optimized_transfer_20261115/experiment.py --run-hardware --expanded --exploratory-measured-lut --map PATH_TO_PINNED_MAP.json --acquire-only --publish
python -u -B tools/run_foreground_logged.py --log results/foreground_logs/optimized-SIM-UNIQUE.log experiments/mnist_optimized_transfer_20261115/experiment.py --run-simulator --expanded --exploratory-measured-lut --map PATH_TO_SAME_PINNED_MAP.json --acquire-only
```

Root publishes code/map through Git and launches hardware. Matched SIM starts
only after that map is pinned; previous 66.53% is not its matched baseline.
No test labels choose pairs or voltage. Preserve the native foreground logs.

After full terminal acquisition and verified restoration, evaluate each frozen
head locally (hardware first, CPU jobs sequentially):

```text
python -u -B tools/run_foreground_logged.py --log results/foreground_logs/optimized-baseline-UNIQUE.log experiments/mnist_optimized_transfer_20261115/experiment.py --evaluate-session ACQUISITION_UUID --baseline-method bracket_centered
```

Then use `mnist_readout_source_20261115` with that acquisition and completed
baseline UUID. Acquisition remains hybrid: measured frozen partial operators,
CPU composition and LIF; not a live board query for each MNIST test image.
