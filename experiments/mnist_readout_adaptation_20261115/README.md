# Local readout adaptation, fixed measured SIM operators

This experiment adapts only the digital linear classifier and feature scaler.
It does not instantiate a Connector, program cells, retrain STDP filters or use
the board. The source is completed SIM acquisition
`605c6b02-4865-44a3-8c85-14d7f22d03e4`, with `bracket_centered` fixed in advance.
The frozen baseline evaluation is `9fc7ac47-e0ed-413f-b3a9-4c55a99499d4`:
53.80% strict / 53.82% argmax on the official 10000-image MNIST test set.

## Fixed protocol

1. Verify terminal manifests against archive receipts and verify reused feature
   counts, map, frozen assets, encoder/LIF/replay source and native converter SHA.
2. Discover masks from the first 10000 official TRAIN images without labels.
   Extend the key set if needed and reconstruct every response from preserved
   measured partial MVM. All original test-key LUT values must match bit for bit.
3. Extract train features with the original deterministic encoder and CPU LIF
   threshold 0.38. Save uint8 counts as a memory-mapped array. No ideal weights.
4. Split only those TRAIN images into exactly 8000 fit / 2000 validation using
   deterministic stratification (seed 42, largest-remainder class quotas).
5. Fit `log1p` scaler on fit examples only. Train a fresh Linear head with SGD,
   learning rate 0.001, weight decay 0.0001, batches 512, 25 epochs. Choose highest
   validation STRICT accuracy; earliest epoch wins ties. Save all epoch results.
6. Discard that head/scaler. Fit a fresh scaler and head on all 10000 TRAIN
   images for the selected number of epochs, resetting seed to 42. Save and
   hash the final checkpoint before any new test-label access.
7. Reuse verified test feature counts and evaluate test once. Silent and tied
   examples are errors for strict accuracy. Preserve predictions and confusion.

The test set was seen in earlier experiments, so this is an **exploratory**
adaptation result, not a new confirmatory generalization claim. It changes the
readout comparison: the old baseline used its previously frozen classifier.
No improvement is guaranteed. Source masks/filters and LIF remain fixed.

## Execution

With repository-relative configured dependencies and restored source sessions:

```text
python -u -B tools/run_foreground_logged.py --log results/foreground_logs/readout-adaptation-UNIQUE.log experiments/mnist_readout_adaptation_20261115/experiment.py --run-local
```

Without `--run-local`, the entrypoint prints the protocol and does no fitting.
The experiment uses CPU with 4 Torch threads; no GPU. The console log is separate
evidence and must be preserved after exit. Progress and terminal manifests are
written under `results/mnist_readout_adaptation_20261115/UUID`. Typical disk need
is under 1 GiB; train LIF extraction and the readout stages determine runtime.
No publication happens automatically. Archive/scan terminal evidence with the
existing preservation helper; root coordinates Git LFS publication.

Offline synthetic tests cover exact disjoint stratification, validation leakage
through scaling, earliest-tie epoch choice, complete selection/refit with sealed
checkpoints before labels, and tampered evidence rejection. These synthetic
tests are not MNIST accuracy measurements.
