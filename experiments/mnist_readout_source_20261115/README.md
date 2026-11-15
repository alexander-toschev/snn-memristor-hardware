# CPU readout adaptation from verified hardware or simulator evidence

This NEW entrypoint preserves the fixed 15 November adaptation experiment and
its source hashes. It takes an explicit acquisition directory/UUID, completed
frozen-head baseline UUID, and expected measurement source. All identities,
archive receipts, saved map, input/model assets, digital/replay code and reused
test feature counts are SHA-verified. A simulator source cannot satisfy a
hardware request. The command creates no Connector and performs no board I/O.

The protocol is unchanged: first 10000 official TRAIN, 8000 fit /2000 validation
stratified seed 42, fit-only scaler, SGD lr.001/weight decay.0001/batch 512 for 25
epochs, highest STRICT validation accuracy with earliest tie, then fresh
scaler/head refit on all 10000 train for selected epochs. Checkpoint precedes
one final TEST 10000 label access. Only measured partial operators produce LIF
features; no ideal W@x fallback. The test was previously seen, so the result is
explicitly exploratory. Baseline accuracy comes from the SAME source evaluation,
not the historical 66.53% of another map.

```text
python -u -B tools/run_foreground_logged.py --log results/foreground_logs/optimized-readout-UNIQUE.log experiments/mnist_readout_source_20261115/experiment.py --run-local --acquisition-base results/mnist_optimized_transfer_20261115 --acquisition ACQUISITION_UUID --baseline BASELINE_UUID --source hardware
```

For the matched SIM use its actual two UUIDs and `--source simulator`. Keep
CPU evaluations sequential, hardware-derived result first. TRAIN mask discovery
uses no labels; any new masks are composed from the complete measured partial
binary operators. Original test-key LUT values must remain identical. Every
outcome writes summary, escaped HTML and terminal inventory. Root archives,
scans and publishes results; native disk logs are separate evidence.
