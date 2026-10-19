# CIFAR with 300 saved effective weights

This offline evaluation compares the complete digital classifier with two mixed
models. Exactly the first four Conv1 filters (300 of 4,800 coefficients) are
replaced by the published `effective` weights from the 16 October hardware or
native SIM run. The other 4,500 Conv1 coefficients and every downstream layer
remain trained digital weights. There is no acquisition or native simulator run.

The historical values already include the frozen independent calibration matrix
and fixed source gain. No new fitting, rescaling or selection using CIFAR labels
is performed. Every one of the 300 records must exist, be finite, unique, within
scope and match the checkpoint target. A recorded zero remains an observed zero;
a missing measurement is an error. Source CSV and provenance SHA are pinned.

Run from the repository with the CPU environment described in
[FULL_CIFAR_CLASSIFIER.md](FULL_CIFAR_CLASSIFIER.md):

```powershell
python -B -m unittest tests.test_partial_measured_cifar -v
python -B tools/evaluate_partial_measured_cifar.py --output results/partial_measured_cifar_20261019/report.json
```

The output must be new. All ten embedded real CIFAR100 subset fixtures use the
same images, labels, TTFS T12, checkpoint, trained BN and digital LIF. Baseline
outputs must exactly match the original model forward. The report saves each
prediction, output membrane/spike trajectory and TWD scores; Conv1, BN and LIF
differences on the affected filters; and output differences and prediction
agreement. It reports each decoder separately, including silent/tied outputs.

The inference applies each saved scalar W_eff at every spatial location as a
stationary linear model. It does not reproduce hardware drift, nonlinearities or
repeat noise. Results are diagnostic fixture accuracy of a **mixed digital
model**, never hardware classifier accuracy or full-test accuracy. The previous
trusted measured-tensor boundary remains disabled and unchanged.
