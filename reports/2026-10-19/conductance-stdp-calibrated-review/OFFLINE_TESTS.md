# STDP-v2 offline checks

7/7 tests passed on19 October2026 with bundled Python3.12, using
`python -B -m unittest tests.test_conductance_stdp_calibrated -v`.
Runtime reported by unittest:2.062s. Test suite:

- Natural fake ADC responses select gain400 and exercise actual digital post
  spikes, nonzero eligibility and both update signs, with no forced spike result.
- Complete fake schedule has28 windows,232 checkpoints,4776mode10 calls,
  64fit measurement steps and832evaluated LIF steps, with one frozen shared gain.
- All-zero centered fake responses preserve the intentionally uninformative
  prefix1536impacts/64preparation pulses/80checkpoints, without training.
- Control-bank fit data do not affect gain selection.
- Spikes at zero-input steps make a candidate ineligible.
- Fixed gain grid, train/fit/probe order and both closed native launch gates
  are checked.
- A separately forced-eligibility unit case reaches exactly6712impacts and
  128pulses, verifying the upper bound; failed close is preserved as failure.

The forced case is a budget unit test, not learning evidence. Every Connector
in this suite is a local fake. No native SIM, board, model experiment, network
connection or real driver execution occurred. Informative native behavior and
its independent raw-data audit remain separate requirements.

Reviewed source snapshot:

- core.py SHA256 `0b3f0bac8762a1cd93a17a6849139ffd8bea4d35cfffc8c10dec67a32d9ea06c`
- experiment.py SHA256 `9d06ebb1a943a73ba211d0307f91f242aba30c034ae99e16c4ea1aaacbee0e2a`
- canonical SPEC SHA256 `6d86180c0a73311753bfa420b66e8f6727d779106025ad23ac9d8cc6a3b414c5`

The old frozen conductance STDP, full CIFAR and MNIST128 directories had no
tracked diff after these changes. The snapshot above preceded registration.

Root subsequently authorized SIM-only registration after scientific review.
The gate-only entrypoint change has SHA256
`cd56bbf243d236ed4b544733f510d16b5de512d8e4aba7f1677af26a6954d75c`;
core and canonical SPEC hashes remain unchanged. The registration unit test now
patches both gates closed before exercising rejection, so it remains offline
even after legitimate SIM registration. All7 tests were rerun and passed in
1.854s. Hardware approval remains empty; native SIM execution is still root's
separate step after auditor preparation and commit.
