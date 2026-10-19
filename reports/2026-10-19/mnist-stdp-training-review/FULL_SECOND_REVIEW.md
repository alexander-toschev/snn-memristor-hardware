# Independent experimenter review: full MNIST STDP route

Reviewed 19 October 2026, without executing a network, training, native SIM or hardware.

The actual qualifying session is `440fc54b-a3bc-48f9-ae3c-43196809b96d`.
The author and independent reports distinguish initial clipping from later
plasticity, runtime telemetry from saved arrays, and a reconstruction from the
unattested historical 69.65% result. No accuracy is claimed for smoke128.
The saved per-image weights support change beyond clipping; they do not permit
independent reconstruction of every individual LTD/LTP event.

The full launcher binds the immutable smoke acquisition helper
`7f2a35d9fe5b832686d73b57d940acbc6fae99a3406de41258377d4d2fd9ee3f`,
the actual smoke manifest and its independent review. The inspected registered
launcher SHA is `87eedc25cc3ca73aa429d5dff9b210c9076b8257538ce2789cbec4aa5a6016fd`.
No frozen acquisition source was changed by this reviewer.

The path starts fresh seed42 training on IDs0…29999, batch1/T100; it does not
resume smoke or load old trained weights. Frozen feature extraction uses its
new final weights, with learning disabled and state hashes checked before and
after each batch. Train-only log1p/sample-SD scaling, fixed25-epoch SGD
(1475 updates), final head/scaler freezing before test, and a single10k test
evaluation match the declared reconstruction contract. The historical broken
post-window E/I/normalization hook remains declared and unchanged.

Planned feature/head counts are separated from confirmed execution events.
Partial head counts explicitly mean the last completed epoch. The8h wall,
8GiB output and initial resource requirements remain finite; the2.61–2.77h
STDP extrapolation from128 examples is not an end-to-end guarantee.

Independent offline execution of `tests.test_mnist_stdp_full_launcher` and
`tests.test_mnist_full_recovery`: **7 tests passed**,2.917s. These cover closed
gate behavior, actual smoke artifact inventory, completed-event accounting,
exact split/tail coverage, streamed sample variance, invalid count rejection,
and fixed readout scope. They do not execute the full model path.

Verdict: no blocking issue found in the reviewed full launch/readout path.
Actual full-run correctness and accuracy remain to be measured and audited.
This is software BindsNET, not a physical memristor result. All future hardware
I/O must still use public MemriBoard Connector, native converters and INI.
