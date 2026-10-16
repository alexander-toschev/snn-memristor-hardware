# Independent B/C audit  final source freeze

Protocol: `cifar-feedback-remaining-reserves-v1`. The filename retains its historical DRAFT suffix; the auditor now accepts only the reviewed final source/config pins and actual completed-A bindings. It never starts an acquisition or imports a collector/driver.

## Final validation

-17 offline tests PASS. Eleven original self-contained routing/accounting/provenance checks plus final-source/parent checks and evidence-dependent mutation tests cover the frozen contract. Five optional evidence-dependent tests explicitly skip on a checkout lacking saved synthetic journals or the historical A session; skips are never counted as passes. All17 ran here.
- Saved synthetic B+C journals:323048 checks,0 errors,12938 calls/20 writes/1200 partials.
- Saved synthetic B capacity miss followed by the single planned C:196954 checks,0 errors,7789 calls/229 writes/600 C partials, no full300 result.
- Real completed-A parent branches: HW1131 checks/0 errors, SIM1440/0. Exact280 retained,20 excluded, A13 retained and recursive42 preserved. Parent cumulative totals are HW394315/32213 and SIM572402/36893. This verifies audit/result commitments and ancestry, not a repeat numerical raw replay.

The historical synthetic checks used the explicitly marked89764591 draft fixtures. Those are FakeConnector journals, not native SIM or hardware. Final runtime changes enable fixed real-parent metadata/CLI; the acquisition mathematics is unchanged. Actual native B/C acquisition and its final session audit remain to be performed by the root agent after clean source commit.

## Independent checks

The auditor verifies chunk SHA/order, native ADC conversions and task/result pairs, latest valid mode7 native-R guards, RESET confirmations, exact pools, preferred reservation before ordinary fill, bounded residual-selected SET, fresh C fitting/selection/freeze, fixed gain and every new CIFAR contribution. B/C budgets are independently derived from the immutable real fixture:21421/2794 and18957/2466, total40378 impacts/5260 decisions. C follows B once. Only complete scientific capacity failure permits continuation; no repeated screening or historical fallback.

A completed manifest must bind completed_microtiles.json and summary.json; any present partial checkpoint, summary or capacity_failures.json must also be hash-bound. Declared limits independently equal40378/5260/5260 and20 new/280 imported coefficients. Mutation tests cover absent commitments and every declaration.

Each completed block produces600 fresh partials. Complete composition requires all300 unique coefficients and2400 time outputs, summing75 taps before trained BN and one digital LIF update. Fitted alpha is diagnostic only; fixed gains are hardware1 and simulator0.03. Parent choice cannot switch at runtime. Only20 coefficients are newly measured;280 retain historical acquisition states and are not claimed to coexist physically.

## Limitations

Consistency PASS does not establish weight fidelity or full CIFAR classification accuracy. Transport/interrupted raw currently fails closed under the inherited complete-ledger replayer; it is never labeled complete or consistency-PASS. Complete scientific route-capacity failures are supported. The baseline-plus-failed-A accounting branch has synthetic tests but no real failed-A end-to-end exercise, and the frozen current config selects completed-A for both backends. No unknown hardware binding was invented.

## Reproduction

From repository root:

```text
python -m unittest test_audit_remaining_reserves -v
python audit_cifar_feedback_remaining_reserves.py --session results/cifar_feedback_remaining_reserves/UUID --output results/cifar_feedback_reviews/remaining_UUID
```

Output must be a new directory outside the immutable session. For the saved synthetic draft fixtures, run from the enclosing workspace root:

```text
python .tmp/hardware-public/audit_cifar_feedback_remaining_reserves.py --repo . --session .tmp/reserve_remaining_draft/synthetic_fixtures/success --synthetic-fixture --synthetic-plan .tmp/reserve_remaining_draft/configs/cifar_feedback_remaining_reserves.json --output .tmp/reserve_remaining_audit_draft/replay_success_NEW
```

Both completed-A parent source/result chains are preserved. A runtime/config/auditor were not changed. Only the programmer's separate approved dispatcher change adds the B/C launch option.
