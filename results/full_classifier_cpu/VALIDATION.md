# CPU classifier validation (no hardware acquisition)

All ten fixed genuine CIFAR100 subset images completed the full trained network.
Each uses all64 Conv1 filters,28x28 positions,75 CHW coefficients and12 logical
timesteps. CPU partials use three tiles of32,32,11 coefficients.

The independent functional downstream and original `TemporalCSNN.forward`
produced exactly identical output membranes, spikes and TWD scores on all ten.
Tiled versus direct Conv1 maximum absolute roundoff was9.5367431640625e-7.
The direct Conv1 downstream also matched the original forward exactly.
Nine offline tests passed, including all ten comparisons, negative membrane,
hard reset, nonzero trained BN at silent t10/t11 and rejected incomplete or
unattested hardware inputs.

|Fixture|Original test index|True subset index|First spike|Membrane sum|TWD|
|---:|---:|---:|---:|---:|---:|
|0|9|0|1|1|1|
|1|132|1|0|1|0|
|2|54|2|2|2|2|
|3|396|3|0|3|0|
|4|334|4|2|4|2|
|5|27|5|5|5|5|
|6|94|6|6|6|6|
|7|7|7|7|7|7|
|8|49|8|8|8|8|
|9|122|9|9|9|9|

Correct counts: first-spike6/10, membrane-sum9/10, TWD6/10. These are ten selected
diagnostic images, not a complete test-set accuracy estimate and not hardware
accuracy. Output indices map to fine labels `[0,1,2,3,5,8,13,14,17,19]`.
The alternative decoder results are reported, not selected after seeing labels.

Final numerical reports are in `final_10/case0.json` through `case9.json` and
bind the final classifier source SHA. Earlier `validated_10/` and `case0.json`
are preserved as historical CPU checks before path-validation hardening.
Use the commands in `FULL_CIFAR_CLASSIFIER.md` for a new output folder.

## Remaining full hardware integration work

This closes the CPU numerical-reference preparation, **not** full hardware
classification integration. The measured importer is disabled. Existing physical
four-filter/spot-patch data cover neither all64 filters nor all784 locations.

1. **Full acquisition planner/collector**: add a separate reviewed scenario,
   not a change to frozen historical experiments. Bind a full image and TTFS,
   map all4800 Conv1 coefficients with physical microtiles, and schedule every
   `(t,filter,y,x,tile)` contribution. Freeze per-tile decoder/calibration before
   acquisition. Each measured response must reference its fresh raw commands;
   do not cache basis responses or compute missing responses on CPU. Produce a
   bounded plan with real programming/read/zero/repetition counts and shutdown
   feasibility before any execution.
2. **Exporter**: consume the collector's chunked raw plus frozen calibration,
   derive native ADC values and paired corrections, group physical microtiles
   into logical32/32/11 partials, reject duplicate/missing keys and retain raw
   task references. Emit the complete float32 tensor, coverage, pinned manifest
   and image/checkpoint/calibration/source identities. It must never import
   checkpoint weights to manufacture a missing measured contribution.
3. **Independent exporter auditor**: separately replay raw-to-partial arithmetic,
   route/CHW/spatial identity, fixed-before-CIFAR calibration, exact coverage,
   source hashes and all safety/transport outcomes. Bind payload digest to raw
   manifest digest. Test missing last-time/edge/filter, duplicate keys, wrong
   image, post-BN input, altered decoder and substituted CPU partial mutations.
4. **Enable consumer only after review**: register the reviewed export-auditor
   source identity and verify its reproducible evidence, then use the existing
   downstream CPU module. Validate this bridge on explicitly labeled synthetic
   measured-contract fixtures and a software acquisition test before physical
   execution. No fixture should be represented as physical measurement.
5. **Classification evaluation**: compare measured-Conv1/digital-rest against
   the fixed CPU reference for the same full images, separating completeness,
   Conv1 fidelity, membrane/spike effects and classification. A full dataset
   accuracy claim requires all intended test examples and preselected decoder.

The minimum logical contract is1806336 partials/image; five repeats yield9031680.
This is not an impact budget. Current physical encoding can require multiple
row/zero/one-hot reads per partial, so a credible native I/O estimate cannot be
borrowed from the old four-filter protocol. These are four implementation units
(planner/collector, exporter, independent auditor, enabled bridge) plus validation;
none requires pretending that today's CPU test was measured on hardware.
