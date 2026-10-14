# SNN Memristor Hardware Experiment (Two-Crossbar, CIFAR)

Code for the real-hardware validation experiment: two physical memristor
crossbars (32x8 = 256 cells each, 512 cells total) used to validate the
MemriBoard simulator's write-noise and differential-mapping predictions,
originally developed for CSNN/CIFAR-100(20) Conv1 filter deployment.

See `TWO_CROSSBAR_PLAN.md` for the experiment plan (phases, endurance/time
analysis, per-cycle protocol), `HARDWARE_REQUIREMENTS.md` for the detailed
simulator-to-silicon requirements analysis (write precision, conductance
window, linearity, ADC/DAC resolution, technology comparison), and
`DRIVER_INTERFACE.md` for the exact driver interface contract the real-hardware
driver (`rp5_fpga_python`) needs to implement to be a drop-in replacement for
the software `BoardSimulator`.

`experiments/` contains the pilot and dry-run scripts used to develop and
rehearse the sequential-reprogram protocol prior to running on physical
hardware.

Extracted from the `snn-mnist` dissertation project (CSNN/CIFAR-100:20,
Alexander Toshchev).
