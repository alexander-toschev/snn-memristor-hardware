# Native MemriBoard interface - REV3

Use the existing `manager.board.Connector`, not a new hardware driver embedded
in the experiment. The current backend is **simulator**.

## Configuration layers

- `cb_type='simulator'|'real'`: selected chip/backend. The GUI Manager reads it
  from chip metadata in `use_chip()`.
- `board_type='rp5_fpga_python'`: hardware driver name from
  `manager/service/drivers.py`; this alone does not select physical hardware.
- `config`: board conversion parameters supplied to Connector.
- `crossbar_serial`: simulator array file stem; not a physical array selector.

The compatible local installation is
`vendor/MemriBoard`.
The obsolete submodule was removed from the working tree and remains in Git
history. Its Connector return format and simulator API differ; the experiment
uses only the vendored version.

## Modern Connector contract

Constructor arguments are `silent`, `logger`, `cb_type`, `board_type`, optional
`driver_attr`, `task_format`, and keyword configuration/serial.
`open_port()` selects BoardSimulator or the registered driver.
`impact(task)` dispatches the same experiment task to the selected backend.

Use `task_format='SI'`: `vol` in volts and `pulse_width` in seconds. Read-only
single-cell tasks have `mode_flag=7`, `vol=0`, `pulse_width=0`, `sign=0`, `id`,
`wl`, `bl`. The return is `(resistance_ohm, id, raw_adc)`, not `(adc,id)`.
Preserve raw ADC and verify ID before analysis. Some stale docstrings say
`task_format='adc'`; source accepts **`'dac'` or `'si'`**.

For MVM, do not apply the single-cell `a2r` formula to a sum-current ADC code.
The simulator suite converts raw ADC to output voltage using the configured
ADC scale, then compares differential tile outputs with ideal and readback
weights. This checks the simulator contract; physical MVM transfer and timing
still require separate validation.

## Simulator settings

In the inspected source, `open_port()` creates `BoardSimulator()` without
passing Connector's config. The vendored BoardSimulator resolves its INI next to its source module.
The experiment reads that same INI and passes it as Connector's `config`.
The simulator loads its configuration normally; the experiment never assigns
`connector.interface` or its configuration. The effective board values are copied
into the saved protocol for analysis.

Loading config does not connect. Importing experiment modules must not instantiate
Connector, HardCore, Application or an FPGA class. The experiment opens Connector only
after the explicit simulator launch flag. No automatic simulator-to-real fallback.

## Hardware observations, not permission

Earlier Pi source inspection found
`MemriCORE.rp5_fpga_python.rpi_FPGAed.RPI_modes_FPGAed` with no-argument
constructor, `mode_7(vDAC,tms,tus,rev,id,wl,bl)`,
`mode_9(vDAC,id,wl,bl)` and `mode_mvm(vDAC_mas,tms,tus,rtms,rums,wl,id)`.
The constructor opens SPI/GPIO, so constructing it is already hardware I/O.
The full mode_7 path and timing have not been reviewed. No measurement was made.

HardCore construction scans the whole matrix; it cannot be used unchanged with
the WL=1 exclusion. Replacing only a `BoardSimulator()` constructor in the old
experiment is insufficient: those scripts also access `board.crossbar`, noise
flags, model resistance, pickle arrays and absolute WSL paths.

## Checked simulator contract

Run `python -m experiments.unn_rev3_20261014.check_connector --run-simulator`
for seven cases on a fresh array: mode 7 read, mode 9 at 0/.1/.2/.3 V,
and mode 10 zero/one-hot input. Each SI command is compared with a native DAC
command from the same saved starting state. All board parameters must match
the vendored INI. This check does not touch an existing experiment session.

Mode 9 returns ID=0 in the native simulator; modes 7 and 10 echo the requested
ID. Zero ADC is valid for zero-input mode 9/10. Returned resistance from mode 9
or MVM is not interpreted as single-cell resistance. Native DAC conversion
rounds voltages to three decimal places; SI/DAC equality is not physical
voltage calibration. See `results/connector_checks/` for recorded commands.

The root experiment uses ordinary `from manager.board import Connector` and
`from simulator.src import create_crossbar_array` imports. The latter creates
the saved array, as Manager.add_chip does; it does not choose or construct
the board interface. `Connector.open_port()` performs that selection.
Only the local package search paths are configured once in `paths.py`.

## Raspberry Pi configuration files

The Raspberry Pi board will have its own INI files. `SETTINGS_FILE` in the root
experiment identifies the local INI passed to `Connector` as a `ConfigParser`;
use the board's supplied file for that deployment. Keep its path relative to
the repository. `CB_TYPE` and `BOARD_TYPE` remain the native selection parameters.
The simulator run uses `vendor/MemriBoard/simulator/simulator_settings.ini`.
Driver-specific INI files remain managed by the native driver; changing the
Connector INI does not automatically replace files read by that driver itself.
The current run is simulator-only and does not provision or connect the Pi.

For the simulator, changing only `SETTINGS_FILE` to a different INI would not
change the model's own INI. Keep Connector and the native simulator settings
consistent. This run reads the same vendored INI on both sides. For Raspberry,
verify the installed driver's INI locations against that driver's source; the
hardware driver source is not bundled here. The current entry point explicitly
allows only simulator runs and requires a separate reviewed hardware launch.
