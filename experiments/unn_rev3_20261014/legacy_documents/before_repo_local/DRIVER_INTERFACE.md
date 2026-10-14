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
`D:/projects/memristor-aware-twd-snn/memristor_hw/MemriBoard`.
The submodule in this hardware repository is older; its Connector return format
and simulator API differ. Fail clearly on the wrong version, never guess.

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
Use the raw ADC and the independently verified MVM transfer function. REV3's
initial read session does not expose MVM or writes.

## Simulator settings

In the inspected source, `open_port()` creates `BoardSimulator()` without
passing Connector's config. BoardSimulator reads a cwd-relative INI by itself.
The new session layer must explicitly assign the same board configuration to
the simulator interface before the first measurement. Do not change cwd globally
or monkeypatch `manager` modules to resolve this.

Loading config does not connect. Importing experiment modules must not instantiate
Connector, HardCore, Application or an FPGA class. The session may open only
through an explicit approved function call. No automatic simulator-to-real fallback.

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
