# MemriBoard hardware conductance STDP v2 — preparation only

Separate launcher and supplemental auditor passed independent source review;
the exact bounded hardware protocol is registered conditionally for root.
The frozen native-SIM directory and auditor are unchanged. Root coordinates
the actual foreground run on Raspberry **after CIFAR releases the board**.
The shared exclusive hardware lock prevents overlap with other board users;
the launcher never deletes another session's lock or automatically schedules
itself after CIFAR.

All physical interactions go through public `manager.board.Connector`, native
converters and the reviewed `configs/raspberry_board.ini` driver selection.
No direct driver/GPIO/SPI/serial access, `.interface`, dynamic module loading,
runtime monkeypatch or native API bypass exists. On-Pi preflight strictly
checks the private MemriCORE source files and their known SHA; missing files
are a blocker, not a reason to omit source checks.

The physical/digital protocol is exactly the qualified v2 protocol:
BL6/8/9/14 at WL5(train) and WL7(control),8cells; readMVM0.05V;SET1.7V;
RESET1.4V;≤128 admitted pulses and≤6712 impacts. Each operation retains fresh
nativeR and unchanged GUI current/ADC guards. Hardware pulse width is submitted
as0 to the native driver; the physical duration is unknown.

Preparation can modify all8selected cells before calibration. Two fresh fit
windows choose from the fixed gain grid using WL5 only; the chosen gain is
shared with control and frozen before pretest/training. SIM's gain800 is never
loaded or reused. No eligible gain stops with `uninformative_calibration`,
exit3, before pretest/training, preserving all earlier preparation/raw data.
No voltage/gain-grid retry or silent fallback is allowed.

The qualifying actual SIM session, independent audit, conditional hardware
review, original source freeze and auditor are SHA-pinned. Every SIM result
artifact is rechecked. The new manifest preserves the original acquisition
`source_sha256` inventory exactly for the frozen protocol auditor and adds
separate `launcher_source_sha256`, `launcher_sha256`, launcherSPEC/SHA and
qualification references. A supplemental HW auditor must verify these extra
fields against real files and start/final manifests, then invoke the unchanged
full raw replay auditor. The old auditor alone does not qualify this launcher.

New results go to `results/conductance_stdp_hardware_20261019/<UUID>` and use
Git LFS for raw gzip. Writer limits remain64MiB expanded,2GiB initial disk
reserve,1GiB periodic reserve. Invalid data closes the Connector and preserves
failed/partial evidence. Driver output during every impact is retained in raw
events; progress remains foreground. Retention uses the unchanged60s interval.

After Git checkout/pull and LFS retrieval on Raspberry, one command at a time:

```text
python experiments/conductance_stdp_hardware_20261019/experiment.py --dry-run
```

Only after launcher/auditor review, registration and CIFAR completion:

```text
python experiments/conductance_stdp_hardware_20261019/experiment.py --run-hardware
```

The result is algorithmic, boundary-updated conductance STDP. NativeSIM
qualification is evidence of functioning software/protocol branches; it does
not demonstrate physical learning improvement or classification accuracy.
