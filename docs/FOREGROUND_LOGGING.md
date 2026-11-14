# Foreground experiment logging

An SSH output pipe closed during the 14 November expanded MNIST acquisition.
The next progress print raised `BrokenPipeError`; the experiment finalized its
partial raw data, marked the session failed and closed the MemriBoard port.
That failed session remains evidence and is not resumed or relabeled completed.

For a separately authorized next launch, keep the experiment in the foreground
and direct both Python and native stdout/stderr to a new local file:

```sh
python -u -B tools/run_foreground_logged.py --log results/foreground_logs/UNIQUE_RUN.log experiments/mnist_expanded_exploratory_20261114/experiment.py --run-hardware --expanded --exploratory-measured-lut --acquire-only --publish
```

Replace `UNIQUE_RUN` before running. This command is an example, not permission
for an additional hardware run. The wrapper refuses an existing log and requires
repository-relative paths. It executes the entrypoint in the same process, with
no detached job, child supervisor, retry or hardware access of its own. Normal
exit codes and interrupts propagate. Use `-u` for prompt Python log flushing.

Regular-file output removes dependence on the SSH output pipe. It does **not**
protect the process against `SIGHUP`, host loss, power loss or every kind of SSH
disconnection. Do not infer completion from disappearance of the SSH handle.
Inspect the actual terminal manifest, cleanup status and publication receipt.

The native log is separate from the immutable session archive. After the process
ends, preserve it with the session reference and SHA-256, scan it for secrets,
and publish it as separate evidence. Do not insert it into an already finalized
session or overwrite earlier logs. Existing structured raw recording, current
guards, finite budgets and Connector cleanup remain unchanged.

Offline tests cover a closed parent output pipe (Python and native writes),
exclusive log creation, path escape rejection and propagation of exit code 7.
