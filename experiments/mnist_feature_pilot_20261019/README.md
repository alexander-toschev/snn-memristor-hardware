# Train-only frozen-weight MNIST feature pilot

New reconstruction, not a reproduction of historical69.65% accuracy. No new
STDP, readout training, test examples, MemriBoard SIM or board access. Source
The fixed CUDA train128×2 pilot is registered after root/scientific source
review. The network has not yet been executed; root schedules the launch.
The archived reference and every active CIFAR/STDP source remain unchanged.

Scope is fixed: first128 train examples, one128-example batch, T100, two passes
of exactly the same128 examples to check repeatability. Saved checkpoint Conv1
is64x1x5x5, output64x24x24. Counts are128x36,864; unused10WTA columns are omitted.
No train/test split or sample-count override is offered.

The archived SHA-pinned builder and encoder are imported normally. Builder
initial normalization occurs before loading checkpoint W and threshold, and
is overwritten by those saved tensors. The broken post-window `connection`
lookup is deliberately left unchanged. With archived w_norm_enable/EI true,
its caught exception skips post-window normalization and E/I. This hook never
ran per timestep in that reference anyway. No alias or silent fix is installed.

The pilot explicitly sets network/layer `learning=False`, unlike historical
counts code that only zeroed STDP nu. This is a declared reconstruction choice
to prevent weight changes. All connection weights and all layer thresholds
are SHA-checked before/after each pass. Connection normalization must be None;
no Dales transform is allowed. The public BindsNET default synchronous run is
preserved, including one-step Input→Conv1 delay. This is a software SNN pilot;
all future real memristor operations still require MemriBoard Connector.

Poisson uses the unchanged deterministic content-hash encoder, base_seed123,
rate_scale0.0015,boost1. It does not use torch RNG for these spikes. Torch/NumPy
and Python seeds42 are fixed for construction. Inputs and counts must match
exactly across two reset passes; input-spike SHA/count are recorded. Negative
or nonfinite source/feature inconsistencies abort instead of substitutions.

Use only the new isolated repository-local `.venv-mnist-pilot`. Existing
`.deps`, `.deps-classifier`, `.venv` and system packages are not modified.
Official BindsNET tag0.3.3 resolves to
`1b9786152db6fc67a38d2ddb3090ac434bca239f`; its archive SHA256 is
`d3e2a214e1c740981e508fcca439ed7509d70e7df1e60ec6ed60fe6ec682458e`.
Its actual dependencies require Python3.9–3.11, Torch2.4.1, torchvision0.19.1,
torchaudio2.4.1 and NumPy2.x. We select Python3.11.9, CUDA12.4 wheel variants,
NumPy2.0.2, and freeze all resolved dependencies. This differs from the later
unattested snn-mnist environment.yml candidate; no historical lock exists.

`INSTALL_PROVENANCE.json` records all63 installed package versions, official
artifact URLs and SHA256. `bootstrap.lock` and `requirements.lock` pin the
Windows/CPython3.11 artifacts. Every preflight checks the entire installed
distribution inventory and all48 BindsNET Python source files against the
official source archive. Imports, NumPy/CUDA tiny tensor interoperability and
`pip check` passed; no neural forward has yet run.

To recreate in a fresh isolated environment using Python3.11.9:

```text
py -3.11 -m venv .venv-mnist-pilot
.venv-mnist-pilot/Scripts/python.exe -m pip install --require-hashes --no-deps -r experiments/mnist_feature_pilot_20261019/bootstrap.lock
.venv-mnist-pilot/Scripts/python.exe -m pip install --require-hashes --no-deps --no-build-isolation -r experiments/mnist_feature_pilot_20261019/requirements.lock
.venv-mnist-pilot/Scripts/python.exe -m pip check
```

The dependency graph was resolved normally and checked before this lock was
made; `--no-deps` during recreation installs exactly the complete pinned graph.

Preflight checks imports, versions, Torch↔NumPy/CUDA tiny-tensor interoperability,
source/data hashes and available memory. It does not construct/run the SNN:

```text
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_feature_pilot_20261019/experiment.py --preflight --device cuda
```

Registered frozen CUDA pilot (launch coordinated by root):

```text
.venv-mnist-pilot/Scripts/python.exe -B experiments/mnist_feature_pilot_20261019/experiment.py --run-pilot --device cuda
```

GTX1660SUPER has6GiB. Preflight requires4GiB free for fixedbatch128/T100; no
silent smaller-batch/CPU fallback. CPU preflight is available, but CPU execution
remains gated until separate review.
Monitor storage defaults to host CPU; float32 reduction alone can occupy1.76GiB
of host memory, in addition to stored spikes and temporary buffers. GPU memory
peaks exclude this host allocation. Explicit CUDA synchronization after net.run
precedes CPU monitor reads, completing nonblocking device-to-host copies.
Recorded timing includes these copies and CPU aggregation. Runtime and peak
memory must be measured by this pilot; no current
40k-example ETA exists. Two counts arrays occupy37,748,736 bytes plus headers;
input spikes uint8 occupy10,035,200 bytes. Allow at least128MiB output space.

New UUID results contain original train IDs/images/labels, input spikes,
frozen weights, both counts arrays, full source/dependency provenance, weight/
threshold hashes, timings and CUDA memory peaks. Restricted `weights_only=True`
checkpoint loading follows a trusted exact file SHA; there is no unrestricted
pickle fallback. Failure output is marked failed, never promoted to accuracy.
