# Publication record — 17 October 2026

The user explicitly authorized publication of the complete simulator journals
and reports to `alexander-toschev/snn-memristor-hardware`, branch
`codex/full-cifar-diagnostic`, and then requested all experiment artifacts be
preserved in Git. This record distinguishes available local files from data
that still require the Raspberry to be online.

## Included local datasets

| Source local commit | Content | Added files | Uncompressed bytes |
|---|---|---:|---:|
| `3accde8af12dbc8c3ba8179cce9d934c753550dd` | Fresh 300-weight native SIM, raw and independent audit/science | 718 | 1,918,152,793 |
| `8a562bd` | No-BL8 native SIM, raw/audit/science and calibrated-MVM SIM | 732 | 1,984,844,395 |

These datasets are incorporated as data commits on the publication branch;
their resulting commit IDs can differ from the original local IDs. Individual
blobs in these source commits are below 100 MiB; largest are 33,554,345 and
69,834,321 bytes respectively. Existing source manifests and raw contents are
preserved, including historical source pins and recorded status.

Seven exact dataset directories were rescanned with Gitleaks before publication,
using `--redact=100 --ignore-gitleaks-allow --max-target-megabytes 0` (no large-file
skip). All seven scans completed with no detected leaks. They cover both fresh
session roots, their four audit/science directories, and calibrated-MVM SIM
session `3ed62521-1a3a-495f-9cd3-f4bbcf807c90`.

Additional previously untracked historical/exploratory outputs are described in
[legacy-publication/README.md](legacy-publication/README.md) and its per-file
inventory. All files retain their original names and bytes; JSONL journals
are stored using Git LFS. They are not relabelled as current
hardware qualification.

All five corresponding historical/exploratory source directories also passed
the same no-size-skip Gitleaks scan, including both uncompressed raw journals.
Two additional audit files were recovered from the main local checkout and
passed its staged secret scan. The 39 preserved files total 796,607,329 original and materialized stored bytes; each copied stored file was SHA256-verified against its inventory.

The publication also includes the Monday experiment plan, repository navigation
and the developer-facing HW/SIM discrepancy report. Documentation organization
does not change the frozen acquisition code or historical results.

## Hardware raw still pending

At the availability check on 17 October, Raspberry Pi Connect showed the device
last seen about eleven hours earlier, and its console showed `Connection closed`.
Full hardware acquisition files remain committed on the Raspberry, including
local commits `454d72c`, `cb91f02` and `995902e`; this publication cannot transfer
those unavailable files. Compact hardware CSV/JSON, maps, LIF trajectories and
the direct HW/SIM comparison are already in the repository.

When the Raspberry is available, inspect its current Git state, verify and
scan the exact unpublished content, incorporate the public branch without
discarding either history, and publish the remaining hardware data. No new
measurement or board access is required to transfer those files.

Secret scans are evidence of no detected credentials, not an absolute guarantee.
Credentials, private operator-provided driver files and local `.tmp` work are
not publication artifacts.

## Git LFS storage

Raw JSONL, archived payloads, array snapshots and network weights use Git LFS.
Install Git LFS and run `git lfs pull` after checkout to obtain their contents.
Public history through `33e182a` is preserved; only unpublished data commits
were rewritten to LFS pointers. No force push is required. Original source
manifest pins and payload SHA256 values are unchanged.
