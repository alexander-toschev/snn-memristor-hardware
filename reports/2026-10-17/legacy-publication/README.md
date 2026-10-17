# Historical simulator and exploratory artifacts preserved on 17 October 2026

These files were previously untracked local experiment outputs. They are
published to preserve the research record, not to qualify a current board state
or replace the reviewed 16 October CIFAR comparison.

- `results/cell_calibrations/8fb315b9-3750-41d0-a810-01a787b48efc/`:
  historical simulator outputs completing the previously tracked manifest,
  protocol, settings, report and offline audit for this session.
- `results/cell_calibrations/9fa4e745-cd50-44e9-97b0-b650292e4db8/`:
  historical simulator outputs and reports; recorded provenance includes a dirty
  checkout and untracked runtime. Preservation does not repair that provenance.
- `exploratory_singletons/` under calibrated-MVM sessions `ea567b92-2428-48d3-b072-914fa31961a9`
  and `f4046fbf-ec5f-4aa7-b949-e2fc2900d51f/`: exploratory derived analyses.
- `mapped_singleton_proposal/` under `f4046fbf-ec5f-4aa7-b949-e2fc2900d51f/`:
  a derived proposal, not proof of a subsequently executed hardware protocol.

`FILE_INVENTORY.json` records original paths, byte counts and SHA256, plus the
stored file paths, byte counts and SHA256. Original files were not edited.
The two large `raw.jsonl` files retain their original names and bytes through
Git LFS. After installing Git LFS, run from repository root:

```sh
git lfs install
git lfs pull
```

No manual decompression is required. Verify payload SHA256 against this
inventory before analysis. Preserving the bytes does not repair incomplete or
exploratory provenance. The publication secret scan covered original payloads.
