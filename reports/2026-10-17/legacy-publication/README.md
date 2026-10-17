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
The two large `raw.jsonl` files are stored as lossless `raw.jsonl.gz` so that
individual repository files remain small. Both were decompressed in a stream
and their SHA256 checked against the complete original before publication.
The original uncompressed files remain in the local research checkout.

Before an analyzer that expects `raw.jsonl`, decompress the corresponding
`raw.jsonl.gz` to that exact original name in a separate analysis checkout and
verify the original SHA256 from this inventory. For example, from repository root:

```sh
python -c "import gzip,shutil; from pathlib import Path; p=Path('results/cell_calibrations/9fa4e745-cd50-44e9-97b0-b650292e4db8/raw.jsonl.gz'); src=gzip.open(p,'rb'); dst=p.with_suffix('').open('xb'); shutil.copyfileobj(src,dst); dst.close(); src.close()"
```

Exclusive creation (`xb`) refuses to overwrite an existing raw file. Merely
restoring the bytes does not make incomplete or exploratory provenance complete.
The publication scan covers the original uncompressed files; compression is not
used to avoid secret scanning.
