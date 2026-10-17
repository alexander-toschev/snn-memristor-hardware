# Repository delivery validation — 17 October 2026

These checks concern packaging, repository organization and secret-scanning
infrastructure. They are not a new hardware measurement or CIFAR accuracy test.

- Organized offline suite: 574 tests, 573 passed, one existing failure, no errors
  or skips (800.830 s). The failure is
  `test_audit_remaining_reserves.test_final_source_bytes_and_actual_parent_bindings`:
  its historical B/C dispatcher pin starts `632c534f`, whereas the already
  published dispatcher at `33e182a` starts `78ab85d6`. The cleanup did not modify
  that dispatcher or relax the scientific provenance check.
- Additional repository layout/CLI/VS Code checks: 4/4 passed.
- 76 runtime Python files and three pinned root tests remained byte-identical
  during organization. Local Markdown-link validation found no missing targets.
- LFS secret-scanner checks: 16/16 passed, no skips (87.017 s), including real
  Git Bash interpreter-selection tests and Gitleaks 8.30.1. The real scanner
  rejected a synthetic unusable token after a 12 MiB binary prefix and accepted
  its clean control. No real credential was used in tests.
- Commit-time LFS payload scan passed for both restored calibration journals;
  payloads were verified against pointer size and SHA256 before scanning.
- All 39 historical inventory files were verified after materialization:
  796,607,329 bytes, every size and SHA256 matched FILE_INVENTORY.json.
- Independent final scientific review of the developer report and reproduction
  instructions found no blocking issues. Full classifier accuracy is not claimed;
  full hardware raw still on the offline Raspberry is explicitly distinguished.

The publication record describes the separate dataset secret scans and remaining
hardware transfer. A passing secret scan does not prove absence of every secret.
