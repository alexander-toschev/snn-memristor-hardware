# Native N1 wrapper audit history

The acquired component was copied without byte changes. wrapper_v1 failed only because its wrapper mistakenly required the production experiment.py entrypoint in a standalone component source manifest. wrapper_v2 checks the actual pinned run_native_adaptive_n1.py harness. Original reports remain unchanged, including temporary paths and auditor hashes. The final archived-path replay is in the separate adaptive_native_n1_20261016_v3 directory.

The component was acquired with staged/untracked work recorded in its original manifest; no fully clean source tree is claimed. Local recorded source hashes are checked, but untracked files not included in the launch source inventory are not retrospectively hash-attested. This is an explicit singleton component, not an acquired 13+1 history or full 300-coefficient result.

Reproduce offline from the repository root (choose a new output directory):

```text
python audit_native_adaptive_n1.py --session results/cifar_feedback_component_tests/49478a54-643e-4009-bcba-89aaf6629b60 --output results/cifar_feedback_reviews/adaptive_native_n1_recheck
```
