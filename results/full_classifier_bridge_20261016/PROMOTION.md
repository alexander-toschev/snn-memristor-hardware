# Software-preparation promotion record

Promoted the five bridge modules, three test modules and synthetic harness to
repository root after the saved goal-control review. Documentation is now
`FULL_CIFAR_BRIDGE.md` and `FULL_CIFAR_NATIVE_LINK.md`; the root README, CPU
classifier, acquisition runtimes and dispatcher were not edited by this action.

One reviewed hardening change precedes promotion: origin reference strings now
reject Windows parent traversal (`..\\`) as well as POSIX parent traversal.
Added one regression test covering both session and folder references.

|File|Reviewed draft SHA256|Promoted SHA256|
|---|---|---|
|native_full_cifar_adapter.py|11b8f7da2199a03d7db266e7b7c355088cc91ccae17ce41be61ece96b05905a0|3ffdd77434c9d60c6e90e82f35baa67cf1ee86cec2ebd8dad7264adf48151e81|
|test_native_full_cifar_adapter.py|166a0b95bf60bf8745461b98e3d9caeae01b77657b7c04de28dcf8edeec531fa|6c069596ddcc4339cfff52850b551dc3adea5b7a45d6911b392cc10fccafe804|

The native parser, planner, exporter, independent numerical replay and synthetic
harness are unchanged. The saved review remains the original review of its
listed draft hashes; it was not rewritten to imply that it reviewed new bytes.

Preserved evidence was copied byte-for-byte:

- `actual_native_300_link.json`: SHA256
  `650984ecd0011bdb75b13b1ebe5932988529fea8c4fbafa722ecbae7d29e8097`.
  Its verifier SHA remains
  `3410a118fa80469878b1f1dd5632d1ad2eee4981f402db8273a22aa2c7c79bc7`.
- `synthetic_full_coverage.json`: SHA256
  `3e44de62d2ca3a80ab634de0d51cb2b9df54be3e2d1ee77f500d437e2b419aa9`.
  Its embedded plan/export/replay/harness hashes still match the promoted code.
- `ten_image_plan.json`: SHA256
  `968f4151ed2836b255ce90637d8349cdc4b2adc3ccdc113bae6e5a7ae1eaea4f`.
- `GOAL_REVIEW.md`: exact copy of the original reviewer report.

Validation from repository root:

```powershell
python -m unittest test_full_cifar_bridge test_native_full_cifar_link test_native_full_cifar_adapter -v
```

**23 tests PASS in0.428seconds.** The expensive actual/synthetic full replay was
not repeated: its parser/arithmetic bytes did not change. No native backend or
hardware calls occurred. Full hardware import remains disabled; historical300
native evidence and full synthetic software coverage remain separate claims.
