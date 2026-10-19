"""Read-only smoke-audit boundaries; no driver, Connector or native simulation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import audit_cifar_smoke as tool

SESSION = tool.ROOT / 'results/cifar_full_case0_20261019/4bfc8cd1-7cc8-4223-99ac-a94cbe546213'


class SmokeAuditTests(unittest.TestCase):
    def test_relative_evidence_paths_cannot_escape(self):
        for value in ('../escape', '/escape', 'D:/escape', '\\escape'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                tool.relative(tool.ROOT, value)

    def test_full_or_failed_never_accepted_as_smoke(self):
        with tempfile.TemporaryDirectory(dir=tool.ROOT / '.tmp') as name:
            folder = Path(name)
            (folder / 'start_manifest.json').write_text('{}')
            for status, scope in (('completed', 'full'), ('failed', 'smoke')):
                (folder / 'manifest.json').write_text(json.dumps(dict(status=status, scope=scope)))
                with self.assertRaisesRegex(ValueError, 'completed smoke'):
                    tool.verify_evidence(folder)

    @unittest.skipUnless((SESSION / 'manifest.json').is_file(), 'Archived native SIM fixture unavailable')
    def test_missing_acquisition_source_is_not_silently_accepted(self):
        final = json.loads((SESSION / 'manifest.json').read_text())
        start = json.loads((SESSION / 'start_manifest.json').read_text())
        removed = 'vendor/MemriBoard/manager/service/converters.py'
        del final['source_sha256'][removed]
        del start['source_sha256'][removed]
        with tempfile.TemporaryDirectory(dir=tool.ROOT / '.tmp') as name:
            folder = Path(name)
            (folder / 'manifest.json').write_text(json.dumps(final))
            (folder / 'start_manifest.json').write_text(json.dumps(start))
            with self.assertRaisesRegex(ValueError, 'source inventory'):
                tool.verify_evidence(folder)

    @unittest.skipUnless((SESSION / 'manifest.json').is_file(), 'Archived native SIM fixture unavailable')
    def test_changed_raw_chunk_hash_rejected_without_mutating_session(self):
        actual_sha = tool.sha
        def changed(path):
            if Path(path).name == 'part-000000.jsonl.gz':
                return '0' * 64
            return actual_sha(path)
        with mock.patch.object(tool, 'sha', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'Result SHA mismatch'):
                tool.verify_evidence(SESSION)


if __name__ == '__main__':
    unittest.main()
