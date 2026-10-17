"""Offline source-verification tests; no native imports or hardware."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import audit_cell_map as audit


class SourceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.name = next(iter(audit.PRIVATE_PINS))
        self.data = b'private test fixture\n'
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.errors = []

    def check(self, condition, label):
        if not condition:
            self.errors.append(label)

    def private(self, data=None, saved_digest=None):
        if data is not None:
            path = self.repo / self.name
            path.parent.mkdir(parents=True)
            path.write_bytes(data)
        with patch.dict(audit.PRIVATE_PINS, {self.name: self.digest}), patch.object(audit.subprocess, 'run') as git:
            result = audit.verify_sources(self.repo, 'commit', {self.name: saved_digest or self.digest}, self.check)
            git.assert_not_called()
        return result

    def test_private_present_uses_pin_and_local_bytes_without_git(self):
        self.assertEqual(self.private(self.data), ([], {}))
        self.assertEqual(self.errors, [])

    def test_private_present_tamper_fails(self):
        self.private(b'tampered')
        self.assertEqual(self.errors, ['local_source/' + self.name])

    def test_private_absent_valid_pin_is_explicitly_unavailable(self):
        self.assertEqual(self.private(), ([], {self.name: self.digest}))
        self.assertEqual(self.errors, [])

    def test_private_absent_wrong_pin_fails(self):
        self.private(saved_digest='0' * 64)
        self.assertEqual(self.errors, ['private_source/pin/' + self.name])

    def test_private_present_matching_tampered_manifest_still_fails_pin(self):
        tampered = b'tampered'
        self.private(tampered, hashlib.sha256(tampered).hexdigest())
        self.assertEqual(self.errors, ['private_source/pin/' + self.name])

    def test_public_checks_git_blob_and_local_bytes(self):
        name = 'public.py'
        (self.repo / name).write_bytes(self.data)
        with patch.object(audit.subprocess, 'run', return_value=SimpleNamespace(stdout=self.data)) as git:
            self.assertEqual(audit.verify_sources(self.repo, 'commit', {name: self.digest}, self.check), ([name], {}))
            git.assert_called_once_with(['git', '-C', str(self.repo), 'show', 'commit:' + name], capture_output=True, check=True)
        self.assertEqual(self.errors, [])

    def test_public_tampered_local_and_git_are_both_reported(self):
        name = 'public.py'
        (self.repo / name).write_bytes(b'local changed')
        with patch.object(audit.subprocess, 'run', return_value=SimpleNamespace(stdout=b'git changed')):
            audit.verify_sources(self.repo, 'commit', {name: self.digest}, self.check)
        self.assertEqual(self.errors, ['local_source/' + name, 'git_source/' + name])


if __name__ == '__main__':
    unittest.main()
