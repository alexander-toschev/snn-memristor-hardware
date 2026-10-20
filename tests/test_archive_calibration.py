import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import archive_calibration as a

class ArchiveTests(unittest.TestCase):
    def test_exact_archive_two_full_streams_and_no_source_changes(self):
        temporary=a.ROOT/'.tmp/archive-unit';temporary.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as directory:
            base=Path(directory);session=base/'test-session';session.mkdir()
            (session/'raw.jsonl.gz').write_bytes(gzip.compress(b'{"event":"complete"}\n'))
            manifest=dict(status='completed',finished_utc='2026-10-20T00:00:00Z',source='simulator',
                          result_sha256={'raw.jsonl.gz':a.b.sha(session/'raw.jsonl.gz')})
            (session/'manifest.json').write_text(json.dumps(manifest))
            original=a.inventory(session)[1];scans=[]
            def scan(scanner,chunks):scans.append(b''.join(chunks));return dict(passed=True)
            with patch.object(a.b.security,'find_scanner',return_value=Path(a.__file__)),patch.object(a.b,'scan_chunks',side_effect=scan):
                report=a.preserve(session,base)
            self.assertEqual(len(scans),2);self.assertIn((session/'raw.jsonl.gz').read_bytes(),scans[0])
            self.assertIn(b'{"event":"complete"}',scans[1]);self.assertEqual(a.inventory(session)[1],original)
            saved=json.loads((base/'archives/test-session.json').read_text())
            self.assertTrue(saved['byte_identical_archive_verified']);self.assertTrue(saved['decoded_gzip']['raw.jsonl.gz']['complete_gzip'])
            self.assertGreater(report['decoded_bytes'],0)
            with self.assertRaises(FileExistsError):a.preserve(session,base)
            wrong=dict(original);wrong['manifest.json']=dict(wrong['manifest.json'],sha256='0'*64)
            with self.assertRaises(ValueError):a.verify_archive(base/'archives/test-session.tar.gz','test-session',wrong)

    def test_live_and_corrupted_payload_rejected(self):
        temporary=a.ROOT/'.tmp/archive-unit';temporary.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as directory:
            base=Path(directory);session=base/'test-session';session.mkdir()
            path=session/'manifest.json';path.write_text(json.dumps(dict(status='running')))
            with self.assertRaises(ValueError):a.preserve(session,base)
            path.write_text(json.dumps(dict(status='completed',finished_utc='now',result_sha256={'missing':'0'*64})))
            with self.assertRaises(ValueError):a.preserve(session,base)

if __name__=='__main__':unittest.main()
