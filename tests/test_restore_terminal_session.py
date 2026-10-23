from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from tools.restore_terminal_session import ROOT, restore

SID = '00000000-0000-4000-8000-000000000001'


def archive(base: Path, name: str) -> None:
    folder = base/'archives'
    folder.mkdir()
    path = folder/f'{SID}.tar.gz'
    data = b'evidence'
    with tarfile.open(path, 'w:gz') as tf:
        member = tarfile.TarInfo(SID+'/'+name)
        member.size = len(data)
        tf.addfile(member, io.BytesIO(data))
    receipt = dict(byte_identical_archive_verified=True, archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        files={name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())})
    (folder/f'{SID}.json').write_text(json.dumps(receipt), encoding='utf-8')


class RestoreTest(unittest.TestCase):
    def test_exact_restore_idempotent_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            base = Path(temp)
            archive(base, 'raw/sample.json')
            self.assertEqual(restore(base, SID), 1)
            self.assertEqual(restore(base, SID), 1)
            target = base/SID/'raw/sample.json'
            self.assertEqual(target.read_bytes(), b'evidence')
            target.write_bytes(b'user change')
            with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                restore(base, SID)
            self.assertEqual(target.read_bytes(), b'user change')

    def test_rejects_traversal_even_with_matching_archive_hash(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            base = Path(temp)
            archive(base, '../escape.json')
            with self.assertRaisesRegex(ValueError, 'escapes'):
                restore(base, SID)
            self.assertFalse((base/'escape.json').exists())


if __name__ == '__main__':
    unittest.main()
