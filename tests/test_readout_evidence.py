"""Synthetic provenance cases: source/UUID mismatch, failure and tampering."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from experiment_common.durable import sha
from experiment_common.readout_evidence import terminal_manifest


class EvidenceTests(unittest.TestCase):
    def test_completed_hardware_and_source_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);sid='fixture';folder=base/sid;folder.mkdir();(base/'archives').mkdir()
            manifest=folder/'manifest.json'
            manifest.write_text(json.dumps(dict(session_id=sid,status='completed',source='hardware')))
            receipt=base/'archives'/f'{sid}.json'
            receipt.write_text(json.dumps(dict(session=sid,files={'manifest.json':{'sha256':sha(manifest)}})))
            self.assertEqual(terminal_manifest(base,sid,'hardware')['source'],'hardware')
            with self.assertRaises(ValueError):
                terminal_manifest(base,sid,'simulator')
            manifest.write_text(json.dumps(dict(session_id=sid,status='failed',source='hardware')))
            with self.assertRaises(ValueError):
                terminal_manifest(base,sid,'hardware')
            receipt.write_text(json.dumps(dict(session=sid,files={'manifest.json':{'sha256':sha(manifest)}})))
            with self.assertRaises(ValueError):
                terminal_manifest(base,sid,'hardware')


if __name__=='__main__':
    unittest.main()
