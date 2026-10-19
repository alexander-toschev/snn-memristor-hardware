"""Closed launcher/provenance and event accounting; no network execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.mnist_stdp_retraining_20261019 import run_full as e


class FullLauncherTests(unittest.TestCase):
    def test_launch_gate_closed_without_review(self):
        with patch.object(e,'APPROVED_FULL_SPEC_SHA256',frozenset()):
            with self.assertRaisesRegex(ValueError,'qualification'):e.main(['--run'])
        with patch.object(e,'QUALIFICATION_REVIEW_SHA',None):
            with self.assertRaisesRegex(ValueError,'qualification'):e.main(['--run'])

    def test_unchanged_acquisition_and_actual_smoke_inventory(self):
        self.assertEqual(e.sha(e.ENTRY),e.ENTRY_SHA)
        helper=e.import_reviewed_helper()
        self.assertFalse(helper.APPROVED_FULL_SPEC_SHA256)
        qualification=e.qualify_smoke()
        self.assertEqual(qualification['smoke_manifest_sha256'],e.SMOKE_MANIFEST_SHA)

    def test_actual_coverage_not_planned_numbers(self):
        with tempfile.TemporaryDirectory(dir=e.ROOT/'.tmp') as directory:
            path=Path(directory)
            self.assertEqual(e.confirmed_events(path)['head_updates_at_last_completed_epoch'],0)
            rows=[dict(kind='stdp_image',image_id=0),dict(kind='new_weight_features',split='train',start=0,end=128),
                  dict(kind='head_epoch',epoch=1,updates=59)]
            (path/'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            result=e.confirmed_events(path)
            self.assertEqual(result['training_images'],1)
            self.assertEqual(result['feature_rows'],dict(train=128,test=0))
            self.assertEqual(result['head_updates_at_last_completed_epoch'],59)
            self.assertEqual(result['test_evaluations'],0)


if __name__=='__main__':unittest.main()
