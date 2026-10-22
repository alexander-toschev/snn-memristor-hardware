from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from typing import TYPE_CHECKING, cast

from experiment_common.calibration_inventory import REMAINING, inventory, mapping_candidates, render_map
from experiment_common.durable import Writer, atomic_json, sha
from experiment_common.programming_validation import CELLS, Collector, Schedule
from experiment_common import terminal_publication as publication
from experiments.remaining_programming_validation_20261022 import experiment
from tests.test_selected_programming_validation import Native, Sink
from tests.test_selective_metrology import Clock, Converters

if TYPE_CHECKING:
    from manager.board import Connector


class AllNative(Native):
    def __init__(self):
        super().__init__()
        self.state = {(w, b): 0 for w in range(8) for b in range(32)}
        self.written = set()

    def impact(self, task):
        if task['mode_flag'] == 7 and task['vol']:
            self.written.add((task['wl'], task['bl']))
        return super().impact(task)


class RemainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native, clock = AllNative(), Clock()
        schedule, _ = experiment.protocol(False)
        cls.collector = Collector(cast('Connector', native), Converters(), cast(Writer, Sink()),
                                  'simulator', lambda _: None, clock=clock.now, sleep=clock.sleep,
                                  schedule=schedule)
        cls.result = cls.collector.run()
        cls.native, cls.clock = native, clock

    def test_exact240_plus16_no_old_writes_and_no_b(self):
        c = self.collector
        self.assertEqual(c.calls, 265600)
        self.assertEqual(c.pulses, 7680)
        self.assertEqual(c.phase_calls, dict(A=264960, B=0, R=640))
        self.assertEqual(len(c.points), 6256)
        self.assertEqual(c.pairs, [])
        self.assertEqual(self.native.written, set(REMAINING))
        self.assertTrue(self.native.written.isdisjoint(CELLS))
        self.assertEqual(self.clock.value, 360)
        self.assertEqual(self.result['status'], 'completed')

    def test_mixed_inventory_and_not_simultaneous_qualification(self):
        prior = experiment.read(experiment.PRIOR / 'observations.json')
        manifest = experiment.read(experiment.PRIOR / 'manifest.json')
        result = inventory(self.collector.cells, prior, manifest, self.collector.points, 'new', 'later')
        self.assertEqual(len(result['cells']), 256)
        self.assertFalse(result['simultaneous_256_calibration'])
        self.assertFalse(result['hardware_mnist_200_qualified'])
        old = [r for r in result['cells'] if not r['reprogrammed_this_session']]
        self.assertEqual(len(old), 16)
        self.assertTrue(all(not r['fresh_readonly_recheck']['current_programmable_range_revalidated'] for r in old))

    def test_remapped_pairs_are_not_total_count_or_common_layout(self):
        uneven = {(w, b) for w in range(7) for b in range(32)}
        self.assertGreaterEqual(len(uneven), 200)
        self.assertIsNone(mapping_candidates(uneven)['four_disjoint_pairs_with_own25bl'])
        remap = {(w, (b + (w // 2)*2) % 32) for w in range(8) for b in range(25)}
        result = mapping_candidates(remap)
        self.assertIsNone(result['common25_candidate'])
        self.assertEqual(len(result['four_disjoint_pairs_with_own25bl']), 4)
        self.assertFalse(result['signed_mvm_qualified'])

    def test_hardware_smoke_rejected_and_default16_preserved(self):
        with self.assertRaises(ValueError):
            experiment.run('hardware', True, False)
        self.assertEqual(Schedule().calls_by_phase, dict(A=17664, B=6656))
        smoke, spec = experiment.protocol(True)
        self.assertEqual(sum(smoke.calls_by_phase.values()), 4496)
        self.assertFalse(spec['retention_qualification_performed'])


class PublicationTests(unittest.TestCase):
    def test_partial_map_is_gray_not_negative_red(self):
        page = render_map([dict(wl=0, bl=0, hard_eligible=False, eligible=False,
                                hard_reasons=['missing_required_state_checkpoint'])], {})
        self.assertIn('неполные данные</td>', page)
        self.assertNotIn('background:#f2b3b3', page)

    def test_existing_staged_is_never_modified(self):
        with tempfile.TemporaryDirectory(dir='.tmp') as temp:
            root = Path(temp).resolve()
            with patch.object(publication, 'git', side_effect=[str(root), 'someone.txt']) as call:
                with self.assertRaises(ValueError):
                    publication.preflight(root, root / 'logs/publish.log')
            self.assertEqual(call.call_count, 2)

    def test_exact_artifacts_failed_session_and_commit_failure_no_push(self):
        with tempfile.TemporaryDirectory(dir='.tmp') as temp:
            root = Path(temp).resolve()
            session = root / 'results/route/session'
            session.mkdir(parents=True)
            atomic_json(session / 'manifest.json', dict(status='failed', finished_utc='done', source='hardware'))
            atomic_json(session / 'summary.json', dict(status='failed'))
            (session / 'map.html').write_text('<p>partial</p>', encoding='utf-8')
            destination = session.parent / 'archives'
            destination.mkdir()
            archive = destination / 'session.tar.gz'
            archive.write_bytes(b'already verified archive fixture')
            atomic_json(destination / 'session.json', dict(archive_sha256=sha(archive), byte_identical_archive_verified=True,
                        files={p.name: dict(sha256=sha(p)) for p in session.iterdir()}))
            calls = []

            def fake_git(_, arguments, log):
                calls.append(arguments)
                if arguments[:2] == ['check-attr', 'filter']:
                    return arguments[-1] + ': filter: lfs'
                if arguments[0] == 'add':
                    self.assertEqual(len(arguments[2:]), 4)
                    self.assertTrue(all(name.startswith('results/route/archives/') or name.startswith('results/route/summaries/') for name in arguments[2:]))
                if arguments[:2] == ['diff', '--cached']:
                    return 'results/route/archives/session.tar.gz'
                if arguments[0] == 'commit':
                    raise RuntimeError('synthetic hook failure')
                return ''

            with patch.object(publication, 'preflight', return_value='head'), patch.object(publication, 'git', side_effect=fake_git):
                with self.assertRaises(RuntimeError):
                    publication.publish(root, session, log=root / 'local.log')
            self.assertFalse(any(c[0] == 'push' for c in calls))
            self.assertEqual(json.loads((session / 'manifest.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
