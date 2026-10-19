"""Independent offline evidence replay, using a fake Connector and real gzip.

This is software validation, not native simulator qualification or hardware data.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.conductance_stdp_20261019 import core, raw
from tests.test_conductance_stdp import Clock, Connector, Converters

ROOT = Path(__file__).resolve().parents[1]


class EvidenceReplayTests(unittest.TestCase):
    def test_real_gzip_complete_fake_run_replays_adc_lif_and_stdp(self):
        clock = Clock()
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as temporary:
            folder = Path(temporary) / 'raw'
            writer = raw.Writer(folder)
            runner = core.Collector(Connector(), Converters, writer, 'simulator', clock.sleep, clock.now)
            with mock.patch('builtins.print'):
                summary = runner.run()
            packed = (folder / 'raw.jsonl.gz').read_bytes()
            expanded = gzip.decompress(packed)
            index = json.loads((folder / 'index.json').read_text())
            self.assertEqual(index['compressed_sha256'], hashlib.sha256(packed).hexdigest())
            self.assertEqual(index['expanded_sha256'], hashlib.sha256(expanded).hexdigest())
            self.assertEqual(index['compressed_bytes'], len(packed))
            self.assertEqual(index['expanded_bytes'], len(expanded))
            rows = [json.loads(line) for line in expanded.splitlines()]
            self.assertEqual(index['lines'], len(rows))

        requests = {r['task_id']: r for r in rows if r['event'] == 'request'}
        responses = {r['task_id']: r for r in rows if r['event'] == 'result'}
        self.assertEqual(set(requests), set(range(1, summary['impacts'] + 1)))
        self.assertEqual(set(requests), set(responses))
        self.assertEqual(sum(r['command_si']['mode_flag'] == 10 for r in requests.values()), 4776)

        def voltage(task):
            request, response = requests[task], responses[task]
            self.assertEqual(request['command_si'], response['command_si'])
            self.assertEqual(request['command_si']['mode_flag'], 10)
            adc = response['native_result'][2]
            return round(adc * 5 / 16383, 5)

        states = {}
        lif_count = 0
        for row in rows:
            if row['event'] != 'lif_step':
                continue
            lif_count += 1
            phase, window, wl, t = (row[k] for k in ('window_phase', 'window_index', 'wl', 't'))
            key = phase, window, wl
            u, a, b, e, next_t = states.get(key, (0., [0.] * 4, 0., [0.] * 4, 0))
            self.assertEqual(t, next_t)
            timing = ((2, 3, 10, 11) if window % 2 == 0 else (10, 11, 2, 3)) if phase == 'train' else (
                (1, 4, 9, 12) if window % 2 == 0 else (9, 12, 1, 4))
            pre = [int(t == value) for value in timing]
            measured = row['measurement']
            v = voltage(measured['signal']['task_id']) - (
                voltage(measured['before']['task_id']) + voltage(measured['after']['task_id'])) / 2
            self.assertEqual(row['pre'], pre)
            self.assertAlmostEqual(row['voltage_v'], v, places=14)
            unreset = u / 2 + v * 100
            post = int(unreset >= 1)
            after = 0. if post else unreset
            a = [x * math.exp(-1 / 4) for x in a]
            b *= math.exp(-1 / 4)
            e = [old + pa * post - 1.05 * b * pi for old, pa, pi in zip(e, a, pre)]
            a = [pa + pi for pa, pi in zip(a, pre)]
            b += post
            self.assertEqual(row['post'], post)
            for name, actual, expected in (('membrane', row['u_after'], after), ('post_trace', row['post_trace'], b)):
                with self.subTest(name=name, key=key, t=t):
                    self.assertAlmostEqual(actual, expected, places=12)
            for actual, expected in zip(row['eligibility'], e):
                self.assertAlmostEqual(actual, expected, places=12)
            for actual, expected in zip(row['pre_trace'], a):
                self.assertAlmostEqual(actual, expected, places=12)
            states[key] = after, a, b, e, t + 1
        self.assertEqual(lif_count, 896)
        self.assertEqual(len(states), 56)
        self.assertTrue(all(value[-1] == 16 for value in states.values()))

        training = [r for r in rows if r['event'] == 'pulse_complete' and r['phase'] == 'training_update']
        expected_updates = {(window, bl): e for (phase, window, wl), state in states.items()
                            if phase == 'train' and wl == 5
                            for bl, e in zip((6, 8, 9, 14), state[3]) if abs(e) >= .25}
        self.assertEqual({(r['window_index'], r['bl']) for r in training}, set(expected_updates))
        self.assertEqual(len(training), len(expected_updates))
        for update in training:
            expected = expected_updates[update['window_index'], update['bl']]
            self.assertEqual(update['sign'], int(expected >= .25))
            sham, guard = update['sham']['task_id'], update['guard']['task_id']
            self.assertEqual(guard, sham + 1)
            self.assertEqual(requests[sham]['command_si']['wl'], 7)
            self.assertEqual(requests[guard]['command_si']['wl'], 5)
            self.assertEqual(requests[sham]['command_si']['vol'], 0.)
            self.assertEqual(update['wl'], 5)
        self.assertEqual(rows[-1]['event'], 'session_end')
        self.assertEqual(rows[-1]['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
