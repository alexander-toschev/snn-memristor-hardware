from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from typing import cast, TYPE_CHECKING

from experiment_common.durable import Writer
from experiment_common.selective_metrology import Collector, load_sample, summarize
from experiment_common import runtime

if TYPE_CHECKING:
    from manager.board import Connector

ROOT = Path(__file__).resolve().parents[1]


class MemoryWriter:
    def __init__(self):
        self.events = []

    def write(self, value):
        self.events.append(copy.deepcopy(value))

    def checkpoint(self, **fields):
        pass


class Clock:
    def __init__(self):
        self.value = 0.

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FakeConnector:
    def __init__(self, resistance=100000.):
        self.commands = []
        self.resistance = resistance

    def impact(self, ticket):
        self.commands.append(copy.deepcopy(ticket))
        if ticket['mode_flag'] == 7:
            return self.resistance, ticket['id'], 1000
        active = any(ticket['vol'])
        # Реальный Connector преобразует vol-list in-place; SI raw не должен меняться.
        ticket['vol'][:] = [41 if v else 0 for v in ticket['vol']]
        return 1., ticket['id'], 100 + ticket['wl'] if active else 0


class Converters:
    @staticmethod
    def convert_volt_to_dac(bits, reference, voltage):
        return 41

    @staticmethod
    def convert_dac_to_volt(bits, reference, code):
        return .05

    @staticmethod
    def convert_adc_to_volt(gain, bits, reference, code):
        return code * 5 / 16383


class SelectiveTests(unittest.TestCase):
    def setUp(self):
        self.data, self.cells = load_sample(ROOT / 'experiments/selective_map_validation_20261022/sample.json')

    def collector(self, resistance=100000.):
        native, writer, clock = FakeConnector(resistance), MemoryWriter(), Clock()
        runner = Collector(cast('Connector', native), Converters(), cast(Writer, writer),
                           'hardware', clock=clock.now, sleep=clock.sleep)
        return runner, native, writer, clock

    def test_fixed_sample_and_original_criteria(self):
        self.assertEqual(self.data['parent_session'], '4c8c70ca-f819-4c82-83c8-8fed7f9fe3b0')
        self.assertFalse(self.data['parent_qualified'])
        self.assertEqual({c.wl for c in self.cells}, set(range(8)))
        self.assertEqual(len(self.cells), 40)
        chosen = {(c.wl, c.bl) for c in self.cells}
        self.assertTrue({(w, b) for w in (0, 7) for b in (2, 3, 6, 10, 11, 12, 14, 15)} <= chosen)

    def test_environment_packages_precede_repo_fallback(self):
        environment_path = 'active-venv/site-packages'
        with patch.object(runtime.sys, 'path', [environment_path]):
            runtime.use_local_dependencies(prefer_environment=True)
            self.assertLess(runtime.sys.path.index(environment_path), runtime.sys.path.index(str(ROOT / '.deps')))
            self.assertEqual(runtime.sys.path[0], str(ROOT / 'vendor/MemriBoard'))

    def test_complete_fake_schedule_exact6400_no_writes_and_si_preserved(self):
        runner, native, writer, clock = self.collector()
        results = runner.run(self.cells, lambda _: None)
        self.assertEqual(runner.calls, 6400)
        self.assertEqual(len(results), 160)
        self.assertEqual(sum(t['mode_flag'] == 7 for t in native.commands), 1600)
        self.assertTrue(all(t['vol'] == 0 and t['pulse_width'] == 0 for t in native.commands if t['mode_flag'] == 7))
        self.assertEqual(clock.value, 60.)
        self.assertTrue(all(s['noise_pass'] and s['drift_pass'] and s['zero_shift_pass'] for s in results))
        requests = [e for e in writer.events if e['event'] == 'request']
        responses = [e for e in writer.events if e['event'] == 'result']
        self.assertEqual([r['command_si'] for r in requests], [r['command_si'] for r in responses])
        self.assertTrue(all(v in (0., .05) for r in requests if r['command_si']['mode_flag'] == 10 for v in r['command_si']['vol']))

    def test_writes_current_invalid_native_and_cap_fail_closed(self):
        runner, native, _, _ = self.collector()
        with self.assertRaises(ValueError):
            runner.call(dict(mode_flag=7, wl=0, bl=0, vol=1.4, sign=0, pulse_width=0.))
        self.assertFalse(native.commands)
        runner.calls = 6400
        with self.assertRaises(TimeoutError):
            runner.vector(0, None)
        for resistance in (0., 1.):
            runner, native, _, _ = self.collector(resistance)
            with self.assertRaises(ValueError):
                runner.checkpoint(self.cells[0], 'initial', None)
            self.assertFalse(any(t['mode_flag'] == 10 and any(t['vol']) for t in native.commands))

    def test_drift_and_noise_not_hidden_by_historical_grade(self):
        runner, _, _, _ = self.collector()
        runner.checkpoint(self.cells[0], 'initial', None)
        rows = runner.checkpoints[0]['brackets']
        baseline = summarize(rows, self.cells[0], None)
        changed = copy.deepcopy(rows)
        changed[0]['centered_adc'] += 10000
        result = summarize(changed, self.cells[0], baseline['mean_adc'])
        self.assertFalse(result['noise_pass'])
        self.assertFalse(result['drift_pass'])


if __name__ == '__main__':
    unittest.main()
