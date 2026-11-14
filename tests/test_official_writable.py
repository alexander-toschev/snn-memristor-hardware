"""Pure fake-Connector tests: boundaries, waveform size and guard accounting."""
from __future__ import annotations
from pathlib import Path
import unittest
from typing import cast
from experiment_common.selective_metrology import number, object_dict

from experiment_common.runtime import use_local_dependencies
use_local_dependencies(prefer_environment=True)
from manager.service import converters
from experiment_common.adaptive_programming import AcquisitionBudget, Collector
from experiment_common.official_writable import characterize, load_tickets, waveform

ROOT = Path(__file__).resolve().parents[1]
TICKETS = load_tickets(ROOT / 'experiments/official_writable_cells_20261114/writable.json')


class FakeCollector(Collector):
    def __init__(self, guard_r: float = 1000.) -> None:
        self.cv = converters
        self.budget = AcquisitionBudget()
        self.decisions = self.pulses = self.calls = 0
        self.guard_r = guard_r
        self.commands: list[dict[str, int | float | list[float]]] = []

    def read_r(self, wl: int, bl: int, **metadata: object) -> float:
        return self.guard_r

    def emit(self, event: str, **fields: object) -> None:
        pass

    def call(self, command: dict[str, int | float | list[float]], **metadata: object) -> tuple[int, float, int]:
        self.calls += 1
        self.commands.append(command)
        resistance = 296. if metadata['stage'] == 'SET' else 2998.
        return self.calls, resistance, 5400


class OfficialTests(unittest.TestCase):
    def test_native_waveform_counts_polarity_and_width(self) -> None:
        c = FakeCollector()
        expected = [('0', 140, 1, .0001), ('1', 180, 0, .001), ('2', 300, 1, .00013)]
        for key, count, first_sign, width in expected:
            commands = waveform(c, object_dict(TICKETS[key]), 0, 8)
            self.assertEqual(len(commands), count)
            self.assertEqual(commands[0]['sign'], first_sign)
            self.assertAlmostEqual(number(commands[0]['pulse_width']), width)
            self.assertTrue(all(cmd['wl'] == 0 and cmd['bl'] == 8 for cmd in commands))

    def test_docx_equality_stops_and_zero_is_not_write(self) -> None:
        c = FakeCollector()
        row = characterize(c, 0, 8, TICKETS)
        stages = cast(list[dict[str, object]], row['stages'])
        self.assertEqual([s['submitted_tasks'] for s in stages], [1, 1, 300])
        self.assertTrue(stages[0]['stopped_by_docx_resistance'])
        self.assertTrue(stages[1]['stopped_by_docx_resistance'])
        self.assertEqual(c.pulses, 288)
        self.assertTrue(row['observed_docx_switching'])
        self.assertFalse(row['qualified_for_mnist'])

    def test_guard_samples_do_not_contaminate_extrema(self) -> None:
        c = FakeCollector(guard_r=50.)
        row = characterize(c, 0, 8, TICKETS)
        self.assertGreater(number(row['skipped_current_guard']), 0)
        self.assertTrue(row['guard_modified_protocol'])
        self.assertEqual(row['min_r_ohm'], 2998.)
        self.assertEqual(row['max_r_ohm'], 2998.)
        self.assertFalse(row['observed_docx_switching'])


if __name__ == '__main__':
    unittest.main()
