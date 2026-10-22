from __future__ import annotations

from typing import TYPE_CHECKING, cast
import unittest

from experiment_common.durable import Writer
from experiment_common.mvm_address_diagnostic import Collector, MODE9_CELLS, summarize
from tests.test_selective_metrology import Converters
from tests.test_selected_programming_validation import Sink

if TYPE_CHECKING:
    from manager.board import Connector


class Fake:
    def __init__(self):
        self.tasks = []

    def impact(self, task):
        self.tasks.append({**task, 'vol': list(task['vol']) if isinstance(task['vol'], list) else task['vol']})
        if task['mode_flag'] == 7:
            assert task['vol'] == 0 and task['pulse_width'] == 0
            return 100000., task['id'], 100
        if task['mode_flag'] == 9:
            return 100000., 0, 7 if task['vol'] else 0
        active = [b for b, v in enumerate(task['vol']) if v]
        assert len(active) <= 1
        adc = 0 if not active or 16 <= active[0] <= 30 else 7
        # Connector действительно может менять список. SI журнал не должен.
        task['vol'][:] = [999] * 32
        return 100000., 0, adc


class Recording(Sink):
    def __init__(self):
        self.rows = []

    def write(self, value):
        self.rows.append(value)


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake, cls.writer = Fake(), Recording()
        cls.c = Collector(cast('Connector', cls.fake), Converters(), cast(Writer, cls.writer), 'simulator')
        cls.rows = cls.c.diagnose(lambda _: None)

    def test_all_addresses_orders_and_readonly_cap(self):
        self.assertEqual(self.c.calls, 3240)
        self.assertEqual(len(self.rows), 810)
        mode10 = [t for t in self.fake.tasks if t['mode_flag'] == 10 and any(t['vol'])]
        self.assertEqual(len(mode10), 768)
        for w in range(8):
            for b in range(32):
                self.assertEqual(sum(t['wl'] == w and t['vol'][b] == .05 for t in mode10), 3)
        self.assertTrue(all(t['id'] == 0 for t in self.fake.tasks if t['mode_flag'] in (9,10)))
        mode9 = [t for t in self.fake.tasks if t['mode_flag'] == 9 and t['vol']]
        self.assertEqual({(t['wl'],t['bl']) for t in mode9}, set(MODE9_CELLS))
        with self.assertRaises(ValueError):
            self.c.call(dict(mode_flag=7,wl=0,bl=0,vol=0.,sign=0,pulse_width=0.))

    def test_different_read_path_not_zero_filled_and_si_preserved(self):
        c10 = {(c['wl'],c['bl']):c for c in summarize(self.rows)}
        c9 = {(c['wl'],c['bl']):c for c in summarize(self.rows,9)}
        self.assertTrue(c10[0,20]['all_mvm_zero'])
        self.assertFalse(c9[0,20]['all_mvm_zero'])
        self.assertEqual(c10[0,31]['mean_adc'],7)
        requests = [e['command_si'] for e in self.writer.rows if e['event'] == 'request']
        self.assertTrue(all(all(v in (0.,.05) for v in t['vol']) for t in requests if t['mode_flag']==10))

    def test_writes_and_unplanned_mode9_rejected_before_impact(self):
        fake = Fake()
        c = Collector(cast('Connector',fake),Converters(),cast(Writer,Sink()),'simulator')
        for task in (dict(mode_flag=7,wl=0,bl=14,vol=.3,sign=0,pulse_width=0.),
                     dict(mode_flag=9,wl=0,bl=20,vol=.3),
                     dict(mode_flag=9,wl=1,bl=20,vol=.05)):
            with self.assertRaises(ValueError):
                c.call(task)
        self.assertEqual(fake.tasks,[])


if __name__ == '__main__':
    unittest.main()
