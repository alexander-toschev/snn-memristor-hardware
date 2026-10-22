from __future__ import annotations
from typing import TYPE_CHECKING, cast
import unittest

from experiment_common.durable import Writer
from experiment_common.mvm_voltage_diagnostic import Collector, CELLS, VOLTAGES, summarize
from tests.test_selective_metrology import Converters
from tests.test_selected_programming_validation import Sink

if TYPE_CHECKING:
    from manager.board import Connector


class Native:
    def __init__(self, resistance=100000.):
        self.tasks=[]
        self.resistance=resistance

    def impact(self, task):
        self.tasks.append(task)
        if task['mode_flag']==7:
            assert task['vol']==0 and task['pulse_width']==0
            return self.resistance,task['id'],100
        if task['mode_flag']==9:
            return 100000.,0,round(task['vol']*2000)
        voltage=max(task['vol'])
        active=[b for b,v in enumerate(task['vol']) if v]
        return 100000.,0,0 if active and active[0]>=16 and voltage<=.05 else round(voltage*1000)


class SweepTests(unittest.TestCase):
    def test_complete_exact_scope_and_distinct_voltage_paths(self):
        fake=Native()
        c=Collector(cast('Connector',fake),Converters(),cast(Writer,Sink()),'simulator')
        c.diagnose(lambda _:None)
        self.assertEqual(c.calls,1344)
        self.assertEqual(len(c.checkpoints),336)
        self.assertEqual(len(summarize(c.checkpoints)),56)
        for wl,bl in CELLS:
            for v in VOLTAGES:
                for mode in (9,10):
                    rows=[r for r in c.checkpoints if (r['wl'],r['bl'],r['voltage_v'],r['diagnostic_mode'])==(wl,bl,v,mode)]
                    self.assertEqual(len(rows),3)
                    self.assertTrue(all(abs(r['guard_current_a']-1e-6)<1e-12 for r in rows))
        upper=[r for r in c.checkpoints if r['bl']==20 and r['diagnostic_mode']==10]
        self.assertTrue(all(r['centered_adc']==0 for r in upper if r['voltage_v']<=.05))
        self.assertTrue(all(r['centered_adc']>0 for r in upper if r['voltage_v']>.05))
        with self.assertRaises(TimeoutError): c.call(9,0,20,.05)

    def test_guard_stops_before_signal_and_writes_rejected(self):
        fake=Native(resistance=10.)
        c=Collector(cast('Connector',fake),Converters(),cast(Writer,Sink()),'simulator')
        with self.assertRaises(ValueError): c.diagnose(lambda _:None)
        self.assertEqual(len(fake.tasks),2)
        self.assertTrue(all(t['vol']==0 for t in fake.tasks))
        with self.assertRaises(ValueError): c.call(7,0,20,.1)
        with self.assertRaises(ValueError): c.call(9,0,20,.3)
        self.assertEqual(len(fake.tasks),2)


if __name__=='__main__': unittest.main()
