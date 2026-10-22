from __future__ import annotations
from typing import TYPE_CHECKING, cast
import unittest
from experiment_common.durable import Writer
from experiment_common.mvm_group_diagnostic import Collector, MASKS, summarize
from tests.test_selective_metrology import Converters
from tests.test_selected_programming_validation import Sink
if TYPE_CHECKING:
    from manager.board import Connector

class Native:
    def __init__(self, resistance=100000.):
        self.tasks=[]
        self.resistance=resistance
    def impact(self,t):
        self.tasks.append(t)
        if t['mode_flag']==7:
            assert t['vol']==0 and t['pulse_width']==0
            return self.resistance,t['id'],100
        active=[b for b,v in enumerate(t['vol']) if v]
        return 100000.,0,sum(10 for b in active if b<16 or b==31)

class GroupTests(unittest.TestCase):
    def test_full_vectors_and_measured_increment(self):
        native=Native();c=Collector(cast('Connector',native),Converters(),cast(Writer,Sink()),'simulator')
        c.diagnose(lambda _:None)
        self.assertEqual(c.calls,648)
        self.assertEqual(len(c.checkpoints),54)
        signals=[t for t in native.tasks if t['mode_flag']==10 and any(t['vol'])]
        self.assertEqual(len(signals),54)
        for name,bls in MASKS:
            self.assertEqual(sum(tuple(b for b,v in enumerate(t['vol']) if v)==bls for t in signals),6)
        mixed=[r for r in summarize(c.checkpoints) if r['mask']=='measured_mixed_minus_BL14']
        self.assertEqual([r['mean_adc'] for r in mixed],[0.,0.])
        self.assertTrue(any(r['mask']=='upper16_31' and r['centered_adc']==10 for r in c.checkpoints))
        with self.assertRaises(TimeoutError):c.call(10,0,())
    def test_sum_guard_stops_group_even_when_single_would_pass(self):
        native=Native(resistance=50.);c=Collector(cast('Connector',native),Converters(),cast(Writer,Sink()),'simulator')
        with self.assertRaises(ValueError):c.diagnose(lambda _:None)
        self.assertEqual(c.calls,9)
        self.assertFalse(any(t['mode_flag']==10 and any(t['vol']) for t in native.tasks))
        with self.assertRaises(ValueError):c.call(8,0,(14,))
        with self.assertRaises(ValueError):c.call(10,0,(16,17,20))

if __name__=='__main__':unittest.main()
