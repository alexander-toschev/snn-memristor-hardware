from __future__ import annotations
import copy
from dataclasses import replace
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast
import unittest

from experiment_common.durable import Writer
from experiment_common.block_readout_validation import BLOCKS, MAX_CALLS, Collector, patterns, summarize
from tests.test_mvm_fixed_mask_diagnostic import Converters
if TYPE_CHECKING:
    from manager.board import Connector


class Sink:
    def __init__(self):self.freeze_count=0;self.validation_before_freeze=False
    def write(self,event):
        if event['event']=='basis_frozen':self.freeze_count+=1
        if event['event']=='request' and event['metadata']['phase']=='validation' and not self.freeze_count:self.validation_before_freeze=True
    def checkpoint(self,**fields):pass


class Native:
    def __init__(self,resistance=100000.):self.calls=[];self.resistance=resistance
    def impact(self,ticket):
        self.calls.append(copy.deepcopy(ticket))
        if ticket['mode_flag']==7:return self.resistance,ticket['id'],100
        # Каждый бинарный уровень аддитивен; между уровнями намеренно нелинейно.
        # Ноги одинаковы: нулевой signed вес обязан остаться допустимым.
        adc=sum({0.:0,.025:20,.05:41,.075:100}[v] for v in ticket['vol'])
        ticket['vol'][:]=[Converters.convert_volt_to_dac(12,5,v) for v in ticket['vol']]
        return 100000.,ticket['id'],adc


class BlockTests(unittest.TestCase):
    def test_global_bias_guards_all32_and_preserves_background(self):
        n=Native();c=Collector(cast('Connector',n),Converters(),cast(Writer,Sink()),'simulator')
        leg=c.measure_leg(BLOCKS[0],0,'basis',0,'fixed_positive',.05,(4,),global_bias=True)
        self.assertEqual(len(leg.guards),32)
        self.assertEqual({t['bl'] for t in n.calls[:32]},set(range(32)))
        self.assertTrue(all(t['mode_flag']==7 and t['vol']==0. for t in n.calls[:32]))
        signals=n.calls[36:40]
        self.assertTrue(all(t['vol'][4]==.075 and all(v==.025 for b,v in enumerate(t['vol']) if b!=4) for t in signals))
        self.assertAlmostEqual(leg.centered('mean4'),80.)

    @classmethod
    def setUpClass(cls):
        cls.native=Native();cls.sink=Sink()
        cls.collector=Collector(cast('Connector',cls.native),Converters(),cast(Writer,cls.sink),'simulator')
        cls.collector.diagnose(lambda _:None)
        cls.report=summarize(cls.collector.pairs)

    def test_schedule_addresses_four_samples_and_freeze(self):
        self.assertEqual(self.collector.calls,MAX_CALLS)
        self.assertEqual(len(self.collector.pairs),2352)
        self.assertEqual(sum(len(patterns(b)) for b in BLOCKS),137)
        for block in BLOCKS:
            self.assertEqual(len(set(patterns(block))),2**len(block.bls)-1)
        self.assertEqual(self.sink.freeze_count,1)
        self.assertFalse(self.sink.validation_before_freeze)
        coords=[(wl,bl) for b in BLOCKS for wl in (b.plus_wl,b.minus_wl) for bl in b.bls]
        self.assertEqual((len(coords),len(set(coords))),(66,66))
        parent=json.loads(Path('reports/2026-10-22/mvm-address-diagnostic/BLOCK_SELECTION.json').read_text(encoding='utf-8'))['primary_candidate']['blocks']
        self.assertEqual([(b.plus_wl,b.minus_wl,list(b.bls)) for b in BLOCKS],[(b['wl_pair'][0],b['wl_pair'][1],b['bls']) for b in parent])
        for pair in self.collector.pairs:
            for leg in (pair.plus,pair.minus):
                self.assertEqual([len(leg.before),len(leg.signal),len(leg.after)],[4,4,4])
                self.assertLess(max(g['task_id'] for g in leg.guards),leg.before[0]['task_id'])
        self.assertTrue(all(t['mode_flag'] in (7,10) for t in self.native.calls))
        self.assertTrue(any(p.leg_order[0]==p.minus.wl for p in self.collector.pairs))

    def test_zero_signed_is_valid_and_cross_amplitude_not_binary_gate(self):
        primary=self.report['primary']
        self.assertEqual(len(primary),45)
        self.assertTrue(all(r['status']=='stable_current_state_readout' for r in primary))
        self.assertTrue(all(v['signed']['zero_or_unresolved'] for r in primary for v in r['validation']))
        self.assertTrue(any(not v['passed'] for r in primary for v in r['multi_amplitude_linearity_diagnostic']))
        self.assertEqual((len(self.report['selected_blocks']),self.report['selected_cells']),(15,66))
        self.assertTrue(all(r['profile']=='ordinary_delta05' for r in self.report['selected_blocks']))

    def test_bad_validation_does_not_refit_or_remove_inputs(self):
        changed=[]
        for p in self.collector.pairs:
            if p.block=='B01' and p.phase=='validation':
                samples=tuple({**r,'adc':r['adc']+50} for r in p.plus.signal)
                p=replace(p,plus=replace(p.plus,signal=samples))
            changed.append(p)
        report=summarize(changed)
        before=[r for r in self.report['primary'] if r['block']=='B01']
        after=[r for r in report['primary'] if r['block']=='B01']
        self.assertEqual([r['basis'] for r in before],[r['basis'] for r in after])
        self.assertTrue(all(r['status']!='stable_current_state_readout' for r in after))
        self.assertFalse(any(r['block']=='B01' for r in report['selected_blocks']))
        self.assertTrue(all(r['bls']==list(BLOCKS[0].bls) for r in after))

    def test_guard_stops_before_mvm_and_commands_are_read_only(self):
        n=Native(5.);c=Collector(cast('Connector',n),Converters(),cast(Writer,Sink()),'simulator')
        with self.assertRaises(ValueError):c.diagnose(lambda _:None)
        self.assertTrue(all(t['mode_flag']==7 for t in n.calls))
        with self.assertRaises(ValueError):c.call(dict(mode_flag=7,wl=0,bl=2,vol=1.4,sign=0,pulse_width=0.))
        with self.assertRaises(ValueError):c.call(dict(mode_flag=8,wl=0,vol=.05))

if __name__=='__main__':unittest.main()
