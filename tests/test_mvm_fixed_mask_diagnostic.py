from __future__ import annotations
import copy
from typing import TYPE_CHECKING, cast
import unittest
from experiment_common.durable import Writer
from experiment_common.mvm_fixed_mask_diagnostic import Collector, CONTEXTS, summarize
from tests.test_selected_programming_validation import Sink
if TYPE_CHECKING:
    from manager.board import Connector

class Converters:
    @staticmethod
    def convert_volt_to_dac(bits, ref, voltage):
        return round(voltage*(2**bits-1)/ref)
    @staticmethod
    def convert_dac_to_volt(bits, ref, code):
        return code*ref/(2**bits-1)
    @staticmethod
    def convert_adc_to_volt(sign, bits, ref, code):
        return code*ref/(2**bits-1)

class Native:
    def __init__(self, resistance=100000.):
        self.tasks=[]
        self.resistance=resistance
    def impact(self, task):
        self.tasks.append(copy.deepcopy(task))
        if task['mode_flag']==7:
            return self.resistance,task['id'],100
        # Аддитивная модель только теста: реальный collector всегда читает Connector.
        codes=[Converters.convert_volt_to_dac(12,5,v) for v in task['vol']]
        task['vol'][:]=codes  # Штатная in-place SI→DAC мутация не портит evidence.
        return 100000.,task['id'],sum(codes)

class FixedMaskTests(unittest.TestCase):
    def collector(self, native):
        return Collector(cast('Connector',native),Converters(),cast(Writer,Sink()),'simulator')

    def test_complete_schedule_constant_masks_and_quantized_delta(self):
        native=Native(); c=self.collector(native); c.diagnose(lambda _:None)
        self.assertEqual((c.calls,len(c.checkpoints)),(4290,156))
        self.assertEqual(sum(t['mode_flag']==7 for t in native.tasks),3822)
        for row in c.checkpoints:
            tasks=[native.tasks[row[k]['task_id']-1] for k in ('baseline_before','signal','baseline_after')]
            self.assertTrue(all(tuple(b for b,v in enumerate(t['vol']) if v)==tuple(row['active_bls']) for t in tasks))
            self.assertIn(tuple(row['active_bls']),[active for _,active in CONTEXTS])
            self.assertEqual(tasks[0]['vol'],tasks[2]['vol'])
        reports=summarize(c.checkpoints)
        for report in reports:
            self.assertTrue(report['baseline_drift_screen_pass'])
            self.assertTrue(all(v['mean_residual_adc']==0 and v['screen_pass'] for v in report['heldout']))
            self.assertTrue(all(v['actual_dac_increment_ratio']==41/21 and v['error_adc']==0 for v in report['scaled_delta_comparison']))
        original=copy.deepcopy(reports[0]['singleton'])
        for row in c.checkpoints:
            if row['phase']=='heldout':row['delta_adc']+=17
        altered=summarize(c.checkpoints)
        self.assertEqual(altered[0]['singleton'],original)
        self.assertEqual(altered[0]['heldout'][0]['mean_residual_adc'],17)

    def test_guard_and_invalid_commands_fail_before_mvm(self):
        native=Native(50.); c=self.collector(native)
        with self.assertRaises(ValueError):c.diagnose(lambda _:None)
        self.assertEqual(c.calls,17)
        self.assertTrue(all(t['mode_flag']==7 for t in native.tasks))
        for command in [dict(mode_flag=8,wl=0,vol=.05),dict(mode_flag=10,wl=0,vol=[0.]*32),dict(mode_flag=7,wl=0,bl=16,vol=1.4,sign=0,pulse_width=0.)]:
            with self.assertRaises(ValueError):c.call(command)

if __name__=='__main__':unittest.main()
