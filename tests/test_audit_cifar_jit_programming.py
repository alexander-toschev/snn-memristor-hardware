"""Synthetic acquisition fixtures test the independent auditor; never hardware."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_cifar_jit_programming as a
# Collector is used solely to GENERATE explicitly fake fixtures, not as audit oracle.
import cifar_jit_programming as collector
from tests.test_cifar_feedback import FakeConnector,converters
from tests.test_cifar_jit_programming import Clock


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.tmp');cls.base=Path(cls.temp.name)
        cls.plan,cls.route=collector.load_inputs()
        for fail in (False,True):
            folder=cls.base/str(fail);folder.mkdir();p=tuple(cls.route[0]['positive_cell']);n=tuple(cls.route[0]['negative_cell'])
            class Disturb(FakeConnector):
                def impact(self,task):
                    disturb=fail and task['mode_flag']==7 and task['vol'] and task['sign']==0 and (task['wl'],task['bl'])==n
                    result=super().impact(task)
                    if disturb:self.states[p]=10
                    return result
            fake=Disturb(reset_step=100,set_step=1)
            if fail:fake.states[n]=200
            clock=Clock()
            runner=collector.JITRun(fake,converters,folder,'OFFLINE_FAKE',cls.plan,'simulator',clock=clock.now,monotonic=clock.mono,sleep=clock.sleep)
            runner.collect(cls.route)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def replay(self,folder=None,route=None,plan=None):
        ck=a.CompactChecks()
        result=a.replay_session(folder or self.base/'False','simulator','OFFLINE_FAKE',plan or self.plan,route or self.route,ck)
        return result,ck

    def test_independent_two_cycle_replay(self):
        result,ck=self.replay();self.assertEqual(ck.errors,[])
        self.assertEqual(result['cycles'],2);self.assertEqual(result['physical_cells'],18)
        self.assertEqual(result['programmed_pair_decisions'],18)

    def test_gate_failed_capture_is_complete_but_not_programming_success(self):
        result,ck=self.replay(self.base/'True');self.assertEqual(ck.errors,[])
        self.assertGreater(result['quiet_gate_failures'],0);self.assertLess(result['programmed_pair_decisions'],18)

    def mutation(self,name,modify):
        original=a.ledger
        def changed(path):
            rows=original(path)
            if path.name==name:modify(rows)
            return rows
        with patch.object(a,'ledger',side_effect=changed):
            result,ck=self.replay()
        self.assertTrue(ck.errors)

    def test_forged_gate_rejected(self):
        self.mutation('paired_snapshots.jsonl',lambda r:r[0].update(both_legs_quiet=False))

    def test_removed_sentinel_rejected(self):self.mutation('sentinel.jsonl',lambda r:r.pop())

    def test_forged_target_rejected(self):self.mutation('pair_programming.jsonl',lambda r:r[0].update(target_signed_v=999))

    def test_deleted_decision_rejected(self):self.mutation('decisions.jsonl',lambda r:r.pop())

    def test_false_retention_quality_rejected(self):
        self.mutation('disturbance_maps.jsonl',lambda r:r[0]['physical_fidelity'].update(raw_target_nrmse=999))

    def test_moved_weight_route_rejected(self):
        route=copy.deepcopy(self.route);route[0]['trained_weight']+=.01
        try:result,ck=self.replay(route=route)
        except (ValueError,IndexError,KeyError):return
        self.assertTrue(ck.errors)

    def test_changed_budget_rejected(self):
        plan=copy.deepcopy(self.plan);plan['maximum_impacts']+=1
        result,ck=self.replay(plan=plan);self.assertTrue(ck.errors)

    def raw_mutation(self,modify):
        original=a.raw_lines
        def changed(path):
            rows=[json.loads(line) for line in original(path)];modify(rows)
            return iter(json.dumps(r)+'\n' for r in rows)
        with patch.object(a,'raw_lines',side_effect=changed):
            try:result,ck=self.replay()
            except (ValueError,IndexError,KeyError):return
        self.assertTrue(ck.errors)

    def test_short_retention_rejected(self):
        self.raw_mutation(lambda r:next(x for x in r if x['event']=='retention_wait').update(elapsed_monotonic_s=.5))

    def test_changed_current_prediction_rejected(self):
        def modify(rows):
            for r in rows:
                if 'predicted_current_a' in r:r['predicted_current_a']=0.;break
        self.raw_mutation(modify)

    def test_raw_converter_tamper_rejected(self):
        self.raw_mutation(lambda r:next(x for x in r if x['event']=='result' and x['command_si']['mode_flag']==10).update(output_voltage_v=7.))

    def test_deadline_marker_never_certified_complete(self):
        self.raw_mutation(lambda r:r.append(dict(event='deadline_stop')))

    def test_unknown_auxiliary_event_rejected(self):
        self.raw_mutation(lambda r:r.append(dict(event='convenient_ignored_command')))

    def test_protocol_reserialization_allowed_but_values_and_original_bytes_pinned(self):
        blob=(Path(__file__).resolve().parents[1]/'configs/cifar_jit_programming.json').read_bytes()
        plan=json.loads(json.dumps(json.loads(blob),ensure_ascii=False,sort_keys=True))
        ck=a.CompactChecks();a.verify_protocol_copy(plan,blob,ck);self.assertEqual(ck.errors,[])
        plan['cycles']=3;ck=a.CompactChecks();a.verify_protocol_copy(plan,blob,ck);self.assertTrue(ck.errors)
        ck=a.CompactChecks();a.verify_protocol_copy(json.loads(blob),blob+b' ',ck);self.assertTrue(ck.errors)

    def test_hardware_private_pins_required_but_SIM_hardware_only_omitted(self):
        hw=a.private_requirements('hardware');sim=a.private_requirements('simulator')
        self.assertEqual(set(hw)-set(sim),{
            'vendor/MemriBoard/MemriCORE/rp5_fpga_python/rpi_FPGAed.py',
            'vendor/MemriBoard/MemriCORE/rp5_fpga_python/FPGA_SIM.py'})
        self.assertEqual(hw,a.PRIVATE)


if __name__=='__main__':unittest.main()
