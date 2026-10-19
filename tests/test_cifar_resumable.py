"""Offline software evidence only. No native driver, model, or physical board."""
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import types
import unittest
from unittest import mock
import numpy as np
from experiments.cifar_resumable_20261019 import collector as c,durable,contract
from experiments.cifar_full_case0_20261019 import collector as original,trace,plan
from tools.classify_cifar_continuation import combine
from tests.test_full_case0_native import FakeConnector,Converters,MemoryTrace


class CheckpointMemory(MemoryTrace):
    def __init__(self):super().__init__();self.checkpoints=[]
    def checkpoint(self,**fields):self.checkpoints.append(copy.deepcopy(fields))


class RecoveryTests(unittest.TestCase):
    def test_durable_checkpoint_closed_crc_before_new_unindexed_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);writer=durable.Writer(root/'raw',minimum_free_bytes=0)
            writer.write({'event':'saved','id':1});writer.checkpoint(reason='test')
            index=(root/'raw/index.json').read_bytes()
            writer.write({'event':'unsaved','id':2});writer.stream.flush();writer.file.flush()
            self.assertEqual((root/'raw/index.json').read_bytes(),index)
            self.assertEqual(list(trace.records(root/'raw')),[{'event':'saved','id':1}])
            self.assertFalse(json.loads(index)['stream_finalized'])
            writer.finish()
            self.assertEqual(len(list(trace.records(root/'raw'))),2)
            self.assertTrue(json.loads((root/'raw/index.json').read_text())['stream_finalized'])

    def test_failed_checkpoint_replace_preserves_previous_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);writer=durable.Writer(root/'raw',minimum_free_bytes=0)
            writer.write({'id':1});writer.checkpoint(reason='one')
            previous=(root/'raw/index.json').read_bytes()
            writer.write({'id':2})
            with mock.patch.object(durable.os,'replace',side_effect=OSError('simulated power loss')):
                with self.assertRaises(OSError):writer.checkpoint(reason='two')
            self.assertEqual((root/'raw/index.json').read_bytes(),previous)
            self.assertEqual(list(trace.records(root/'raw')),[{'id':1}])
            writer.finish()

    def test_real_subprocess_loss_keeps16_durable_without_finalizing_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'raw'
            code="""import sys
from pathlib import Path
from experiments.cifar_resumable_20261019.durable import Writer,simulated_process_loss
w=Writer(Path(sys.argv[1]),minimum_free_bytes=0)
for x in range(16):w.write({'event':'position_complete','y':0,'x':x})
w.checkpoint(reason='position16')
w.write({'event':'position_complete','y':0,'x':16})
simulated_process_loss(w,'simulator',1,17,123)
"""
            result=subprocess.run([sys.executable,'-B','-c',code,str(path)],cwd=contract.ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,86,result.stderr)
            rows=list(trace.records(path));self.assertEqual(len(rows),16)
            self.assertFalse(json.loads((path/'index.json').read_text())['stream_finalized'])
            self.assertEqual(json.loads((path.parent/'fault_injection.json').read_text())['software_observed_positions'],17)
            self.assertTrue((path/'part-000001.jsonl.gz').exists())
            self.assertFalse((path.parent/'manifest.json').exists())
        with self.assertRaises(ValueError):durable.simulated_process_loss(None,'hardware',1,17,123)

    def test_all48_full_stage_mapping_and_no_fill(self):
        stages={i:{k:np.full((12,4,28,28),i+(100 if k=='calibrated' else 0),dtype='<f4').tobytes() for k in ('raw','calibrated')} for i in range(48)}
        result=combine(stages)
        for label,offset in (('raw',0),('calibrated',100)):
            values=np.frombuffer(result[label],dtype='<f4').reshape(12,64,28,28,3)
            for f in range(64):
                for tile in range(3):self.assertTrue(np.all(values[:,f,:,:,tile]==(f//4)*3+tile+offset))
        del stages[13]
        with self.assertRaises(ValueError):combine(stages)

    def test_reject_partial_nonfinite_or_wrong_payload(self):
        stages={i:{k:np.zeros((12,4,28,28),dtype='<f4').tobytes() for k in ('raw','calibrated')} for i in range(48)}
        stages[4]['raw']=b'bad'
        with self.assertRaises(ValueError):combine(stages)
        stages[4]['raw']=np.full((12,4,28,28),np.nan,dtype='<f4').tobytes()
        with self.assertRaises(ValueError):combine(stages)

    def test_stage_upper_bounds_sum_original_full_bound(self):
        image=plan.fixture()['image_chw_uint8']
        bounds=[c.stage_bounds(image,s) for s in range(48)]
        self.assertEqual(sum(v['maximum_impacts'] for v in bounds),14457648)
        self.assertEqual(sum(v['maximum_pulses'] for v in bounds),115200)

    def test_single_stage_clean_close_and_inherited_semantic_replay(self):
        from tools.cifar_prefix_replay import audit_range
        p=json.loads((Path(original.__file__).parent/'protocol.json').read_text())
        image=[0]*3072;weights=np.zeros((64,75),dtype=np.float32)
        connector=FakeConnector();writer=CheckpointMemory()
        with mock.patch.object(original,'load_assets',return_value=(image,weights)),mock.patch('builtins.print'):
            runner=c.Collector(connector,Converters,writer,p,original.digest(p),'simulator',stage=2,scope='smoke')
            result=runner.run_stage(image,weights)
        payload,report=audit_range(iter(writer.rows),image,weights,original.digest(p),first_stage=2,stop_stage=3)
        self.assertTrue(connector.closed and writer.finished)
        self.assertEqual(result['status'],'completed');self.assertEqual(set(payload),{2})
        self.assertEqual(report['accepted_stages'][0]['shape'],[12,4,1,1])
        self.assertFalse(report['complete_full_tensor'])

    def test_failed_open_close_and_qualification_limit(self):
        p=json.loads((Path(original.__file__).parent/'protocol.json').read_text());image=[0]*3072;weights=np.zeros((64,75),dtype=np.float32)
        for outcome in (None,False,0):
            connector=FakeConnector();connector.open_port=lambda:outcome;writer=CheckpointMemory()
            with mock.patch.object(original,'load_assets',return_value=(image,weights)):
                runner=c.Collector(connector,Converters,writer,p,original.digest(p),'simulator',stage=0)
                with self.assertRaises(ConnectionError):runner.run_stage(image,weights)
            self.assertTrue(connector.closed and writer.finished);self.assertEqual(connector.calls,[])
            self.assertEqual(writer.rows[-1]['status'],'failed')
        with self.assertRaises(ValueError):c.Collector(FakeConnector(),Converters,CheckpointMemory(),p,original.digest(p),'simulator',stage=0,interrupt_after_positions=17)

    def test_connector_closed_if_collector_construction_fails(self):
        from experiments.cifar_resumable_20261019 import experiment as entry
        import read_all
        import experiment_hardware
        p=json.loads((Path(original.__file__).parent/'protocol.json').read_text())
        connector=FakeConnector()
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);protocol_path=folder/'experiments/cifar_full_case0_20261019/protocol.json'
            protocol_path.parent.mkdir(parents=True);protocol_path.write_text(json.dumps(p))
            plan_path=folder/'plan.json';plan_path.write_text('{}')
            grant_path=folder/'grant.json';grant_path.write_text('{}')
            grant=dict(path=grant_path.relative_to(contract.ROOT).as_posix(),sha256=contract.sha(grant_path))
            fake_service=types.ModuleType('manager.service');fake_service.converters=Converters
            with mock.patch.object(entry,'ROOT',folder),mock.patch.object(read_all,'native_connector',return_value=connector),\
                 mock.patch.object(experiment_hardware,'launch_provenance',return_value={'fake_offline_test':True}),\
                 mock.patch.object(entry.importlib.metadata,'version',return_value='fake'),\
                 mock.patch.object(original,'load_assets',return_value=([0]*3072,np.zeros((64,75),dtype=np.float32))),\
                 mock.patch.object(c,'Collector',side_effect=RuntimeError('construction failed')),\
                 mock.patch.dict(sys.modules,{'manager.service':fake_service}),mock.patch('builtins.print'):
                output,result=entry.run_segment('hardware',13,None,b'',{},p,plan_path,False,1024**2,grant)
            self.assertTrue(connector.closed)
            self.assertEqual(connector.calls,[])
            self.assertEqual(result['status'],'failed')
            self.assertIs(result['pre_run_cleanup_close_result'],True)
            self.assertTrue((output/'manifest.json').exists())

    def test_prefix_helper_identity_and_failed_report_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);path=folder/'report.json'
            report=dict(schema='cifar-stage-prefix-v1',source='hardware',scope='full',protocol_sha256=contract.PHYSICAL_PROTOCOL_SHA,
                status='verified_complete_stage_prefix',complete_full_tensor=False,
                audit_source_sha256={'tools/audit_cifar_interrupted_prefix.py':next(iter(contract.APPROVED_PREFIX_AUDITOR_SHA256)),
                    'tools/cifar_prefix_replay.py':'0'*64},accepted_stages=[])
            path.write_text(json.dumps(report));name=path.relative_to(contract.ROOT).as_posix()
            plan_dict=dict(schema='cifar-continuation-plan-v1',source='hardware',purpose='full',spec_sha256=contract.digest(contract.SPEC),
                physical_protocol_sha256=contract.PHYSICAL_PROTOCOL_SHA,first_stage=0,stop_stage_exclusive=1,target_stop_stage_exclusive=48,
                accepted_stage_ids=[],prefix_reports=[dict(path=name,sha256=contract.sha(path))])
            with self.assertRaisesRegex(ValueError,'Exact reviewed prefix'):contract.validate_plan(plan_dict,'hardware')
            report['status']='failed';path.write_text(json.dumps(report));plan_dict['prefix_reports'][0]['sha256']=contract.sha(path)
            with self.assertRaisesRegex(ValueError,'Failed diagnostic'):contract.validate_plan(plan_dict,'hardware')

    def test_unreviewed_or_missing_prefix_rejected(self):
        p=dict(schema='cifar-continuation-plan-v1',source='hardware',purpose='full',spec_sha256=contract.digest(contract.SPEC),
               physical_protocol_sha256=contract.PHYSICAL_PROTOCOL_SHA,first_stage=1,stop_stage_exclusive=48,target_stop_stage_exclusive=48,accepted_stage_ids=[0],prefix_reports=[])
        with self.assertRaises(ValueError):contract.validate_plan(p,'hardware')
        p.update(first_stage=0,accepted_stage_ids=[])
        self.assertEqual(contract.validate_plan(p,'hardware'),(0,48))
        p['stop_stage_exclusive']=6
        self.assertEqual(contract.validate_plan(p,'hardware'),(0,6))
        p['stop_stage_exclusive']=49
        with self.assertRaises(ValueError):contract.validate_plan(p,'hardware')

    def test_controlled_interruption_restarts_whole_stage_with_fresh_programming(self):
        from tools.cifar_prefix_replay import audit_range
        p=json.loads((Path(original.__file__).parent/'protocol.json').read_text())
        image=[0]*3072;weights=np.zeros((64,75),dtype=np.float32)
        failed=CheckpointMemory();connector=FakeConnector()
        with mock.patch.object(original,'load_assets',return_value=(image,weights)),mock.patch('builtins.print'):
            runner=c.Collector(connector,Converters,failed,p,original.digest(p),'simulator',stage=1,interrupt_after_positions=17)
            with self.assertRaisesRegex(RuntimeError,'position17'):runner.run_stage(image,weights)
        self.assertTrue(connector.closed and failed.finished)
        self.assertEqual(runner.completed_positions,17)
        self.assertTrue(any(v.get('completed_positions')==16 for v in failed.checkpoints))
        self.assertEqual(failed.rows[-1]['status'],'failed')
        with self.assertRaises(ValueError):audit_range(iter(failed.rows),image,weights,original.digest(p),first_stage=1,stop_stage=2)
        fresh=CheckpointMemory();connector2=FakeConnector()
        with mock.patch.object(original,'load_assets',return_value=(image,weights)),mock.patch('builtins.print'):
            runner2=c.Collector(connector2,Converters,fresh,p,original.digest(p),'simulator',stage=1)
            runner2.run_stage(image,weights)
        payload,report=audit_range(iter(fresh.rows),image,weights,original.digest(p),first_stage=1,stop_stage=2)
        self.assertEqual(runner2.completed_positions,784)
        self.assertEqual(report['accepted_stages'][0]['task_first'],1)
        self.assertEqual(report['accepted_stages'][0]['shape'],[12,4,28,28])
        self.assertEqual(sum(row['event']=='coefficient_begin' for row in fresh.rows),128)
        self.assertTrue(connector2.closed)


if __name__=='__main__':unittest.main()
