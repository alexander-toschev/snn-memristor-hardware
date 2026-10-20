"""Synthetic saved evidence only; no live acquisition, raw numerical replay or Git."""
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import review_cifar_batch as tool


class BatchReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=tool.ROOT/'.tmp')
        self.root=Path(self.temp.name)
        self.stack=ExitStack();self.stack.enter_context(patch.object(tool,'ROOT',self.root))
        self.stack.enter_context(patch.object(tool,'source_pins',return_value={'synthetic_source_fixture':True}))
    def tearDown(self):self.stack.close();self.temp.cleanup()
    def write(self,path,value):
        target=self.root/path;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
        return target
    def fixture(self,stage=24,footer_status='completed'):
        session='results/cifar_resumable_20261019/synthetic'
        plan=self.write(session+'/continuation_plan.json',{'synthetic_only':True})
        plan_sha=tool.sha(plan)
        self.write(session+'/start_manifest.json',dict(continuation_plan_sha256=plan_sha))
        footer=dict(event='session_complete',full_classifier_accuracy_measured=False,impact_attempts=1,
                    programming_pulses=0,scope='full',source='hardware',stage=stage,status=footer_status)
        raw_path=self.write(session+'/raw/part-000000.jsonl.gz',gzip.compress(json.dumps(footer).encode()+b'\n'))
        raw=dict(parts=[dict(path=raw_path.name,sha256=tool.sha(raw_path),bytes=raw_path.stat().st_size,admitted=True)],
                 excluded_crc_valid_tail_records=1)
        payloads={}
        for branch in ('raw','calibrated'):
            path=self.write('reports/synthetic/'+branch+'.f32',bytes(tool.compare.STAGE_BYTES))
            payloads[branch]=dict(path=path.relative_to(self.root).as_posix(),sha256=tool.sha(path),shape=[12,4,28,28],dtype='<f4')
        cert=dict(stage=stage,session_id='synthetic',source='hardware',scope='full',shape=[12,4,28,28],
            dtype='<f4',protocol_sha256=tool.compare.metrics.PROTOCOL_SHA,
            source_provenance_sha256=tool.compare.digest({}),raw_inventory_sha256=tool.compare.digest(raw),
            native_impacts=1,programming_pulses=0,task_first=1,task_last=1,payloads=payloads,
            payload_sha256={k:v['sha256'] for k,v in payloads.items()})
        cert['certificate_sha256']=tool.compare.digest(cert)
        rows=[dict(stage=stage,filter=(stage//3)*4+p,tap=(stage%3)*32+b,bl=b,
            positive_cell=[2*p,b],negative_cell=[2*p+1,b],per_coefficient_fitting_used=False)
            for p in range(4) for b in range((32,32,11)[stage%3])]
        report=dict(schema='cifar-stage-prefix-v1',status='completed_range',source='hardware',scope='full',
            protocol_sha256=tool.compare.metrics.PROTOCOL_SHA,complete_full_tensor=False,session=session,session_id='synthetic',
            first_stage=stage,stop_stage_exclusive=stage+1,accepted_stages=[cert],native_impacts=1,programming_pulses=0,
            provenance={},raw_inventory=raw,audit_source_sha256={n:tool.compare.PINNED_TOOLS[n] for n in
                ('tools/audit_cifar_interrupted_prefix.py','tools/cifar_prefix_replay.py')},
            strict_numeric_replay='PASS',numeric_diagnostics=[dict(stage=stage,strict_numeric_replay='PASS',
                decision_equivalence='exact_within_original_tolerance')],
            heldout=[dict(stage=stage)],retention=[dict(stage=stage)],covered_tensor_elements=12*4*28*28,
            observed_coefficients=len(rows),weight_observations=rows,maximum_shared_zero_shift_v=0.,
            maximum_position_bracket_duration_s=1.,frozen_full_auditor_pass_claimed=False,
            physical_state_now_inferred=False,terminal_session_event_verified=True,unknown_actual_tail_operations=False)
        directory=self.root/session
        self.write(session+'/manifest.json',dict(stage_range=[stage,stage+1],continuation_plan_sha256=plan_sha,
            result_sha256={p.relative_to(directory).as_posix():tool.sha(p) for p in directory.rglob('*') if p.is_file()}))
        path=self.write('reports/synthetic/PREFIX_AUDIT.json',report)
        self.stack.enter_context(patch.object(tool.compare,'bind_stage_sidecars',return_value={'synthetic_binding_stub':True}))
        self.stack.enter_context(patch.object(tool.composer,'verify_stage_grant',return_value={'synthetic_grant_stub':True}))
        return path.relative_to(self.root).as_posix(),plan_sha,report
    def test_completed_arbitrary_stage_actual_hash_and_payload_checks(self):
        name,pin,_=self.fixture(stage=33)
        result=tool.review_report(name,pin)
        self.assertEqual(result['stages'],[33]);self.assertEqual(result['payloads_rehashed'],2)
        self.assertEqual(result['strict_numeric_replay'],'PASS')
        (self.root/'reports/synthetic/raw.f32').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):tool.review_report(name,pin)
    def test_wrong_plan_missing_close_and_raw_mutation_rejected(self):
        name,pin,report=self.fixture(footer_status='failed')
        with self.assertRaisesRegex(ValueError,'another finite plan'):tool.review_report(name,'0'*64)
        with self.assertRaisesRegex(ValueError,'close footer'):tool.review_report(name,pin)
        (self.root/report['session']/'raw/part-000000.jsonl.gz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Result bytes'):tool.review_report(name,pin)
    def test_portable_condition_remains_explicit_and_invalid_policy_rejected(self):
        name,pin,report=self.fixture()
        report['strict_numeric_replay']='FAIL_condition_only'
        report['portable_condition_policy']='excluded-rank-condition-v1'
        report['numeric_diagnostics']=[dict(stage=24,strict_numeric_replay='FAIL_condition_only',decision_equivalence='PASS',
            policy='excluded-rank-condition-v1',recorded_condition=None,local_condition=1e16,
            unchanged_other_fields_tolerance=dict(absolute=1e-10,relative=1e-10),
            spectra={k:dict(singular_values=[1.]*7+[0.],numpy_rank_tolerance=1e-12,margin_factor=16,stable=True)
                     for k in ('fit','reference')})]
        self.write(name,report)
        self.assertEqual(tool.review_report(name,pin)['condition_only_stages'],[24])
        report['numeric_diagnostics'][0]['decision_equivalence']='FAIL';self.write(name,report)
        with self.assertRaises(ValueError):tool.review_report(name,pin)
    def test_exact_prefix_batch_coverage_no_duplicate_sessions(self):
        old=[dict(stages=list(range(19)),session='old')]
        new=[dict(stages=[s],session=str(s)) for s in range(19,25)]
        tool.check_coverage(old,new,19,25)
        for changed in (new[:-1],new+[new[-1]],list(reversed(new)),[dict(stages=[19],session='old')]+new[1:]):
            with self.assertRaises(ValueError):tool.check_coverage(old,changed,19,25)
    def test_duplicate_coefficients_not_accepted_as_same_count(self):
        _,_,report=self.fixture()
        rows=report['weight_observations'];rows[-1]=dict(rows[0])
        with self.assertRaisesRegex(ValueError,'coefficient'):tool.check_coefficients(rows,[24])
    def test_default_review_creates_no_plan_and_failure_creates_no_outputs(self):
        plan={'purpose':'full','prefix_reports':[dict(path='old.json',sha256='old')]}
        self.write('plan.json',plan);self.write('old.json',{});self.write('new.json',{})
        plan['prefix_reports'][0]['sha256']=tool.sha(self.root/'old.json');self.write('plan.json',plan)
        summaries=[dict(path=p,sha256=tool.sha(self.root/p),stages=stages,session=p,
            native_impacts=1,programming_pulses=0,unknown_prior_tail=p=='old.json')
            for p,stages in [('old.json',[0]),('new.json',[1])]]
        with patch.object(tool,'verify_tools'),patch.object(tool.contract,'validate_plan',return_value=(1,2)), \
             patch.object(tool.contract,'execution_grant',return_value={'sha256':'grant'}), \
             patch.object(tool,'review_report',side_effect=summaries),patch.object(tool.planner,'build') as build:
            result=tool.review('plan.json',['new.json'],'reports/review')
        build.assert_not_called();self.assertIsNone(result['next_finite_plan'])
        self.assertFalse((self.root/'reports/review/NEXT_PLAN.json').exists())
        with patch.object(tool,'verify_tools'),patch.object(tool.contract,'validate_plan',return_value=(1,2)), \
             patch.object(tool,'review_report',side_effect=ValueError('failed audit')):
            with self.assertRaises(ValueError):tool.review('plan.json',['new.json'],'reports/failed',3)
        self.assertFalse((self.root/'reports/failed').exists())
    def test_real_dependency_pins_file_only(self):
        with patch.object(tool,'ROOT',Path(__file__).resolve().parents[1]):tool.verify_tools()
    def test_output_never_changes_acquisition_inventory(self):
        self.write('plan.json',{'purpose':'full','prefix_reports':[]})
        with patch.object(tool,'verify_tools'),patch.object(tool.contract,'validate_plan',return_value=(0,1)):
            with self.assertRaisesRegex(ValueError,'Output must'):
                tool.review('plan.json',['not_read.json'],'results/immutable-session/analysis')
        self.assertFalse((self.root/'results').exists())
    def test_terminal_decompression_budget_and_crc_fail_closed(self):
        path=self.write('oversized.gz',gzip.compress(b'x'*(tool.TERMINAL_PART_LIMIT+1)))
        with self.assertRaisesRegex(ValueError,'8MiB'):tool.terminal_record(path)
        data=bytearray(gzip.compress(b'{"event":"session_complete"}\n'));data[-8]^=1
        path.write_bytes(data)
        with self.assertRaises(gzip.BadGzipFile):tool.terminal_record(path)
    def test_next_plan_only_on_explicit_option_with_reviewed_bounds(self):
        from experiments.cifar_resumable_20261019 import collector
        from experiments.cifar_full_case0_20261019 import plan as physical
        plan={'purpose':'full','prefix_reports':[]};self.write('plan.json',plan);self.write('new.json',{})
        summary=dict(path='new.json',sha256=tool.sha(self.root/'new.json'),stages=[0],session='new',
                     native_impacts=1,programming_pulses=0,unknown_prior_tail=False)
        next_plan=dict(first_stage=1,stop_stage_exclusive=2,target_stop_stage_exclusive=48)
        with patch.object(tool,'verify_tools'),patch.object(tool.contract,'validate_plan',side_effect=[(0,1),(1,2)]), \
             patch.object(tool.contract,'execution_grant',return_value={'sha256':'grant'}), \
             patch.object(tool,'review_report',return_value=summary), \
             patch.object(tool.planner,'build',return_value=next_plan) as build, \
             patch.object(physical,'fixture',return_value={'image_chw_uint8':[]}), \
             patch.object(collector,'stage_bounds',return_value={'stage':1,'maximum_impacts':123,'maximum_pulses':24}):
            result=tool.review('plan.json',['new.json'],'reports/explicit',2)
        build.assert_called_once_with('hardware',['new.json'],'full',2)
        self.assertEqual(result['next_finite_plan']['maximum_impacts'],123)
        self.assertEqual(json.loads((self.root/'reports/explicit/NEXT_PLAN.json').read_text()),next_plan)
        self.assertFalse(result['autorun_qualified'])


if __name__=='__main__':unittest.main()
