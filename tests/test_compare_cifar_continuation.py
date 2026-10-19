"""Synthetic-only comparison qualification; no fabricated real HW reports or I/O."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
from tools import compare_cifar_continuation as adapter
from tests.test_compare_full_cifar_reports import report as legacy_fixture


def composite_fixture():
    base = legacy_fixture('hardware')
    report = dict(schema='cifar-continuation-classification-v1', source='hardware',
        complete_full_tensor=True, accepted_stage_ids=list(range(48)),
        full_dataset_accuracy_measured=False, hardware_acquisition_performed=False,
        physical_protocol_sha256=adapter.metrics.PROTOCOL_SHA, spec_sha256=adapter.SPEC_SHA,
        analyzer_sha256=adapter.PINNED_TOOLS['tools/classify_cifar_continuation.py'],
        frozen_digital_bridge_sha256=adapter.metrics.PINS['classifier.py'],
        digital_downstream_sha256=adapter.metrics.DIGITAL_SHA,
        case=base['case'], output_class_fine_labels=base['output_class_fine_labels'],
        artifact_sha256=base['artifact_sha256'], source_gain_v_per_weight=.125,
        gain_confounds_source_comparison=True, predictions=base['predictions'],
        conv1_errors=base['conv1_errors'], cpu_validation=base['cpu_validation'],
        payload_sha256={'raw': '0'*64, 'calibrated': '0'*64}, stage_reports=[],
        validated_stage_impacts=48, validated_stage_pulses=0, unknown_prior_lost_tail_operations=True)
    audit = base['native_audit']
    audit.update(native_impacts=48, programming_pulses=0)
    return report, audit


def write_synthetic_payload_evidence(root):
    """Ephemeral test fixture only, never a claimed acquisition or saved report."""
    directory = root/'reports/synthetic'; directory.mkdir(parents=True)
    report, aggregate = composite_fixture()
    provenance = {}; raw_inventory = {}
    one = bytes(adapter.STAGE_BYTES)
    stages = {s: {'raw': one, 'calibrated': one} for s in range(48)}
    certificates = []
    for stage in range(48):
        payloads = {}
        for branch in ('raw', 'calibrated'):
            path = directory/f'{stage}-{branch}.f32'; path.write_bytes(one)
            payloads[branch] = dict(path=path.relative_to(root).as_posix(), sha256=adapter.sha(path),
                                    shape=[12,4,28,28], dtype='<f4')
        cert = dict(stage=stage, session_id='synthetic-only', source='hardware', scope='full', shape=[12,4,28,28], dtype='<f4',
                    protocol_sha256=adapter.metrics.PROTOCOL_SHA, source_provenance_sha256=adapter.digest(provenance),
                    raw_inventory_sha256=adapter.digest(raw_inventory), native_impacts=1, programming_pulses=0,
                    task_first=stage+1, task_last=stage+1, payloads=payloads,
                    payload_sha256={k:v['sha256'] for k,v in payloads.items()})
        cert['certificate_sha256'] = adapter.digest(cert); certificates.append(cert)
    prefix = dict(schema='cifar-stage-prefix-v1', source='hardware', scope='full',
        status='verified_complete_stage_prefix', complete_full_tensor=False,
        protocol_sha256=adapter.metrics.PROTOCOL_SHA, first_stage=0, stop_stage_exclusive=48,
        session='results/cifar_full_case0_20261019/synthetic-only', session_id='synthetic-only', accepted_stages=certificates,
        native_impacts=48, programming_pulses=0, provenance=provenance, raw_inventory=raw_inventory,
        audit_source_sha256={n:adapter.PINNED_TOOLS[n] for n in
            ('tools/audit_cifar_interrupted_prefix.py','tools/cifar_prefix_replay.py')},
        strict_numeric_replay='PASS', portable_condition_policy=None,
        numeric_diagnostics=[dict(stage=s, strict_numeric_replay='PASS',
            decision_equivalence='exact_within_original_tolerance') for s in range(48)],
        covered_tensor_elements=1806336, observed_coefficients=4800,
        weight_observations=aggregate['weight_observations'], heldout=aggregate['heldout'], retention=aggregate['retention'],
        maximum_shared_zero_shift_v=0., maximum_position_bracket_duration_s=1.)
    audit_path = directory/'prefix.json'; audit_path.write_text(json.dumps(prefix))
    reference = dict(path=audit_path.relative_to(root).as_posix(), sha256=adapter.sha(audit_path),
        acquisition_session=prefix['session'], accepted_stages=list(range(48)), native_impacts=48,
        programming_pulses=0, execution_grant=None, unknown_prior_tail=True)
    report['stage_reports'] = [reference]
    for branch in ('raw','calibrated'):
        combined = adapter.assemble(stages, branch)
        (directory/(branch+'.f32')).write_bytes(combined)
        report['payload_sha256'][branch] = hashlib.sha256(combined).hexdigest()
    return report, prefix, directory/'composite.json', audit_path


class ContinuationComparisonTests(unittest.TestCase):
    def test_synthetic_complete_silent_mechanical_zero_not_success(self):
        hw, audit = composite_fixture()
        result = adapter.compare_reports(legacy_fixture('simulator'), hw, audit)
        self.assertEqual(result['status'], 'consistent_reports')
        decoder = result['sources']['hardware']['downstream']['calibrated']['decoders']['first_spike']
        self.assertTrue(decoder['mechanical_matches_true_label'])
        self.assertFalse(decoder['valid_unique_decision'])
        self.assertIsNone(decoder['valid_correct'])
        self.assertFalse(result['full_dataset_accuracy_measured'])
        self.assertIn('.015V/weight', ' '.join(result['limitations']))

    def test_missing48_wrongcase_protocol_source_or_gain_rejected(self):
        for name in ('coverage','case','protocol','source','gain','analyzer','checkpoint','partial'):
            hw, audit = composite_fixture()
            if name == 'coverage': hw['accepted_stage_ids'].pop()
            elif name == 'case': hw['case']['fine_label']=1
            elif name == 'protocol': hw['physical_protocol_sha256']='bad'
            elif name == 'source': hw['source']='simulator'
            elif name == 'gain': hw['source_gain_v_per_weight']=.015
            elif name == 'analyzer': hw['analyzer_sha256']='bad'
            elif name == 'checkpoint': hw['artifact_sha256']['weights/best_fs.pt']='bad'
            else: hw['complete_full_tensor']=False
            with self.subTest(name=name), self.assertRaises(ValueError):
                adapter.compare_reports(legacy_fixture('simulator'), hw, audit)

    def test_coefficient_duplicate_layout_and_target_differences_rejected(self):
        for name in ('duplicate','layout','target','heldout'):
            hw, audit = composite_fixture()
            if name == 'duplicate': audit['weight_observations'][-1]=copy.deepcopy(audit['weight_observations'][0])
            elif name == 'layout': audit['weight_observations'][0]['positive_cell']=[2,0]
            elif name == 'target': audit['weight_observations'][0]['trained_weight']=.2
            else: audit['heldout'][-1]=copy.deepcopy(audit['heldout'][0])
            with self.subTest(name=name), self.assertRaises(ValueError):
                adapter.compare_reports(legacy_fixture('simulator'), hw, audit)

    def test_cpu_platform_difference_explicit_review(self):
        hw, audit = composite_fixture(); sim=legacy_fixture('simulator')
        hw['predictions']['cpu_reference']['output_membranes_post_reset'][0][0]=1e-7
        self.assertEqual(adapter.compare_reports(sim,hw,audit)['status'],'consistent_reports')
        hw['predictions']['cpu_reference']['output_membranes_post_reset'][0][0]=.1
        self.assertEqual(adapter.compare_reports(sim,hw,audit)['status'],'review_required')

    def test_exact_byte_stage_interleave_no_missing_or_nonfinite_fill(self):
        stages={s:{b:struct.pack('<f',float(s+1))* (adapter.STAGE_BYTES//4) for b in ('raw','calibrated')} for s in range(48)}
        output=adapter.assemble(stages,'raw')
        for t in (0,11):
            for f in range(64):
                for pos in (0,783):
                    for tile in range(3):
                        index=(((t*64+f)*784+pos)*3+tile)*4
                        self.assertEqual(struct.unpack_from('<f',output,index)[0], f//4*3+tile+1)
        bad=copy.deepcopy(stages); del bad[13]
        with self.assertRaises(ValueError):adapter.assemble(bad,'raw')
        stages[13]['raw']=struct.pack('<f',float('nan'))+stages[13]['raw'][4:]
        with self.assertRaises(ValueError):adapter.assemble(stages,'raw')

    def test_bound_synthetic_payload_files_then_mutations(self):
        with tempfile.TemporaryDirectory(dir=adapter.ROOT/'.tmp') as folder:
            root=Path(folder)
            report,prefix,path,audit_path=write_synthetic_payload_evidence(root)
            with mock.patch.object(adapter.metrics,'ROOT',root),mock.patch.object(adapter,'ROOT',root),\
                 mock.patch.object(adapter,'bind_stage_sidecars',return_value={'synthetic_binding_stub':True}):
                aggregate,binding=adapter.bind_composite(report,path)
                self.assertEqual(len(aggregate['weight_observations']),4800)
                self.assertTrue(binding['payload_assembly_verified'])
                report['stage_reports'].append(copy.deepcopy(report['stage_reports'][0]))
                with self.assertRaisesRegex(ValueError,'Duplicate acquisition'):adapter.bind_composite(report,path)
                report['stage_reports'].pop()
                (path.parent/'raw.f32').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'Composite tensor'):adapter.bind_composite(report,path)
                prefix['audit_source_sha256']['tools/cifar_prefix_replay.py']='bad'
                audit_path.write_text(json.dumps(prefix));report['stage_reports'][0]['sha256']=adapter.sha(audit_path)
                with self.assertRaisesRegex(ValueError,'dependency inventory'):adapter.bind_composite(report,path)

    def test_stage_certificate_and_task_counter_mutations_rejected(self):
        with tempfile.TemporaryDirectory(dir=adapter.ROOT/'.tmp') as folder:
            report,prefix,_,_=write_synthetic_payload_evidence(Path(folder))
            ref=report['stage_reports'][0]
            adapter.validate_stage_report(prefix,ref)
            prefix['accepted_stages'][0]['task_first']=2
            with self.assertRaisesRegex(ValueError,'certificate changed'):adapter.validate_stage_report(prefix,ref)
            cert=prefix['accepted_stages'][0]
            cert['certificate_sha256']=adapter.digest({k:v for k,v in cert.items() if k!='certificate_sha256'})
            with self.assertRaisesRegex(ValueError,'task interval'):adapter.validate_stage_report(prefix,ref)

    def test_partial_current_native_sim_qualification_is_never_full_hw(self):
        path=adapter.ROOT/'reports/2026-10-19/cifar-resume-review/native-sim-recovery-qualification/COMPOSITE_REPORT.json'
        if not path.exists():self.skipTest('Saved nativeSIM qualification unavailable in this checkout')
        with self.assertRaises(ValueError):adapter.validate_composite(adapter.read_json(path))

    def test_portable_diagnostic_scope_and_session_mutations_rejected(self):
        with tempfile.TemporaryDirectory(dir=adapter.ROOT/'.tmp') as folder:
            report, prefix, _, _ = write_synthetic_payload_evidence(Path(folder))
            ref = report['stage_reports'][0]
            portable = dict(stage=0, strict_numeric_replay='FAIL_condition_only', decision_equivalence='PASS',
                policy='excluded-rank-condition-v1', local_condition=1e16, recorded_condition=None,
                unchanged_other_fields_tolerance=dict(absolute=1e-10, relative=1e-10),
                spectra={k:dict(singular_values=[1.] * 7 + [0.], numpy_rank_tolerance=1e-12,
                               margin_factor=16, stable=True) for k in ('fit', 'reference')})
            prefix['strict_numeric_replay'] = 'FAIL_condition_only'
            prefix['portable_condition_policy'] = 'excluded-rank-condition-v1'
            prefix['numeric_diagnostics'][0] = portable
            adapter.validate_stage_report(prefix, ref)
            for name in ('missing', 'duplicate', 'top_status', 'decision', 'domain', 'margin', 'stable', 'spectrum', 'session'):
                bad = copy.deepcopy(prefix)
                if name == 'missing': bad['numeric_diagnostics'].pop()
                elif name == 'duplicate': bad['numeric_diagnostics'][-1] = copy.deepcopy(portable)
                elif name == 'top_status': bad['strict_numeric_replay'] = 'PASS'
                elif name == 'decision': bad['numeric_diagnostics'][0]['decision_equivalence'] = 'FAIL'
                elif name == 'domain': bad['numeric_diagnostics'][0]['local_condition'] = 10.
                elif name == 'margin': bad['numeric_diagnostics'][0]['spectra']['fit']['margin_factor'] = 8
                elif name == 'stable': bad['numeric_diagnostics'][0]['spectra']['fit']['stable'] = False
                elif name == 'spectrum': bad['numeric_diagnostics'][0]['spectra']['fit']['singular_values'][-1] = 1e-12
                else:
                    cert = bad['accepted_stages'][0]; cert['session_id'] = 'different-session'
                    cert['certificate_sha256'] = adapter.digest({k:v for k,v in cert.items() if k != 'certificate_sha256'})
                with self.subTest(name=name), self.assertRaises(ValueError):
                    adapter.validate_stage_report(bad, ref)


if __name__=='__main__':unittest.main()
