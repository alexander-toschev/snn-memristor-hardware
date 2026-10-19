"""Synthetic JSON only; no actual HW data, model, raw audit, or device access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from tools import compare_full_cifar_reports as c


def prediction(silent=False):
    spikes = [[0.] * 10 for _ in range(12)]
    if not silent: spikes[2][1] = 1.
    membranes = [[0.] * 10 for _ in range(12)]; membranes[-1][1] = 1.
    twd = [0.] * 10
    if not silent: twd[1] = 1.
    indices = dict(first_spike=0 if silent else 1, membrane=1, twd=0 if silent else 1)
    return dict(predicted_subset_indices=indices, predicted_cifar100_fine_labels={k: c.LABELS[v] for k, v in indices.items()},
                all_output_classes_silent=silent, first_spike_winner_count=10 if silent else 1,
                output_membranes_post_reset=membranes, output_spikes=spikes, twd_scores=twd)


def report(source):
    rows = []
    for f in range(64):
        for tap in range(75):
            pair=f%4; bl=tap%32
            row=dict(stage=f//4*3+tap//32, filter=f, tap=tap, bl=bl, positive_cell=[2*pair,bl], negative_cell=[2*pair+1,bl],
                     trained_weight=.1, immediate_within_tolerance=True, programming_stop='synthetic', per_coefficient_fitting_used=False)
            row.update({k:.1 for k in ['immediate_weight','post_all_raw_weight','post_all_calibrated_weight','final_raw_weight','final_calibrated_weight']})
            rows.append(row)
    return dict(schema='full-case0-classification-v1', source=source+'_conv1_digital_remaining_network',
        registered_classifier_import_verified=True, full_dataset_accuracy_measured=False,
        classifier_sha256=c.PINS['classifier.py'], auditor_sha256=c.PINS['audit_native.py'], digital_downstream_sha256=c.DIGITAL_SHA,
        artifact_sha256={'weights/best_fs.pt':c.CHECKPOINT_SHA,'weights/cifar100_conv1_fixture.json':c.FIXTURE_SHA},
        case=dict(c.CASE), output_class_fine_labels=list(c.LABELS), acquisition_manifest_sha256='0'*64,
        native_audit=dict(source=source,scope='full',status='native_trace_replayed',complete_full_tensor=True,
            observed_coefficients=4800,covered_tensor_elements=1806336,native_impacts=100,programming_pulses=20,
            maximum_shared_zero_shift_v=0.,maximum_position_bracket_duration_s=1.,weight_observations=rows,
            heldout=[dict(stage=i,raw_rmse_v=.02,calibrated_rmse_v=.01) for i in range(48)],
            retention=[dict(stage=i,final_minus_initial_basis_rmse_v=0.,max_abs_v=0.) for i in range(48)]),
        predictions={k:prediction(k=='calibrated') for k in ['cpu_reference','raw','calibrated']},
        conv1_errors={k:dict(conv1_rmse=.1,conv1_max_abs=.2,relative_l2=.5) for k in ['raw','calibrated']},
        cpu_validation=dict(direct_forward_max_abs=0.,tiled_forward_max_abs=dict(membrane=0.,spikes=0.,twd=0.),conv1_tiled_vs_direct_max_abs=1e-7))


class ComparisonTests(unittest.TestCase):
    def test_synthetic_complete_and_silent_tie_not_success(self):
        out=c.compare(report('simulator'),report('hardware'))
        self.assertEqual(out['status'],'pass')
        calibrated=out['sources']['hardware']['downstream']['calibrated']['decoders']
        self.assertEqual(calibrated['first_spike']['mechanical_subset_index'],0)
        self.assertTrue(calibrated['first_spike']['mechanical_matches_true_label'])
        self.assertIsNone(calibrated['first_spike']['valid_correct'])
        self.assertEqual(calibrated['twd']['reason'],'no_output_spike')

    def test_missing_duplicate_layout_and_case_rejected(self):
        for kind in ['missing','duplicate','layout','case','scope','source','checkpoint']:
            sim=report('simulator'); hw=report('hardware'); rows=hw['native_audit']['weight_observations']
            if kind=='missing': rows.pop()
            elif kind=='duplicate': rows[-1]=copy.deepcopy(rows[0])
            elif kind=='layout': rows[0]['positive_cell']=[2,0]
            elif kind=='case': hw['case']['original_test_index']=10
            elif kind=='scope': hw['native_audit']['scope']='smoke'
            elif kind=='source': hw['native_audit']['source']='simulator'
            else: hw['artifact_sha256']['weights/best_fs.pt']='bad'
            with self.subTest(kind=kind), self.assertRaises(ValueError): c.compare(sim,hw)

    def test_cpu_crossplatform_tolerance_and_difference_review(self):
        sim=report('simulator'); hw=report('hardware')
        hw['predictions']['cpu_reference']['output_membranes_post_reset'][0][0]=1e-7
        self.assertEqual(c.compare(sim,hw)['status'],'pass')
        hw['predictions']['cpu_reference']['output_membranes_post_reset'][0][0]=.1
        self.assertEqual(c.compare(sim,hw)['status'],'review_required')

    def test_per_source_tolerance_and_wrong_sign(self):
        r=report('simulator')['native_audit']['weight_observations'][:2]
        r[0]['trained_weight']=.01
        r[0]['immediate_weight']=.05; r[0]['immediate_within_tolerance']=True
        self.assertEqual(c.weight_summary(r,'simulator')['phases']['immediate_weight']['direct_within_tolerance'],2)
        with self.assertRaises(ValueError): c.weight_summary(r,'hardware')
        r[1]['post_all_raw_weight']=-.1; r[0]['post_all_raw_weight']=0
        stats=c.weight_summary(r,'simulator')['phases']['post_all_raw_weight']
        self.assertEqual(stats['wrong_sign_nonzero_target'],1); self.assertEqual(stats['zero_effective_nonzero_target'],1)

    def test_nan_and_forged_mechanical_prediction_rejected(self):
        p=prediction(); p['twd_scores'][0]=float('nan')
        with self.assertRaises(ValueError): c.decode(p,0)
        p=prediction(); p['predicted_subset_indices']['first_spike']=0
        with self.assertRaises(ValueError): c.decode(p,0)

    def test_manifest_incomplete_or_sha_mismatch_rejected_before_sidecars(self):
        base=c.ROOT/'.tmp'; base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            root=Path(folder); session=root/'results/cifar_full_case0_20261019/synthetic'; session.mkdir(parents=True)
            path=session/'manifest.json'; path.write_text(json.dumps(dict(status='running',scope='full',source='hardware',session_id='synthetic')))
            r=report('hardware')
            with mock.patch.object(c,'ROOT',root), self.assertRaisesRegex(ValueError,'bound'):c.bind_session(r,session,'hardware')
            r['acquisition_manifest_sha256']=c.sha(path)
            with mock.patch.object(c,'ROOT',root), self.assertRaisesRegex(ValueError,'incomplete'):c.bind_session(r,session,'hardware')

    def test_complete_synthetic_binding_then_protocol_source_budget_mutations(self):
        base=c.ROOT/'.tmp'; base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            root=Path(folder); session=root/'results/cifar_full_case0_20261019/synthetic'; session.mkdir(parents=True)
            pins={}; sources={}
            names=[c.DIRECTORY+n for n in c.PINS]+['vendor/MemriBoard/manager/board.py',
                'vendor/MemriBoard/manager/service/converters.py','vendor/MemriBoard/manager/service/drivers.py',
                'configs/raspberry_board.ini','weights/conv1_weights.npy']
            for name in names:
                path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('synthetic '+name)
                sources[name]=c.sha(path)
                if name.startswith(c.DIRECTORY): pins[name[len(c.DIRECTORY):]]=c.sha(path)
            digital=root/'full_cifar_classifier.py'; digital.write_text('synthetic digital')
            protocol=dict(simulator_gain=.015,hardware_gain=.125,maximum_impacts=15000000,maximum_programming_pulses=115200)
            pp=session/'protocol.json'; pp.write_text(json.dumps(protocol)); protocol_sha=c.digest(protocol)
            manifest=dict(status='completed',scope='full',source='hardware',session_id='synthetic',protocol_sha256=protocol_sha,
                impacts=100,pulses=20,source_sha256=sources,result_sha256={'protocol.json':c.sha(pp)},execution_environment={'synthetic':True})
            r=report('hardware'); r['artifact_sha256']={}
            mp=session/'manifest.json'
            def save(): mp.write_text(json.dumps(manifest)); r['acquisition_manifest_sha256']=c.sha(mp)
            save()
            with mock.patch.multiple(c,ROOT=root,PINS=pins,DIGITAL_SHA=c.sha(digital),PROTOCOL_SHA=protocol_sha):
                self.assertEqual(c.bind_session(r,session,'hardware')['protocol_sha256'],protocol_sha)
                private='vendor/MemriBoard/MemriCORE/rp5_fpga_python/rpi_FPGAed.py'
                sources[private]='0'*64; save()
                with self.assertRaisesRegex(ValueError,'Required local source missing'):c.bind_session(r,session,'hardware')
                del sources[private]; save()
                manifest['impacts']=15000001; r['native_audit']['native_impacts']=15000001; save()
                with self.assertRaisesRegex(ValueError,'budget'):c.bind_session(r,session,'hardware')
                manifest['impacts']=100; r['native_audit']['native_impacts']=100
                key=c.DIRECTORY+'collector.py'; original=sources[key]; sources[key]='bad'; save()
                with self.assertRaisesRegex(ValueError,'source pin'):c.bind_session(r,session,'hardware')
                sources[key]=original; save(); protocol['hardware_gain']=.2; pp.write_text(json.dumps(protocol))
                with self.assertRaisesRegex(ValueError,'Protocol sidecar'):c.bind_session(r,session,'hardware')


if __name__=='__main__': unittest.main()
