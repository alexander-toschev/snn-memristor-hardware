"""Offline tests only: FakeConnector never opens native hardware/simulator."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cifar_memristor as c
from cifar_memristor_analysis import scalar_calibration,temporal_response,aggregate,select_matrix,apply_matrix
from manager.service import converters
from native_conversions import readback,voltage


class FakeConnector:
    cb_type='simulator'
    def __init__(self,wrong=False,fail=False,zero=False):self.calls=[];self.wrong=wrong;self.fail=fail;self.zero=zero
    def impact(self,task):
        self.calls.append(copy.deepcopy(task))
        if self.fail:raise ConnectionError('lost response')
        if task['mode_flag']==7:
            adc=300;r=readback(adc,c.EXPECTED_BOARD,converters,read_voltage=.3)['resistance_ohm']
            return r+int(self.wrong),task['id'],adc
        self.assert_vector(task)
        active=next((i for i,v in enumerate(task['vol']) if v),None)
        adc=0 if self.zero or active is None else {(0,0):200,(1,0):100,(0,1):100,(1,1):200}[task['wl'],active]
        task['vol'][:]=[999]*32 # native conversion is allowed to mutate its input
        return 1.,0,adc
    @staticmethod
    def assert_vector(task):
        assert len(task['vol'])==32 and sum(x!=0 for x in task['vol'])<=1 and max(task['vol'])<=.1


def small_inputs():
    active=[dict(slot=i,tap_index=i,padding=False,trained_weight=w,positive_cell=[0,i],negative_cell=[1,i]) for i,w in enumerate((.1,-.1))]
    tile=dict(tile_id=0,slots=active);f=dict(filter_id=0,tiles=[tile])
    bntt=dict(eps=1e-5,parameters_t12x64={key:[[value]*64 for _ in range(12)] for key,value in
        [('weight',1.),('bias',0.),('running_mean',0.),('running_var',1.)]})
    fixture=dict(bntt=bntt,cases=[dict(original_test_index=i,spikes_t12x75=[[int(t==0),int(t==1)]+[0]*73 for t in range(12)],
        reference_t12x64={key:[[0.]*64 for _ in range(12)] for key in ('conv_sum','bntt_current','lif_before_reset','lif_after_reset','lif_spikes')}) for i in range(10)])
    return f,tile,fixture


class CifarTests(unittest.TestCase):
    def test_exact_budgets_and_all_weights(self):
        plan,routing,fixture=c.load_inputs();all_tiles=c.tiles(routing)
        self.assertEqual(c.counts(all_tiles,fixture)['maximum_impacts'],222118)
        self.assertEqual(c.counts(all_tiles[:1],fixture)['maximum_impacts'],9350)
        self.assertEqual(c.counts(all_tiles[1:],fixture)['maximum_impacts'],212768)
        self.assertEqual(c.counts(all_tiles,fixture)['maximum_programming_pulses'],66000)
        for f in routing['filters']:
            self.assertEqual(sorted(s['tap_index'] for t in f['tiles'] for s in c.slots(t)),list(range(75)))
        self.assertEqual(len({voltage(v,c.EXPECTED_BOARD,converters)['dac_code'] for v in plan['ramp_v']}),55)

    def test_scalar_no_per_weight_force_or_test_leak(self):
        cal=scalar_calibration([1.,-1.],[[.002,-.001]]*5,.00153)
        self.assertTrue(cal['decode_usable']);self.assertNotEqual(cal['effective_weights'],[1.,-1.])
        self.assertGreater(cal['weight_rmse'],0)
        self.assertTrue(scalar_calibration([1.],[[.00031]]*5,.00153)['decode_usable'])
        self.assertFalse(scalar_calibration([1.],[[-.01]]*5,.00153)['decode_usable'])
        self.assertFalse(scalar_calibration([1.],[[0.]]*5,.00153)['decode_usable'])
        with self.assertRaises(ValueError):scalar_calibration([float('nan')],[[1.]]*5,.00153)

    def test_matrix_independent_selection_and_gain(self):
        train=[dict(measured_mean_v=x,reference_v=[x[0]*2,x[1]*.5]) for x in ([.1,0.],[0.,.1],[.1,.1])]
        select=[dict(measured_repeats_v=[x]*3,reference_v=[x[0]*2,x[1]*.5]) for x in ([0.,0.],[.1,0.],[0.,.1])]
        result=select_matrix(train,select,.00031)
        self.assertEqual(result['selected_model'],'diagonal')
        self.assertEqual(apply_matrix(result['matrix'],[.2,.4]),[.4,.2])
        self.assertEqual(apply_matrix(result['matrix'],[0.,0.]),[0.,0.])
        singular=[dict(measured_mean_v=[.1,.1],reference_v=[2.,2.])]*3
        report=select_matrix(singular,select,.00031)
        self.assertEqual(report['training_design_rank'],1)
        self.assertTrue(all(r['name']!='full_ridge_0.01' for r in report['candidates']))
        diagonal=next(r for r in report['candidates'] if r['name']=='diagonal')
        self.assertFalse(diagonal['eligible'])
        self.assertEqual(report['selected_model'],'identity')

    def test_full_matrix_recovers_known_distortion_and_rejects_collapse(self):
        distortion=[[1.,.2],[.1,1.]]
        references=[[.1,0.],[0.,.1],[.1,.1],[.05,.02],[.02,.07]]
        train=[dict(measured_mean_v=apply_matrix(distortion,v),reference_v=v) for v in references]
        selection=[dict(measured_repeats_v=[apply_matrix(distortion,v)]*3,reference_v=v)
                   for v in ([0.,0.],[.03,.08],[.07,.02])]
        result=select_matrix(train,selection,.00031)
        self.assertEqual(result['selected_model'],'full_ridge_0.01')
        corrected=apply_matrix(result['matrix'],apply_matrix(distortion,[.04,.06]))
        self.assertLess(max(abs(a-b) for a,b in zip(corrected,[.04,.06])),.002)
        collapsed=[dict(measured_mean_v=v,reference_v=[v[0],0.00001*v[1]]) for v in references]
        out=select_matrix(collapsed,selection,.00031)
        self.assertTrue(all(not row['eligible'] for row in out['candidates'] if row['name']!='identity'))

    def test_selection_uses_repeat_mse_and_not_quality_gate(self):
        train=[dict(measured_mean_v=v,reference_v=[2*v[0],2*v[1]]) for v in ([.1,0.],[0.,.1],[.1,.1])]
        selection=[dict(measured_repeats_v=[[.08,.08],[.1,.1],[.12,.12]],reference_v=[.2,.2])]
        result=select_matrix(train,selection,.00031)
        self.assertFalse(result['selection_pass']) # too noisy, but still the best eligible correction
        self.assertNotEqual(result['selected_model'],'identity')
        for candidate in result['candidates']:
            corrected=[apply_matrix(candidate['matrix'],v) for v in selection[0]['measured_repeats_v']]
            expected=sum((v[j]-.2)**2/.01**2 for v in corrected for j in range(2))/6
            self.assertAlmostEqual(candidate['selection_score'],expected)
        with self.assertRaises(ValueError):select_matrix(train,[dict(measured_repeats_v=[[1.,2.]],reference_v=[1.,2.])],.00031)

    def test_actual_partial_measurements_and_unresolved_decode(self):
        f,tile,fixture=small_inputs()
        for zero in (False,True):
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
                fake=FakeConnector(zero=zero);run=c.TileRun(fake,converters,Path(tmp),'fake',c.SPEC,'simulator')
                result=run.collect_tile(f,tile,fixture)
                self.assertEqual(len(result['partials']),600)
                self.assertEqual(result['decode_usable'],not zero)
                self.assertTrue(all(p['trained_unit_contribution'] is None for p in result['partials']) if zero else
                    any(p['trained_unit_contribution']!=0 for p in result['partials']))
                self.assertEqual(result['programming'][0]['endpoint'],'set_ramp_exhausted')
                empty=result['partials'][2]['measured_rows'][0]
                self.assertEqual(len({empty['zero_before']['task_id'],empty['signals'][0]['task_id'],empty['zero_after']['task_id']}),3)
                raw=[json.loads(line) for line in (Path(tmp)/'raw.jsonl').read_text().splitlines()]
                frozen=next(i for i,row in enumerate(raw) if row['event']=='calibration_frozen')
                first=next(i for i,row in enumerate(raw) if row.get('phase')=='cifar')
                self.assertLess(frozen,first)
                before=copy.deepcopy(result['calibration']);result['final_anchors'][0][0]['zero']['voltage_v']=3.
                self.assertEqual(result['calibration'],before)

    def test_native_r_mismatch_and_transport_error_retained(self):
        for fake in (FakeConnector(wrong=True),FakeConnector(fail=True)):
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
                run=c.TileRun(fake,converters,Path(tmp),'fake',c.SPEC,'simulator')
                with self.assertRaises((ValueError,ConnectionError)):run.read([0,0],1,'test')
                run.log.close();raw=[json.loads(line) for line in (Path(tmp)/'raw.jsonl').read_text().splitlines()]
                self.assertEqual(raw[-1]['status'],'error');self.assertEqual(raw[0]['event'],'request')

    def test_current_guard_ends_ramp_without_fake_response(self):
        class Guarded(c.TileRun):
            def read(self,cell,n,phase,**meta):
                self.last_mode7[tuple(cell)]=(1.,1)
                return [dict(q=.1,task_id=i) for i in range(n)]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            fake=FakeConnector();run=Guarded(fake,converters,Path(tmp),'fake',c.SPEC,'simulator')
            result=run.program([0,0],.01,{})
            self.assertEqual(fake.calls,[]);self.assertEqual(run.skipped,1)
            self.assertFalse(result['reset_confirmed']);self.assertEqual(result['endpoint'],'reset_unconfirmed');run.log.close()

    def test_lif_one_step_and_negative_state_and_late_bn(self):
        _,_,fixture=small_inputs();bn=fixture['bntt'];bn['parameters_t12x64']['bias'][10][0]=2.
        result=temporal_response([-1.]+[0.]*11,bn,0)
        self.assertLess(result[0]['lif_after_reset'],0);self.assertEqual(result[10]['lif_spike'],1)
        bn['parameters_t12x64']['bias'][0][0]=1.
        self.assertEqual(temporal_response([0.]*12,bn,0)[0]['lif_spike'],1)
        self.assertIsNone(temporal_response([None]+[0.]*11,bn,0))

    def test_duplicate_or_missing_partial_rejected(self):
        f,tile,fixture=small_inputs()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            run=c.TileRun(FakeConnector(),converters,Path(tmp),'fake',c.SPEC,'simulator')
            result=run.collect_tile(f,tile,fixture)
        routing=dict(filters=[f])
        self.assertEqual(len(aggregate([result],routing,fixture)['trajectories']),50)
        with self.assertRaises(ValueError):aggregate([result,result],routing,fixture)
        result['partials'][-1]=result['partials'][0]
        with self.assertRaises(ValueError):aggregate([result],routing,fixture)

    def test_cli_no_io_and_forwarding(self):
        import experiment
        with patch.object(c,'native_connector',side_effect=AssertionError('No I/O')):
            self.assertEqual(c.main([]),0)
        with patch.object(c,'main',return_value=0) as routed:
            self.assertEqual(experiment.main(['--cifar-conv1','--dry-run']),0)
            routed.assert_called_once_with(['--dry-run','--stage','pilot'])
        with self.assertRaises(SystemExit):experiment.main(['--cifar-stage','remaining','--run-simulator'])

    def test_pilot_import_rejects_missing_or_changed_measured_data(self):
        _,routing,fixture=c.load_inputs();f,tile=c.tiles(routing)[0]
        response={}
        for s in c.slots(tile):
            delta=round(s['trained_weight']*1000)
            response[tuple(s['positive_cell'])]=500+delta
            response[tuple(s['negative_cell'])]=500-delta
        class PilotFake(FakeConnector):
            def impact(self,task):
                if task['mode_flag']==7:return super().impact(task)
                self.calls.append(copy.deepcopy(task));bl=next((i for i,v in enumerate(task['vol']) if v),None)
                return 1.,0,0 if bl is None else response[task['wl'],bl]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            directory=Path(tmp);folder=directory/'tiles/f0_t0';folder.mkdir(parents=True)
            run=c.TileRun(PilotFake(),converters,folder,'pilot',c.SPEC,'simulator')
            result=run.collect_tile(f,tile,fixture)
            self.assertTrue(result['decode_usable'])
            for name in ('summary.json','protocol.json'):(directory/name).write_text('{}')
            (directory/'board.ini').write_text('fixture')
            manifest=dict(session_id='pilot',stage='pilot',status='completed',source='simulator',source_sha256={},
                result_sha256={p.relative_to(directory).as_posix():c.sha256(p) for p in directory.rglob('*') if p.is_file()})
            def save(): (directory/'manifest.json').write_text(json.dumps(manifest))
            save();relative=directory.relative_to(c.ROOT).as_posix()
            imported,parent=c.verify_pilot(relative,'simulator',{},fixture,routing)
            self.assertEqual(len(imported['partials']),600);self.assertTrue(parent['not_new_measurement'])
            with self.assertRaises(ValueError):c.verify_pilot(relative,'hardware',{},fixture,routing)
            with self.assertRaises(ValueError):c.verify_pilot(relative,'simulator',{'changed':'source'},fixture,routing)
            path=folder/'tile.json';original=path.read_bytes();changed=json.loads(original);changed['partials'].pop()
            path.write_text(json.dumps(changed));manifest['result_sha256']['tiles/f0_t0/tile.json']=c.sha256(path);save()
            with self.assertRaises(ValueError):c.verify_pilot(relative,'simulator',{},fixture,routing)
            path.write_bytes(original);manifest['result_sha256']['tiles/f0_t0/tile.json']=c.sha256(path)
            raw=folder/'raw.jsonl';lines=raw.read_text().splitlines();raw.write_text('\n'.join(lines[:-1])+'\n')
            manifest['result_sha256']['tiles/f0_t0/raw.jsonl']=c.sha256(raw);save()
            with self.assertRaises(ValueError):c.verify_pilot(relative,'simulator',{},fixture,routing)


if __name__=='__main__':unittest.main()
