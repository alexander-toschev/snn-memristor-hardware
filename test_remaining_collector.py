"""Draft acquisition tests use FakeConnector only; production entrypoint stays blocked."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cifar_feedback_remaining_reserves as r
from test_cifar_feedback_adaptive import QuietSubsetFake
from test_cifar_feedback import converters


def totals():return dict(impact_attempts=0,programming_pulse_attempts=0,programming_decisions=0,programming_skipped_current=0)


def synthetic_parent(routing):
    chunks=[]
    for f,parent in r.adaptive.tiles(routing):
        active=r.slots(parent)
        result=dict(filter_id=f['filter_id'],tile_id=2*parent['tile_id'],technical_completed=True,
            effective_route=active,physical_weight_quality_pass=False,calibration=dict(effective_weights=[0.]*len(active)),
            partials=[dict(case=c,repeat=repeat,t=t,trained_unit_contribution=0.) for c in range(10) for repeat in range(5) for t in range(12)])
        chunks.append(dict(filter_id=f['filter_id'],parent_tile_id=parent['tile_id'],microtile_id=0,result=result,
            origin=dict(session='synthetic_parent',reused=False,new_measurement=False)))
    return chunks


class CollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.original_plan,cls.routing,cls.fixture=r.adaptive.load_inputs()

    def test_combined20_actual_fake_partials_and_exact280_composition(self):
        old=synthetic_parent(self.routing);chunks=r.import_other280(old,'synthetic_parent','test_only_digest')
        self.assertEqual(sum(len(c['result']['effective_route']) for c in chunks),280)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp,contextlib.redirect_stdout(io.StringIO()):
            output=Path(tmp);counter=totals();fake=QuietSubsetFake([])
            with patch.object(r,'ROOT',output):
                r.collect_blocks(fake,converters,output,'synthetic_BC',r.SPEC,'simulator',self.routing,self.fixture,chunks,counter)
            fresh=[c for c in chunks if not c['origin']['reused']]
            self.assertEqual([len(c['result']['effective_route']) for c in fresh],[11,9])
            self.assertEqual([len(c['result']['partials']) for c in fresh],[600,600])
            self.assertEqual(counter['impact_attempts'],len(fake.calls));self.assertLessEqual(counter['impact_attempts'],40378)
            self.assertLessEqual(counter['programming_pulse_attempts'],5260)
            for block,c in zip(r.BLOCKS,fresh):
                result=c['result'];route=result['effective_route']
                self.assertTrue(all(d['actual_cifar_assigned'] for d in result['preferred_disposition']))
                for short in ('positive','negative'):
                    for tap,cell in block['preferred_'+short].items():
                        self.assertEqual(route[tap][short+'_cell'],cell)
                folder=output/Path(c['origin']['path']).parent
                index=json.loads((folder/'raw_parts.json').read_text())
                rows=[json.loads(line) for part in index['parts'] for line in (folder/part['path']).read_text().splitlines()]
                freeze=next(i for i,row in enumerate(rows) if row['event']=='calibration_frozen')
                first=next(i for i,row in enumerate(rows) if row.get('phase')=='cifar')
                self.assertLess(freeze,first)
            aggregate=r.adaptive.aggregate_microtiles(chunks,self.routing,self.fixture)
            self.assertEqual(len(aggregate['trajectories']),200)
            self.assertEqual(sum(len(t['actual']) for t in aggregate['trajectories']),2400)
            with self.assertRaises(ValueError):r.adaptive.aggregate_microtiles(chunks+old,self.routing,self.fixture)
            fresh[-1]['result']['partials'].pop()
            with self.assertRaises(ValueError):r.adaptive.aggregate_microtiles(chunks,self.routing,self.fixture)

    def test_nonquiet_preferred_has_measured_fallback_and_retained_disposition(self):
        block=r.BLOCKS[0];f=self.routing['filters'][2];parent=f['tiles'][0]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            fake=QuietSubsetFake([[4,7],[5,15]]);run=r.RemainingReserveRun(fake,converters,Path(tmp),'synthetic',r.SPEC,'simulator')
            try:result=run.collect_block(f,parent,block,self.fixture)
            finally:run.log.close()
            self.assertTrue(result['technical_completed']);self.assertEqual(len(result['partials']),600)
            negative=[d for d in result['preferred_disposition'] if d['cell'] in ([4,7],[5,15])]
            self.assertTrue(all(d['status']=='screened_not_quiet' and not d['actual_cifar_assigned'] for d in negative))
            self.assertNotIn([4,7],[s['positive_cell'] for s in result['effective_route']])
            self.assertNotIn([5,15],[s['negative_cell'] for s in result['effective_route']])
            self.assertEqual(result['effective_route'][0]['negative_cell'],[5,6])

    def test_B_capacity_failure_no_B_SET_but_independent_C_600_no_retry(self):
        pools=r.candidate_pools(r.BLOCKS[0]);fake=QuietSubsetFake(pools['positive_cell'][:4])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            output=Path(tmp);counter=totals();chunks=[]
            with patch.object(r,'ROOT',output),contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(r.RouteCapacityError,'incomplete B/C'):
                r.collect_blocks(fake,converters,output,'synthetic',r.SPEC,'simulator',self.routing,self.fixture,chunks,counter)
            self.assertFalse(any(t['mode_flag']==7 and t['vol'] and t['sign']==1 and t['wl'] in (4,5) for t in fake.calls))
            self.assertEqual(len(chunks),1);self.assertEqual(chunks[0]['filter_id'],1)
            self.assertEqual(len(chunks[0]['result']['partials']),600)
            result=json.loads((output/'tiles/f2_p0_reserve_B/tile.json').read_text())
            self.assertTrue(all(d['status']=='quiet_but_route_capacity_abort' for d in result['preferred_disposition']))
            self.assertEqual(result['partials'],[]);self.assertEqual(counter['impact_attempts'],len(fake.calls))
            self.assertEqual(len(result['reset_candidates']),29)
            self.assertLessEqual(counter['impact_attempts'],40378)

    def test_noncapacity_ValueError_stops_before_C(self):
        class Invalid(QuietSubsetFake):
            def impact(self,task):raise ValueError('synthetic invalid ADC/native R')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            output=Path(tmp);counter=totals()
            with patch.object(r,'ROOT',output),self.assertRaisesRegex(ValueError,'invalid ADC'):
                r.collect_blocks(Invalid([]),converters,output,'synthetic',r.SPEC,'simulator',self.routing,self.fixture,[],counter)
            self.assertFalse((output/'tiles/f1_p0_reserve_C').exists())
            self.assertEqual(counter['impact_attempts'],1)

    def test_transport_failure_retains_unassessed_and_closes_journal(self):
        class Broken(QuietSubsetFake):
            def impact(self,task):raise ConnectionError('synthetic transport lost')
        block=r.BLOCKS[0];f=self.routing['filters'][2];parent=f['tiles'][0]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            run=r.RemainingReserveRun(Broken([]),converters,Path(tmp),'synthetic',r.SPEC,'simulator')
            with self.assertRaises(ConnectionError):run.collect_block(f,parent,block,self.fixture)
            self.assertTrue(run.log.closed)
            result=json.loads((Path(tmp)/'tile.json').read_text())
            self.assertTrue(all(d['status']=='technical_unassessed' and d['quiet'] is None for d in result['preferred_disposition']))
            self.assertEqual(result['impact_attempts'],1)

    def test_failedA_counter_added_once_without_its_parent_cumulative(self):
        parent=dict(cumulative_impacts_including_all_parent_attempts=1000,cumulative_pulses_including_all_parent_attempts=100)
        failed=dict(status='failed',parent_import=dict(session='baseline'),new_counters=dict(impact_attempts=30,programming_pulse_attempts=3),
                    cumulative_impacts_including_all_ancestry=1030,cumulative_pulses_including_all_ancestry=103)
        out=r.historical_accounting('baseline',parent,('failed_A',failed))
        self.assertEqual((out['parent_cumulative_impacts'],out['parent_cumulative_pulses']),(1030,103))
        self.assertEqual(len(out['historical_attempt_ledger']),2)
        with self.assertRaises(ValueError):r.historical_accounting('baseline',parent,('baseline',failed))
        wrong=copy.deepcopy(failed);wrong['parent_import']['session']='unrelated'
        with self.assertRaises(ValueError):r.historical_accounting('baseline',parent,('failed_A',wrong))

    def test_main_failure_manifest_counts_and_close_without_native_backend(self):
        class ManagedFake(QuietSubsetFake):
            def __init__(self):super().__init__([]);self.closed=0;self.opened=0
            def open_port(self):self.opened+=1;return True
            def close_port(self):self.closed+=1;return True
        fake=ManagedFake()
        parent=dict(parent_cumulative_impacts=100,parent_cumulative_pulses=10)
        def fail_after_one(connector,converters,output,sid,plan,source,routing,fixture,chunks,counter):
            counter['impact_attempts']=1
            raise ConnectionError('synthetic response lost')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            with patch.object(r,'ROOT',root),patch.object(r,'local_path',side_effect=lambda name:root/name), \
                 patch.object(r,'sha256',return_value='synthetic_test_only'), \
                 patch.object(r,'load_inputs',return_value=(r.SPEC,self.routing,self.fixture)), \
                 patch.object(r,'verify_parent',return_value=([],parent)), \
                 patch.object(r,'check_sources',return_value=(None,b'[board]\n',{})), \
                 patch.object(r,'launch_provenance',return_value={'synthetic_test_only':True}), \
                 patch.object(r,'native_connector',return_value=fake), \
                 patch.object(r,'collect_blocks',side_effect=fail_after_one):
                self.assertEqual(r.main(['--run-hardware']),2)
            self.assertEqual((fake.opened,fake.closed),(1,1))
            manifest=json.loads(next((root/'results').rglob('manifest.json')).read_text())
            self.assertEqual(manifest['status'],'failed')
            self.assertEqual(manifest['new_counters']['impact_attempts'],1)
            self.assertEqual(manifest['cumulative_impacts_including_all_ancestry'],101)
            self.assertIn('imported_microtiles.json',manifest['result_sha256'])
            self.assertFalse((root/r.LOCK).exists())

    def test_draft_cannot_open_native_backend_and_missing_import_rejected(self):
        with patch.object(r,'native_connector',side_effect=AssertionError('no backend')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(r.main([]),0)
            with self.assertRaisesRegex(ValueError,'Draft'):r.main(['--run-simulator'])
            with self.assertRaisesRegex(ValueError,'Draft'):r.main(['--run-hardware'])
            with self.assertRaisesRegex(ValueError,'Draft'):r.main(['--dry-run'])
        old=synthetic_parent(self.routing)
        with self.assertRaises(ValueError):r.import_other280(old[:-1],'synthetic','digest')


if __name__=='__main__':unittest.main()
