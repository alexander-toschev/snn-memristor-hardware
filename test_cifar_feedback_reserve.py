"""Offline FakeConnector tests; neither native simulator nor board is opened."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cifar_feedback_reserve as r
from test_cifar_feedback_adaptive import QuietSubsetFake
from test_cifar_feedback import converters


class ReserveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan,cls.routing,cls.fixture=r.load_inputs()
        cls.f=cls.routing['filters'][3];cls.parent=cls.f['tiles'][4]

    def test_exact_budget_pool_and_fresh_priority(self):
        pools=r.reserve_pools(self.parent)
        self.assertEqual([len(pools[s]) for s in ('positive_cell','negative_cell')],[16,17])
        self.assertEqual(244*33+811*13+r.adaptive.measurement_tail(r.slots(self.parent),self.fixture),24323)
        self.assertEqual(55*33+109*13,3232)
        selected=r.select_route(pools,pools)
        self.assertEqual(selected['positive_cell'][:2],[[6,2],[6,11]])
        self.assertEqual(selected['negative_cell'][-1],[7,15])
        self.assertEqual(len({tuple(c) for cells in selected.values() for c in cells}),26)
        eligible=copy.deepcopy(pools);eligible['positive_cell'].remove([6,2])
        selected=r.select_route(pools,eligible)
        self.assertNotIn([6,2],selected['positive_cell']);self.assertEqual(selected['positive_cell'][0],[6,11])

    def test_required_reserve_and_both_capacities_gate_before_any_SET(self):
        pools=r.reserve_pools(self.parent)
        for blocked in ([[7,15]],pools['negative_cell'][:5],pools['positive_cell'][:4]):
            with self.subTest(blocked=blocked),tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
                fake=QuietSubsetFake(blocked);run=r.ReserveRun(fake,converters,Path(tmp),'fake',self.plan,'simulator')
                try:
                    with self.assertRaisesRegex(ValueError,'reserve_route_gate_failed_before_SET'):
                        run.collect_reserve(self.f,self.parent,self.fixture)
                finally:run.log.close()
                self.assertFalse(any(t['mode_flag']==7 and t['vol'] and t['sign']==1 for t in fake.calls))
                saved=json.loads((Path(tmp)/'tile.json').read_text())
                self.assertEqual(len(saved['reset_candidates']),33);self.assertEqual(saved['partials'],[])
                self.assertFalse(saved['technical_completed']);self.assertFalse((Path(tmp)/'frozen_route.json').exists())
                self.assertTrue((Path(tmp)/'raw_parts.json').exists())

    def test_actual_fake_full13_freeze_600_and_fixed_decoder(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            fake=QuietSubsetFake([]);run=r.ReserveRun(fake,converters,Path(tmp),'fake',self.plan,'simulator')
            try:result=run.collect_reserve(self.f,self.parent,self.fixture)
            finally:run.log.close()
            self.assertTrue(result['technical_completed']);self.assertEqual(result['assigned_taps'],list(range(52,65)))
            self.assertEqual(result['effective_route'][-1]['negative_cell'],[7,15])
            self.assertEqual([s['positive_cell'] for s in result['effective_route'][:2]],[[6,2],[6,11]])
            self.assertEqual(len(result['partials']),600);self.assertEqual(len(result['reset_candidates']),33)
            self.assertLessEqual(run.sequence,24323);self.assertLessEqual(run.pulses,3232)
            self.assertFalse(result['physical_weight_quality_pass']) # diagnostic data are retained
            for p in result['partials']:
                self.assertEqual(p['trained_unit_contribution'],r.decode_measured(result['matrix']['matrix'],p['raw_row_sums_v'],.03))
            index=json.loads((Path(tmp)/'raw_parts.json').read_text())
            events=[json.loads(line) for part in index['parts'] for line in (Path(tmp)/part['path']).read_text().splitlines()]
            frozen=next(i for i,e in enumerate(events) if e['event']=='calibration_frozen')
            first=next(i for i,e in enumerate(events) if e.get('phase')=='cifar')
            self.assertLess(frozen,first)
            self.assertTrue(all(t['mode_flag'] in (7,10) for t in fake.calls))
            self.assertTrue(all(t['id']==0 for t in fake.calls if t['mode_flag']==10))
            # Distinct logical tap64 really uses the reserved physical negative cell.
            self.assertTrue(any(t['mode_flag']==10 and t['wl']==7 and t['vol'][15]==.1 for t in fake.calls))

    def test_bound_parent287_and_exact_replacement_aggregate(self):
        for source in ('hardware','simulator'):
            binding=self.plan['parent_bindings'][source]
            chunks,metadata=r.verify_parent(binding['session'],source,self.routing,self.fixture,self.plan)
            self.assertEqual(sum(len(c['result']['effective_route']) for c in chunks),287)
            self.assertFalse(any((c['filter_id'],c['parent_tile_id'])==(3,4) for c in chunks))
            self.assertEqual(metadata['ancestry42']['imported_trained_coefficients'],42)
            self.assertTrue(all(c['origin']['reused'] and 'ancestry_original_origin' in c['origin'] for c in chunks))
            parent=json.loads((r.local_path(binding['session'])/'completed_microtiles.json').read_text())
            old=[c for c in parent if (c['filter_id'],c['parent_tile_id'])==(3,4)]
            # Reassemble only for arithmetic/coverage regression; this is not new acquisition.
            out=r.adaptive.aggregate_microtiles(chunks+old,self.routing,self.fixture)
            self.assertEqual(len(out['trajectories']),200)
            self.assertEqual(sum(len(x['actual']) for x in out['trajectories']),2400)
            with self.assertRaises(ValueError):r.adaptive.aggregate_microtiles(chunks+old+old,self.routing,self.fixture)
            with self.assertRaises(ValueError):r.adaptive.aggregate_microtiles(chunks,self.routing,self.fixture)
            wrong=copy.deepcopy(self.plan);wrong['parent_bindings'][source]['audit_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                r.verify_parent(binding['session'],source,self.routing,self.fixture,wrong)
            with self.assertRaises(ValueError):r.verify_parent(binding['session'],'hardware' if source=='simulator' else 'simulator',self.routing,self.fixture,self.plan)

    def test_no_default_io_explicit_route_and_dry_run(self):
        import experiment
        with patch.object(r,'main',return_value=0) as launch,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            experiment.main([]);launch.assert_not_called()
            experiment.main(['--cifar-feedback-reserve','--dry-run'])
            launch.assert_called_once_with(['--dry-run'])
            for args in (['--cifar-feedback-reserve'],['--cifar-feedback-reserve','--dry-run','--cifar-stage','pilot'],
                         ['--cifar-feedback-reserve','--cifar-feedback-adaptive','--dry-run']):
                with self.assertRaises(SystemExit):experiment.main(args)
        # Exercise actual dry-run control flow without pretending private driver files exist locally.
        with patch.object(r,'verify_parent',return_value=([],{})),patch.object(r,'check_sources',return_value=(None,b'',{})),\
             patch.object(r,'native_connector',side_effect=AssertionError('dry-run cannot open Connector')),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(r.main(['--dry-run']),0)
            description=json.loads(output.getvalue());self.assertFalse(description['hardware_access'])
            self.assertEqual(description['maximum_impacts'],24323)


if __name__=='__main__':unittest.main()
