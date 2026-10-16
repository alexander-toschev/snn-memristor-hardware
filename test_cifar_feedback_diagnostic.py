"""Offline diagnostic decoder and fake-acquisition regression tests."""
import contextlib
import io
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cifar_feedback_diagnostic as d
from test_cifar_feedback import FakeConnector,converters
from test_cifar_memristor import small_inputs


class WrongSignFake(FakeConnector):
    def __init__(self):super().__init__();self.written=False
    def impact(self,task):
        mode=task['mode_flag'];active=mode==10 and any(task['vol']);wl=task['wl']
        if mode==7 and task['vol'] and task['sign']==1:self.written=True
        result=super().impact(task)
        if mode==10 and self.written:return 1.,0,(50 if wl==0 else 100) if active else 0
        return result


class DiagnosticTests(unittest.TestCase):
    def test_chunk_journal_lossless_unicode_and_cross_part_pair(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder=Path(tmp)
            stream=(folder/'raw.jsonl').open('x',encoding='utf-8',newline='\n')
            journal=d.ChunkedJournal(stream,folder,200)
            lines=[json.dumps(dict(event=event,task_id=i,text='ячейка'),ensure_ascii=False)+'\n'
                   for i in range(20) for event in ('request','result')]
            for line in lines:journal.write(line);journal.flush()
            journal.close();journal.close()
            index=json.loads((folder/'raw_parts.json').read_text(encoding='utf-8'))
            rebuilt=b''.join((folder/p['path']).read_bytes() for p in index['parts'])
            self.assertEqual(rebuilt,''.join(lines).encode('utf-8'))
            self.assertEqual(index['sha256'],hashlib.sha256(rebuilt).hexdigest())
            self.assertEqual(index['records'],40)
            for part in index['parts']:
                data=(folder/part['path']).read_bytes()
                self.assertLessEqual(len(data),200);self.assertEqual(part['sha256'],hashlib.sha256(data).hexdigest())
            self.assertTrue(any(json.loads((folder/p['path']).read_text(encoding='utf-8').splitlines()[-1])['event']=='request' for p in index['parts'][:-1]))

    def test_fixed_decoder_negative_fit_is_not_abs_or_gain_fit(self):
        cal=d.fixed_calibration([.1,.05],[[-.1,-.05]]*5,1.,.00155)
        self.assertLess(cal['fitted_alpha_diagnostic']['alpha_v_per_weight'],0)
        self.assertTrue(cal['decode_usable']);self.assertEqual(cal['effective_weights'],[-.1,-.05])
        self.assertEqual(d.decode_measured([[1.,0.],[0.,1.]],[.1,.3],1.),-.19999999999999998)
        self.assertEqual(d.decode_measured([[2.,0.],[0.,.5]],[.1,.3],.03),(.2-.15)/.03)
        for gain in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):d.decode_measured([[1.,0.],[0.,1.]],[.1,.3],gain)

    def test_quality_failure_and_negative_alpha_still_acquire_cifar(self):
        f,tile,fixture=small_inputs();tile['slots'][1]['trained_weight']=.05
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            run=d.DiagnosticRun(WrongSignFake(),converters,Path(tmp),'fake',d.SPEC,'simulator')
            result=run.collect_tile(f,tile,fixture)
            self.assertTrue(result['technical_completed']);self.assertFalse(result['physical_weight_quality_pass'])
            self.assertLess(result['calibration']['fitted_alpha_diagnostic']['alpha_v_per_weight'],0)
            self.assertEqual(len(result['partials']),600)
            self.assertTrue(all(p['trained_unit_contribution'] is not None for p in result['partials']))
            for part in result['partials']:
                self.assertEqual(part['trained_unit_contribution'],
                    d.decode_measured(result['matrix']['matrix'],part['raw_row_sums_v'],.03))
            raw=[json.loads(line) for line in (Path(tmp)/'raw.jsonl').read_text(encoding='utf-8').splitlines()]
            freeze=next(i for i,r in enumerate(raw) if r['event']=='calibration_frozen')
            first=next(i for i,r in enumerate(raw) if r.get('phase')=='cifar')
            self.assertLess(freeze,first)
            coverage=d.address_coverage([result])
            self.assertEqual(coverage['reset_probed']['count'],10)
            self.assertEqual(coverage['represented']['count'],4)
            saved=result['partials'][0]['trained_unit_contribution']
            result['calibration']['fitted_alpha_diagnostic']['alpha_v_per_weight']=999.
            self.assertEqual(result['partials'][0]['trained_unit_contribution'],saved)

    def test_exact_fresh_coverage_and_budgets(self):
        _,routing,fixture=d.load_inputs();selected=d.tiles(routing);budget=d.counts(selected,fixture)
        self.assertEqual((budget['maximum_impacts'],budget['maximum_programming_pulses']),(580210,74940))
        self.assertEqual((len(selected),budget['trained_coefficients']),(28,300))
        records=[]
        for f,tile in selected:
            active=d.slots(tile)
            records.append(dict(filter_id=f['filter_id'],tile_id=tile['tile_id'],
                calibration=dict(effective_weights=[0.]*len(active)),
                partials=[dict(case=ci,repeat=r,t=t,trained_unit_contribution=0.)
                    for ci in range(10) for r in range(5) for t in range(12)]))
        summary=d.aggregate(records,routing,fixture)
        self.assertEqual(len(summary['trajectories']),200)
        self.assertEqual(sum(len(r['actual']) for r in summary['trajectories']),2400)
        self.assertTrue(all(len(r['actual'])==12 for r in summary['trajectories']))
        records[-1]['partials'].pop()
        with self.assertRaises(ValueError):d.aggregate(records,routing,fixture)

    def test_insufficient_route_aborts_preserving_raw(self):
        f,tile,fixture=small_inputs()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            run=d.DiagnosticRun(FakeConnector(initial=100,reset_step=0),converters,Path(tmp),'fake',d.SPEC,'simulator')
            with self.assertRaisesRegex(ValueError,'insufficient_quiet'):run.collect_tile(f,tile,fixture)
            result=json.loads((Path(tmp)/'tile.json').read_text(encoding='utf-8'))
            self.assertFalse(result['technical_completed']);self.assertEqual(result['partials'],[])
            self.assertTrue((Path(tmp)/'raw.jsonl').stat().st_size>0)

    def test_no_default_io_and_no_pilot_import_route(self):
        import experiment
        with patch.object(d,'main',return_value=0) as launch,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            experiment.main([]);launch.assert_not_called()
            experiment.main(['--cifar-feedback-diagnostic','--dry-run'])
            launch.assert_called_once_with(['--dry-run'])
            for args in (['--cifar-feedback-diagnostic'],['--cifar-feedback-diagnostic','--dry-run','--cifar-stage','pilot'],
                         ['--cifar-feedback-diagnostic','--dry-run','--cifar-pilot-session','old'],
                         ['--cifar-feedback-diagnostic','--cifar-feedback','--dry-run']):
                with self.assertRaises(SystemExit):experiment.main(args)
        with patch.object(d,'native_connector',side_effect=AssertionError('No backend')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(d.main([]),0)


if __name__=='__main__':unittest.main()
