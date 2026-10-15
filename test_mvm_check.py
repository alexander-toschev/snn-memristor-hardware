"""Offline diagnostic tests with the real converters and a fake public Connector."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
for relative in (".deps", "vendor/MemriBoard"):
    if (ROOT / relative).is_dir():
        sys.path.insert(0, str(ROOT / relative))
from manager.service import converters
import experiment
import mvm_check


class FakeConnector:
    def __init__(self, fail=False, bad_id=False, high_read=False, gui=False):
        self.calls = []
        self.fail, self.bad_id, self.high_read = fail, bad_id, high_read
        self.opened = self.closed = 0
        self.gui = gui

    def open_port(self):
        self.opened += 1
        return True

    def close_port(self):
        self.closed += 1
        return True

    def impact(self, task):
        assert task['mode_flag'] in (9, 10)
        assert task['wl'] in (range(8) if self.gui else (0, 2))
        self.calls.append(json.loads(json.dumps(task)))
        if self.fail and len(self.calls) == 14:
            raise TimeoutError('response missing')
        if task['mode_flag'] == 9:
            assert task['bl'] in (range(32) if self.gui else (2, 3)) and task['vol'] == .1
            return (1000, 99 if self.bad_id else 0, self.high_read if self.high_read else 500)
        vector = task['vol']
        assert len(vector) == 32 and set(vector) <= {0.0, .1}
        assert self.gui or all(v == 0 or i in (2, 3) for i, v in enumerate(vector))
        if self.gui:
            assert set(task) == {'mode_flag', 'vol', 'id', 'wl'} and task['id'] == 0
        adc = 100 * sum(v != 0 for v in vector)
        vector[:] = [82] * 32  # Native SI conversion is permitted to mutate its input.
        return (1e-9, task['id'], adc)


class DiagnosticTests(unittest.TestCase):
    def run_fake(self, fake, *, gui_ordered=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        summary = mvm_check.collect(fake, converters, output, 'offline', gui_ordered=gui_ordered)
        raw = [json.loads(line) for line in (output/'raw.jsonl').read_text().splitlines()]
        return summary, raw

    def test_64_calls_modes_order_zero_id_mutation_and_statistics(self):
        fake = FakeConnector()
        summary, raw = self.run_fake(fake)
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual((fake.opened, fake.closed), (1, 1))
        self.assertEqual(len(fake.calls), 64)
        self.assertEqual(sum(c['mode_flag']==9 for c in fake.calls), 24)
        self.assertEqual(sum(c['mode_flag']==10 for c in fake.calls), 40)
        self.assertEqual([c['id'] for c in fake.calls], list(range(1,65)))
        self.assertEqual(len(raw), 128)
        self.assertTrue(all(r['returned_id']==0 for r in summary['readbacks']))
        self.assertEqual([r['vector'] for r in summary['mvm'][:8]], ['zero','A','B','AB']*2)
        self.assertTrue(all(set(r['command_si']['vol']) <= {0.0,.1} for r in summary['mvm']))
        self.assertEqual(summary['row_statistics'][0]['vectors']['zero']['adc']['mean'], 0)
        self.assertEqual(summary['row_statistics'][0]['offset_adjusted_superposition_epsilon']['adc'], 0)
        self.assertTrue(all(r['mean_delta_siemens']==0 for r in summary['conductance_comparison']))
        self.assertIsNone(summary['acceptance_threshold'])

    def test_response_failure_is_flushed_and_stops(self):
        fake = FakeConnector(fail=True)
        summary, raw = self.run_fake(fake)
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['impact_attempts'], 14)
        self.assertEqual(summary['valid_responses'], 13)
        self.assertEqual(raw[-2]['event'], 'request')
        self.assertEqual(raw[-1]['status'], 'error')
        self.assertIn('response missing', raw[-1]['error'])
        self.assertEqual(fake.closed, 1)

    def test_mode9_id_must_be_native_zero(self):
        summary, raw = self.run_fake(FakeConnector(bad_id=True))
        self.assertEqual(summary['impact_attempts'], 1)
        self.assertEqual(raw[-1]['returned_id'], 99)
        self.assertEqual(summary['status'], 'failed')

    def test_invalid_read_stops_before_any_mvm(self):
        fake = FakeConnector(high_read=4000)
        summary, raw = self.run_fake(fake)
        self.assertEqual(summary['status'], 'failed')
        self.assertTrue(all(c['mode_flag']==9 for c in fake.calls))
        self.assertEqual(raw[-1]['raw_adc'], 4000)

    def test_model_current_bound_stops_before_pair(self):
        fake = FakeConnector(high_read=3560)
        summary, raw = self.run_fake(fake)
        self.assertEqual(summary['status'], 'failed')
        self.assertIn('Pre-read model', summary['error'])
        self.assertTrue(all(sum(v != 0 for v in c['vol']) <= 1 for c in fake.calls if c['mode_flag']==10))

    def test_dry_run_has_no_connector_creation(self):
        with patch.object(mvm_check, 'sha256', return_value=mvm_check.READ_ALL_SHA256), \
             patch.object(mvm_check, 'preflight', return_value=(None,b'',{})), \
             patch.object(mvm_check, 'native_connector') as factory, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(mvm_check.main(['--dry-run']),0)
            factory.assert_not_called()

    def test_frontdoor_routes_and_rejects_phase_options(self):
        with patch.object(mvm_check, 'main', return_value=0) as check:
            self.assertEqual(experiment.main(['--dry-run','--check-mvm']),0)
            check.assert_called_once_with(['--dry-run'])
            check.reset_mock()
            for args in (['--check-mvm'], ['--run-simulator','--check-mvm'], ['--run-hardware','--check-mvm','--all'], ['--run-hardware','--check-mvm','--max-pulses','1']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    experiment.main(args)
            check.assert_not_called()

    def test_gui_all_cells_1736_impacts_order_id0_dense_dac(self):
        fake = FakeConnector(gui=True)
        summary, raw = self.run_fake(fake, gui_ordered=True)
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(len(fake.calls), 1736)
        self.assertEqual(len(summary['readbacks']), 1536)
        self.assertEqual(len(summary['mvm']), 200)
        self.assertEqual(len(summary['conductance_comparison']), 256)
        self.assertEqual(len(raw), 3472)
        expected = [(repeat, label, wl) for repeat in range(5)
                    for label in ('zero','A','B','AB','ALL32') for wl in range(8)]
        self.assertEqual([(r['repeat'],r['vector'],r['command_si']['wl']) for r in summary['mvm']],expected)
        self.assertTrue(all(r['command_si']['id']==0 and r['returned_id']==0 for r in summary['mvm']))
        self.assertEqual([r['task_id'] for r in raw if r['event']=='request'],list(range(1,1737)))
        dense = [r for r in summary['mvm'] if r['vector']=='ALL32']
        self.assertEqual(len(dense),40)
        self.assertTrue(all(r['command_si']['vol']==[.1]*32 and r['vol_dac']==[82]*32 for r in dense))

    def test_gui_dense_model_guard_stops_before_any_mvm(self):
        fake = FakeConnector(gui=True,high_read=3500)
        summary, raw = self.run_fake(fake,gui_ordered=True)
        self.assertEqual(summary['status'],'failed')
        self.assertIn('no MVM sent',summary['error'])
        self.assertEqual(len(fake.calls),768)
        self.assertTrue(all(c['mode_flag']==9 for c in fake.calls))
        self.assertEqual(len(summary['pre_mvm_guard_predictions']),40)

    def test_gui_dense_observed_output_stops_and_preserves_raw(self):
        class DenseOverrange(FakeConnector):
            def impact(self, task):
                dense = task['mode_flag']==10 and task['vol']==[.1]*32
                result = super().impact(task)
                return (result[0],result[1],14000) if dense else result
        fake = DenseOverrange(gui=True)
        summary, raw = self.run_fake(fake,gui_ordered=True)
        self.assertEqual(summary['status'],'failed')
        self.assertEqual(len(fake.calls),801)  # 768 pre-read +32 sparse/zero +first dense WL0.
        self.assertEqual(summary['valid_responses'],800)
        self.assertEqual(len(summary['mvm']),32)
        self.assertTrue(all(r['stage']=='before' for r in summary['readbacks']))
        self.assertEqual((fake.opened,fake.closed),(1,1))
        self.assertEqual(raw[-2]['event'],'request')
        self.assertEqual(raw[-2]['vector'],'ALL32')
        self.assertEqual(raw[-1]['status'],'error')
        self.assertEqual(raw[-1]['command_si']['wl'],0)
        self.assertEqual(raw[-1]['raw_adc'],14000)
        self.assertGreater(raw[-1]['output_voltage_v'],4)
        self.assertIn('80%',raw[-1]['error'])

    def test_gui_frontdoor_routing_and_mutual_exclusion(self):
        with patch.object(mvm_check,'main',return_value=0) as check:
            experiment.main(['--dry-run','--check-mvm-gui'])
            check.assert_called_once_with(['--dry-run','--gui-ordered'])
            check.reset_mock()
            for args in (['--check-mvm-gui'],['--run-hardware','--check-mvm','--check-mvm-gui'],['--run-hardware','--check-mvm-gui','--all']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    experiment.main(args)
            check.assert_not_called()


if __name__ == '__main__':
    unittest.main()
