"""Offline protocol/guard/evidence tests; no native SIM or board connection."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'.deps'))
sys.path.insert(0, str(ROOT/'vendor/MemriBoard'))
from manager.service import converters
from experiments.survey256_20261019 import experiment as survey


class FakeConnector:
    cb_type = 'real'

    def __init__(self, stuck_reset=False, invalid=False, high_g=False, fail_set=False):
        self.commands = []
        self.closed = 0
        self.q = {}
        self.stuck_reset, self.invalid, self.high_g, self.fail_set = stuck_reset, invalid, high_g, fail_set

    def open_port(self):
        return True

    def close_port(self):
        self.closed += 1
        return True

    def impact(self, task):
        self.commands.append(copy.deepcopy(task))
        if task['mode_flag'] == 10:
            # Descriptive linear one-hot response, not a native simulator.
            return 1e-9, 0, round(sum(task['vol'])*1000)
        key = task['wl'], task['bl']
        self.q.setdefault(key, 50. if self.high_g else .06)
        if task['vol']:
            if task['sign'] == 0 and not self.stuck_reset:
                self.q[key] = .004
            if task['sign'] == 1:
                if self.fail_set:
                    raise TimeoutError('Write may have reached device')
                self.q[key] = .051
        adc = 0 if self.invalid else converters.convert_res_to_adc(11, 3000, .3, 14, 5, 10, 330/self.q[key])
        native = converters.convert_adc_to_res(11, 3000, .3, 14, 5, 10, adc)
        return native, task['id'], adc


class SurveyTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'.tmp').mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT/'.tmp')
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)

    def worker(self, fake):
        return survey.Survey(fake, converters, self.output, 'offline-test', copy.deepcopy(survey.SPEC), 'hardware')

    def run_small(self, fake, cells=((0, 0), (7, 31))):
        worker = self.worker(fake)
        with patch.object(survey, 'CELLS', cells), contextlib.redirect_stdout(io.StringIO()):
            result = worker.run()
        return result

    def test_full_pre_array_before_first_write_and_final_mvm_after_last(self):
        fake = FakeConnector()
        result = self.run_small(fake)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['impact_attempts'], 2*75+4)
        self.assertEqual(result['programming_pulse_attempts'], 4)
        first = next(i for i, t in enumerate(fake.commands) if t['mode_flag'] == 7 and t['vol'])
        self.assertEqual(first, 30+15)
        self.assertEqual(len(result['initial_array']), 2)
        self.assertEqual(len(result['final_array']), 2)
        self.assertTrue(all(len(s['mvm']) == 12 for s in result['final_array']))
        self.assertEqual(fake.closed, 1)
        self.assertTrue((self.output/'cell_map.html').exists())
        self.assertTrue(all(t.get('pulse_width', 0) == 0 for t in fake.commands))

    def test_failed_reset_endpoint_does_not_block_set_and_q_high_still_excited(self):
        fake = FakeConnector(stuck_reset=True)
        result = self.run_small(fake, ((0, 0),))
        self.assertEqual(result['status'], 'completed')
        phases = result['cells'][0]['programming']
        self.assertEqual(phases['reset']['admitted_pulses'], 55)
        self.assertFalse(phases['reset']['endpoint_reached'])
        self.assertEqual(phases['set']['admitted_pulses'], 1)
        self.assertTrue(phases['set']['endpoint_reached'])
        report = json.loads((self.output/'cell_map.json').read_text())
        self.assertFalse(report['cells'][0]['automatically_excluded'])

    def test_invalid_initial_adc_globally_stops_with_full_map(self):
        fake = FakeConnector(invalid=True)
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.worker(fake).run()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['programming_pulse_attempts'], 0)
        rows = json.loads((self.output/'cell_map.json').read_text())['cells']
        self.assertEqual(len(rows), 256)
        self.assertEqual(rows[0]['status'], 'incomplete')
        self.assertEqual(rows[-1]['status'], 'not_measured')
        self.assertEqual(result['failed_checkpoints'][0]['reads'], [])
        records = [json.loads(line) for line in (self.output/'raw.jsonl').read_text().splitlines()]
        self.assertEqual(records[-1]['raw_adc'], 0)
        self.assertEqual(fake.closed, 1)

    def test_model_guard_keeps_reads_skips_mvm_then_all_current_decisions(self):
        fake = FakeConnector(stuck_reset=True, high_g=True)
        result = self.run_small(fake, ((0, 0),))
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(all(t['mode_flag'] == 7 for t in fake.commands))
        self.assertEqual(result['programming_pulse_attempts'], 0)
        self.assertEqual(result['programming_skipped_current'], 110)
        self.assertEqual(result['initial_array'][0]['mvm_status'], 'skipped_model_guard')
        self.assertEqual(len(result['initial_array'][0]['reads']), 3)

    def test_transport_failure_counts_attempt_and_preserves_partial_evidence(self):
        fake = FakeConnector(fail_set=True)
        result = self.run_small(fake, ((0, 0),))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['cells'][0]['programming']['set']['admitted_pulses'], 1)
        self.assertEqual(result['final_array'], [])
        records = [json.loads(line) for line in (self.output/'raw.jsonl').read_text().splitlines()]
        self.assertIn('Write may have reached', records[-1]['error'])
        self.assertEqual(fake.closed, 1)

    def test_guard_boundaries_and_budget_before_io(self):
        worker = self.worker(FakeConnector())
        try:
            worker.last_mode7[0, 0] = (500., 99)
            admitted = worker.impact(0, 0, 7, 1., 'set', sign=1)
            self.assertEqual(admitted['predicted_current_a'], .002)
            worker.last_mode7[0, 0] = (500., 99)
            skipped = worker.impact(0, 0, 7, 1.001, 'set', sign=1)
            self.assertIsNone(skipped)
            self.assertEqual(worker.last_mode7[0, 0], (500., 99))
            worker.sequence = survey.SPEC['maximum_impacts']
            with self.assertRaisesRegex(ValueError, 'impact budget'):
                worker.impact(0, 0, 7, 0., 'read')
        finally:
            worker.log.close()

    def test_frozen_config_geometry_exact_budget_and_no_default_io(self):
        configured = json.loads((ROOT/survey.CONFIG).read_text())
        survey.validate(configured)
        self.assertEqual(len(configured['cells']), 256)
        self.assertEqual(len(set(map(tuple, configured['cells']))), 256)
        self.assertEqual(configured['ramp_v'][0], .3)
        self.assertEqual(configured['ramp_v'][-1], 3.)
        self.assertEqual(len(configured['ramp_v']), 55)
        self.assertEqual(configured['maximum_impacts'], 256*(5*(3+3*4)+2*55))
        configured['ramp_v'][0] = .35
        with self.assertRaises(ValueError):
            survey.validate(configured)
        with patch.object(survey, 'check_sources') as check, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(survey.main([]), 0)
            check.assert_not_called()

    def test_zero_drift_is_flagged_without_excluding_cell(self):
        fake = FakeConnector()
        result = self.run_small(fake, ((0, 0),))
        final = result['final_array'][0]
        for row in final['mvm']:
            if row['position'] == 3:
                row['raw_adc'] += 20
        rows = survey.build_map(result)
        self.assertTrue(rows[0]['final']['zero_shift_flag'])
        self.assertIn('final_zero_baseline_shift', rows[0]['reasons'])
        self.assertFalse(rows[0]['automatically_excluded'])


if __name__ == '__main__':
    unittest.main()
