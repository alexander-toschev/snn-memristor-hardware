from dataclasses import asdict
import copy
import unittest

from experiment_common.selective_metrology import Cell
from tools.report_selective_metrology import analyze_rows, count_group, render_html


def fixture(cell, drift=0):
    records = []
    for index, role in enumerate(('initial', 'delay_0', 'delay_10', 'delay_60')):
        signal = 100 + (drift if index else 0)
        brackets = []
        for repeat in range(10):
            brackets.append(dict(
                before=dict(adc=0), signal=dict(adc=signal), after=dict(adc=0),
                guard=dict(adc=1000, resistance_ohm=100000.), centered_adc=float(signal)))
        records.append(dict(cell=asdict(cell), role=role, brackets=brackets, monotonic_s=index*60.,
                            summary=dict(mean_adc=float(signal), sd_adc=0., mean_g_s=1e-5,
                                         zero_difference_adc=0., zero_shift_limit_adc=2.,
                                         noise_limit_adc=2., drift_limit_adc=3.,
                                         drift_from_initial_adc=float(drift if index else 0),
                                         noise_pass=True, zero_shift_pass=True,
                                         drift_pass=index == 0 or abs(drift) <= 3)))
    return records


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.selected = Cell(0, 2, 20., 'original_selected_rectangle', True, False)
        self.other = Cell(1, 3, 20., 'span_boundary', False, False)

    def test_adc_recomputed_grade_and_256map(self):
        rows = analyze_rows([self.selected, self.other], fixture(self.selected) + fixture(self.other, 10))
        self.assertEqual(count_group(rows), dict(cells=2, passed=1, failed=1, incomplete=0))
        self.assertEqual(rows[1].reasons, ['drift:delay_0', 'drift:delay_10', 'drift:delay_60'])
        text = render_html(rows, 'hardware', 'fake')
        self.assertEqual(text.count('class="gray"'), 254)
        self.assertEqual(text.count('class="green"'), 1)
        self.assertEqual(text.count('class="red"'), 1)
        self.assertIn('не проверка программируемого диапазона', text)

    def test_missing_is_not_green_and_duplicate_rejected(self):
        records = fixture(self.selected)
        rows = analyze_rows([self.selected], records[:-1])
        self.assertFalse(rows[0].passed)
        self.assertEqual(rows[0].reasons, ['missing:delay_60'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            analyze_rows([self.selected], records + [copy.deepcopy(records[0])])

    def test_altered_saved_verdict_or_centered_value_rejected(self):
        for mutation in ('summary', 'centered'):
            records = fixture(self.selected)
            if mutation == 'summary':
                records[1]['summary']['noise_pass'] = False
            else:
                records[1]['brackets'][0]['centered_adc'] += 1
            with self.assertRaises(ValueError):
                analyze_rows([self.selected], records)


if __name__ == '__main__':
    unittest.main()
