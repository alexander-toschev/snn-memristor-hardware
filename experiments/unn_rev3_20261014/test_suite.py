"""Unit tests with a tiny deterministic Connector double; no MemriBoard import."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .suite import SimulatorSuite, comparison


class ConnectorDouble:
    cb_type = "simulator"

    def __init__(self, board):
        self.board = board
        self.invalid_adc = False
        self.call_count = 0

    def impact(self, task):
        self.call_count += 1
        if self.invalid_adc:
            return (999, task["id"], 0)
        if task["mode_flag"] == 10:
            value = sum(task["vol"])*0.02
            task["vol"][:] = [999]*len(task["vol"])
            adc = int(value*2**self.board.adc_bit/self.board.vol_ref_adc)
            return (999, task["id"], adc)
        resistance = self.board.sum_gain/0.02
        adc = int(self.board.gain*self.board.res_load*self.board.vol_read*2**self.board.adc_bit
                  / ((resistance+self.board.res_switches+self.board.res_load)*self.board.vol_ref_adc))
        return (999, task["id"], adc)


class SimulatorSuiteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.board = SimpleNamespace(dac_bit=12, adc_bit=14, vol_ref_dac=5,
                                     vol_ref_adc=5, gain=11, res_load=3000,
                                     res_switches=10, vol_read=0.3, sum_gain=330)
        self.protocol = SimpleNamespace(board=self.board, excluded_wl=(1,),
                                         initial_cells=((0, 2), (2, 6), (2, 4)))
        self.connections = {name: ConnectorDouble(self.board) for name in ("A", "B")}
        self.suite = SimulatorSuite(self.connections, self.protocol, self.directory.name)

    def tearDown(self):
        self.suite.close()
        self.directory.cleanup()

    def test_mvm_uses_fresh_list_and_accepts_zero(self):
        volts = [0.1] + [0.0]*31
        first = self.suite.mvm(0, volts)
        second = self.suite.mvm(0, volts)
        self.assertEqual(first, second)
        self.assertEqual(volts, [0.1] + [0.0]*31)
        self.assertEqual(self.suite.mvm(0, [0.0]*32), 0.0)

    def test_invalid_adc_saved_before_exception(self):
        self.connections["A"].invalid_adc = True
        with self.assertRaises(RuntimeError):
            self.suite.read(0, 2, "first_check")
        row = json.loads((Path(self.directory.name)/"raw_tasks.jsonl").read_text())
        self.assertEqual(row["raw_adc"], 0)
        self.assertEqual(row["status"], "error")
        self.assertNotIn("resistance_ohm", row)

    def test_reset_has_finite_budget_and_does_not_set_after_failure(self):
        result = self.suite.program(0, 2, 0.03, "test", target_q=0.03)
        self.assertFalse(result["reset_ok"])
        self.assertEqual(result["reset_pulses"], 271)
        self.assertEqual(result["set_pulses"], 0)
        self.assertEqual(self.connections["A"].call_count, 272)

    def test_excluded_wl_never_reaches_connector(self):
        with self.assertRaises(ValueError):
            self.suite.read(1, 0, "test")
        self.assertEqual(self.connections["A"].call_count, 0)

    def test_baseline_and_mapping_cover_two_arrays_and_prefix_means(self):
        baseline = self.suite.run_baseline()
        self.assertEqual(baseline["read_count"], 63)
        self.suite.program = lambda *args, **kwargs: {
            "reset_ok": True, "target_reached": True, "reset_pulses": 0,
            "set_pulses": 0, "last_q": 0.02}
        weights = [[(i-37)/100 for i in range(75)] for _ in range(64)]
        result = self.suite.run_mapping(weights, filters=[41], label="phase4", repetitions=3)
        self.assertEqual(result["filter_count"], 1)
        self.assertEqual(result["programmed_cell_count"], 450)
        self.assertEqual([row["K"] for row in result["prefix_means"]], [1, 2, 3])
        self.assertEqual(len(result["reconstructed_weights"]["41"]), 3)
        self.assertGreater(self.connections["B"].call_count, 0)

    def test_metrics_use_signed_bias(self):
        metric = comparison([0, 4], [1, 3])
        self.assertEqual(metric["bias"], 0)
        self.assertEqual(metric["rmse"], 1)

    def test_characterization_logs_all_failed_attempts(self):
        result = self.suite.run_characterization()
        self.assertEqual(result["attempt_count"], 75)
        self.assertEqual(result["post_read_count"], 375)
        self.assertEqual(result["target_reached_count"], 0)
        self.assertTrue(all(r["reset_success_count"] == 0 for r in result["groups"]))

    def test_paths_are_portable_and_contained(self):
        from .paths import PROJECT_DIR, repo_path
        self.assertEqual(repo_path("weights/conv1_weights.npy"), PROJECT_DIR / "weights/conv1_weights.npy")
        for value in ("/tmp/weights.npy", "C:/weights.npy", "C:weights.npy", "../weights.npy", "..\\weights.npy"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                repo_path(value)


if __name__ == "__main__":
    unittest.main()
