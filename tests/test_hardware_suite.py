"""Offline tests with real converters and a fake public Connector only."""

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
for relative in (".deps", "vendor/MemriBoard"):
    if (ROOT / relative).is_dir():
        sys.path.insert(0, str(ROOT / relative))
from manager.service import converters
import numpy as np
from read_all import EXPECTED_BOARD
import experiment_hardware as entry
from hardware_suite import HardwareSuite, ScientificGateError


class FakeConnector:
    cb_type = "real"
    def __init__(self, *, reset_stuck=False, fail_set=False, invalid=False):
        self.commands = []
        self.q = .02
        self.reset_stuck, self.fail_set, self.invalid = reset_stuck, fail_set, invalid
        self.after_set = False
        self.opened = self.closed = 0

    def open_port(self):
        self.opened += 1
        return True

    def close_port(self):
        self.closed += 1
        return True

    def impact(self, task):
        self.commands.append(copy.deepcopy(task))
        assert task["mode_flag"] in (7, 10)
        assert task["pulse_width"] == 0 and task["wl"] in (0, 2)
        if task["mode_flag"] == 10:
            assert all(0 <= v <= .1 for v in task["vol"])
            for i, v in enumerate(task["vol"]):
                task["vol"][i] = converters.convert_volt_to_dac(12, 5, v)
            adc = sum(task["vol"])
        else:
            if task["vol"]:
                if task["sign"] == 0:
                    if not self.reset_stuck:
                        self.q = .004
                    self.after_set = False
                else:
                    if self.fail_set:
                        raise TimeoutError("SET sent; response lost")
                    self.q += .01
                    self.after_set = True
            q = self.q-.005 if task["vol"] == 0 and self.after_set else self.q
            adc = converters.convert_res_to_adc(11, 3000, .3, 14, 5, 10, 330/q)
        if self.invalid:
            adc = 0
        native_r = converters.convert_adc_to_res(11, 3000, .3, 14, 5, 10, adc)
        return native_r, task["id"], adc

    def __getattr__(self, name):
        raise AssertionError(f"Forbidden Connector API: {name}")


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.output = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def suite(self, connector=None, budget=6625950):
        suite = HardwareSuite(connector or FakeConnector(), EXPECTED_BOARD, self.output,
                              "offline", converters, max_pulses=budget)
        self.addCleanup(suite.close)
        return suite

    def records(self, name="raw.jsonl"):
        return [json.loads(line) for line in (self.output/name).read_text(encoding="utf-8").splitlines()]

    def test_phase1_keeps_negative_post_read_errors_without_selection(self):
        suite = self.suite()
        with contextlib.redirect_stdout(io.StringIO()):
            result = suite.phase1()
        self.assertTrue(result["completion_conditions_pass"])
        self.assertFalse(result["precision_qualified"])
        self.assertEqual(len(result["attempts"]), 75)
        self.assertTrue(any(r["signed_error_siemens"] < 0 for r in result["attempts"]))
        self.assertTrue(any(not r["post_read_target_reached"] for r in result["attempts"]))
        self.assertEqual(sum(r["phase"] == "baseline" and r["event"] == "result" for r in self.records()), 63)

    def test_first_unreachable_reset_stops_and_persists_attempt(self):
        connector = FakeConnector(reset_stuck=True)
        suite = self.suite(connector)
        with self.assertRaises(ScientificGateError):
            suite.program(0, 2, .01, "phase1")
        attempts = self.records("programming_attempts.jsonl")
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["reset_pulses"], 271)
        self.assertEqual(attempts[0]["set_pulses"], 0)
        self.assertEqual(len(connector.commands), 272)
        self.assertFalse(attempts[0]["reset_ok"])

    def test_set_transport_error_preserves_attempted_pulse_and_raw(self):
        suite = self.suite(FakeConnector(fail_set=True))
        with self.assertRaises(TimeoutError):
            suite.program(0, 2, .01, "phase1")
        record = self.records("programming_attempts.jsonl")[0]
        self.assertEqual(record["set_pulses"], 1)
        self.assertEqual(record["reset_pulses"], 1)
        self.assertEqual(record["last_command_id"], 3)
        self.assertEqual(self.records()[-1]["status"], "error")
        self.assertIn("SET sent", self.records()[-1]["error"])

    def test_budget_stops_before_next_nonzero_command(self):
        connector = FakeConnector()
        suite = self.suite(connector, budget=1)
        with self.assertRaisesRegex(ScientificGateError, "budget"):
            suite.program(0, 2, .01, "phase1")
        self.assertEqual(suite.pulses, 1)
        self.assertEqual(len(connector.commands), 2)
        self.assertEqual(self.records("programming_attempts.jsonl")[0]["set_pulses"], 0)

    def test_zero_mvm_and_list_mutation_keep_original_si(self):
        suite = self.suite()
        zero = suite.mvm(0, [0.0]*32, [0.0]*32, "mvm")
        self.assertEqual(zero["raw_adc"], 0)
        self.assertEqual(zero["output_voltage"], 0)
        vector = [.1]+[0.0]*31
        suite.mvm(0, vector, [.00001]*32, "mvm")
        self.assertEqual(vector[0], .1)
        self.assertEqual(self.records()[-1]["command"]["vol"][0], .1)
        self.assertEqual(self.records()[-1]["native_voltage_conversion"][0]["dac_code"], 82)

    def test_invalid_adc_saved_before_stop(self):
        suite = self.suite(FakeConnector(invalid=True))
        with self.assertRaises(ValueError):
            suite.read(0, 2, "baseline")
        row = self.records()[-1]
        self.assertEqual(row["raw_adc"], 0)
        self.assertEqual(row["status"], "error")

    def test_current_gate_sends_no_mvm(self):
        connector = FakeConnector()
        suite = self.suite(connector)
        with self.assertRaises(ScientificGateError):
            suite.mvm(0, [.1]*8+[0.0]*24, [.01]*32, "mvm")
        self.assertEqual(connector.commands, [])
        self.assertTrue(self.records("rejected_commands.jsonl"))

    def test_mapping_reads_completed_tile_before_mvm_and_next_tile(self):
        suite = self.suite()
        weights = np.ones((64, 3, 5, 5), dtype=np.float32)
        with contextlib.redirect_stdout(io.StringIO()):
            result = suite.mapping(weights, [0], "phase2")
        rows = [r for r in self.records() if r["event"] == "request"]
        self.assertEqual(len([r for r in rows if r["command"]["mode_flag"] == 10]), 552)
        for tile in range(3):
            before_mvm = [i for i, r in enumerate(rows) if r.get("tile") == tile and r["phase"] == "phase2_pre_mvm_tile"]
            mvm = [i for i, r in enumerate(rows) if r.get("tile") == tile and r["command"]["mode_flag"] == 10]
            self.assertLess(max(before_mvm), min(mvm))
            if tile < 2:
                next_tile = [i for i, r in enumerate(rows) if r.get("tile") == tile+1]
                self.assertLess(max(mvm), min(next_tile))
        self.assertFalse(result["zero_offset_correction"])
        self.assertEqual(len(self.records("pre_mvm_tile_readback.jsonl")), 150)

    def test_default_no_io_and_dry_run_no_connector(self):
        with patch.object(entry, "preflight", side_effect=AssertionError), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main([]), 0)
        plan = json.loads((ROOT/"configs/hardware_four_phase.json").read_text(encoding="utf-8"))
        def hashes(path):
            return entry.READ_ALL_SHA256 if path.name == "read_all.py" else plan["weights_sha256"]
        with patch.object(entry, "local_path", side_effect=lambda p: ROOT/p), patch.object(entry, "sha256", side_effect=hashes), patch.object(entry, "preflight", return_value=(None, b"", {})), patch.object(entry, "native_connector", side_effect=AssertionError), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main(["--dry-run"]), 0)

    def test_fingerprint_failure_before_connector(self):
        with patch.object(entry, "local_path", side_effect=lambda p: ROOT/p), patch.object(entry, "sha256", return_value="bad"), patch.object(entry, "native_connector", side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError, "helper"):
                entry.main(["--run-hardware"])

    def test_weight_failure_before_connector(self):
        def hashes(path):
            return entry.READ_ALL_SHA256 if path.name == "read_all.py" else "bad"
        with patch.object(entry, "local_path", side_effect=lambda p: ROOT/p), patch.object(entry, "sha256", side_effect=hashes), patch.object(entry, "preflight", return_value=(None, b"", {})), patch.object(entry, "native_connector", side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError, "weight"):
                entry.main(["--run-hardware"])

    def test_config_phase_drift_rejected(self):
        plan = json.loads((ROOT/"configs/hardware_four_phase.json").read_text(encoding="utf-8"))
        entry.validate_plan(plan)
        plan["phases"]["phase1"]["cycles"] = 1
        with self.assertRaises(ValueError):
            entry.validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
