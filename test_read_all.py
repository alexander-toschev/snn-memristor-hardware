"""Offline tests: no driver packages, GPIO, SPI, or simulator imports."""

import contextlib
import ast
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import read_all


class FakeConnector:
    def __init__(self, responses=None, fail_open=False):
        self.commands = []
        self.open_count = 0
        self.close_count = 0
        self.responses = iter(responses) if responses is not None else None
        self.fail_open = fail_open

    def open_port(self):
        self.open_count += 1
        return not self.fail_open

    def impact(self, task):
        assert self.open_count == 1 and self.close_count == 0
        assert task["mode_flag"] == 9
        assert task["vol"] == 0.1
        assert task["pulse_width"] == 0 and task["sign"] == 0
        assert 0 <= task["wl"] < 8 and 0 <= task["bl"] < 32
        assert task["id"] == len(self.commands) + 1
        self.commands.append(task.copy())
        result = next(self.responses) if self.responses is not None else (99999.0, 0, 1000)
        if isinstance(result, (Exception, KeyboardInterrupt)):
            raise result
        return result

    def close_port(self):
        self.close_count += 1
        return True

    def __getattr__(self, name):
        raise AssertionError(f"Unexpected Connector interaction: {name}")


class ReadAllTests(unittest.TestCase):
    def test_no_driver_import_or_interface_bypass(self):
        tree = ast.parse(Path(read_all.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith(("MemriCORE", "RRAMPiDriver")))
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.startswith(("MemriCORE", "RRAMPiDriver", "spidev", "gpiod")) for alias in node.names))
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, ("interface", "mode_7", "mode_9", "mode_mvm", "custom_impact"))

    def scan(self, connector, cells):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            output = Path(directory)
            audit = read_all.collect(connector, cells, output, "offline-test", read_all.EXPECTED_BOARD)
            rows = [json.loads(line) for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(audit, json.loads((output / "audit.json").read_text(encoding="utf-8")))
            self.assertEqual(audit["raw_sha256"], read_all.sha256(output / "raw.jsonl"))
            return audit, rows

    def test_all_256_including_wl1_exactly_once(self):
        connector = FakeConnector()
        cells = [(wl, bl) for wl in range(8) for bl in range(32)]
        audit, rows = self.scan(connector, cells)
        self.assertEqual(audit["status"], "completed")
        self.assertEqual(audit["unique_cells"], 256)
        self.assertEqual(audit["missing_cells"], [])
        self.assertEqual([(t["wl"], t["bl"]) for t in connector.commands], cells)
        self.assertEqual((connector.open_count, connector.close_count), (1, 1))
        self.assertEqual(len(rows), 512)
        self.assertEqual([r["event"] for r in rows], ["request", "result"] * 256)
        for row in rows[1::2]:
            self.assertEqual(row["raw_adc"], 1000)
            self.assertEqual(row["returned_id"], 0)
            self.assertEqual(row["native_r_uses_ini_vol_read"], 99999)
            self.assertAlmostEqual(row["resistance_nominal_ohm"], 10813440 / 1000 - 3010)
            self.assertNotEqual(row["resistance_nominal_ohm"], row["resistance_dac_estimate_ohm"])

    def test_invalid_adc_retained_without_clamp_or_retry(self):
        raw = [0, 16383, -1, 16384, True, 10000]
        connector = FakeConnector([(1, 0, value) for value in raw])
        audit, rows = self.scan(connector, [(0, bl) for bl in range(len(raw))])
        self.assertEqual([row["raw_adc"] for row in rows[1::2]], raw)
        self.assertEqual(audit["flagged"], len(raw))
        self.assertTrue(all(row["resistance_nominal_ohm"] is None for row in rows[1::2]))
        self.assertEqual(len(connector.commands), len(raw))

    def test_transport_error_flushes_attempt_and_stops(self):
        connector = FakeConnector([TimeoutError("таймаут")])
        audit, rows = self.scan(connector, [(0, 0), (0, 1)])
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["attempted"], 1)
        self.assertEqual(rows[1]["status"], "error")
        self.assertIn("таймаут", rows[1]["error"])
        self.assertEqual(connector.close_count, 1)

    def test_wrong_returned_id_preserves_native_tuple(self):
        connector = FakeConnector([(100, 42, 1234)])
        audit, rows = self.scan(connector, [(1, 31)])
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(rows[1]["native_result"], [100, 42, 1234])
        self.assertEqual(rows[1]["raw_adc"], 1234)
        self.assertEqual(rows[1]["returned_id"], 42)

    def test_interruption_preserves_error_record_and_closes(self):
        connector = FakeConnector([KeyboardInterrupt()])
        audit, rows = self.scan(connector, [(0, 0), (0, 1)])
        self.assertEqual(audit["status"], "failed")
        self.assertIn("KeyboardInterrupt", rows[1]["error"])
        self.assertEqual(connector.close_count, 1)

    def test_open_failure_creates_no_fake_measurements(self):
        connector = FakeConnector(fail_open=True)
        audit, rows = self.scan(connector, [(0, 0)])
        self.assertEqual(rows, [])
        self.assertEqual(audit["received"], 0)
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(connector.close_count, 1)

    def test_close_failure_prevents_completed_status(self):
        for failure in (False, RuntimeError("cleanup failed")):
            with self.subTest(failure=failure):
                connector = FakeConnector()
                def close():
                    connector.close_count += 1
                    if isinstance(failure, Exception):
                        raise failure
                    return failure
                connector.close_port = close
                audit, rows = self.scan(connector, [(0, 0)])
                self.assertEqual(audit["status"], "failed")
                self.assertIn("close_error", audit)
                self.assertEqual(audit["received"], 1)
                self.assertEqual(connector.close_count, 1)

    def test_invalid_address_rejected_before_io(self):
        connector = FakeConnector()
        with self.assertRaises(ValueError):
            read_all.collect(connector, [(8, 0)], Path("unused"), "offline", read_all.EXPECTED_BOARD)
        self.assertEqual(connector.open_count, 0)

    def test_no_flag_no_preflight_or_connector(self):
        with patch.object(read_all, "preflight", side_effect=AssertionError), patch.object(read_all, "native_connector", side_effect=AssertionError):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(read_all.main([]), 0)

    def test_dry_run_does_not_construct_connector(self):
        with patch.object(read_all, "preflight", return_value=(None, b"", {})), patch.object(read_all, "native_connector", side_effect=AssertionError):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(read_all.main(["--dry-run", "--cell", "1,31"]), 0)
            self.assertEqual(json.loads(stdout.getvalue())["cell_count"], 1)

    def test_relative_path_guard(self):
        for value in ("../outside", "D:/outside", "/outside", "..\\outside"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                read_all.local_path(value)

    def test_preflight_rejects_source_drift_and_changed_ini(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory).resolve()
            (root / "configs").mkdir()
            ini = root / "configs/raspberry_board.ini"
            ini.write_bytes((Path(__file__).parent / "configs/raspberry_board.ini").read_bytes())
            source = root / "vendor/MemriBoard/manager/board.py"
            source.parent.mkdir(parents=True)
            source.write_text("# synthetic reviewed source\n", encoding="utf-8")
            fpga = root / "vendor/MemriBoard/fpga.py"
            fpga.write_text("# synthetic encoder\n", encoding="utf-8")
            (root / "read_all.py").write_text("# synthetic entrypoint\n", encoding="utf-8")
            with patch.object(read_all, "ROOT", root), patch.object(read_all, "EXPECTED_SOURCES", {"manager/board.py": read_all.sha256(source)}), patch.object(read_all, "FPGA_SOURCE", "fpga.py"), patch.object(read_all, "FPGA_SHA256", read_all.sha256(fpga)):
                _, snapshot, hashes = read_all.preflight(read_all.INI)
                self.assertEqual(snapshot, ini.read_bytes())
                self.assertEqual(hashes["vendor/MemriBoard/manager/board.py"], read_all.sha256(source))
                original = source.read_bytes()
                source.write_bytes(original + b"# drift")
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    read_all.preflight(read_all.INI)
                source.write_bytes(original)
                ini.write_text(snapshot.decode().replace("gain = 11.0", "gain = 12.0"), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "gain"):
                    read_all.preflight(read_all.INI)


if __name__ == "__main__":
    unittest.main()
