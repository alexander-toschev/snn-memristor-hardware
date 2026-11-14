"""Offline tests only: harmless fixture, no experiment or hardware import."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).resolve().parents[1] / "tools" / "run_foreground_logged.py"


class ForegroundLogTests(unittest.TestCase):
    def test_closed_parent_pipe_keeps_native_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.py"
            fixture.write_text(
                "import os, sys\n"
                "print('python stdout', flush=True)\n"
                "os.write(1,b'native stdout\\n')\n"
                "os.write(2,b'native stderr\\n')\n"
                "raise SystemExit(0)\n", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, str(RUNNER), "--log", "results/native.log", "fixture.py"],
                cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert process.stdout is not None and process.stderr is not None
            process.stdout.close()
            process.stderr.close()
            self.assertEqual(process.wait(timeout=20), 0)
            output = (root / "results/native.log").read_text(encoding="utf-8")
            self.assertIn("python stdout", output)
            self.assertIn("native stdout", output)
            self.assertIn("native stderr", output)
            # Re-running with the same log cannot truncate previous evidence.
            repeated = subprocess.run(
                [sys.executable, str(RUNNER), "--log", "results/native.log", "fixture.py"],
                cwd=root, capture_output=True, check=False)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual((root / "results/native.log").read_text(encoding="utf-8"), output)

    def test_reject_parent_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, str(RUNNER), "--log", "../bad.log", "fixture.py"],
                cwd=temporary, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)

    def test_preserves_failure_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fixture.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(RUNNER), "--log", "results/failure.log", "fixture.py"],
                cwd=root, capture_output=True, check=False)
            self.assertEqual(result.returncode, 7)


if __name__ == "__main__":
    unittest.main()
