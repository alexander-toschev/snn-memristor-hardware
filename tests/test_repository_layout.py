"""Navigation and launch-path regression checks; no backend or data writes."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    def test_document_destinations_and_compatibility_redirects(self):
        layout = json.loads((ROOT / 'docs/layout.json').read_text(encoding='utf-8'))
        for old, new in layout['documents'].items():
            self.assertTrue((ROOT / new).is_file(), new)
            if old in layout['compatibility_redirects']:
                self.assertIn(new, (ROOT / old).read_text(encoding='utf-8'))
            else:
                self.assertFalse((ROOT / old).exists(), old)
        for name in layout['pinned_root_tests']:
            self.assertTrue((ROOT / name).is_file())

    def test_monday_plan_and_local_scratch_ignore(self):
        plan = (ROOT / 'docs/plans/EXPERIMENTS_20261019.md').read_text(encoding='utf-8')
        self.assertIn('Начать с эксперимента №1', plan)
        self.assertIn('checkpoint', plan)
        self.assertIn('/.tmp/', (ROOT / '.gitignore').read_text())

    def test_vscode_default_is_help_and_cpu_inputs_are_preserved(self):
        launch = json.loads((ROOT / ".vscode/launch.json").read_text(encoding="utf-8"))
        first = launch["configurations"][0]
        self.assertEqual(first["args"], ["--help"])
        self.assertNotIn("preLaunchTask", first)
        self.assertTrue({"fullCpuCase", "fullCpuOutput"}.issubset({x["id"] for x in launch["inputs"]}))
        self.assertEqual(len({x["name"] for x in launch["configurations"]}), len(launch["configurations"]))

    def test_offline_frontdoor_help_from_other_working_directory(self):
        for args in (['--help'], ['compare-traces', '--help'], ['export-trace', '--help']):
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/offline.py'), *args],
                                    cwd=ROOT / 'tests', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout.lower())


if __name__ == '__main__':
    unittest.main()
