"""Offline CLI/provenance/setup tests; never connect to a board."""
import contextlib
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import experiment
import experiment_hardware as hardware
import setup_driver


class LaunchTests(unittest.TestCase):
    def test_default_no_dispatch(self):
        with patch.object(experiment, 'run_simulator') as sim, patch.object(hardware, 'main') as hw, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiment.main([]), 0)
            sim.assert_not_called()
            hw.assert_not_called()

    def test_simulator_only(self):
        with patch.object(experiment, 'run_simulator', return_value=7) as sim, patch.object(hardware, 'main') as hw:
            self.assertEqual(experiment.main(['--run-simulator']), 7)
            sim.assert_called_once_with()
            hw.assert_not_called()

    def test_forward_modes_and_options(self):
        for args in (['--dry-run','--all'], ['--run-hardware','--phase','phase2','--phase1-session','results/example','--max-pulses','40']):
            with self.subTest(args=args), patch.object(hardware, 'main', return_value=2) as hw:
                self.assertEqual(experiment.main(args), 2)
                hw.assert_called_once_with(args)

    def test_conflicts_never_dispatch(self):
        cases = [['--run-simulator','--run-hardware'], ['--run-hardware','--dry-run'], ['--run-simulator','--phase','phase1'], ['--phase','phase1'], ['--dry-run','--phase','phase1','--all']]
        with patch.object(experiment, 'run_simulator') as sim, patch.object(hardware, 'main') as hw:
            for args in cases:
                with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    experiment.main(args)
                self.assertEqual(caught.exception.code, 2)
            sim.assert_not_called()
            hw.assert_not_called()

    def test_foreign_cwd_default_no_native_import(self):
        path = str(Path(experiment.__file__).resolve())
        code = "import runpy,sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(sys.argv[1]).parent))\nclass Guard:\n def find_spec(self,name,*args):\n  if name in ('manager','MemriCORE'): raise AssertionError(name)\nsys.meta_path.insert(0,Guard())\nsys.argv=[sys.argv[1]]\nrunpy.run_path(sys.argv[0],run_name='__main__')"
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable,'-B','-c',code,path], cwd=temp, capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--dry-run',result.stdout)


class ProvenanceTests(unittest.TestCase):
    def test_dirty_and_untracked_separate_exact_status(self):
        outputs = [subprocess.CompletedProcess([],0,'a'*40+'\n',''), subprocess.CompletedProcess([],0,' M experiment.py\n?? results/new/\n','')]
        with patch.object(hardware.subprocess,'run',side_effect=outputs) as run, patch.object(sys,'argv',['experiment.py','--run-hardware']):
            result = hardware.launch_provenance(['--run-hardware','--phase','phase1'])
        self.assertTrue(result['git']['dirty_tracked'])
        self.assertEqual(result['git']['untracked'],['results/new/'])
        self.assertTrue(result['git']['status_porcelain'].startswith(' M'))
        self.assertEqual(result['process_argv'],['experiment.py','--run-hardware'])
        self.assertEqual(result['hardware_argv'][-1],'phase1')
        self.assertEqual([c.args[0][3] for c in run.call_args_list],['rev-parse','status'])

    def test_untracked_results_not_tracked_dirty(self):
        outputs = [subprocess.CompletedProcess([],0,'a'*40+'\n',''), subprocess.CompletedProcess([],0,'?? results/new/\n','')]
        with patch.object(hardware.subprocess,'run',side_effect=outputs):
            self.assertFalse(hardware.launch_provenance([])['git']['dirty_tracked'])

    def test_missing_git_explicit(self):
        with patch.object(hardware.subprocess,'run',side_effect=FileNotFoundError('git')):
            result = hardware.launch_provenance([])['git']
        self.assertIsNone(result['commit'])
        self.assertIsNone(result['dirty_tracked'])
        self.assertIn('FileNotFoundError',result['error'])


class SetupTests(unittest.TestCase):
    def test_roundtrip_idempotence_no_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/'operator/MemriCORE/rp5_fpga_python'
            source.mkdir(parents=True)
            content=b'raise AssertionError("must never import")\n'
            hashes={name:hashlib.sha256(content).hexdigest() for name in setup_driver.FILES}
            for name in hashes:
                (source/name).write_bytes(content)
            destination=root/'repo/driver'
            with patch.object(setup_driver,'FILES',hashes):
                result=setup_driver.copy_verified(root/'operator',destination)
                self.assertEqual(len(result['copied']),2)
                self.assertFalse(result['hardware_access'])
                self.assertEqual(setup_driver.copy_verified(root/'operator',destination)['copied'],[])
            self.assertTrue(all((destination/name).read_bytes()==content for name in hashes))

    def test_second_hash_failure_copies_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/'operator/MemriCORE/rp5_fpga_python'
            source.mkdir(parents=True)
            names=list(setup_driver.FILES)
            (source/names[0]).write_bytes(b'ok')
            (source/names[1]).write_bytes(b'bad')
            hashes={name:hashlib.sha256(b'ok').hexdigest() for name in names}
            with patch.object(setup_driver,'FILES',hashes),self.assertRaises(ValueError):
                setup_driver.copy_verified(root/'operator',root/'destination')
            self.assertFalse((root/'destination').exists())


if __name__ == '__main__':
    unittest.main()
