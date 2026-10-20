"""Offline Git/LFS compatibility; temporary repositories, no network/acquisition."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from tools import run_cifar_batch as publication

ROOT=Path(__file__).resolve().parents[1]
HOOK_NAMES=('pre-push','post-commit','post-checkout','post-merge')


class OwnedHookTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'.tmp').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.tmp',prefix='hook-compat-')
        self.repo=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def write(self,name,content):
        path=self.repo/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(content);path.chmod(0o755)
        return path
    def run_git(self,*args):
        return subprocess.run(['git','-c','core.hooksPath=.githooks','-c','gc.auto=0',*args],
            cwd=self.repo,env=publication.safe_env(),check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
    def install_hooks(self):
        for name in HOOK_NAMES:self.write('.githooks/'+name,(ROOT/'.githooks'/name).read_bytes())
    def snapshot(self):
        return {n:hashlib.sha256((self.repo/'.githooks'/n).read_bytes()).hexdigest() for n in HOOK_NAMES}
    def test_delegation_exact_arguments_stdin_exit_and_missing_lfs(self):
        shell=shutil.which('sh')
        if shell is None and os.name=='nt':
            candidate=Path(shutil.which('git')).resolve().parents[1]/'usr/bin/sh.exe'
            if candidate.is_file():shell=str(candidate)
        self.assertIsNotNone(shell,'POSIX shell required (Git for Windows includes sh)')
        self.install_hooks()
        self.write('bin/git-lfs',b'#!/bin/sh\nexit 0\n')
        self.write('bin/git',b'#!/bin/sh\nprintf "<%s>\\n" "$@"\nwhile IFS= read -r line; do printf "%s\\n" "$line"; done\nexit "${TEST_EXIT:?}"\n')
        env=publication.safe_env();env['PATH']=str(self.repo/'bin')
        refs=b'refs/heads/a aaaa refs/heads/a bbbb\nrefs/heads/b cccc refs/heads/b dddd\n'
        for name in HOOK_NAMES:
            for code in (0,23):
                with self.subTest(hook=name,exit=code):
                    env['TEST_EXIT']=str(code)
                    result=subprocess.run([shell,str(self.repo/'.githooks'/name),'origin space','url argument'],
                        cwd=self.repo,env=env,input=refs,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    self.assertEqual(result.returncode,code,result.stderr)
                    self.assertEqual(result.stdout,('<lfs>\n<'+name+'>\n<origin space>\n<url argument>\n').encode()+refs)
        (self.repo/'bin/git-lfs').unlink()
        for name in HOOK_NAMES:
            result=subprocess.run([shell,str(self.repo/'.githooks'/name)],cwd=self.repo,env=env,
                input=refs,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertEqual(result.returncode,2);self.assertEqual(result.stdout,b'')
    def test_real_filter_process_commit_diff_preserve_all_hook_bytes(self):
        self.run_git('init')
        self.run_git('config','user.name','Synthetic hook test')
        self.run_git('config','user.email','hook-test@example.invalid')
        self.run_git('config','filter.lfs.process','git-lfs filter-process')
        self.run_git('config','filter.lfs.required','true')
        self.install_hooks()
        self.write('.gitattributes',b'* -text\n*.bin filter=lfs diff=lfs merge=lfs -text\n')
        self.write('sample.bin',b'fixture LFS contents\n'*1024)
        before=self.snapshot()
        self.run_git('add','.githooks','.gitattributes','sample.bin')
        self.assertEqual(before,self.snapshot())
        pointer=self.run_git('show',':sample.bin')
        self.assertEqual(publication.security.parse_pointer(pointer,True),
                         (publication.sha(self.repo/'sample.bin'),(self.repo/'sample.bin').stat().st_size))
        self.run_git('commit','-m','Synthetic real LFS hook compatibility')
        self.assertEqual(before,self.snapshot())
        for _ in range(3):
            os.utime(self.repo/'sample.bin',None)  # Force stat refresh/clean-filter comparison.
            self.assertEqual(self.run_git('diff','--name-only'),b'')
            self.assertEqual(before,self.snapshot())
        self.assertEqual(self.run_git('status','--porcelain'),b'')
        # Explicit template installation refuses custom hooks; no force/restore workaround.
        update=subprocess.run(['git','-c','core.hooksPath=.githooks','lfs','update'],
            cwd=self.repo,env=publication.safe_env(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.assertNotEqual(update.returncode,0)
        self.assertEqual(before,self.snapshot())
        self.assertEqual(self.run_git('diff','--name-only'),b'')
    def test_reviewed_hook_pins_are_exact(self):
        publication.verify_hooks()
        for name in HOOK_NAMES:
            content=(ROOT/'.githooks'/name).read_text()
            self.assertIn('Repository-owned LFS delegation',content)
            self.assertTrue(content.endswith('exec git lfs '+name+' "$@"\n'))
    def test_posix_nonexecutable_hook_fails_closed(self):
        with patch.object(publication,'os',SimpleNamespace(name='posix',X_OK=os.X_OK,access=lambda *_:False)):
            with self.assertRaisesRegex(publication.Pending,'not_executable'):
                publication.verify_hooks()


if __name__=='__main__':unittest.main()
