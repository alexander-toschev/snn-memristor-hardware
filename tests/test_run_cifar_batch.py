"""Synthetic files/mocked Git only; no Connector, model, network or real repo mutation."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch, Mock
import uuid

from tools import run_cifar_batch as b


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name).resolve()
        self.patch=patch.object(b,'ROOT',self.root);self.patch.start()
    def tearDown(self):
        self.patch.stop();self.temp.cleanup()
    def write(self,name,data):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(data);return p
    def test_relative_paths_and_private_namespace(self):
        for name in ('../escape','D:/secret','/absolute',r'a\..\b'):
            with self.assertRaises(b.Pending):b.local(name)
        with self.assertRaises(b.Pending):
            b.session_inventory('vendor/private',{},'abc')
    def test_remote_has_no_credentials_or_other_destination(self):
        self.assertTrue(b.remote_allowed('https://github.com/'+b.REMOTE_REPO+'.git'))
        for url in ('https://user:credential@github.com/'+b.REMOTE_REPO,
                    'https://github.com/other/repo','https://github.com/'+b.REMOTE_REPO+'\nother'):
            self.assertFalse(b.remote_allowed(url))
    def test_environment_cannot_override_hooks_or_prompt(self):
        with patch.dict(b.os.environ,{'GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'core.hooksPath',
                    'GIT_INDEX_FILE':'other','GITLEAKS_CONFIG':'other','GIT_LFS_SKIP_PUSH':'1'}):
            env=b.safe_env()
        self.assertNotIn('GIT_CONFIG_COUNT',env);self.assertNotIn('GIT_INDEX_FILE',env)
        self.assertNotIn('GITLEAKS_CONFIG',env);self.assertEqual(env['GIT_TERMINAL_PROMPT'],'0')
        self.assertNotIn('GIT_LFS_SKIP_PUSH',env)
    def publication(self,inventory,exit_record):
        return dict(schema='cifar-evidence-preservation-v1',base_commit='base',
                    wrapper_sha256=b.sha(b.__file__),acquisition_exit=exit_record,sessions=[],
                    semantic_audit_pass=False,push_destination=b.DESTINATION,inventory=inventory)
    def test_gzip_all_members_and_truncated_final_stream(self):
        data=b'one\n'*1000000+b'two\n'
        path=self.write('part.gz',gzip.compress(data)+gzip.compress(b'last payload'))
        meta={};self.assertEqual(b''.join(b.decoded_chunks(path,meta)),data+b'last payload')
        self.assertEqual(meta['gzip_members'],2);self.assertTrue(meta['complete_gzip'])
        path.write_bytes(path.read_bytes()[:-5]);meta={}
        self.assertEqual(b''.join(b.decoded_chunks(path,meta)),data+b'last payload')
        self.assertTrue(meta['truncated_prefix']);self.assertFalse(meta['complete_gzip'])
    def test_gzip_corruption_and_budget_fail_closed(self):
        payload=bytearray(gzip.compress(b'not a secret'));payload[-8]^=1
        path=self.write('bad.gz',payload)
        with self.assertRaises(b.Pending):list(b.decoded_chunks(path,{}))
        path.write_bytes(gzip.compress(b'a'*1000))
        with patch.object(b,'MAX_DECODED_FILE',100),self.assertRaises(b.Pending):
            list(b.decoded_chunks(path,{}))
    def fixture(self):
        plan={'source':'hardware','first_stage':19,'stop_stage_exclusive':24}
        raw=json.dumps(plan).encode();pin=hashlib.sha256(raw).hexdigest()
        sid=str(uuid.uuid4());name=b.SESSION_ROOT+'/'+sid
        self.write(name+'/continuation_plan.json',raw)
        start={'session_id':sid,'source':'hardware','stage_range':[19,20],
               'continuation_plan_sha256':pin,'source_sha256':{}}
        self.write(name+'/start_manifest.json',json.dumps(start).encode())
        self.write(name+'/raw/part-000000.jsonl.gz',gzip.compress(b'{"kind":"request"}\n')[:-5])
        return name,plan,pin,start
    def test_interrupted_bytes_preserved_without_claiming_complete(self):
        name,plan,pin,start=self.fixture()
        report=b.session_inventory(name,plan,pin)
        self.assertEqual(report['status'],'interrupted_no_final_manifest')
        self.assertFalse(report['semantic_audit_pass'])
        with self.assertRaises(b.Pending):b.session_inventory(name,plan,pin,require_final=True)
    def test_final_inventory_and_private_extra_rejected(self):
        name,plan,pin,start=self.fixture()
        directory=self.root/name
        final=dict(start,status='failed',result_sha256={p.relative_to(directory).as_posix():b.sha(p)
                         for p in directory.rglob('*') if p.is_file()})
        self.write(name+'/manifest.json',json.dumps(final).encode())
        self.assertEqual(b.session_inventory(name,plan,pin,True)['status'],'failed')
        self.write(name+'/private_driver.py',b'not publishable')
        with self.assertRaises(b.Pending):b.session_inventory(name,plan,pin,True)
        (directory/'private_driver.py').unlink()
        self.write(name+'/raw/part-000000.jsonl.gz',b'changed')
        with self.assertRaises(b.Pending):b.session_inventory(name,plan,pin,True)
    def test_finalize_never_launches_acquisition(self):
        batch=self.root/b.BATCH_ROOT/str(uuid.uuid4());batch.mkdir(parents=True)
        with patch.object(b,'run_child') as child,patch.object(b,'new_batch') as new, \
             patch.object(b,'commit_or_retry',return_value={'push':'not_requested','acquisition':0}) as finalize:
            self.assertEqual(b.main(['finalize','--batch',batch.relative_to(self.root).as_posix()]),0)
        child.assert_not_called();new.assert_not_called();finalize.assert_called_once()
    def test_publication_failure_retains_batch_and_never_relaunches(self):
        batch=self.root/b.BATCH_ROOT/str(uuid.uuid4());batch.mkdir(parents=True)
        evidence=self.write(batch.relative_to(self.root).as_posix()+'/kept',b'evidence')
        with patch.object(b,'run_child') as child,patch.object(b,'commit_or_retry',side_effect=b.Pending('hook_failed')):
            self.assertEqual(b.main(['finalize','--batch',batch.relative_to(self.root).as_posix()]),3)
        child.assert_not_called();self.assertEqual(evidence.read_bytes(),b'evidence')
    def test_failed_scanner_is_not_ignored(self):
        process=Mock();process.stdin=Mock();process.wait.return_value=1;process.poll.return_value=1
        with patch.object(b.subprocess,'Popen',return_value=process),self.assertRaises(b.Pending):
            b.scan_chunks('fake-scanner',[b'payload'])
        process.stdin.write.assert_called_once_with(b'payload')
    def test_frozen_entry_pin_matches_real_source(self):
        real=Path(__file__).resolve().parents[1]
        self.assertEqual(b.sha(real/b.ENTRY),b.ENTRY_SHA)
    def test_committed_retry_pushes_same_commit_without_add_or_commit(self):
        batch=self.root/b.BATCH_ROOT/str(uuid.uuid4());batch.mkdir(parents=True)
        evidence=self.write(batch.relative_to(self.root).as_posix()+'/plan.json',b'{}')
        inventory={evidence.relative_to(self.root).as_posix():{'sha256':b.sha(evidence),'bytes':2}}
        b.write_new(batch/'PUBLICATION.json',self.publication(inventory,{'exit_code':2}))
        names=sorted([*inventory,(batch/'PUBLICATION.json').relative_to(self.root).as_posix()])
        fake=Mock()
        def text(*args):
            if args==('rev-parse','HEAD^'):return 'base'
            if args[0]=='ls-remote':return 'owncommit\t'+b.DESTINATION
            raise AssertionError(args)
        def run(*args,**kw):
            if args[0]=='diff-tree':return ('\0'.join(names)+'\0').encode()
            if args[0] in ('diff','push'):return b''
            raise AssertionError(args)
        fake.text.side_effect=text;fake.run.side_effect=run
        with patch.object(b,'Git',return_value=fake),patch.object(b,'git_preflight',return_value='owncommit'), \
             patch.object(b,'collect_batch',return_value=({'base_commit':'base'},{'exit_code':2},[],inventory)),patch.object(b,'verify_hooks'):
            result=b.commit_or_retry(batch,push=True)
        self.assertEqual(result['commit'],'owncommit');self.assertEqual(result['acquisition'],2)
        self.assertEqual(result['push'],'completed');self.assertFalse(result['semantic_audit_pass'])
        self.assertFalse(any(c.args[0] in ('commit','add','--literal-pathspecs') for c in fake.run.call_args_list))
        self.assertIn(unittest.mock.call('push','origin','owncommit:'+b.DESTINATION),fake.run.call_args_list)
    def test_unrelated_retry_commit_refused_without_mutation(self):
        batch=self.root/b.BATCH_ROOT/str(uuid.uuid4());batch.mkdir(parents=True)
        b.write_new(batch/'PUBLICATION.json',self.publication({}, {'exit_code':0}))
        fake=Mock();fake.run.return_value=b'';fake.text.return_value='unexpectedparent'
        with patch.object(b,'Git',return_value=fake),patch.object(b,'git_preflight',return_value='othercommit'), \
             patch.object(b,'collect_batch',return_value=({'base_commit':'base'},{'exit_code':0},[],{})),self.assertRaises(b.Pending):
            b.commit_or_retry(batch)
        self.assertFalse(any(c.args[0] in ('commit','add','push','--literal-pathspecs') for c in fake.run.call_args_list))
    def test_fixed_git_flags_enforce_hooks_and_lfs_upload(self):
        result=Mock(returncode=0,stdout=b'')
        with patch.object(b.subprocess,'run',return_value=result) as run:
            b.Git().run('push','origin','oid:'+b.DESTINATION)
        command=run.call_args.args[0]
        for config in ('core.hooksPath=.githooks','lfs.allowincompletepush=false','credential.helper=cache --timeout=43200'):
            self.assertIn(config,command)
    def test_retry_publication_cannot_rebind_exit_or_destination(self):
        batch=self.root/b.BATCH_ROOT/str(uuid.uuid4());batch.mkdir(parents=True)
        original=self.publication({}, {'exit_code':2})
        for key,value in (('push_destination','refs/heads/other'),
                          ('acquisition_exit',{'exit_code':0}),('wrapper_sha256','0'*64)):
            changed=dict(original);changed[key]=value
            (batch/'PUBLICATION.json').write_text(json.dumps(changed))
            fake=Mock()
            with patch.object(b,'Git',return_value=fake),patch.object(b,'git_preflight',return_value='base'), \
                 patch.object(b,'collect_batch',return_value=({'base_commit':'base'},{'exit_code':2},[],{})),self.assertRaises(b.Pending):
                b.commit_or_retry(batch)
            fake.run.assert_not_called()
    def test_nested_batch_filename_not_allowlisted(self):
        batch=self.root/b.BATCH_ROOT/str(uuid.uuid4());batch.mkdir(parents=True)
        b.write_new(batch/'plan.json',{'source':'hardware','first_stage':19})
        b.write_new(batch/'batch.json',{'plan_sha256':b.sha(batch/'plan.json'),'mode':'run'})
        b.write_new(batch/'acquisition_exit.json',{'child_terminated':True,'sessions':[]})
        self.write(batch.relative_to(self.root).as_posix()+'/nested/PUBLICATION.json',b'{}')
        with self.assertRaises(b.Pending):b.collect_batch(batch)
    def test_untracked_extra_commit_hook_rejected(self):
        self.write('.githooks/pre-commit',b'approved')
        with patch.object(b,'HOOKS',{'.githooks/pre-commit':b.sha(self.root/'.githooks/pre-commit')}):
            b.verify_hooks()
            self.write('.githooks/commit-msg',b'unapproved')
            with self.assertRaises(b.Pending):b.verify_hooks()
    def test_real_isolated_git_hook_failure_commit_retry_and_lfs(self):
        """Only temp Git repository; stub security scanner; no remote operations."""
        def git(*args):
            return subprocess.run(['git',*args],cwd=self.root,env=b.safe_env(),check=True,
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
        git('init');git('config','user.name','Synthetic Test');git('config','user.email','synthetic@example.invalid')
        git('config','filter.lfs.process','git-lfs filter-process');git('config','filter.lfs.required','true')
        git('remote','add','origin','https://github.com/'+b.REMOTE_REPO+'.git')
        hook=b'#!/bin/sh\nprintf "called\\n" >> .git/test-hook-calls\n[ ! -f .git/test-hook-fail ]\n'
        self.write('.githooks/pre-commit',hook)
        self.write('.githooks/pre-push',b'#!/bin/sh\nexit 1\n')
        self.write('.gitattributes',b'* -text\nresults/**/*.log filter=lfs diff=lfs merge=lfs -text\n')
        git('add','.githooks','.gitattributes');git('commit','-m','synthetic baseline')
        base=git('rev-parse','HEAD').decode().strip()
        pins={p:b.sha(self.root/p) for p in ('.githooks/pre-commit','.githooks/pre-push')}
        batch=self.root/b.BATCH_ROOT/str(uuid.uuid4());batch.mkdir(parents=True)
        plan={'source':'simulator','first_stage':0,'stop_stage_exclusive':3}
        b.write_new(batch/'plan.json',plan)
        b.write_new(batch/'batch.json',{'base_commit':base,'mode':'run','plan_sha256':b.sha(batch/'plan.json')})
        b.write_new(batch/'acquisition_exit.json',{'child_terminated':True,'exit_code':2,'sessions':[]})
        (batch/'console.log').write_bytes(b'Synthetic child failed before acquisition\n')
        (self.root/'.git/test-hook-fail').write_text('fail')
        with patch.object(b,'HOOKS',pins),patch.object(b.security,'find_scanner',return_value='synthetic-scanner'), \
             patch.object(b,'scan_files',side_effect=lambda names:{n:{'synthetic_scan_only':True} for n in names}) as scans:
            with self.assertRaises(b.Pending):b.commit_or_retry(batch)
            self.assertEqual(git('rev-parse','HEAD').decode().strip(),base)
            (self.root/'.git/test-hook-fail').unlink()
            result=b.commit_or_retry(batch)
            again=b.commit_or_retry(batch)
        self.assertEqual(result['commit'],again['commit'])
        self.assertEqual(result['acquisition'],2);self.assertFalse(result['semantic_audit_pass'])
        pointer=git('show','HEAD:'+ (batch/'console.log').relative_to(self.root).as_posix())
        self.assertEqual(b.security.parse_pointer(pointer,True),(b.sha(batch/'console.log'),(batch/'console.log').stat().st_size))
        self.assertEqual((self.root/'.git/test-hook-calls').read_text().count('called'),2)
        self.assertTrue(any(len(call.args[0])==5 for call in scans.call_args_list))
        self.assertEqual(git('status','--porcelain'),b'')


if __name__=='__main__':unittest.main()
