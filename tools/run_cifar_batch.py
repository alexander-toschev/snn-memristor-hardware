"""Foreground reviewed CIFAR batch, then evidence-only preservation/commit.

No MemriBoard calls, physical retries, credential reads or scientific qualification.
Finalization and push retries never invoke acquisition.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.security import scan_staged_lfs as security

ENTRY='experiments/cifar_resumable_20261019/experiment.py'
ENTRY_SHA='f52a623436d73c30a368733dbb75c7666603905f5440201900f7862211b44379'
# Actual immutable entry bytes are checked below; no shell/source rewriting.
HOOKS={'.githooks/pre-commit':'7a25e5dbed9dd3fc71c37fa0888b554c9aebc674554ebaff6e1009676920f425',
       '.githooks/pre-push':'193f2f6be35f2285fe7b965c22b01319b6b1e2b666280d8235371e8fd3e26aeb',
       '.githooks/post-commit':'796dd3beaafc4b715b655d759a841eb133b22512f8b17a252fd68780bae215d7',
       '.githooks/post-checkout':'bbda59754e9e30c8ad41942885b98d57bd4e3561c852364fa8c36a843cfaa44a',
       '.githooks/post-merge':'d42e319e56564b9a730e65b4e52862f748e5486f7a0d1a535f48a5a16b9a096c',
       'tools/security/scan_staged_lfs.py':'af6246330a77446304fc6e7f3f44dcf2dcf057e609986c6bff68756567158669'}
SESSION_ROOT='results/cifar_resumable_20261019'
BATCH_ROOT='results/cifar_batch_preservation'
DESTINATION='refs/heads/codex/full-cifar-diagnostic'
REMOTE_REPO='alexander-toschev/snn-memristor-hardware'
SESSION_FILE=re.compile(r'(?:protocol\.json|board\.ini|continuation_plan\.json|execution_grant\.json|start_manifest\.json|manifest\.json|progress\.json(?:\.tmp)?|native_stdout\.txt\.gz|array\.cb|simulator_settings\.ini|fault_injection\.json|raw/index\.json(?:\.tmp)?|raw/part-[0-9]{6}\.jsonl\.gz)')
MAX_DECODED_FILE=2*1024**3


class Pending(Exception):
    """Safe error codes only; never include Git/scanner/native output."""


def require(value,code):
    if not value:raise Pending(code)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def now():return datetime.now(timezone.utc).isoformat()


def status(**fields):print(json.dumps(fields,sort_keys=True),flush=True)


def write_new(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def local(name):
    p,w=Path(name),PureWindowsPath(name)
    require(bool(name) and not p.is_absolute() and not w.drive and not w.root
            and '..' not in p.parts and '..' not in w.parts,'relative_path_required')
    path=ROOT/p
    require(path.resolve().is_relative_to(ROOT.resolve()),'path_escape')
    for parent in (path,*path.parents):
        if parent==ROOT.parent:break
        if parent.exists():
            info=parent.lstat()
            require(not parent.is_symlink() and not getattr(info,'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',1024),
                    'symlink_or_reparse_not_allowed')
    return path


def safe_env():
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_CONFIG_')
         and k not in ('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE','GIT_ASKPASS','SSH_ASKPASS',
                      'GIT_LFS_SKIP_PUSH','GITLEAKS_CONFIG','GITLEAKS_CONFIG_TOML')}
    env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never',GIT_ASKPASS='',SSH_ASKPASS='',
               GIT_SSH_COMMAND='ssh -oBatchMode=yes')
    return env


class Git:
    def run(self,*args,data=None,timeout=3600):
        fixed=['-c','core.hooksPath=.githooks','-c','gc.auto=0','-c','lfs.allowincompletepush=false']
        if args[0] in ('push','ls-remote'):
            fixed+=['-c','credential.helper=','-c','credential.helper=cache --timeout=43200']
        try:
            result=subprocess.run(['git',*fixed,*args],cwd=ROOT,env=safe_env(),input=data,
                stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=timeout)
        except (OSError,subprocess.SubprocessError):raise Pending('git_command_unavailable_or_timeout') from None
        require(result.returncode==0,'git_'+args[0]+'_failed')
        return result.stdout
    def text(self,*args):return self.run(*args).decode('utf-8').strip()


def remote_allowed(value):
    return value in ('https://github.com/'+REMOTE_REPO,'https://github.com/'+REMOTE_REPO+'.git',
                     'git@github.com:'+REMOTE_REPO,'git@github.com:'+REMOTE_REPO+'.git')


def verify_hooks():
    require(set(files_under(local('.githooks')))=={n for n in HOOKS if n.startswith('.githooks/')},
            'unexpected_hook_inventory')
    for name,pin in HOOKS.items():require(sha(local(name))==pin,'reviewed_hook_changed')


def git_preflight(git,require_clean_index=True):
    require(Path(git.text('rev-parse','--show-toplevel')).resolve()==ROOT.resolve(),'wrong_git_root')
    verify_hooks()
    hooks=git.text('config','--get','core.hooksPath')
    require(hooks=='.githooks','standard_hooks_required')
    require(git.text('config','--get','filter.lfs.process')=='git-lfs filter-process'
            and git.text('config','--get','filter.lfs.required')=='true','standard_lfs_filter_required')
    require(remote_allowed(git.text('remote','get-url','--all','origin')),'unexpected_or_credential_bearing_remote')
    require(remote_allowed(git.text('remote','get-url','--push','--all','origin')),'unexpected_push_remote')
    require(not git.run('ls-files','-u'),'unmerged_index')
    if require_clean_index:require(not git.run('diff','--cached','--name-only','-z'),'nonempty_initial_index')
    require(not git.run('diff','--name-only','-z'),'tracked_worktree_changes')
    security.find_scanner(ROOT)  # Fail before board launch when publication prerequisites are absent.
    return git.text('rev-parse','HEAD')


def files_under(directory):
    result=[]
    for path in directory.rglob('*'):
        local(path.relative_to(ROOT).as_posix())
        require(not path.is_symlink(),'symlink_evidence')
        if path.is_file():result.append(path.relative_to(ROOT).as_posix())
        else:require(path.is_dir(),'unsupported_evidence_type')
    return sorted(result)


def session_inventory(name,plan,plan_sha,require_final=False):
    directory=local(name)
    require(directory.parent==ROOT/SESSION_ROOT,'outside_cifar_sessions')
    try:require(str(uuid.UUID(directory.name))==directory.name,'session_uuid_required')
    except ValueError:raise Pending('session_uuid_required') from None
    names=files_under(directory)
    require(all(SESSION_FILE.fullmatch(Path(p).relative_to(directory.relative_to(ROOT)).as_posix()) for p in names),'unexpected_session_file')
    start_path=directory/'start_manifest.json'
    require(start_path.is_file(),'session_attribution_missing')
    start=json.loads(start_path.read_text())
    stage=start.get('stage_range')
    require(start.get('session_id')==directory.name and start.get('source')==plan['source']
        and start.get('continuation_plan_sha256')==plan_sha and stage and len(stage)==2
        and stage[1]==stage[0]+1 and plan['first_stage']<=stage[0]<plan['stop_stage_exclusive'], 'session_plan_binding_failed')
    require(sha(directory/'continuation_plan.json')==plan_sha,'session_plan_bytes_changed')
    manifest_path=directory/'manifest.json'; final=None
    if manifest_path.exists():
        final=json.loads(manifest_path.read_text())
        require(final.get('status') in ('completed','failed'),'unknown_session_status')
        for key in ('session_id','source','stage_range','continuation_plan_sha256','source_sha256'):
            require(final.get(key)==start.get(key),'start_final_identity_mismatch')
        observed={Path(p).relative_to(directory.relative_to(ROOT)).as_posix():sha(local(p)) for p in names if local(p)!=manifest_path}
        require(observed==final.get('result_sha256'),'final_inventory_mismatch')
    elif require_final:raise Pending('existing_session_needs_final_manifest')
    return dict(path=name,stage=stage[0],status=final['status'] if final else 'interrupted_no_final_manifest',
        semantic_audit_pass=False,files={p:dict(sha256=sha(local(p)),bytes=local(p).stat().st_size) for p in names})


def raw_chunks(path):
    with path.open('rb') as stream:
        yield from iter(lambda:stream.read(1024**2),b'')


def decoded_chunks(path,summary):
    """All gzip members; retain truncated-prefix status; corruption fails closed."""
    decoder=zlib.decompressobj(31); total=members=0; touched=False
    for chunk in raw_chunks(path):
        touched=True; pending=chunk
        while pending:
            if decoder.eof:members+=1;decoder=zlib.decompressobj(31)
            try:data=decoder.decompress(pending,1024**2)
            except zlib.error:raise Pending('gzip_corruption_cannot_fully_scan') from None
            total+=len(data);require(total<=MAX_DECODED_FILE,'decoded_file_budget_exceeded')
            if data:yield data
            pending=decoder.unconsumed_tail or decoder.unused_data
    summary.update(decoded_bytes=total,complete_gzip=bool(touched and decoder.eof),
        gzip_members=members+int(decoder.eof),truncated_prefix=bool(touched and not decoder.eof),empty_file=not touched,
        raw_bytes_scanned_separately=True)


def scan_chunks(scanner,chunks):
    with tempfile.TemporaryDirectory(prefix='cifar-secret-scan-') as isolated:
        process=None
        try:
            process=subprocess.Popen([scanner,'stdin','--redact=100','--no-banner','--ignore-gitleaks-allow',
                '--max-target-megabytes=0','--exit-code=1'],cwd=isolated,env=safe_env(),stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            for data in chunks:process.stdin.write(data)
            process.stdin.close();require(process.wait(timeout=300)==0,'secret_scan_failed_or_finding')
        except Pending:raise
        except (OSError,subprocess.SubprocessError):raise Pending('secret_scan_failed_or_finding') from None
        finally:
            if process is not None and process.poll() is None:
                process.kill();process.wait()


def scan_files(names):
    scanner=security.find_scanner(ROOT); result={}
    for name in names:
        path=local(name);scan_chunks(scanner,raw_chunks(path)); entry=dict(raw_scanned=True)
        if name.endswith('.gz'):scan_chunks(scanner,decoded_chunks(path,entry))
        result[name]=entry
    return result


def collect_batch(batch):
    declaration=json.loads((batch/'batch.json').read_text());exit_record=json.loads((batch/'acquisition_exit.json').read_text())
    require(exit_record.get('child_terminated') is True,'child_exit_not_confirmed')
    require(exit_record.get('attribution_verified',True) is True,'session_attribution_pending')
    plan=json.loads((batch/'plan.json').read_text());plan_sha=sha(batch/'plan.json')
    require(plan_sha==declaration['plan_sha256'],'batch_plan_changed')
    sessions=[session_inventory(name,plan,plan_sha,require_final=declaration['mode']=='preserve') for name in exit_record['sessions']]
    require(len({s['stage'] for s in sessions})==len(sessions),'ambiguous_duplicate_stage_attempts')
    require([s['stage'] for s in sessions]==list(range(plan['first_stage'],plan['first_stage']+len(sessions))), 'noncontiguous_new_stage_sessions')
    names=[name for session in sessions for name in session['files']]
    batch_names=files_under(batch)
    require({local(n).relative_to(batch).as_posix() for n in batch_names}<=
            {'batch.json','plan.json','console.log','acquisition_exit.json','PUBLICATION.json'},'unexpected_batch_file')
    names+=batch_names
    inventory={name:dict(sha256=sha(local(name)),bytes=local(name).stat().st_size) for name in names if not name.endswith('/PUBLICATION.json')}
    return declaration,exit_record,sessions,inventory


def commit_or_retry(batch,push=False):
    git=Git(); declaration,exit_record,sessions,inventory=collect_batch(batch)
    require(not local('results/read_all/.hardware.lock').exists(),'hardware_lock_present_preservation_pending')
    publication=batch/'PUBLICATION.json'; retried=publication.exists()
    head=git_preflight(git,require_clean_index=not retried)
    if retried:
        report=json.loads(publication.read_text());require(report['inventory']==inventory,'evidence_changed_since_scan')
        require(report.get('schema')=='cifar-evidence-preservation-v1'
            and report.get('base_commit')==declaration['base_commit']
            and report.get('wrapper_sha256')==sha(__file__)
            and report.get('acquisition_exit')==exit_record
            and report.get('sessions')==[{k:v for k,v in s.items() if k!='files'} for s in sessions]
            and report.get('semantic_audit_pass') is False
            and report.get('push_destination')==DESTINATION,'publication_binding_changed')
    else:
        require(head==declaration['base_commit'],'head_changed_since_launch')
        status(phase='secret_scan',files=len(inventory))
        scans=scan_files(sorted(inventory))
        require(collect_batch(batch)[3]==inventory,'evidence_changed_during_scan')
        report=dict(schema='cifar-evidence-preservation-v1',created_utc=now(),base_commit=head,
            wrapper_sha256=sha(__file__),acquisition_exit=exit_record,sessions=[{k:v for k,v in s.items() if k!='files'} for s in sessions],
            inventory=inventory,scan=scans,semantic_audit_pass=False,
            status='prepared_for_standard_hooks_commit',push_destination=DESTINATION)
        write_new(publication,report)
    names=sorted([*inventory,publication.relative_to(ROOT).as_posix()])
    allowed=set(names)
    staged=set(git.run('diff','--cached','--name-only','-z').decode().split('\0'))-{''}
    require(staged<=allowed,'unrelated_staged_changes')
    if head!=declaration['base_commit']:
        require(not staged and git.text('rev-parse','HEAD^')==declaration['base_commit'],'unexpected_head_on_retry')
        committed=set(git.run('diff-tree','--no-commit-id','--name-only','-r','-z','HEAD').decode().split('\0'))-{''}
        require(committed==allowed,'retry_commit_contains_other_paths')
        require(not git.run('diff','HEAD','--',*names),'committed_evidence_changed')
        commit=head
    else:
        # Metadata is scanned too; no unscanned logs or reports enter the commit.
        scan_files(names if retried else [publication.relative_to(ROOT).as_posix()])
        status(phase='stage_exact_evidence',files=len(names))
        git.run('--literal-pathspecs','add','--pathspec-from-file=-','--pathspec-file-nul',data=b'\0'.join(n.encode() for n in names)+b'\0')
        staged=set(git.run('diff','--cached','--name-only','-z').decode().split('\0'))-{''}
        require(staged==allowed,'staged_allowlist_mismatch')
        entries=security.staged_entries(ROOT)
        require({os.fsdecode(row[0]) for row in entries}==allowed,'staged_entry_coverage_mismatch')
        for path,mode,oid,attribute in entries:
            name=os.fsdecode(path);require(mode in (b'100644',b'100755'),'unsupported_staged_type')
            require(int(git.text('cat-file','-s',oid.decode()))<=2*1024**2,'oversized_non_pointer_git_blob')
            blob=git.run('cat-file','blob',oid.decode())
            pointer=security.parse_pointer(blob,attribute==b'lfs')
            if name.endswith(('.gz','.log','.cb','.npy','.f32')):require(pointer is not None,'required_lfs_pointer_missing')
            expected=sha(local(name))
            if pointer:require(pointer==(expected,local(name).stat().st_size),'staged_lfs_bytes_differ')
            else:require(hashlib.sha256(blob).hexdigest()==expected,'staged_plain_bytes_differ')
        require(collect_batch(batch)[3]==inventory,'evidence_changed_after_staging')
        require(git.text('rev-parse','HEAD')==declaration['base_commit'],'head_changed_before_commit')
        verify_hooks()
        # Existing pre-commit verifies staged plain/LFS bytes; no bypass flags/config.
        git.run('commit','-m','Preserve CIFAR finite batch '+batch.name)
        commit=git.text('rev-parse','HEAD')
        require(git.text('rev-parse','HEAD^')==declaration['base_commit'],'commit_parent_changed')
        require(set(git.run('diff-tree','--no-commit-id','--name-only','-r','-z','HEAD').decode().split('\0'))-{''}==allowed,'hook_changed_commit_allowlist')
        require(not git.run('diff','HEAD','--',*names) and collect_batch(batch)[3]==inventory,'evidence_changed_during_hooks')
    result=dict(acquisition=exit_record['exit_code'],preservation='committed',commit=commit,
        semantic_audit_pass=False,push='not_requested',remote_ref_verified=False,lfs_remote_status='not_checked')
    if push:
        try:
            status(phase='push_committed_evidence',commit=commit)
            verify_hooks()
            git.run('push','origin',commit+':'+DESTINATION)
            remote=git.text('ls-remote','--exit-code','origin',DESTINATION).split()
            require(remote==[commit,DESTINATION],'remote_ref_not_confirmed')
            result.update(push='completed',remote_ref_verified=True,
                lfs_remote_status='standard_pre_push_hook_succeeded; remote_payload_readback_not_performed')
        except Pending as exc:result.update(push='pending',push_error=str(exc))
    return result


def new_batch(plan_name,source,mode,sessions=None):
    git=Git();base=git_preflight(git)
    require(sha(local(ENTRY))==ENTRY_SHA,'frozen_entry_changed')
    from experiments.cifar_resumable_20261019 import contract
    plan_path=local(plan_name);plan=json.loads(plan_path.read_text())
    first,stop=contract.validate_plan(plan,source);contract.verify_pins()
    require(contract.execution_grant(source) is not None,'reviewed_execution_grant_missing')
    require(not local('results/read_all/.hardware.lock').exists(),'hardware_lock_present')
    batch=local(BATCH_ROOT+'/'+str(uuid.uuid4()));batch.mkdir(parents=True,exist_ok=False)
    (batch/'plan.json').write_bytes(plan_path.read_bytes())
    before=sorted(p.name for p in (ROOT/SESSION_ROOT).iterdir() if p.is_dir()) if (ROOT/SESSION_ROOT).exists() else []
    write_new(batch/'batch.json',dict(schema='cifar-finite-batch-wrapper-v1',mode=mode,created_utc=now(),
        base_commit=base,wrapper_sha256=sha(__file__),entry_sha256=ENTRY_SHA,source=source,
        plan_path=plan_name,plan_sha256=sha(plan_path),stage_range=[first,stop],sessions_before=before,
        publication_policy='after_child_exit_only; no scientific PASS; no physical retry'))
    if mode=='preserve':
        require(bool(sessions),'explicit_sessions_required')
        for name in sessions:session_inventory(name,plan,sha(plan_path),require_final=True)
        ordered=sorted(sessions,key=lambda n:json.loads((local(n)/'start_manifest.json').read_text())['stage_range'][0])
        write_new(batch/'acquisition_exit.json',dict(exit_code=None,child_terminated=True,
            observation='final manifests present and no hardware lock; wrapper did not execute acquisition',sessions=ordered))
    return batch


def run_child(batch):
    declaration=json.loads((batch/'batch.json').read_text());source=declaration['source']
    require(sha(local(declaration['plan_path']))==declaration['plan_sha256'],'plan_changed_before_launch')
    require(sha(local(ENTRY))==ENTRY_SHA,'entry_changed_before_launch')
    command=[sys.executable,'-B',ENTRY,'--run-'+source,'--plan',declaration['plan_path']]
    process=None;exit_code=None
    with (batch/'console.log').open('xb') as console:
        try:
            process=subprocess.Popen(command,cwd=ROOT,env=safe_env(),stdout=console,stderr=subprocess.STDOUT)
            status(phase='acquiring',batch=batch.relative_to(ROOT).as_posix(),pid=process.pid,stage_range=declaration['stage_range'])
            started=time.monotonic()
            while True:
                try:exit_code=process.wait(timeout=300);break
                except subprocess.TimeoutExpired:status(phase='acquiring',elapsed_seconds=int(time.monotonic()-started),pid=process.pid)
        except KeyboardInterrupt:
            # Same foreground process group receives Ctrl+C too. Never respawn it.
            if process is not None:
                status(phase='waiting_for_interrupted_child_close',pid=process.pid)
                exit_code=process.wait()
        finally:
            console.flush();os.fsync(console.fileno())
    require(process is not None and process.poll() is not None,'child_exit_not_confirmed')
    before=set(declaration['sessions_before'])
    created=[p for p in (ROOT/SESSION_ROOT).iterdir() if p.is_dir() and p.name not in before] if (ROOT/SESSION_ROOT).exists() else []
    plan=json.loads((batch/'plan.json').read_text()); attributed=[]; attribution_ok=True
    try:
        for session in created:
            item=session_inventory(session.relative_to(ROOT).as_posix(),plan,declaration['plan_sha256'])
            attributed.append((item['stage'],item['path']))
    except (Pending,OSError,ValueError,KeyError):
        attribution_ok=False
    write_new(batch/'acquisition_exit.json',dict(exit_code=exit_code,child_pid=process.pid,child_terminated=True,
        finished_utc=now(),attribution_verified=attribution_ok,
        sessions=[name for _,name in sorted(attributed)],console_sha256=sha(batch/'console.log')))
    require(attribution_ok,'new_session_attribution_pending')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('run','preserve'):
        p=sub.add_parser(mode);p.add_argument('--plan',required=True);p.add_argument('--source',choices=('hardware','simulator'),required=True)
        p.add_argument('--push',action='store_true')
        if mode=='preserve':p.add_argument('--session',action='append',required=True)
    p=sub.add_parser('finalize');p.add_argument('--batch',required=True);p.add_argument('--push',action='store_true')
    args=parser.parse_args(argv)
    lock=ROOT/'.tmp/cifar-publication.lock';lock.parent.mkdir(parents=True,exist_ok=True);owned=False;token=str(uuid.uuid4())
    try:
        with lock.open('x') as stream:stream.write(token)
        owned=True
        if args.mode=='finalize':
            batch=local(args.batch);require(batch.parent==ROOT/BATCH_ROOT,'wrong_batch_namespace')
        else:
            batch=new_batch(args.plan,args.source,args.mode,getattr(args,'session',None))
            if args.mode=='run':run_child(batch)
        result=commit_or_retry(batch,args.push)
        status(batch=batch.relative_to(ROOT).as_posix(),**result)
        return 0 if result['push']!='pending' and result['acquisition'] in (None,0) else 2
    except (Pending,security.Blocked,OSError,ValueError,KeyError,subprocess.SubprocessError):
        # No arbitrary scanner/Git/native exception text, URLs, credentials or payloads.
        status(preservation='pending_local_review',push='not_attempted_or_unconfirmed',
               evidence_kept=True,physical_retry_performed=False)
        return 3
    finally:
        if owned and lock.exists() and lock.read_text()==token:lock.unlink()


if __name__=='__main__':raise SystemExit(main())
