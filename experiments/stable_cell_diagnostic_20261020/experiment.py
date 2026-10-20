"""Saved-A reselect and finite fresh B/C continuation. No repeated phase A."""
import argparse
import contextlib
import json
import logging
from pathlib import Path
import random
import shutil
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'.deps'):
    if p.is_dir():sys.path.insert(0,str(p))
from experiments.stable_cell_calibration_20261020 import core,experiment as base,plan
from experiments.stable_cell_diagnostic_20261020 import reselect
from experiments.cifar_resumable_20261019.durable import Writer,atomic_json,sha
from experiments.unn_rev3_20261014.paths import repo_path,use_local_dependencies
from read_all import now


def limits(selection):
    selected=selection['selected'];m=len(selected['pairs']);k=len(selected['bls']);d=2*m
    masks=plan.masks(selected['bls'])
    b=4*m*k*(48+160)
    c=2*3*(k*d*4+d*3)+sum(3*d*(3+len(mask)) for group in masks.values() for mask in group)
    return dict(maximum_impacts=b+c,maximum_pulses=4*m*k*24,
                phase_b_impacts=b,phase_c_impacts=c,wait_s=240.,soft_deadline_s=1800.)


class Continuation(core.Collector):
    def __init__(self,connector,converters,writer,source,candidate,**kwargs):
        super().__init__(connector,converters,writer,source,**kwargs)
        self.selection=candidate['selection'];self.cells=candidate['cells'];self.candidate=candidate
        self.caps=limits(self.selection);self.phase='B'

    def budget(self):
        if self.calls>=self.caps['maximum_impacts']:raise ValueError('Diagnostic continuation impact cap')
        if self.clock()-self.started>=self.caps['soft_deadline_s']:raise TimeoutError('Diagnostic finite deadline')

    def run_continuation(self):
        self.emit('diagnostic_session_begin',rule=reselect.RULE,candidate=self.candidate,caps=self.caps,
                  phase_A_repeated=False)
        try:
            if not self.connector.open_port():raise ConnectionError('Connector open failed')
            self.phase_b()
            self.pair_qualification=reselect.pair_diagnostic(
                self.selection['selected'],self.cells,self.pair_observations)
            self.emit('diagnostic_pairs_evaluated',qualification=self.pair_qualification)
            if not self.pair_qualification['mandatory_pass']:self.status='pair_qualification_failed'
            else:
                self.phase_c();self.status='completed'
            if self.pulses>self.caps['maximum_pulses']:raise ValueError('Diagnostic pulse cap mismatch')
        except (Exception,KeyboardInterrupt) as exc:
            self.status='failed';self.error=type(exc).__name__+': '+str(exc)
        finally:
            try:
                if not self.connector.close_port():raise ConnectionError('Connector close failed')
            except Exception as exc:self.status='failed';self.error=repr(exc)
            self.emit('diagnostic_session_end',status=self.status,error=self.error,impacts=self.calls,pulses=self.pulses)
        result=self.result();result.update(qualified=False,diagnostic=True,phase_A_repeated=False,
                                          original_verdict=self.candidate['original_verdict'])
        return result


def acquire(source,candidate_path):
    candidate=json.loads(candidate_path.read_text(encoding='utf-8'))
    if candidate['rule']!=reselect.RULE or candidate['source']!=source:raise ValueError('Rule/source mismatch')
    expected=base.metrics.select_rectangle(candidate['cells'])
    if expected!=candidate['selection'] or expected['selected'] is None:
        raise ValueError('No fixed diagnostic rectangle')
    original=repo_path(candidate['original_session'])
    if sha(original/'manifest.json')!=candidate['original_manifest_sha256']:
        raise ValueError('Original session identity changed')
    original_manifest=json.loads((original/'manifest.json').read_text(encoding='utf-8'))
    if sha(original/'observations.json')!=candidate['observations_sha256']:
        raise ValueError('Original observations changed')
    replay=reselect.select(json.loads((original/'observations.json').read_text(encoding='utf-8')))
    if any(candidate[k]!=v for k,v in replay.items()):raise ValueError('Diagnostic candidate differs from saved A')
    config,hashes=base.preflight(source)
    hashes.update({p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.iterdir() if p.suffix=='.py'})
    out=ROOT/'results/stable_cell_diagnostic_20261020'/str(uuid.uuid4());out.mkdir(parents=True)
    if shutil.disk_usage(out).free<4*1024**3:raise ValueError('Insufficient disk reserve')
    (out/'candidate.json').write_bytes(candidate_path.read_bytes())
    (out/'board.ini').write_bytes((ROOT/base.INI).read_bytes())
    manifest=dict(status='failed',source=source,session_id=out.name,started_utc=now(),
                  candidate_sha256=sha(out/'candidate.json'),source_sha256=hashes,caps=limits(expected),
                  phase_A_repeated=False,private_runtime_status='private_runtime_not_independently_reverified')
    lock=ROOT/'results/read_all/.hardware.lock';locked=False;writer=runner=None
    try:
        if source=='hardware':
            lock.parent.mkdir(parents=True,exist_ok=True)
            with lock.open('x',encoding='utf-8') as f:f.write(out.name+'\n')
            locked=True
        use_local_dependencies()
        from manager.board import Connector
        from manager.service import converters
        writer=Writer(out/'raw',limit=8*1024**2,maximum_expanded_bytes=1024**3,minimum_free_bytes=1024**3)
        with (out/'lifecycle_output.txt').open('w',encoding='utf-8',newline='\n') as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                kwargs={}
                if source=='simulator':
                    import numpy as np
                    random.seed(plan.SPEC['seed']);np.random.seed(plan.SPEC['seed'])
                    if sha(original/'array.cb')!=original_manifest['result_sha256']['array.cb']:
                        raise ValueError('Original final simulator state changed')
                    serial=out/'array';shutil.copyfile(original/'array.cb',serial.with_suffix('.cb'))
                    shutil.copyfile(serial.with_suffix('.cb'),out/'array_initial.cb')
                    manifest['array_initial_sha256']=sha(out/'array_initial.cb')
                    kwargs['crossbar_serial']=str(serial)
                atomic_json(out/'start_manifest.json',manifest)
                connector=Connector(silent=True,logger=logging.getLogger(__name__),
                    cb_type='real' if source=='hardware' else 'simulator',board_type=config.get('board','board_type'),
                    task_format='SI',config=config,**kwargs)
                runner=Continuation(connector,converters,writer,source,candidate)
                result=runner.run_continuation()
        atomic_json(out/'observations.json',result)
        for split in ('before','fit','selection','heldout','after'):
            if split in runner.calibration:atomic_json(out/('C_'+split+'.json'),runner.calibration[split])
        atomic_json(out/'map.json',dict(cells=result['cells'],selection=result['selection'],
            pair_qualification={**(result['pair_qualification'] or {}),'cells':
                (result['pair_qualification'] or {}).get('original_quality_evaluation',{}).get('cells',[])},
            calibration=result['calibration'],qualified=False,diagnostic_best_available=True,
            original_verdict=candidate['original_verdict'],new_rule=reselect.RULE))
        manifest.update(status=result['status'],error=result['error'],impacts=result['impacts'],pulses=result['pulses'])
    except (Exception,KeyboardInterrupt) as exc:manifest.update(error=type(exc).__name__+': '+str(exc))
    finally:
        if writer:
            try:writer.finish()
            except Exception as exc:manifest.update(status='failed',raw_finish_error=repr(exc))
        if runner and not (out/'observations.json').exists():atomic_json(out/'observations.json',runner.result())
        manifest['finished_utc']=now()
        manifest['result_sha256']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
        atomic_json(out/'manifest.json',manifest)
        if locked:
            if lock.read_text(encoding='utf-8').strip()!=out.name:raise ValueError('Lock owner changed')
            lock.unlink()
    return dict(session=out.relative_to(ROOT).as_posix(),status=manifest['status'],error=manifest.get('error'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);m=parser.add_mutually_exclusive_group(required=True)
    m.add_argument('--reselect',action='store_true');m.add_argument('--run-simulator',action='store_true')
    m.add_argument('--run-hardware',action='store_true')
    for name in ('session','output','candidate'):parser.add_argument('--'+name)
    args=parser.parse_args()
    if args.reselect:result=reselect.make_candidate(repo_path(args.session),repo_path(args.output))
    else:result=acquire('hardware' if args.run_hardware else 'simulator',repo_path(args.candidate))
    print(json.dumps(result,allow_nan=False));return 2 if result.get('status')=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
