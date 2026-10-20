"""Bounded public-MemriBoard SIM programming/readback; no CIFAR inference or labels."""
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
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
import numpy as np
from experiments.stable_cell_calibration_20261020 import core as io,experiment as calibration
from experiments.cifar_resumable_20261019.durable import Writer,atomic_json,sha
from experiments.unn_rev3_20261014.paths import use_local_dependencies,repo_path
from read_all import now

LEVELS=[.3,.6,.9,1.2,1.5,2.,2.5,3.]
BASE=ROOT/'results/cifar_confirmatory_native_20261020'
SPEC=dict(schema='confirmatory-conv1-native-readback-v1',source='native_memriboard_simulator',
    physical_hardware=False,native_mvm=False,mode=7,shape=[32,8],levels_v=LEVELS,
    pulse_width_s=.000125,physical_duration_equivalence_claim=False,
    reset_current_limit_a=.04,set_current_limit_a=.002,
    units='G=1/nativeR in Siemens; not GUI weight, not330/R proxy',
    mapping='L/H=10%/90% between median RESET and SET endpoints of8cells*2cycles',
    signed_mapping='Gplus=L+positive(w)*span/maxabs_filter; Gminus=L+negative(w)*span/maxabs_filter',
    programming='each leg8RESET then up to8SET; stop at crossing or tolerance max(.01span,3SE), no retries',
    read_repeats=3,characterization_cells=[[wl,4*wl] for wl in range(8)],
    characterization_cycles=2,characterization_max_impacts=1304,characterization_max_pulses=256,
    programming_max_impacts=825600,programming_max_pulses=153600,
    all_filters=64,coefficients=4800,pairs=[[0,1],[2,3],[4,5],[6,7]],
    target_misses='retain measured values; approximate continuation; never ideal-weight replacement',
    soft_deadline_s=14400,labels_used=False)


class Collector(io.Collector):
    def __init__(self,*args,kind,**kwargs):
        super().__init__(*args,source='simulator',**kwargs)
        self.kind=kind;self.status='failed';self.measures=[]

    def budget(self):
        cap=SPEC['characterization_max_impacts'] if self.kind=='characterize' else SPEC['programming_max_impacts']
        if self.calls>=cap:raise ValueError('Finite native readback impact cap')
        if self.clock()-self.started>=SPEC['soft_deadline_s']:raise TimeoutError('Finite native readback deadline')

    def read_g(self,wl,bl,**metadata):
        reads=[self.read_r(wl,bl,repeat=i,**metadata) for i in range(3)]
        g=[1/r['resistance_ohm'] for r in reads]
        row=dict(wl=wl,bl=bl,metadata=metadata,reads=reads,mean_g_s=float(np.mean(g)),
                 se_g_s=float(np.std(g,ddof=1)/np.sqrt(3)))
        self.emit('conductance_read',measurement=row)
        return row

    def decision(self,wl,bl,sign,level,**metadata):
        if sign not in (0,1) or level not in LEVELS:raise ValueError('Invalid pulse')
        cap=SPEC['characterization_max_pulses'] if self.kind=='characterize' else SPEC['programming_max_pulses']
        if self.decisions>=cap:raise ValueError('Finite pulse-decision cap')
        guard=self.read_r(wl,bl,role='pulse_guard',sign=sign,level=level,**metadata)
        self.decisions+=1
        predicted=level/guard['resistance_ohm'];admitted=predicted<=(.002 if sign else .04)
        result=dict(wl=wl,bl=bl,sign=sign,level_v=level,guard=guard,predicted_current_a=predicted,
                    admitted=admitted,decision=self.decisions,metadata=metadata)
        if admitted:
            self.pulses+=1
            result['write']=self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=level,sign=sign,pulse_width=.000125),
                                      role='pulse',**metadata)
        result['post']=self.read_g(wl,bl,role='post_decision',sign=sign,level=level,**metadata)
        result['delta_g_s']=result['post']['mean_g_s']-1/guard['resistance_ohm']
        self.emit('program_decision_complete',decision=result)
        return result['post']

    def characterize(self):
        endpoints=[]
        for wl,bl in SPEC['characterization_cells']:
            self.read_g(wl,bl,role='initial')
            for cycle in range(2):
                for sign in (0,1):
                    for step,level in enumerate(LEVELS):
                        row=self.decision(wl,bl,sign,level,cycle=cycle,step=step)
                    endpoints.append(dict(wl=wl,bl=bl,cycle=cycle,sign=sign,measurement=row))
                self.durable(wl=wl,bl=bl,cycle=cycle)
        low=float(np.median([r['measurement']['mean_g_s'] for r in endpoints if r['sign']==0]))
        high=float(np.median([r['measurement']['mean_g_s'] for r in endpoints if r['sign']==1]))
        if not high>low:raise ValueError('Characterization has no positive median endpoint span')
        span=high-low
        mapping=dict(low_g_s=low+.1*span,high_g_s=high-.1*span,span_g_s=.8*span,
                     median_reset_g_s=low,median_set_g_s=high,selection_uses_labels=False)
        self.emit('mapping_frozen',mapping=mapping)
        return dict(endpoints=endpoints,mapping=mapping)

    def program(self,weights,mapping,out):
        if weights.shape!=(64,75) or not np.isfinite(weights).all():raise ValueError('Complete Conv1 weights required')
        low,span=mapping['low_g_s'],mapping['span_g_s']
        maxabs=np.max(np.abs(weights),axis=1)
        if not (span>0 and low>0 and np.all(maxabs>0)):raise ValueError('Invalid mapping')
        measured=np.full((64,75,2),np.nan);targets=np.full_like(measured,np.nan)
        counts=np.zeros((64,75,2),dtype=np.int8);stage=0
        for first_filter in range(0,64,4):
            for first_tap in (0,32,64):
                cells=[]
                self.emit('readback_stage_begin',stage=stage,first_filter=first_filter,first_tap=first_tap)
                for offset,pair in enumerate(SPEC['pairs']):
                    f=first_filter+offset
                    for tap in range(first_tap,min(first_tap+32,75)):
                        bl=tap-first_tap
                        for leg,wl in enumerate(pair):
                            target=low+max(float(weights[f,tap])*(1 if leg==0 else -1),0)*span/float(maxabs[f])
                            targets[f,tap,leg]=target
                            meta=dict(stage=stage,filter=f,tap=tap,leg=leg,target_g_s=target)
                            current=self.read_g(wl,bl,role='initial_leg',**meta)
                            for step,level in enumerate(LEVELS):
                                current=self.decision(wl,bl,0,level,step=step,**meta)
                            for step,level in enumerate(LEVELS):
                                tolerance=max(.01*span,3*current['se_g_s'])
                                if current['mean_g_s']>=target or abs(current['mean_g_s']-target)<=tolerance:break
                                current=self.decision(wl,bl,1,level,step=step,**meta)
                                counts[f,tap,leg]+=1
                            self.emit('leg_programmed',**meta,actual=current,set_decisions=int(counts[f,tap,leg]))
                            cells.append((wl,bl,meta))
                # Read only after all other writes in this tile, then preserve its state.
                for wl,bl,meta in cells:
                    r=self.read_g(wl,bl,role='final_readback',**meta)
                    measured[meta['filter'],meta['tap'],meta['leg']]=r['mean_g_s']
                shutil.copyfile(out/'array.cb',out/('array_stage_%02d.cb'%stage))
                self.emit('readback_stage_complete',stage=stage,cells=len(cells))
                self.durable(stage=stage);stage+=1
        if not np.isfinite(measured).all():raise ValueError('Missing measured weight; no CPU fill')
        effective=(measured[:,:,0]-measured[:,:,1])*maxabs[:,None]/span
        error=effective-weights
        np.save(out/'effective_conv1.npy',effective.reshape(64,3,5,5).astype(np.float32),allow_pickle=False)
        return dict(mapping=mapping,measured_g_s=measured.tolist(),target_g_s=targets.tolist(),
            maxabs_per_filter=maxabs.tolist(),set_decisions=counts.tolist(),completed_stages=stage,
            coefficients=4800,weight_rmse=float(np.sqrt(np.mean(error**2))),
            weight_relative_rmse=float(np.sqrt(np.mean(error**2))/np.std(weights)),
            signed_match_fraction=float(np.mean(np.sign(effective)==np.sign(weights))),
            target_miss_fraction=float(np.mean(np.abs(measured-targets)>.01*span)),
            effective_conv1_sha256=sha(out/'effective_conv1.npy'),ideal_weight_replacement=False)


def execute(kind,seed,characterization=None):
    config,public=calibration.preflight('simulator')
    mapping=None;weights=None;binding=None
    if kind=='program':
        sys.path.insert(0,str(ROOT/'.deps-classifier'))
        import torch
        from experiments.cifar_confirmatory_20261020.evaluate import assets
        folder,_,_,_=assets(seed)
        weights=torch.load(folder/'best_fs.pt',map_location='cpu',weights_only=True)['conv_layers.0.weight'].numpy().astype(float).reshape(64,75)
        cm=json.loads((characterization/'manifest.json').read_text())
        cr=characterization/'observations.json'
        if cm['status']!='completed' or cm['kind']!='characterize' or sha(cr)!=cm['result_sha256']['observations.json']:
            raise ValueError('Completed unchanged characterization required')
        mapping=json.loads(cr.read_text())['mapping']
        binding=dict(session=characterization.relative_to(ROOT).as_posix(),manifest_sha256=sha(characterization/'manifest.json'),
                     mapping=mapping,checkpoint_sha256=sha(folder/'best_fs.pt'),seed=seed)
    sid=str(uuid.uuid4());out=BASE/sid
    if shutil.disk_usage(ROOT).free<6*1024**3:raise ValueError('Need6GiB disk for raw evidence')
    out.mkdir(parents=True)
    manifest=dict(schema=SPEC['schema'],kind=kind,session_id=sid,status='failed',source='simulator',
        started_utc=now(),protocol=SPEC,binding=binding,random_seed=20261020+seed if kind=='program' else 20261021,
        source_sha256={**public,**{p.relative_to(ROOT).as_posix():sha(p) for p in
            [Path(__file__),Path(io.__file__),ROOT/'experiments/cifar_resumable_20261019/durable.py']}},
        private_runtime_status='private_runtime_not_independently_reverified',classification_performed=False)
    writer=runner=connector=None;opened=False
    try:
        shutil.copyfile(ROOT/calibration.INI,out/'board.ini')
        shutil.copyfile(ROOT/calibration.SIM_INI,out/'simulator_settings.ini')
        if binding:atomic_json(out/'mapping.json',binding)
        with (out/'lifecycle_output.txt').open('w',encoding='utf-8',newline='\n') as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                use_local_dependencies()
                from manager.board import Connector
                from manager.service import converters
                from simulator.src import create_crossbar_array
                random.seed(manifest['random_seed']);np.random.seed(manifest['random_seed'])
                create_crossbar_array(str(out/'array'),32,8)
                shutil.copyfile(out/'array.cb',out/'array_initial.cb')
                manifest['initial_array_sha256']=sha(out/'array_initial.cb')
                atomic_json(out/'start_manifest.json',manifest)
                writer=Writer(out/'raw',limit=8*1024**2,maximum_expanded_bytes=4*1024**3,minimum_free_bytes=2*1024**3)
                connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                    board_type=config.get('board','board_type'),task_format='SI',config=config,crossbar_serial=str(out/'array'))
                runner=Collector(connector,converters,writer,kind=kind)
                runner.emit('readback_session_begin',protocol=SPEC,kind=kind,binding=binding)
                opened=True;result=connector.open_port();runner.emit('port_open',result=io.safe(result))
                if not result:raise ValueError('Public open failed')
                result=runner.characterize() if kind=='characterize' else runner.program(weights,mapping,out)
                atomic_json(out/'observations.json',result)
                manifest['status']='completed'
    except (Exception,KeyboardInterrupt) as exc:manifest['error']=type(exc).__name__+': '+str(exc)
    finally:
        if opened:
            try:
                result=connector.close_port();runner.emit('port_close',result=io.safe(result))
                if not result:raise ValueError('Public close failed')
            except Exception as exc:manifest.update(status='failed',cleanup_error=repr(exc))
        if runner:
            manifest.update(impacts=runner.calls,pulses=runner.pulses,decisions=runner.decisions)
            runner.emit('readback_session_end',status=manifest['status'],error=manifest.get('error'),
                        impacts=runner.calls,pulses=runner.pulses)
        if writer:
            try:writer.finish()
            except Exception as exc:manifest.update(status='failed',raw_finish_error=repr(exc))
        manifest['finished_utc']=now()
        manifest['result_sha256']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
        atomic_json(out/'manifest.json',manifest)
    return dict(session=out.relative_to(ROOT).as_posix(),status=manifest['status'],error=manifest.get('error'),
                impacts=manifest.get('impacts'),pulses=manifest.get('pulses'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--characterize',action='store_true')
    p.add_argument('--seed',type=int,choices=[0,1,2,42]);p.add_argument('--characterization')
    a=p.parse_args()
    if a.characterize:
        if a.seed is not None or a.characterization:p.error('Characterization takes no checkpoint')
        result=execute('characterize',0)
    else:
        if a.seed is None or not a.characterization:p.error('Programming requires seed and fixed characterization')
        result=execute('program',a.seed,repo_path(a.characterization))
    print(json.dumps(result));raise SystemExit(0 if result['status']=='completed' else 2)
