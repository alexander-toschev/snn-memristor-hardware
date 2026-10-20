"""Independent mode7 replay -> measured Conv1 weights -> explicitly hybrid Torch eval."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.deps-classifier'),str(ROOT/'.deps'),str(ROOT)]
import numpy as np
import torch
from experiments.cifar_confirmatory_20261020 import evaluate
from experiments.cifar_full_case0_20261019.trace import records
from experiments.cifar_resumable_20261019.durable import sha,atomic_json
from experiments.unn_rev3_20261014.paths import repo_path
from tools.audit_mnist_conductance import require,same,exact_command

LEVELS=[.3,.6,.9,1.2,1.5,2.,2.5,3.]
PAIRS=[[0,1],[2,3],[4,5],[6,7]]


class Replay:
    def __init__(self,rows):
        self.rows=iter(rows);self.calls=self.pulses=self.decisions=0
        self.last_clock=None

    def take(self,event):
        row=next(self.rows)
        require(row['event']==event and row['source']=='simulator' and row['phase']=='A','Unexpected raw event/source')
        clock=row['monotonic_s']
        require(math.isfinite(clock) and (self.last_clock is None or clock>=self.last_clock),'Nonmonotonic raw')
        self.last_clock=clock
        return {k:v for k,v in row.items() if k not in ('event','source','phase','utc','monotonic_s')}

    def expect(self,event,**fields):
        same(self.take(event),fields,event)

    def call(self,command,metadata):
        self.calls+=1;task={**command,'id':self.calls}
        request=self.take('request')
        exact_command(request['command_si'],task)
        same(request,dict(task_id=self.calls,command_si=task,metadata=metadata),'request ownership')
        row=next(self.rows)
        while row['event']=='driver_output':
            require(row['task_id']==self.calls and row['source']=='simulator','Wrong driver output ownership')
            row=next(self.rows)
        require(row['event']=='result' and row['source']=='simulator','Missing native result')
        require(row['task_id']==self.calls and row['metadata']==metadata,'Native result ownership')
        exact_command(row['command_si'],task)
        r,echo,adc=row['native_result']
        require(type(r) in (int,float) and math.isfinite(r) and r>1e-8 and
                type(echo) is int and echo==self.calls and type(adc) is int and 20<=adc<=16382,'Invalid native R/ADC/ID')
        require(math.isfinite(row['latency_s']) and row['latency_s']>=0,'Invalid latency')
        return dict(task_id=self.calls,resistance_ohm=r,adc=adc)

    def read(self,wl,bl,**metadata):
        return self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=0.,sign=0,pulse_width=0.),metadata)

    def read_g(self,wl,bl,**metadata):
        reads=[self.read(wl,bl,repeat=i,**metadata) for i in range(3)]
        g=[1/r['resistance_ohm'] for r in reads]
        result=dict(wl=wl,bl=bl,metadata=metadata,reads=reads,mean_g_s=float(np.mean(g)),se_g_s=float(np.std(g,ddof=1)/np.sqrt(3)))
        self.expect('conductance_read',measurement=result)
        return result

    def decision(self,wl,bl,sign,level,**metadata):
        guard=self.read(wl,bl,role='pulse_guard',sign=sign,level=level,**metadata)
        self.decisions+=1;predicted=level/guard['resistance_ohm'];admitted=predicted<=(.002 if sign else .04)
        result=dict(wl=wl,bl=bl,sign=sign,level_v=level,guard=guard,predicted_current_a=predicted,
            admitted=admitted,decision=self.decisions,metadata=metadata)
        if admitted:
            self.pulses+=1
            result['write']=self.call(dict(mode_flag=7,wl=wl,bl=bl,vol=level,sign=sign,pulse_width=.000125),dict(role='pulse',**metadata))
        result['post']=self.read_g(wl,bl,role='post_decision',sign=sign,level=level,**metadata)
        result['delta_g_s']=result['post']['mean_g_s']-1/guard['resistance_ohm']
        self.expect('program_decision_complete',decision=result)
        return result['post']

    def characterize(self):
        endpoints=[]
        for wl in range(8):
            bl=4*wl;self.read_g(wl,bl,role='initial')
            for cycle in range(2):
                for sign in (0,1):
                    for step,level in enumerate(LEVELS):
                        result=self.decision(wl,bl,sign,level,cycle=cycle,step=step)
                    endpoints.append(dict(wl=wl,bl=bl,cycle=cycle,sign=sign,measurement=result))
        low=float(np.median([r['measurement']['mean_g_s'] for r in endpoints if r['sign']==0]))
        high=float(np.median([r['measurement']['mean_g_s'] for r in endpoints if r['sign']==1]))
        require(high>low,'No mapping span')
        span=high-low
        mapping=dict(low_g_s=low+.1*span,high_g_s=high-.1*span,span_g_s=.8*span,
                     median_reset_g_s=low,median_set_g_s=high,selection_uses_labels=False)
        self.expect('mapping_frozen',mapping=mapping)
        return dict(endpoints=endpoints,mapping=mapping),None

    def program(self,weights,mapping):
        low,span=mapping['low_g_s'],mapping['span_g_s'];maxabs=np.max(np.abs(weights),axis=1)
        measured=np.full((64,75,2),np.nan);targets=np.full_like(measured,np.nan)
        counts=np.zeros((64,75,2),dtype=np.int8);stage=0
        for first_filter in range(0,64,4):
            for first_tap in (0,32,64):
                self.expect('readback_stage_begin',stage=stage,first_filter=first_filter,first_tap=first_tap)
                cells=[]
                for offset,pair in enumerate(PAIRS):
                    f=first_filter+offset
                    for tap in range(first_tap,min(first_tap+32,75)):
                        bl=tap-first_tap
                        for leg,wl in enumerate(pair):
                            target=low+max(float(weights[f,tap])*(1 if leg==0 else -1),0)*span/float(maxabs[f])
                            targets[f,tap,leg]=target
                            meta=dict(stage=stage,filter=f,tap=tap,leg=leg,target_g_s=target)
                            current=self.read_g(wl,bl,role='initial_leg',**meta)
                            for step,level in enumerate(LEVELS):current=self.decision(wl,bl,0,level,step=step,**meta)
                            for step,level in enumerate(LEVELS):
                                tolerance=max(.01*span,3*current['se_g_s'])
                                if current['mean_g_s']>=target or abs(current['mean_g_s']-target)<=tolerance:break
                                current=self.decision(wl,bl,1,level,step=step,**meta);counts[f,tap,leg]+=1
                            self.expect('leg_programmed',**meta,actual=current,set_decisions=int(counts[f,tap,leg]))
                            cells.append((wl,bl,meta))
                for wl,bl,meta in cells:
                    row=self.read_g(wl,bl,role='final_readback',**meta)
                    measured[meta['filter'],meta['tap'],meta['leg']]=row['mean_g_s']
                self.expect('readback_stage_complete',stage=stage,cells=len(cells));stage+=1
        require(np.isfinite(measured).all(),'Missing readback')
        effective=(measured[:,:,0]-measured[:,:,1])*maxabs[:,None]/span
        error=effective-weights
        result=dict(mapping=mapping,measured_g_s=measured.tolist(),target_g_s=targets.tolist(),
            maxabs_per_filter=maxabs.tolist(),set_decisions=counts.tolist(),completed_stages=stage,
            coefficients=4800,weight_rmse=float(np.sqrt(np.mean(error**2))),
            weight_relative_rmse=float(np.sqrt(np.mean(error**2))/np.std(weights)),
            signed_match_fraction=float(np.mean(np.sign(effective)==np.sign(weights))),
            target_miss_fraction=float(np.mean(np.abs(measured-targets)>.01*span)),ideal_weight_replacement=False)
        return result,effective.reshape(64,3,5,5).astype(np.float32)


def verify(session):
    m=json.loads((session/'manifest.json').read_text())
    require(m['status']=='completed' and m['source']=='simulator' and m['session_id']==session.name,'Wrong completed native session')
    require(not any(m.get(k) for k in ('error','cleanup_error','raw_finish_error')),'Acquisition error')
    names={p.relative_to(session).as_posix() for p in session.rglob('*') if p.is_file()}-{'manifest.json'}
    require(names==set(m['result_sha256']),'Inventory differs')
    for name,pin in m['result_sha256'].items():require(sha(repo_relative(session,name))==pin,'Result SHA differs: '+name)
    for name,pin in m['source_sha256'].items():
        if name.startswith('vendor/MemriBoard/MemriCORE/'):continue
        require(sha(repo_path(name))==pin,'Public source differs: '+name)
    start=json.loads((session/'start_manifest.json').read_text())
    for k,v in start.items():
        if k!='status':same(m[k],v,'start.'+k)
    require(sha(session/'array_initial.cb')==m['initial_array_sha256'],'Initial array differs')
    require((session/'board.ini').read_bytes()==(ROOT/'configs/raspberry_board.ini').read_bytes(),'Board INI differs')
    require((session/'simulator_settings.ini').read_bytes()==(ROOT/'vendor/MemriBoard/simulator/simulator_settings.ini').read_bytes(),'SIM INI differs')
    spec=m['protocol']
    require(spec['schema']=='confirmatory-conv1-native-readback-v1' and spec['levels_v']==LEVELS and
            spec['pulse_width_s']==.000125 and spec['pairs']==PAIRS and spec['read_repeats']==3 and
            spec['programming_max_impacts']==825600 and spec['programming_max_pulses']==153600,'Protocol differs')
    return m


def repo_relative(folder,name):
    p=(folder/name).resolve()
    require(p.is_relative_to(folder.resolve()),'Escaping artifact')
    return p


def audit(session):
    m=verify(session);kind=m['kind'];weights=None;char_report=None
    if kind=='program':
        binding=m['binding'];char=repo_path(binding['session'])
        require(sha(char/'manifest.json')==binding['manifest_sha256'],'Characterization binding differs')
        char_report,char_result,_=audit(char)
        same(char_result['mapping'],binding['mapping'],'Mapping must be selected before checkpoint inference')
        folder,_,_,_=evaluate.assets(binding['seed'])
        require(sha(folder/'best_fs.pt')==binding['checkpoint_sha256'],'Programming checkpoint differs')
        weights=torch.load(folder/'best_fs.pt',map_location='cpu',weights_only=True)['conv_layers.0.weight'].numpy().astype(float).reshape(64,75)
    else:require(kind=='characterize','Unknown native run kind')
    rows=records(session/'raw');checker=Replay(rows)
    try:
        checker.expect('readback_session_begin',protocol=m['protocol'],kind=kind,binding=m['binding'])
        require(bool(checker.take('port_open')['result']),'Open not confirmed')
        actual,effective=checker.characterize() if kind=='characterize' else checker.program(weights,m['binding']['mapping'])
        require(bool(checker.take('port_close')['result']),'Close not confirmed')
        checker.expect('readback_session_end',status='completed',error=None,impacts=checker.calls,pulses=checker.pulses)
        require(next(checker.rows,None) is None,'Trailing raw records')
    finally:rows.close()
    cap,pulse_cap=(1304,256) if kind=='characterize' else (825600,153600)
    require(checker.calls<=cap and checker.pulses<=checker.decisions<=pulse_cap,'Finite cap violated')
    for key,value in [('impacts',checker.calls),('pulses',checker.pulses),('decisions',checker.decisions)]:
        same(m[key],value,'manifest.'+key)
    if kind=='program':
        require(np.array_equal(effective,np.load(session/'effective_conv1.npy',allow_pickle=False)),'Measured Conv1 differs from raw')
        actual['effective_conv1_sha256']=sha(session/'effective_conv1.npy')
    same(actual,json.loads((session/'observations.json').read_text()),'observations raw replay')
    report=dict(status='PASS',session=session.relative_to(ROOT).as_posix(),source='native_memriboard_simulator',
        kind=kind,manifest_sha256=sha(session/'manifest.json'),auditor_sha256=sha(__file__),
        impacts=checker.calls,pulses=checker.pulses,decisions=checker.decisions,
        private_runtime_status='private_runtime_not_independently_reverified',native_mvm=False,
        characterization_audit=char_report)
    return report,actual,effective


def classify(session,output):
    require(not output.exists(),'Choose a new report directory')
    audit_result,readback,effective=audit(session)
    require(effective is not None,'Programming session required for hybrid')
    output.mkdir(parents=True);atomic_json(output/'AUDIT.json',audit_result)
    m=json.loads((session/'manifest.json').read_text());seed=m['binding']['seed']
    folder,cfg,_,_=evaluate.assets(seed)
    state=torch.load(folder/'best_fs.pt',map_location='cpu',weights_only=True)
    # The ONLY model replacement is complete raw-derived Conv1. No partial/ideal fill.
    state['conv_layers.0.weight']=torch.from_numpy(effective.copy())
    model=evaluate.model_for(cfg,state)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    images,labels=evaluate.load_dataset('test');ids=np.flatnonzero(np.isin(labels,evaluate.CLASSES))
    require(len(ids)==1000,'Official test coverage differs')
    remap={c:i for i,c in enumerate(evaluate.CLASSES)}
    y=torch.tensor([remap[int(labels[i])] for i in ids]);rows=[];begin=time.monotonic()
    with torch.inference_mode():
        for first in range(0,1000,16):
            end=min(first+16,1000)
            _,spikes,twd=model(evaluate.encode(images[ids[first:end]]))
            rows.extend(evaluate.scores(spikes,twd,y[first:end]))
    for row,index in zip(rows,ids):row['original_test_index']=int(index)
    result=dict(schema='confirmatory-native-readback-hybrid-v1',seed=seed,
        source='nativeSIM programming/readback then Torch Conv1/remaining network',
        native_mvm=False,physical_hardware=False,training=False,coefficient_coverage=4800,
        actual_conv1_sha256=sha(session/'effective_conv1.npy'),test_count=1000,wall_seconds=time.monotonic()-begin,
        weight_statistics={k:readback[k] for k in ('weight_rmse','weight_relative_rmse','signed_match_fraction','target_miss_fraction')},
        fs=sum(r['first_spike_correct'] for r in rows)/1000,twd=sum(r['twd_correct'] for r in rows)/1000,
        strict_fs=sum(r['strict_first_spike_correct'] for r in rows)/1000,
        strict_twd=sum(r['strict_twd_correct'] for r in rows)/1000,
        silent=sum(r['silent'] for r in rows),fs_tied=sum(r['first_spike_tied'] for r in rows),
        twd_tied=sum(r['twd_tied'] for r in rows),predictions=rows,mapping=readback['mapping'])
    clean=json.loads((ROOT/'results/cifar_confirmatory_20261020/clean-four-seeds-20261020'/('seed'+str(seed)+'.json')).read_text())
    require(clean['checkpoint_sha256']==m['binding']['checkpoint_sha256'],'Clean comparison checkpoint mismatch')
    result['clean']={k:clean[k] for k in ('fs','twd','strict_fs','strict_twd','silent')}
    result['delta_vs_clean']={k:result[k]-clean[k] for k in ('fs','twd','strict_fs','strict_twd')}
    atomic_json(output/'HYBRID.json',result)
    return {k:v for k,v in result.items() if k not in ('predictions',)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--audit-only',action='store_true');a=p.parse_args();out=repo_path(a.output_dir)
    if a.audit_only:
        require(not out.exists(),'Choose new audit directory')
        report,_,_=audit(repo_path(a.session));out.mkdir(parents=True);atomic_json(out/'AUDIT.json',report);print(json.dumps(report))
    else:print(json.dumps(classify(repo_path(a.session),out)))
