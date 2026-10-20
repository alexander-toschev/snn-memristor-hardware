"""Offline STDP8 numeric replay without private-driver attestations or launch registries."""
import argparse
from collections import Counter
from configparser import ConfigParser
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
from tools.audit_conductance_stdp_calibrated import BL,require,digest,close
from experiments.conductance_stdp_calibrated_20261019.core import SPEC
from experiments.cifar_resumable_20261019.durable import sha
from experiments.unn_rev3_20261014.paths import repo_path
from read_all import EXPECTED_BOARD


def canonical_sha(value):return digest(json.dumps(value,sort_keys=True,separators=(',',':')).encode())


def recorded_sources(manifest):
    private=[];verified=[]
    for name,pin in manifest['source_sha256'].items():
        path=repo_path(name)
        if name.startswith('vendor/MemriBoard/MemriCORE/'):
            private.append(name);continue
        require(path.is_file() and sha(path)==pin,'Recorded public source differs: '+name)
        verified.append(name)
    return dict(private_runtime_status='private_runtime_not_independently_reverified',
                public_files_verified=len(verified),private_recorded_paths_not_checked=private)


def replay(session):
    session=Path(session).resolve();require(session.is_relative_to(ROOT/'results'),'Session outside results')
    m=json.loads((session/'manifest.json').read_text());start=json.loads((session/'start_manifest.json').read_text())
    require(m['status'] in ('completed','uninformative_calibration') and m['source'] in ('hardware','simulator'),'Terminal native outcome required')
    completed=m['status']=='completed'
    for key in ('session_id','source','source_sha256','protocol_sha256','invocation'):
        require(m[key]==start[key],'Start/final identity differs: '+key)
    require(m['session_id']==session.name,'Wrong session directory')
    require(m.get('physical_pulse_duration_s') is None,'Physical pulse duration is unknown')
    protocol=json.loads((session/'protocol.json').read_text())
    require(protocol==SPEC and canonical_sha(protocol)==m['protocol_sha256'],'STDP protocol changed')
    recorded_sources(m);expected=m['source_sha256']
    config=ConfigParser();config.read_string((session/'board.ini').read_text(encoding='utf-8-sig'))
    require(config.get('board','board_type')=='rp5_fpga_python','Unexpected INI driver')
    for key,value in EXPECTED_BOARD.items():require(config.getfloat('board',key)==value,'INI setting differs: '+key)
# BEGIN UNCHANGED NUMERIC REPLAY
    actual = {p.relative_to(session).as_posix() for p in session.rglob('*') if p.is_file()} - {'manifest.json'}
    require(actual == set(m['result_sha256']), 'Result inventory mismatch')
    for name,pin in m['result_sha256'].items():
        require(digest((session/name).read_bytes()) == pin, 'Result bytes mismatch: '+name)
    packed=(session/'raw/raw.jsonl.gz').read_bytes(); data=gzip.decompress(packed)
    index=json.loads((session/'raw/index.json').read_text())
    require(index['compressed_sha256']==digest(packed) and index['expanded_sha256']==digest(data), 'Raw hashes mismatch')
    require(index['compressed_bytes']==len(packed) and index['expanded_bytes']==len(data)<=64*1024**2, 'Raw size mismatch/budget')
    rows=[json.loads(line) for line in data.splitlines()]
    require(len(rows)==index['lines'], 'Raw line count mismatch')
    require(rows[0]['event']=='session_begin' and rows[0]['protocol']==protocol, 'Missing initial protocol')
    require(rows[-1]['event']=='session_end' and rows[-1]['status']==m['status'], 'Missing completed end')
    counts=Counter(r['event'] for r in rows); req={}; res={}; pending=None
    for row in rows:
        require(row['source']==m['source'], 'Mixed source')
        if row['event']=='request':
            require(pending is None and row['task_id']==len(req)+1, 'Nonsequential request')
            pending=row['task_id'];req[pending]=row
        elif row['event']=='driver_output':
            require(row['task_id']==pending, 'Unassociated driver output')
        elif row['event']=='result':
            require(row['task_id']==pending, 'Response order mismatch')
            res[pending]=row;require(row['command_si']==req[pending]['command_si'], 'Echoed command mismatch');pending=None
        require(row['event']!='call_error', 'Recorded call error')
    require(pending is None and len(req)==len(res)==m['impacts']<=6712, 'Unpaired/budget calls')
    mode=Counter(); writes=[]; max_current=0.
    for tid,r in req.items():
        c=r['command_si'];native=res[tid]['native_result'];mode[c['mode_flag']]+=1
        require(c['wl'] in (5,7) and len(native)==3, 'Cell/response invalid')
        resistance,echo,adc=native
        require(type(echo) is int and echo==c['id'] and type(adc) is int, 'Native echo/type invalid')
        if c['mode_flag']==7:
            require(c['bl'] in BL and c['id']==tid and math.isfinite(resistance) and resistance>1e-8 and 20<=adc<=16382, 'Native R invalid')
            if c['vol']:
                writes.append(tid);g=req[tid-1]['command_si'];gr=res[tid-1]['native_result'][0]
                require(g['mode_flag']==7 and g['vol']==0 and g['wl']==c['wl'] and g['bl']==c['bl'] and req[tid-1]['phase']=='pulse_guard', 'Missing immediate write guard')
                require(c['sign'] in (0,1) and c['vol']==(1.7 if c['sign'] else 1.4) and c['pulse_width']==(.0001 if m['source']=='simulator' else 0.), 'Pulse differs')
                require(c['vol']/gr<=(.002 if c['sign'] else .04), 'Pulse current exceeded')
            else:
                require(c['sign']==0 and c['pulse_width']==0, 'Nonzero read parameters')
        else:
            require(c['mode_flag']==10 and c['id']==0 and 0<=adc<=16382 and round(adc*5/16383,5)<=4, 'MVM invalid')
            v=c['vol'];active=[i for i,x in enumerate(v) if x]
            require(len(v)==32 and len(active)<=1 and all(i in BL for i in active) and all(x in (0.,.05) for x in v), 'MVM vector invalid')
            require(active==r['mask'], 'Mask mismatch')
            if active:
                g=req[tid-1]['command_si'];gr=res[tid-1]['native_result'][0]
                require(g['mode_flag']==7 and g['vol']==0 and g['wl']==c['wl'] and g['bl']==active[0] and req[tid-1]['phase']=='vector_guard', 'Missing MVM guard')
                current=max(.05,round(round(.05*4095/5)*5/4095,3))/gr
                require(current<=.005 and 330*current<3.3,'MVM current exceeded');max_current=max(current,max_current)
    def volts(tid):
        require(req[tid]['command_si']['mode_flag']==10,'Expected ADC mode10')
        return round(res[tid]['native_result'][2]*5/16383,5)
    def bracket(b):
        before,signal,after=(b[n]['task_id'] for n in ('before','signal','after'))
        require(signal==before+1+bool(b['mask']) and after==signal+1,
                'Bracket task IDs must be distinct and chronological with only its fresh guard between')
        for name in ('before','signal','after'):
            sample=b[name];tid=sample['task_id'];c=req[tid]['command_si']
            require(c['wl']==b['wl'] and sample['adc']==res[tid]['native_result'][2], 'Bracket ADC mismatch')
            close(sample['voltage_v'],volts(tid))
            require(req[tid]['mask']==(b['mask'] if name=='signal' else []),'Bracket mask differs')
            expected_guards=[]
            if name=='signal' and b['mask']:
                expected_guards=[dict(task_id=tid-1,r=res[tid-1]['native_result'][0],adc=res[tid-1]['native_result'][2])]
            require(sample['guards']==expected_guards,'Bracket guard reference differs')
        value=volts(b['signal']['task_id'])-(volts(b['before']['task_id'])+volts(b['after']['task_id']))/2
        close(b['centered_v'],value);return value
    bracket_fields=('wl','mask','before','signal','after','centered_v')
    brackets={}; mode10_references=[]; owned_brackets=set(); checkpoint_reads=set()
    for r in rows:
        if r['event']!='bracket':continue
        bracket(r);key=tuple(r[n]['task_id'] for n in ('before','signal','after'))
        require(key not in brackets,'Duplicate top-level bracket')
        brackets[key]=r;mode10_references.extend(key)
        for name,suffix in [('before','_zero_before'),('signal','_signal'),('after','_zero_after')]:
            request=req[r[name]['task_id']]
            require(request['phase']==r['phase']+suffix,'Bracket request phase differs')
            for meta in ('checkpoint_phase','repeat','preparation_step','window_phase','window_index','t'):
                require(request.get(meta)==r.get(meta),'Bracket request metadata differs: '+meta)
    require(len(mode10_references)==len(set(mode10_references)) and set(mode10_references)=={i for i,r in req.items() if r['command_si']['mode_flag']==10},'Mode10 must belong to exactly one bracket')
    def bind_bracket(b,phase,metadata):
        key=tuple(b[n]['task_id'] for n in ('before','signal','after'))
        require(key in brackets and key not in owned_brackets,'Nested bracket missing or reused')
        top=brackets[key]
        require(b=={name:top[name] for name in bracket_fields},'Nested bracket content differs from raw event')
        require(top['phase']==phase,'Nested bracket phase differs')
        for name,value in metadata.items():require(top.get(name)==value,'Nested bracket owner metadata differs: '+name)
        owned_brackets.add(key)
        return bracket(b)
    # Independent numerical implementation: no acquisition Neuron/choose_gain import.
    def step(state,pre,v,gain):
        u,a,b,e=state
        unreset=.5*u+gain*v;post=int(unreset>=1);after=0. if post else unreset
        a=[x*math.exp(-.25) for x in a];b*=math.exp(-.25)
        increment=[pa*post-1.05*b*pi for pa,pi in zip(a,pre)]
        e=[x+y for x,y in zip(e,increment)];a=[x+y for x,y in zip(a,pre)];b+=post
        fields=dict(gain=gain,u_before=u,u_unreset=unreset,u_after=after,post=post,
                    pre=pre,voltage_v=v,eligibility=e,eligibility_increment=increment,
                    pre_trace=a,post_trace=b)
        return (after,a,b,e),fields
    def equal(a,b):
        if isinstance(b,dict):
            require(set(a)==set(b),'Replay dictionary fields differ')
            for k in b:equal(a[k],b[k])
        elif isinstance(b,list):
            require(len(a)==len(b),'Replay list length differs')
            for x,y in zip(a,b):equal(x,y)
        elif type(b) is float:close(a,b)
        else:require(a==b,'Replay scalar differs: '+str((a,b)))
    def times(phase,index):
        if phase=='fit':return ([1,5,9,13],[13,9,5,1])[index%2]
        if phase=='train':return ([3,10,11,2],[2,11,10,3])[index%2]
        return ([1,4,9,12],[9,12,1,4])[index%2]
    fitrows=[r for r in rows if r['event']=='fit_measurement_step']
    expected_fit=[('fit',i,t,wl) for i in range(2) for t in range(16) for wl in ((5,7) if t%2==0 else (7,5))]
    require([(r['window_phase'],r['window_index'],r['t'],r['wl']) for r in fitrows]==expected_fit,'Fit coverage/order differs')
    fit_values={}
    for r in fitrows:
        i,t,wl=r['window_index'],r['t'],r['wl'];pre=[int(t==x) for x in times('fit',i)]
        require(r['pre']==pre and r['measurement']['wl']==wl and r['measurement']['mask']==[bl for bl,p in zip(BL,pre) if p],'Fit input/bank differs')
        fit_values[i,t,wl]=bind_bracket(r['measurement'],'window',dict(window_phase='fit',window_index=i,t=t))
    frozen=[r for r in rows if r['event']=='gain_frozen']
    require(len(frozen)==1,'Exactly one gain freeze required');frozen=frozen[0]
    freeze_index=rows.index(frozen)
    require(all(rows.index(r)<freeze_index for r in fitrows),'Fit occurs after freeze')
    require(frozen['last_fit_task_id']==max(r['task_id'] for r in rows[:freeze_index] if r['event']=='request')==fitrows[-1]['measurement']['after']['task_id'],'Fit freeze task boundary differs')
    require(all(rows.index(r)>freeze_index for r in rows if r['event']=='lif_step'),'Non-fit LIF occurs before freeze')
    positions={id(row):i for i,row in enumerate(rows)}
    for row in rows:
        if row['event']=='request' and row.get('window_phase') not in (None,'fit'):
            require(positions[id(row)]>freeze_index,'Non-fit physical request before gain freeze')
    require(all(positions[id(r)]<freeze_index for r in rows if r['event']=='window_complete' and r['phase']=='fit'),'Fit window completed after freeze')
    gains=[10.,20.,40.,80.,100.,200.,400.,800.,1600.];candidates=[]
    for gain in gains:
        windows=[]
        for i in range(2):
            state=(0.,[0.]*4,0.,[0.]*4);steps=[]
            for t in range(16):
                pre=[int(t==x) for x in times('fit',i)]
                state,result=step(state,pre,fit_values[i,t,5],gain)
                record=next(r for r in fitrows if (r['window_index'],r['t'],r['wl'])==(i,t,5))
                steps.append(dict(t=t,signal_task_id=record['measurement']['signal']['task_id'],**result))
            posts=sum(r['post'] for r in steps);zero=sum(r['post'] for r in steps if not any(r['pre']))
            windows.append(dict(index=i,posts=posts,zero_input_posts=zero,steps=steps))
        total=sum(w['posts'] for w in windows)
        eligible=all(1<=w['posts']<=3 and w['zero_input_posts']==0 for w in windows)
        candidates.append(dict(gain=gain,eligible=eligible,total_posts=total,objective=[abs(total-4),gain],windows=windows))
    eligible=[c for c in candidates if c['eligible']]
    gain=min(eligible,key=lambda c:tuple(c['objective']))['gain'] if eligible else None
    equal(frozen['candidates'],candidates)
    require(frozen['selected_gain']==gain and frozen['fit_wl']==5 and frozen['fit_window_indices']==[0,1] and frozen['control_used_for_selection'] is False,'Gain selection/fit ownership differs')
    require(completed==(gain is not None),'Outcome/gain eligibility differs')
    if not completed:
        require([r['event'] for r in rows[freeze_index+1:]]==['calibration_stop','port_close','session_end'],'No-eligible prefix continued acquisition')
    states={}; maximum_u=0.;min_v=math.inf;max_v=-math.inf;spikes=Counter();phase_spikes=Counter()
    for r in rows:
        if r['event']!='lif_step':continue
        phase,window,wl,t=(r[k] for k in ('window_phase','window_index','wl','t'))
        key=phase,window,wl;saved=states.get(key,(0.,[0.]*4,0.,[0.]*4,0));state=saved[:4];next_t=saved[4]
        require(t==next_t,'LIF timestep mismatch')
        pre=[int(t==x) for x in times(phase,window)];require(r['pre']==pre,'Input timing mismatch')
        require(r['measurement']['mask']==[bl for bl,p in zip(BL,pre) if p] and r['measurement']['wl']==wl,'Input mask/bank mismatch')
        v=bind_bracket(r['measurement'],'window',dict(window_phase=phase,window_index=window,t=t))
        state,result=step(state,pre,v,gain)
        for name,value in result.items():equal(r[name],value)
        spikes[str(wl)]+=result['post'];phase_spikes[phase+'/'+str(wl)]+=result['post']
        maximum_u=max(maximum_u,result['u_unreset']);min_v=min(min_v,v);max_v=max(max_v,v)
        states[key]=(*state,t+1)
        # Keep state representation compatible with eligibility checks below.
    # The next iteration needs just the four numerical state entries and next_t.
    require(len(states)==(52 if completed else 0) and all(s[-1]==16 for s in states.values()) and counts['lif_step']==(832 if completed else 0),'LIF coverage mismatch')
    window_schedule=[('fit',i) for i in range(2)]
    if completed:window_schedule += [(p,i) for p,n in [('pretest',2),('train',16),('posttest',4),('retention',4)] for i in range(n)]
    windows=[r for r in rows if r['event']=='window_complete']
    require([(r['phase'],r['index']) for r in windows]==window_schedule,'Window phase/index schedule differs')
    expected_lif=[(p,i,t,wl) for p,i in window_schedule if p!='fit' for t in range(16) for wl in ((5,7) if t%2==0 else (7,5))]
    require([(r['window_phase'],r['window_index'],r['t'],r['wl']) for r in rows if r['event']=='lif_step']==expected_lif,'Alternating read order differs')
    for w in windows:
        event='fit_measurement_step' if w['phase']=='fit' else 'lif_step'
        linked=[{k:v for k,v in r.items() if k not in ('event','source','utc','monotonic_s','window_phase','window_index')} for r in rows if r['event']==event and (r['window_phase'],r['window_index'])==(w['phase'],w['index'])]
        require(w['steps']==linked and w['times']==times(w['phase'],w['index']),'Window nested measurement ownership differs')
        expected_e=None if w['phase']=='fit' else {str(wl):states[w['phase'],w['index'],wl][3] for wl in (5,7)}
        equal(w['eligibility'],expected_e);require(w['gain']==(None if w['phase']=='fit' else gain),'Window gain differs')
    complete=[r for r in rows if r['event']=='pulse_complete']
    prep=[r for r in complete if r['phase']=='preparation'];train=[r for r in complete if r['phase']=='training_update']
    expected_prep=[(step,wl,bl,int((j<2)==(step<4))) for step in range(8) for j,bl in enumerate(BL) for wl in ((5,7) if (step+j)%2==0 else (7,5))]
    require([(r['preparation_step'],r['wl'],r['bl'],r['sign']) for r in prep]==expected_prep,'Preparation schedule differs')
    expected_updates={(w,bl):e for (p,w,wl),s in states.items() if p=='train' and wl==5 for bl,e in zip(BL,s[3]) if abs(e)>=.25}
    require(len(train)==len(expected_updates) and {(r['window_index'],r['bl']) for r in train}==set(expected_updates),'STDP update decisions differ')
    expected_none={(w,bl):e for (p,w,wl),s in states.items() if p=='train' and wl==5 for bl,e in zip(BL,s[3]) if abs(e)<.25}
    none=[r for r in rows if r['event']=='no_update']
    require(len(none)==len(expected_none) and {(r['window_index'],r['bl']) for r in none}==set(expected_none),'No-update decisions differ')
    for r in none:close(r['eligibility'],expected_none[r['window_index'],r['bl']])
    decisions=[r for r in rows if r['event']=='pulse_decision']
    require(len(decisions)==len(complete),'Pulse decision/complete coverage differs')
    pulse_refs=set();guard_refs=set();sham_refs=set()
    strip_event=lambda r:{k:v for k,v in r.items() if k not in ('event','source','utc','monotonic_s')}
    for r,d in zip(complete,decisions):
        payload=strip_event(r);decision=strip_event(d)
        require({k:v for k,v in payload.items() if k not in ('pulse_task_id','pulse_native_r','pulse_adc')}==decision,'Pulse decision/complete fields differ')
        guard=r['guard'];tid=guard['task_id'];request=req[tid];command=request['command_si']
        require(tid not in guard_refs,'Repeated pulse guard reference');guard_refs.add(tid)
        require(guard==dict(task_id=tid,r=res[tid]['native_result'][0],adc=res[tid]['native_result'][2]),'Pulse guard native values differ')
        require(command==dict(mode_flag=7,wl=r['wl'],bl=r['bl'],vol=0.,sign=0,pulse_width=0.,id=tid),'Pulse guard cell/mode differs')
        require(request['phase']=='pulse_guard' and request['intended_phase']==r['phase'],'Pulse guard phase differs')
        for key in ('window_index','eligibility','preparation_step'):require(request.get(key)==r.get(key),'Pulse guard metadata differs: '+key)
        require(positions[id(res[tid])]<positions[id(d)]<positions[id(r)],'Pulse decision chronology differs')
        require(r['voltage_v']==(1.7 if r['sign'] else 1.4),'Pulse event amplitude differs')
        require(r['admitted']==(r['voltage_v']/guard['r']<=(.002 if r['sign'] else .04)), 'Admission differs')
        if r['admitted']:
            pt=r['pulse_task_id'];require(pt==tid+1 and pt in writes and pt not in pulse_refs,'Pulse reference missing/reused');pulse_refs.add(pt)
            write=req[pt];wc=write['command_si'];native=res[pt]['native_result']
            require(wc==dict(mode_flag=7,wl=r['wl'],bl=r['bl'],vol=r['voltage_v'],sign=r['sign'],pulse_width=.0001 if m['source']=='simulator' else 0.,id=pt),'Write request/event differs')
            require(write['phase']==r['phase'],'Write event phase differs')
            for key in ('window_index','eligibility','preparation_step'):require(write.get(key)==r.get(key),'Write event metadata differs: '+key)
            require(r['pulse_native_r']==native[0] and r['pulse_adc']==native[2],'Pulse native reply differs')
            require(positions[id(d)]<positions[id(write)]<positions[id(res[pt])]<positions[id(r)],'Write chronology differs')
        else:require(not any(k in r for k in ('pulse_task_id','pulse_native_r','pulse_adc')),'Skipped pulse claims native write')
        if r['phase']=='training_update':
            expected_e=expected_updates[r['window_index'],r['bl']]
            require(r['wl']==5 and r['sign']==int(expected_e>=.25),'Learning sign/cell differs');close(r['eligibility'],expected_e)
            sham=r['sham'];st=sham['task_id'];sr=req[st];sc=sr['command_si']
            require(st==tid-1 and st not in sham_refs,'Paired sham missing/reused');sham_refs.add(st)
            require(sc==dict(mode_flag=7,wl=7,bl=r['bl'],vol=0.,sign=0,pulse_width=0.,id=st) and sr['phase']=='paired_sham','Paired sham cell/phase differs')
            require(sham==dict(task_id=st,r=res[st]['native_result'][0],adc=res[st]['native_result'][2]),'Paired sham native values differ')
            require(sr['window_index']==r['window_index'] and sr['eligibility']==r['eligibility'],'Paired sham metadata differs')
            w=next(w for w in windows if w['phase']=='train' and w['index']==r['window_index'])
            require(positions[id(w)]<positions[id(sr)],'Update before completed training window')
        else:require(r['sham'] is None,'Preparation has unexpected sham')
    require(pulse_refs==set(writes),'Write requests lack unique pulse owner')
    require(guard_refs=={tid for tid,r in req.items() if r['phase']=='pulse_guard'},'Pulse guard ownership incomplete')
    require(sham_refs=={tid for tid,r in req.items() if r['phase']=='paired_sham'},'Paired sham ownership incomplete')
    require([r['decision'] for r in complete]==list(range(1,len(complete)+1)), 'Decision sequence differs')
    require(len(writes)==sum(r['admitted'] for r in complete)==m['pulses']<=128 and len(complete)==m['decisions'],'Pulse totals differ')
    a=sum(r['admitted'] for r in prep);u=len(train);b=sum(r['admitted'] for r in train)
    require(len(req)==(6456+a+2*u+b if completed else 1472+a) and mode[10]==(4776 if completed else 912) and counts['checkpoint']==(232 if completed else 80) and counts['window_complete']==len(window_schedule),'Schedule counts differ')
    checkpoints=[r for r in rows if r['event']=='checkpoint']
    cells=[(wl,bl) for wl in (5,7) for bl in BL]
    checkpoint_schedule=[('initial',wl,bl,None,None) for wl,bl in cells]
    checkpoint_schedule += [('after_preparation_decision',wl,bl,step,None) for step,wl,bl,_ in expected_prep]
    checkpoint_schedule += [(p,wl,bl,None,None) for p in (('after_preparation','baseline_before_training') if completed else ('after_preparation',)) for wl,bl in cells]
    checkpoint_schedule += [('after_training_window',wl,bl,None,i) for i in range(16 if completed else 0) for wl,bl in cells]
    checkpoint_schedule += [(p,wl,bl,None,None) for p in (('after_posttest','after_retention') if completed else ()) for wl,bl in cells]
    require([(r['phase'],r['wl'],r['bl'],r.get('preparation_step'),r.get('window_index')) for r in checkpoints]==checkpoint_schedule,'Checkpoint schedule differs')
    trajectory=[]
    for r in checkpoints:
        require(len(r['reads'])==len(r['q'])==len(r['native_inverse_r'])==len(r['responses'])==3,'Checkpoint repetition count differs')
        first=r['reads'][0]['task_id']
        for repeat,(rr,q,inv) in enumerate(zip(r['reads'],r['q'],r['native_inverse_r'])):
            tid=rr['task_id'];request=req[tid];command=request['command_si']
            require(tid==first+repeat and tid not in checkpoint_reads,'Checkpoint read order/reuse differs')
            require(command['mode_flag']==7 and command['vol']==0 and command['wl']==r['wl'] and command['bl']==r['bl'], 'Checkpoint read cell/mode differs')
            require(request['phase']=='checkpoint_r' and request['checkpoint_phase']==r['phase'] and request['repeat']==repeat,'Checkpoint read phase/repeat differs')
            for name in ('preparation_step','window_index'):require(request.get(name)==r.get(name),'Checkpoint read owner differs: '+name)
            require(rr['adc']==res[tid]['native_result'][2],'Checkpoint read ADC differs')
            resistance=res[tid]['native_result'][0];close(rr['r'],resistance);close(q,330/resistance);close(inv,1/resistance)
            checkpoint_reads.add(tid)
        for repeat,response in enumerate(r['responses']):
            require(response['wl']==r['wl'] and response['mask']==[r['bl']],'Checkpoint MVM cell differs')
            require(response['before']['task_id']==first+3+4*repeat,'Checkpoint MVM order differs')
            bind_bracket(response,'checkpoint_mvm',dict(checkpoint_phase=r['phase'],repeat=repeat,
                preparation_step=r.get('preparation_step'),window_index=r.get('window_index')))
        if r['phase'] in ('initial','after_preparation','baseline_before_training','after_posttest','after_retention'):
            trajectory.append(dict(wl=r['wl'],bl=r['bl'],phase=r['phase'],q_mean=statistics.mean(r['q']),mvm_mean_v=statistics.mean(bracket(x) for x in r['responses'])))
    require(owned_brackets==set(brackets),'Top-level bracket has no unique LIF/checkpoint owner')
    require(checkpoint_reads=={i for i,r in req.items() if r['phase']=='checkpoint_r'},'Checkpoint read coverage differs')
    retention_elapsed=None
    if completed:
        begin=next(i for i,r in enumerate(rows) if r['event']=='retention_begin');end=next(i for i,r in enumerate(rows) if r['event']=='retention_end')
        require(end==begin+1 and rows[end]['elapsed_s']>=60.,'Retention interval/I-O differs')
        retention_elapsed=rows[end]['elapsed_s']
    require(counts['port_open']==counts['port_close']==1 and all(r['result'] for r in rows if r['event'] in ('port_open','port_close')),'Port lifecycle invalid')
    obs=json.loads((session/'observations.json').read_text())
    strip=lambda r:{k:v for k,v in r.items() if k not in ('event','source','utc','monotonic_s')}
    for name,event in [('checkpoints','checkpoint'),('windows','window_complete'),('updates','pulse_complete')]:
        require(obs[name]==[strip(r) for r in rows if r['event']==event], 'Compact observations differ: '+name)
    for key in ('impacts','pulses','decisions'):require(obs[key]==m[key]==rows[-1][key],'Final counter differs')
    require(obs['gain']==gain and obs['calibration']=={k:v for k,v in frozen.items() if k not in ('event','source','utc','monotonic_s','last_fit_task_id')},'Compact calibration differs')
    require(obs['uninformative_all_silent']==all(n==0 for n in spikes.values()),'Silent summary differs')
    return dict(status='pass',auditor_sha256=digest(Path(__file__).read_bytes()),session=session.relative_to(ROOT).as_posix(),manifest_sha256=digest((session/'manifest.json').read_bytes()),runtime=m['invocation'],source_count=len(expected),result_file_count=len(actual),source_sha256=m['source_sha256'],protocol_sha256=m['protocol_sha256'],raw_index=index,event_counts=dict(counts),impacts=len(req),mode_counts=dict(mode),preparation_pulses=a,training_attempts=u,training_pulses=b,current_skips=len(complete)-len(writes),maximum_mvm_predicted_current_a=max_current,outcome=m['status'],selected_gain=gain,gain_candidates=[{k:v for k,v in c.items() if k!='windows'} for c in candidates],checkpoints=counts['checkpoint'],windows=len(windows),lif_steps=counts['lif_step'],fit_measurement_steps=len(fitrows),phase_spikes=dict(phase_spikes),spikes=dict(spikes),maximum_unreset_membrane=maximum_u,window_centered_voltage_min_v=min_v if completed else None,window_centered_voltage_max_v=max_v if completed else None,eligibility_max_abs=max((abs(e) for s in states.values() for e in s[3]),default=0.),retention_elapsed_s=retention_elapsed,trajectory=trajectory,hardware_approved=False,qualification='Integrity and independent fit-selection/LIF/eligibility replay passed. Activity and pulse coverage are descriptive, not learning efficacy or intrinsic physical STDP.')
# END UNCHANGED NUMERIC REPLAY


def audit(session):
    result=replay(session)
    result.update(schema='stdp8-public-local-audit-v2',private_runtime_status='private_runtime_not_independently_reverified',
        private_receipt_required=False,hardware_io_executed=False,old_strict_attestation_pass_claimed=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=repo_path(args.output)
    if output.exists():raise FileExistsError('Audit output already exists')
    result=audit(repo_path(args.session));output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],output=args.output,
                         impacts=result['impacts'],training_pulses=result['training_pulses'],selected_gain=result['selected_gain'])))

if __name__=='__main__':main()
