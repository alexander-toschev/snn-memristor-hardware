"""Offline native STDP evidence audit; no Connector is instantiated or opened."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.deps'), str(ROOT)]
EXP = 'experiments/conductance_stdp_20261019'
REVIEW = ROOT / 'reports/2026-10-19/conductance-stdp-review'
BL = (6, 8, 9, 14)
HELPERS = {
    'cell_calibration.py':'eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8',
    'read_all.py':'b30ae2f4dd3cd166dbc5cefb07b558127de2bbdba1f6674c40b1b45f4dd74d5f',
    'native_conversions.py':'2ad2e9c057bb14979b083a6e846e8c417c080a6c5273d848e31f893b7970253e',
    'experiment_hardware.py':'059dc4f827943a9df631631f52e776b57182afc9d36db0c1ad5246427f73548c',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def close(a, b):
    if isinstance(b, list):
        require(len(a) == len(b), 'List length mismatch')
        for x, y in zip(a, b):
            close(x, y)
    else:
        require(math.isfinite(a) and abs(a-b) <= 1e-12, f'Numeric replay mismatch: {a} != {b}')


def audit(session):
    m = json.loads((session/'manifest.json').read_text())
    start = json.loads((session/'start_manifest.json').read_text())
    freeze = json.loads((REVIEW/'SOURCE_FREEZE.json').read_text())
    require(m['status'] == 'completed' and m['source'] == 'simulator', 'Completed native SIM required')
    for key in ('session_id', 'source', 'source_sha256', 'protocol_sha256', 'invocation'):
        require(m[key] == start[key], 'Start/final differs: '+key)
    protocol = json.loads((session/'protocol.json').read_text())
    require(protocol == freeze['protocol'], 'Protocol differs from reviewed specification')
    require(digest(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()) == m['protocol_sha256'] == freeze['protocol_sha256'], 'Protocol hash mismatch')
    for name, pin in freeze['file_sha256'].items():
        require(digest((ROOT/EXP/name).read_bytes()) == pin, 'Frozen file differs: '+name)
    # This reviewed helper reads source/INI files only; it never opens a board.
    for name,pin in HELPERS.items():
        require(digest((ROOT/name).read_bytes())==pin,'Reviewed helper differs: '+name)
    from cell_calibration import check_sources
    _, ini, expected = check_sources('simulator')
    expected.update(HELPERS)
    expected.update({EXP+'/'+name:pin for name,pin in freeze['file_sha256'].items()})
    require(expected == m['source_sha256'], 'Source inventory mismatch')
    for name,pin in expected.items():
        require(digest((ROOT/name).read_bytes()) == pin, 'Source bytes mismatch: '+name)
    require((session/'board.ini').read_bytes() == ini, 'INI mismatch')
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
    require(rows[-1]['event']=='session_end' and rows[-1]['status']=='completed', 'Missing completed end')
    counts=Counter(r['event'] for r in rows); req={}; res={}; pending=None
    for row in rows:
        require(row['source']=='simulator', 'Mixed source')
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
                require(c['sign'] in (0,1) and c['vol']==(1.7 if c['sign'] else 1.4) and c['pulse_width']==.0001, 'Pulse differs')
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
    states={}; maximum_u=0.;min_v=math.inf;max_v=-math.inf;spikes=Counter()
    for r in rows:
        if r['event']!='lif_step':continue
        phase,window,wl,t=(r[k] for k in ('window_phase','window_index','wl','t'))
        key=phase,window,wl;u,a,b,e,next_t=states.get(key,(0.,[0.]*4,0.,[0.]*4,0))
        require(t==next_t,'LIF timestep mismatch')
        times=((2,3,10,11) if window%2==0 else (10,11,2,3)) if phase=='train' else ((1,4,9,12) if window%2==0 else (9,12,1,4))
        pre=[int(t==x) for x in times];require(r['pre']==pre,'Input timing mismatch')
        require(r['measurement']['mask']==[bl for bl,p in zip(BL,pre) if p],'Input mask mismatch')
        require(r['measurement']['wl']==wl,'LIF measurement WL differs')
        v=bind_bracket(r['measurement'],'window',dict(window_phase=phase,window_index=window,t=t))
        unreset=.5*u+100*v;post=int(unreset>=1);after=0. if post else unreset
        a=[x*math.exp(-.25) for x in a];b*=math.exp(-.25)
        increment=[pa*post-1.05*b*pi for pa,pi in zip(a,pre)];e=[x+y for x,y in zip(e,increment)]
        a=[x+y for x,y in zip(a,pre)];b+=post
        for name,value in [('voltage_v',v),('u_before',u),('u_unreset',unreset),('u_after',after),('pre_trace',a),('post_trace',b),('eligibility',e),('eligibility_increment',increment)]:close(r[name],value)
        require(r['post']==post,'Spike mismatch');spikes[str(wl)]+=post
        maximum_u=max(maximum_u,unreset);min_v=min(min_v,v);max_v=max(max_v,v)
        states[key]=(after,a,b,e,t+1)
    require(len(states)==56 and all(s[-1]==16 for s in states.values()) and counts['lif_step']==896,'LIF coverage mismatch')
    window_schedule=[(p,i) for p,n in [('pretest',4),('train',16),('posttest',4),('retention',4)] for i in range(n)]
    require([(r['phase'],r['index']) for r in rows if r['event']=='window_complete']==window_schedule,'Window phase/index schedule differs')
    expected_lif=[(p,i,t,wl) for p,i in window_schedule for t in range(16) for wl in ((5,7) if t%2==0 else (7,5))]
    require([(r['window_phase'],r['window_index'],r['t'],r['wl']) for r in rows if r['event']=='lif_step']==expected_lif,'Alternating read order differs')
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
    for r in complete:
        guard=r['guard'];tid=guard['task_id'];close(guard['r'],res[tid]['native_result'][0])
        require(r['admitted']==(r['voltage_v']/guard['r']<=(.002 if r['sign'] else .04)), 'Admission differs')
        if r['admitted']:require(r['pulse_task_id']==tid+1 and r['pulse_task_id'] in writes, 'Pulse correspondence differs')
        if r['phase']=='training_update':
            require(r['wl']==5 and r['sign']==int(expected_updates[r['window_index'],r['bl']]>=.25),'Learning sign/cell differs')
            require(tid==r['sham']['task_id']+1 and req[tid-1]['command_si']['wl']==7,'Paired sham missing')
    require([r['decision'] for r in complete]==list(range(1,len(complete)+1)), 'Decision sequence differs')
    require(len(writes)==sum(r['admitted'] for r in complete)==m['pulses']<=128 and len(complete)==m['decisions'],'Pulse totals differ')
    a=sum(r['admitted'] for r in prep);u=len(train);b=sum(r['admitted'] for r in train)
    require(len(req)==6456+a+2*u+b and mode[10]==4776 and counts['checkpoint']==232 and counts['window_complete']==28,'Schedule counts differ')
    checkpoints=[r for r in rows if r['event']=='checkpoint']
    cells=[(wl,bl) for wl in (5,7) for bl in BL]
    checkpoint_schedule=[('initial',wl,bl,None,None) for wl,bl in cells]
    checkpoint_schedule += [('after_preparation_decision',wl,bl,step,None) for step,wl,bl,_ in expected_prep]
    checkpoint_schedule += [(p,wl,bl,None,None) for p in ('after_preparation','baseline_before_training') for wl,bl in cells]
    checkpoint_schedule += [('after_training_window',wl,bl,None,i) for i in range(16) for wl,bl in cells]
    checkpoint_schedule += [(p,wl,bl,None,None) for p in ('after_posttest','after_retention') for wl,bl in cells]
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
    begin=next(i for i,r in enumerate(rows) if r['event']=='retention_begin');end=next(i for i,r in enumerate(rows) if r['event']=='retention_end')
    require(end==begin+1 and rows[end]['elapsed_s']>=60.,'Retention interval/I-O differs')
    require(counts['port_open']==counts['port_close']==1 and all(r['result'] for r in rows if r['event'] in ('port_open','port_close')),'Port lifecycle invalid')
    obs=json.loads((session/'observations.json').read_text())
    strip=lambda r:{k:v for k,v in r.items() if k not in ('event','source','utc','monotonic_s')}
    for name,event in [('checkpoints','checkpoint'),('windows','window_complete'),('updates','pulse_complete')]:
        require(obs[name]==[strip(r) for r in rows if r['event']==event], 'Compact observations differ: '+name)
    for key in ('impacts','pulses','decisions'):require(obs[key]==m[key]==rows[-1][key],'Final counter differs')
    require(obs['uninformative_all_silent']==all(n==0 for n in spikes.values()),'Silent summary differs')
    return dict(status='pass',auditor_sha256=digest(Path(__file__).read_bytes()),session=session.relative_to(ROOT).as_posix(),manifest_sha256=digest((session/'manifest.json').read_bytes()),runtime=m['invocation'],source_count=len(expected),result_file_count=len(actual),source_sha256=m['source_sha256'],protocol_sha256=m['protocol_sha256'],raw_index=index,event_counts=dict(counts),impacts=len(req),mode_counts=dict(mode),preparation_pulses=a,training_attempts=u,training_pulses=b,current_skips=len(complete)-len(writes),maximum_mvm_predicted_current_a=max_current,checkpoints=232,windows=28,lif_steps=896,spikes=dict(spikes),maximum_unreset_membrane=maximum_u,window_centered_voltage_min_v=min_v,window_centered_voltage_max_v=max_v,eligibility_max_abs=max(abs(e) for s in states.values() for e in s[3]),retention_elapsed_s=rows[end]['elapsed_s'],trajectory=trajectory,hardware_approved=False,qualification='Integrity and independent numeric replay passed; all-silent native SIM does not demonstrate learning efficacy or activate the native training pulse branch.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    session=(ROOT/args.session).resolve();output=(ROOT/args.output).resolve()
    require(session.is_relative_to(ROOT/'results/conductance_stdp_20261019'),'Session outside expected results')
    require(output.is_relative_to(ROOT/'reports') and not output.exists(),'Output must be a new report')
    result=audit(session);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','impacts','preparation_pulses','training_pulses','spikes','maximum_unreset_membrane')}))
