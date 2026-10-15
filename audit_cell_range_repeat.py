"""Independent offline audit of the bounded three-cell interval experiment. No Connector calls."""
import argparse
from collections import Counter
from configparser import ConfigParser
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from interval_audit_core import CELLS,TARGETS,MAX_DECISIONS,MAX_IMPACTS,expected_band,inside,summarize,screen_adc,protocol_profile,stop_band

RUNTIME_SHA='f03860a42755337fcf1a5f71a0edc291bbc8308684a358d7dbbcc386de5b7161'
REFINED_SHA='bd95f0a5e43e0b2088413353b15106676de58135912206a84d2916d3aedad8b8'
BASE_SHA='eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8'
CONVERTER_SHA='28c6495c56afa225501dd6b0dea999215f08b73f30e016ff7ea34d6c5412701c'
EQUALIN_SHA='00b385405bf19b6adff5e3d8dfa02e6c80dd10b3f4693476730d3bdcc2c3b779'
PRIVATE={
 'vendor/MemriBoard/MemriCORE/rp5_fpga_python/rpi_FPGAed.py':'bc717659fa5d4d36d2037eef978bf1cd8dcf0fe92f9bbfad51f1eb1d0d27df9c',
 'vendor/MemriBoard/MemriCORE/rp5_fpga_python/FPGA_SIM.py':'5bcb02c04bc185e332b2ec96acabc2d4303d21724c9c052de540ef92d603b634'}
LEVELS=(0.,.025,.05,.075,.1,0.)
GUARD_FIELDS={'programming_decision_id','previous_native_r_ohm','resistance_source_task_id','predicted_current_a','current_limit_a'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def jsonl(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.is_file() else []


def verify_public_source(repo,name,digest,blob,ck):
    """Acquisition code is checked against Git; every vendor Python import remains byte-pinned locally."""
    path=(repo/name).resolve()
    if Path(name).is_absolute() or '..' in Path(name).parts or not path.is_relative_to(repo.resolve()):
        raise ValueError('Source path escapes repository')
    ck.check(hashlib.sha256(blob).hexdigest()==digest,'source/Git_blob/'+name)
    local_digest=sha(path) if path.is_file() else None
    importable=name.startswith('vendor/MemriBoard/') and name.endswith('.py')
    if importable:ck.check(local_digest==digest,'source/imported_local/'+name)
    return None if local_digest==digest else dict(path=name,recorded_sha256=digest,local_sha256=local_digest,
        verification='recorded Git blob; local runtime differs',imported_by_audit=importable)


class Checks:
    def __init__(self):self.count=0;self.errors=[];self.maxdiff=0.
    def check(self,ok,label):
        self.count+=1
        if not ok:self.errors.append(label)
    def compare(self,a,b,label):
        if isinstance(b,dict):
            self.check(isinstance(a,dict) and set(a)==set(b),label+'/keys')
            if isinstance(a,dict):
                for k in b:
                    if k in a:self.compare(a[k],b[k],label+'/'+str(k))
        elif isinstance(b,list):
            self.check(isinstance(a,list) and len(a)==len(b),label+'/length')
            if isinstance(a,list):
                for i,(x,y) in enumerate(zip(a,b)):self.compare(x,y,label+'/'+str(i))
        elif type(a) in (int,float) and type(b) in (int,float):
            self.maxdiff=max(self.maxdiff,abs(a-b))
            self.check(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),label)
        else:self.check(a==b,label)


def audit_raw(path,manifest,board,c,ck,profile=None):
    """Retain original events for independent protocol replay; at most 6120 pairs."""
    check,compare=ck.check,ck.compare
    profile=profile or protocol_profile()
    source=manifest['source'];records=[];actions=[];delays=[];latest={};decisions=skips=pulses=0
    lastmono=-math.inf;lasttime=None;fault=False;modes=Counter()
    def conversion(v):
        code=c.convert_volt_to_dac(board['dac_bit'],board['vol_ref_dac'],v)
        return dict(requested_v=v,dac_code=code,native_decoded_v=c.convert_dac_to_volt(board['dac_bit'],board['vol_ref_dac'],code))
    def timestamp(row,label):
        nonlocal lastmono,lasttime
        stamp=datetime.fromisoformat(row['timestamp_utc'].replace('Z','+00:00'));mono=row['monotonic_time_s']
        check(stamp.tzinfo is not None and (lasttime is None or stamp>=lasttime),label+'/timestamp')
        check(math.isfinite(mono) and mono>=lastmono,label+'/monotonic')
        lasttime,lastmono=stamp,mono
    with path.open(encoding='utf-8') as stream:
        while line:=stream.readline():
            row=json.loads(line);event=row['event'];label='raw/'+str(len(actions)+1)
            check(not fault,label+'/no_events_after_fault')
            if event in ('delay','rejected_mvm'):
                check((row['wl'],row['bl']) in CELLS,label+'/event_address')
                row['_before_record_count']=len(records);actions.append(row)
                if event=='delay':
                    check(row['requested_s']==1 and math.isfinite(row['elapsed_s']) and row['elapsed_s']>=0,label+'/delay')
                    delays.append(row)
                else:fault=True
                continue
            check(event in ('request','skipped'),label+'/event')
            timestamp(row,label)
            key=(row['wl'],row['bl']);check(key in CELLS,label+'/address')
            check(row['session_id']==manifest['session_id'] and row['source']==source,label+'/source_session')
            skip=event=='skipped';task=row['planned_command_si'] if skip else row['command_si']
            mode=task['mode_flag'];v=task['vol'];write=mode==7 and v!=0
            check(mode in (7,10),label+'/mode')
            if write:
                decisions+=1
                check(key in latest,label+'/previous_mode7_exists')
                previous=latest[key];r=previous['native_r'];limit=.04 if task['sign']==0 else board['soft_cc']
                predicted=v/r
                for k,value in dict(programming_decision_id=decisions,previous_native_r_ohm=r,
                    resistance_source_task_id=previous['task_id'],predicted_current_a=predicted,current_limit_a=limit).items():compare(row.get(k),value,label+'/'+k)
                check((predicted>limit)==skip,label+'/correct_guard_decision')
                check(row['phase']==('interval_reset' if task['sign']==0 else 'interval_set'),label+'/write_phase')
                check(type(row['step']) is int and 0<=row['step']<271 and v==round(.3+.01*row['step'],2),label+'/step_voltage')
            else:check(not GUARD_FIELDS&set(row),label+'/no_programming_fields')
            width=.000125 if source=='simulator' and write else 0.
            if mode==7:
                check(type(v) in (int,float) and math.isfinite(v) and (v==0 or .3<=v<=3) and task['sign'] in (0,1),label+'/mode7_bounds')
                expected=dict(mode_flag=7,vol=v,pulse_width=width,sign=task['sign'],wl=key[0],bl=key[1])
                if not skip:expected['id']=len(records)+1
                compare(task,expected,label+'/mode7_task')
                compare(row['native_voltage_conversion'],conversion(v),label+'/DAC')
            else:
                check(isinstance(v,list) and len(v)==32 and all(x in LEVELS for x in v) and sum(x!=0 for x in v)<=1,label+'/basis')
                compare(task,dict(mode_flag=10,vol=v,wl=key[0],id=0),label+'/mode10_task')
                compare(row['native_voltage_conversion'],[conversion(x) for x in v],label+'/DAC')
            if skip:
                skips+=1
                check(write and row['reason']=='predicted_write_current_exceeds_limit',label+'/skip_reason')
                check(not {'task_id','command_si','native_result','raw_adc','native_r','returned_id','q','status','latency_s'}&set(row),label+'/no_fabricated_result')
                row['_before_record_count']=len(records);actions.append(row);continue
            result_line=stream.readline()
            if not result_line:raise ValueError('Unpaired final request')
            result=json.loads(result_line)
            check(result['event']=='result',label+'/paired_result')
            for k,value in row.items():
                if k!='event':compare(result.get(k),value,label+'/'+k)
            tid=len(records)+1
            check(result['task_id']==tid,label+'/task_id')
            check(result['status'] in ('ok','error') and math.isfinite(result['latency_s']) and result['latency_s']>=0,label+'/result_status_latency')
            check(result['physical_pulse_width_s'] is None and result['simulator_duration_steps']==(125 if width else None),label+'/width_metadata')
            check(result['native_adc_samples']==(10 if source=='hardware' and mode==7 else 1),label+'/native_adc_samples')
            native=result.get('native_result');validshape=isinstance(native,list) and len(native)==3
            if validshape:
                compare([result.get('native_r'),result.get('returned_id'),result.get('raw_adc')],native,label+'/native_tuple')
                r,rid,adc=native
                valid=type(rid) is int and rid==task['id'] and type(adc) is int and (20 if mode==7 else 0)<=adc<16383
                if result['status']=='ok':
                    check(valid,label+'/valid_ID_ADC')
                    expected_r=c.convert_adc_to_res(board['gain'],board['res_load'],.3,board['adc_bit'],board['vol_ref_adc'],board['res_switches'],adc)
                    compare(r,expected_r,label+'/native_R')
                    if mode==7:
                        check(expected_r>1e-8 and type(r) in (int,float) and math.isfinite(r) and r>0,label+'/valid_R')
                        for k,value in dict(resistance_ohm=expected_r,conductance_siemens=1/expected_r,q=board['sum_gain']/expected_r,read_voltage_parameter_v=.3).items():compare(result[k],value,label+'/'+k)
                        latest[key]=result
                    else:
                        output=c.convert_adc_to_volt(1,board['adc_bit'],board['vol_ref_adc'],adc)
                        compare(result['output_voltage_v'],output,label+'/output_V')
                        check(output<=.8*board['vol_ref_adc'],label+'/observed_headroom')
            else:check(result['status']=='error',label+'/malformed_response_failed')
            if result['status']=='error':check(bool(result.get('error')),label+'/error_preserved');fault=True
            pulses+=int(write);modes[mode]+=1
            records.append(result);actions.append(result)
    check(len(records)<=profile['maximum_impacts'] and decisions<=MAX_DECISIONS and pulses+skips==decisions,'raw/global_bounds')
    return dict(records=records,actions=actions,decisions=decisions,skips=skips,pulses=pulses,modes=dict(modes),delays=delays,fault=fault)


def replay_completed(raw,cycles,targets,board,ck,profile=None):
    check,compare=ck.check,ck.compare
    profile=profile or protocol_profile()
    target_plan=profile['targets']
    actions=raw['actions'];cursor=0;latest={};cid=0;target_index=0
    def take(phase,wl,bl,**meta):
        nonlocal cursor
        row=actions[cursor];cursor+=1
        check(row['event']=='result' and row['status']=='ok','replay/successful_result')
        compare([row['phase'],row['wl'],row['bl'],row['cycle_id']],[phase,wl,bl,cid],'replay/phase_address_cycle')
        for k,v in meta.items():compare(row.get(k),v,'replay/'+k)
        if row['command_si']['mode_flag']==7:latest[wl,bl]=row
        return row
    def reads(wl,bl,n,phase,target=None):
        rows=[]
        for repeat in range(n):
            metadata=dict(repeat=repeat)
            if target is not None:metadata['target_q']=target
            row=take(phase,wl,bl,**metadata)
            check(row['command_si']['mode_flag']==7 and row['command_si']['vol']==0,'replay/read_only_mode7')
            rows.append(row)
        return rows
    def read_summary(rows,band=None):
        return dict(task_ids=[r['task_id'] for r in rows],raw_adc=[r['raw_adc'] for r in rows],
            native_r_ohm=[r['native_r'] for r in rows],q=summarize([r['q'] for r in rows]),
            all_inside=all(inside(r['native_r'],band) for r in rows) if band else None)
    def program(wl,bl,sign,step,target=None):
        nonlocal cursor
        row=actions[cursor]
        phase='interval_reset' if sign==0 else 'interval_set'
        compare([row['phase'],row['wl'],row['bl'],row['cycle_id'],row['step']],
            [phase,wl,bl,cid,step],'replay/program_schedule')
        task=row['planned_command_si'] if row['event']=='skipped' else row['command_si']
        check(task['mode_flag']==7 and task['sign']==sign and task['vol']==round(.3+.01*step,2),'replay/program_command')
        if target is not None:compare(row['target_q'],target,'replay/program_target')
        if row['event']=='skipped':cursor+=1;return None
        return take(phase,wl,bl,step=step)
    def checkpoint(wl,bl,target,band):
        nonlocal cursor
        q=target['target_q']
        before=reads(wl,bl,5,'interval_pre',q)
        compare(target['pre'],read_summary(before,band),'replay/pre')
        g=sum(1/r['native_r'] for r in before)/5
        prediction=dict(current_a=.1*g,output_v=.1*g*board['sum_gain'])
        compare(target['mvm_prediction'],prediction,'replay/mvm_prediction')
        check(g>0 and prediction['current_a']<=.005 and prediction['output_v']<=4,'replay/model_guard')
        mvms=[]
        for repeat in range(5):
            for position,v in enumerate(LEVELS):
                row=take('interval_mvm',wl,bl,target_q=q,repeat=repeat,position=position,nominal_v=v)
                compare(row['command_si']['vol'],[v if i==bl else 0. for i in range(32)],'replay/basis_vector')
                mvms.append(row)
        compare(target['mvm'],[{k:r[k] for k in ('task_id','raw_adc','output_voltage_v','repeat','position','nominal_v')} for r in mvms],'replay/mvm_records')
        post=reads(wl,bl,5,'interval_post',q)
        compare(target['post'],read_summary(post,band),'replay/post')
        delay=actions[cursor];cursor+=1
        compare([delay['event'],delay['cycle_id'],delay['wl'],delay['bl'],delay['target_q']],['delay',cid,wl,bl,q],'replay/delay_event')
        expected_delay={k:delay[k] for k in ('requested_s','elapsed_s','started_monotonic_s','delay_met')}
        compare(target['delay'],expected_delay,'replay/delay_record')
        check(delay['started_monotonic_s']>=post[-1]['monotonic_time_s']+post[-1]['latency_s'],'replay/delay_after_post')
        delayed=reads(wl,bl,3,'interval_delayed',q)
        compare(target['delayed'],read_summary(delayed,band),'replay/delayed')
        check(delayed[0]['monotonic_time_s']>=delay['started_monotonic_s']+delay['elapsed_s'],'replay/delayed_after_wait')
        delay_met=delay['elapsed_s']>=1.
        compare(target['delay']['delay_met'],delay_met,'replay/delay_met')
        all_inside=all(inside(r['native_r'],band) for r in before+post+delayed)
        compare(target['all_readbacks_inside'],all_inside,'replay/all_inside')
        screen=screen_adc(*[[r['raw_adc'] for r in mvms if r['position']==p] for p in (0,2,4)])
        expected_screen={k:screen[k] for k in ('snr_low','snr_high','cv_high','passes')}
        expected_screen.update(epsilon_adc=screen['linearity_epsilon_adc'],tolerance_adc=screen['linearity_tolerance_adc'])
        compare(target['mvm_screen'],expected_screen,'replay/mvm_screen')
        compare(target['interval_and_mvm_pass'],all_inside and screen['passes'] and delay_met,'replay/qualification')
    for wl,bl in CELLS:
        for cycle in range(1,4):
            cid+=1;record=cycles[cid-1];start=cursor
            compare([record['cycle_id'],record['cycle'],record['wl'],record['bl']],[cid,cycle,wl,bl],'replay/cycle_identity')
            current=reads(wl,bl,1,'cycle_initial')[0]
            for step in range(271):
                if current['q']<=.005:break
                response=program(wl,bl,0,step)
                if response is not None:current=response
            crossed=current['q']<=.005
            compare(record['reset_crossed'],crossed,'replay/reset_crossing')
            confirmation=reads(wl,bl,5,'reset_confirmation')
            compare(record['reset_confirmation'],read_summary(confirmation),'replay/reset_confirmation')
            confirmed=crossed and sum(r['q'] for r in confirmation)/5<=.005 and confirmation[-1]['q']<=.005
            compare(record['reset_confirmed'],confirmed,'replay/reset_confirmed')
            check(len(record['targets'])==len(target_plan[wl,bl]),'replay/target_count')
            step=0
            for q,target in zip(target_plan[wl,bl],record['targets']):
                compare(target['target_q'],q,'replay/target_q')
                band=expected_band(q,board['sum_gain'])
                compare(target['native_r_interval_ohm'],[band['r_min'],band['r_max']],'replay/Rband')
                stop,tolerance=stop_band(q,profile,board['sum_gain'])
                if profile['refined']:
                    compare(target.get('stop_native_r_interval_ohm'),[stop['r_min'],stop['r_max']],'replay/stop_Rband')
                    compare(target.get('stop_q_relative_tolerance'),tolerance,'replay/stop_tolerance')
                flat=dict(cycle_id=cid,wl=wl,bl=bl,**target)
                compare(targets[target_index],flat,'replay/target_jsonl');target_index+=1
                if not confirmed:
                    check(target['status']=='reset_unconfirmed' and target['immediate_hit'] is False,'replay/reset_prevents_set')
                    continue
                while True:
                    current=latest[wl,bl]
                    if inside(current['native_r'],stop):
                        check(target['status']=='hit' and target['immediate_hit'] is True,'replay/real_hit')
                        compare(target['evidence_task_id'],current['task_id'],'replay/hit_evidence_task')
                        compare(target['evidence_native_r_ohm'],current['native_r'],'replay/hit_R')
                        compare(target['evidence_q'],current['q'],'replay/hit_q')
                        origin='programming_response' if current['command_si']['vol'] else 'readback_without_new_write'
                        compare(target['hit_origin'],origin,'replay/hit_origin')
                        checkpoint(wl,bl,target,band);break
                    if current['native_r']<stop['r_min'] or step>=271:
                        expected='overshoot' if current['native_r']<stop['r_min'] else 'set_ramp_exhausted'
                        check(target['status']==expected and target['immediate_hit'] is False,'replay/scientific_miss')
                        compare(target['evidence_task_id'],current['task_id'],'replay/miss_evidence_task')
                        compare(target['evidence_q'],current['q'],'replay/miss_q')
                        check('interval_and_mvm_pass' not in target and 'mvm' not in target,'replay/no_fabricated_measurement_on_miss')
                        break
                    program(wl,bl,1,step,q);step+=1
            final=reads(wl,bl,3,'cycle_final')
            compare(record['final'],read_summary(final),'replay/final')
            group=actions[start:cursor]
            nr=sum(r['event']=='result' for r in group)
            nw=sum(r['event']=='result' and r['command_si']['mode_flag']==7 and bool(r['command_si']['vol']) for r in group)
            ns=sum(r['event']=='skipped' for r in group)
            compare({k:record[k] for k in ('impact_attempts','programming_pulse_attempts','programming_skipped_current','programming_decisions')},
                dict(impact_attempts=nr,programming_pulse_attempts=nw,programming_skipped_current=ns,programming_decisions=nw+ns),'replay/cycle_ledger')
            check(nr<=551+43*len(target_plan[wl,bl]) and nw+ns<=542 and record['status']=='completed','replay/cycle_bounds_status')
    check(cursor==len(actions) and cid==len(cycles)==9 and target_index==len(targets)==profile['target_count'],'replay/consumed_all')


def audit_partial(raw,cycles,targets,summary,ck,profile=None):
    """Conservative partial consistency checks; never labels this a complete replay."""
    check,compare=ck.check,ck.compare
    profile=profile or protocol_profile()
    target_plan=profile['targets']
    check(bool(summary.get('error') or summary.get('close_error')),'partial/error_preserved')
    byid={r['task_id']:r for r in raw['records']}
    for cycle in cycles:
        cid=cycle['cycle_id'];group=[r for r in raw['actions'] if r['cycle_id']==cid]
        key=(cycle['wl'],cycle['bl'])
        check(all((r['wl'],r['bl'])==key for r in group),'partial/cycle_address')
        n=sum(r['event']=='result' for r in group)
        w=sum(r['event']=='result' and r['command_si']['mode_flag']==7 and bool(r['command_si']['vol']) for r in group)
        s=sum(r['event']=='skipped' for r in group)
        compare([cycle[k] for k in ('impact_attempts','programming_pulse_attempts','programming_skipped_current','programming_decisions')],[n,w,s,w+s],'partial/cycle_ledger')
        check(n<=551+43*len(target_plan[key]) and w+s<=542,'partial/cycle_budget')
        for phase in ('interval_reset','interval_set'):
            steps=[r['step'] for r in group if r.get('phase')==phase]
            check(steps==list(range(len(steps))) and len(steps)<=271,'partial/shared_ramp_prefix')
        if any(r.get('phase')=='interval_set' for r in group):check(cycle.get('reset_confirmed') is True,'partial/set_after_reset_confirmation')
        check([t['target_q'] for t in cycle['targets']]==list(target_plan[key])[:len(cycle['targets'])],'partial/target_prefix')
    for target in targets:
        band=expected_band(target['target_q']);key=(target['wl'],target['bl'])
        compare(target['native_r_interval_ohm'],[band['r_min'],band['r_max']],'partial/target_band')
        stop,tolerance=stop_band(target['target_q'],profile)
        if profile['refined']:
            compare(target.get('stop_native_r_interval_ohm'),[stop['r_min'],stop['r_max']],'partial/stop_band')
            compare(target.get('stop_q_relative_tolerance'),tolerance,'partial/stop_tolerance')
        if 'evidence_task_id' in target:
            check(target['evidence_task_id'] in byid,'partial/evidence_exists')
            row=byid[target['evidence_task_id']]
            check(row['status']=='ok' and row['command_si']['mode_flag']==7 and (row['wl'],row['bl'])==key and row['cycle_id']==target['cycle_id'],'partial/evidence_identity')
            compare(target['evidence_q'],row['q'],'partial/evidence_q')
            if target['immediate_hit']:
                check(inside(row['native_r'],stop),'partial/real_hit')
                compare(target['evidence_native_r_ohm'],row['native_r'],'partial/evidence_R')
                compare(target['hit_origin'],'programming_response' if row['command_si']['vol'] else 'readback_without_new_write','partial/hit_origin')
            if target['status']=='overshoot':check(row['native_r']<stop['r_min'],'partial/overshoot')
        for field,n in (('pre',5),('post',5),('delayed',3)):
            if field not in target:continue
            saved=target[field];rows=[byid[i] for i in saved['task_ids']]
            check(len(rows)==n,'partial/read_count')
            for r in rows:check(r['status']=='ok' and r['command_si']['mode_flag']==7 and r['command_si']['vol']==0 and r['phase']=='interval_'+field and (r['wl'],r['bl'])==key and r['target_q']==target['target_q'] and r['cycle_id']==target['cycle_id'],'partial/read_identity')
            compare(saved['raw_adc'],[r['raw_adc'] for r in rows],'partial/read_ADC')
            compare(saved['native_r_ohm'],[r['native_r'] for r in rows],'partial/read_R')
            compare(saved['q'],summarize([r['q'] for r in rows]),'partial/read_qstats')
            compare(saved['all_inside'],all(inside(r['native_r'],band) for r in rows),'partial/all_inside')
        mvms=target.get('mvm',[])
        check([(r['repeat'],r['position']) for r in mvms]==[(i,p) for i in range(5) for p in range(6)][:len(mvms)],'partial/mvm_prefix')
        for r in mvms:
            original=byid[r['task_id']]
            check(original['status']=='ok' and original['phase']=='interval_mvm' and original['cycle_id']==target['cycle_id'] and (original['wl'],original['bl'])==key and original['target_q']==target['target_q'],'partial/mvm_identity')
            compare(r,{k:original[k] for k in ('task_id','raw_adc','output_voltage_v','repeat','position','nominal_v')},'partial/mvm_record')
        if 'mvm_screen' in target:
            check(len(mvms)==30,'partial/complete_screen_requires_30')
            sc=screen_adc(*[[r['raw_adc'] for r in mvms if r['position']==p] for p in (0,2,4)])
            compare(target['mvm_screen']['passes'],sc['passes'],'partial/screen_pass')
        if 'interval_and_mvm_pass' in target:
            expected=all(target.get(k,{}).get('all_inside') is True for k in ('pre','post','delayed')) and target.get('delay',{}).get('elapsed_s',0)>=1 and target.get('mvm_screen',{}).get('passes') is True
            compare(target['interval_and_mvm_pass'],expected,'partial/qualification')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    parser.add_argument('--session',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();repo=Path(args.repo).resolve();ck=Checks();check,compare=ck.check,ck.compare
    def local(name):
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Use repository-relative paths')
        p=(repo/name).resolve()
        if not p.is_relative_to(repo):raise ValueError('Path escapes repository')
        return p
    session,out=local(args.session),local(args.output_dir)
    if out.exists() or out==session:raise ValueError('Use a new audit output folder')
    manifest,summary,plan=(load(session/name) for name in ('manifest.json','summary.json','protocol.json'))
    profile=protocol_profile(plan['protocol_id'])
    start=load(session/'start_manifest.json');source=manifest['source'];commit=manifest['invocation']['git']['commit']
    check(source in ('hardware','simulator') and source==summary['source'],'source')
    check(manifest['status']==summary['status'] and summary['status'] in ('completed','failed'),'status')
    check(manifest['hardware_access']==(source=='hardware'),'hardware_access')
    check(summary['continuous_range_qualified'] is False,'no_continuous_range_claim')
    compare(manifest['invocation'],start['invocation'],'invocation/start_final')
    argv=manifest['invocation']['process_argv']
    check(len(argv)==3 and argv[0]=='experiment.py' and sorted(argv[1:])==sorted([profile['flag'],'--run-'+source]),'invocation/explicit_scope')
    if source=='hardware':check(manifest['invocation']['git']['dirty_tracked'] is False,'hardware/clean_tracked')
    pins={'cell_range_repeat.py':RUNTIME_SHA,'cell_calibration.py':BASE_SHA,
        'vendor/MemriBoard/manager/service/converters.py':CONVERTER_SHA,
        'vendor/MemriBoard/manager/terminate.py':EQUALIN_SHA}
    if profile['refined']:pins['cell_range_refined.py']=REFINED_SHA
    missing_private=[];local_differences=[];source_blobs={}
    for name,pin in pins.items():check(manifest['source_sha256'].get(name)==pin,'source/reviewed_pin/'+name)
    for name,digest in manifest['source_sha256'].items():
        path=local(name)
        if name in PRIVATE:
            check(digest==PRIVATE[name],'source/private_pin/'+name)
            if path.is_file():check(sha(path)==digest,'source/private_local/'+name)
            else:missing_private.append(name)
        else:
            blob=subprocess.run(['git','-C',str(repo),'show',commit+':'+name],capture_output=True,check=True).stdout
            source_blobs[name]=blob
            difference=verify_public_source(repo,name,digest,blob,ck)
            if difference:local_differences.append(difference)
    for path in (repo/'vendor/MemriBoard').rglob('*.py'):
        check(path.relative_to(repo).as_posix() in manifest['source_sha256'],'source/all_importable_files_recorded')
    if source=='hardware':
        for name,pin in PRIVATE.items():check(manifest['source_sha256'].get(name)==pin,'source/private_required/'+name)
    if ck.errors:raise ValueError('Source validation failed before converter import: '+repr(ck.errors))
    for name,digest in manifest['result_sha256'].items():
        p=(session/name).resolve()
        if not p.is_relative_to(session):raise ValueError('Result path escapes session')
        check(sha(p)==digest,'result/SHA/'+name)
    base=json.loads(source_blobs['configs/cell_calibration.json']);spec=json.loads(source_blobs[profile['config']])
    compare(plan,{**base,**spec},'protocol/exact_source_merge')
    expected_spec=dict(protocol_id=plan['protocol_id'],
        cells=[dict(wl=wl,bl=bl,targets_q=list(profile['targets'][wl,bl])) for wl,bl in CELLS],cycles_per_cell=3,
        q_relative_tolerance=.01,reset_endpoint_q=.005,ramp=dict(start_v=.3,step_v=.01,stop_v=3.,maximum_steps=271),
        initial_reads=1,reset_confirmation_reads=5,pre_reads=5,post_reads=5,delayed_reads=3,delay_s=1.,final_reads=3,
        maximum_programming_decisions=MAX_DECISIONS,maximum_programming_pulses=MAX_DECISIONS,maximum_impacts=profile['maximum_impacts'],
        baseline_scope='per_cell_cycle_before_reset')
    if profile['refined']:expected_spec.update(stop_q_relative_tolerance=.0075,control_targets_q=[.045,.05],control_stop_q_relative_tolerance=.01)
    compare({k:v for k,v in spec.items() if k!='qualification'},expected_spec,'protocol/reviewed_spec')
    check(manifest['protocol_id']==summary['protocol_id']==plan['protocol_id'],'protocol/id_agreement')
    for k in ('maximum_impacts','maximum_programming_decisions','maximum_programming_pulses'):compare(manifest[k],plan[k],'manifest/'+k)
    compare(manifest['cells'],expected_spec['cells'],'manifest/cells')
    check(manifest['cycles_per_cell']==3 and manifest['physical_pulse_width_s'] is None,'manifest/cycles_duration')
    cfg=ConfigParser();cfg.read_string((session/'board.ini').read_text(encoding='utf-8-sig'))
    board={k:cfg.getfloat('board',k) for k in ('gain','res_load','res_switches','vol_read','vol_ref_adc','vol_ref_dac','sum_gain','soft_cc')}
    board.update(adc_bit=cfg.getint('board','adc_bit'),dac_bit=cfg.getint('board','dac_bit'))
    compare(board,dict(gain=11,res_load=3000,res_switches=10,vol_read=.3,vol_ref_adc=5,vol_ref_dac=5,sum_gain=330,soft_cc=.002,adc_bit=14,dac_bit=12),'board/reviewed_values')
    if ck.errors:raise ValueError('Input validation failed before converter import: '+repr(ck.errors))
    sys.dont_write_bytecode=True
    for name in ('.deps','vendor/MemriBoard'):
        if (repo/name).is_dir():sys.path.insert(0,str(repo/name))
    from manager.service import converters
    raw=audit_raw(session/'raw.jsonl',manifest,board,converters,ck,profile)
    for name,value in dict(impact_attempts=len(raw['records']),programming_pulse_attempts=raw['pulses'],
        programming_decisions=raw['decisions'],programming_skipped_current=raw['skips']).items():
        compare(manifest[name],value,'manifest/'+name);compare(summary[name],value,'summary/'+name)
    cycles,targets=jsonl(session/'cycles.jsonl'),jsonl(session/'targets.jsonl')
    compare(summary['cycles'],cycles,'summary/cycles_jsonl')
    compare(targets,[dict(cycle_id=c['cycle_id'],wl=c['wl'],bl=c['bl'],**t) for c in cycles for t in c['targets']],'targets/flattened')
    expected_cycles=[(i+1,wl,bl,cycle) for i,(wl,bl,cycle) in enumerate((wl,bl,cycle) for wl,bl in CELLS for cycle in range(1,4))]
    check([(c['cycle_id'],c['wl'],c['bl'],c['cycle']) for c in cycles]==expected_cycles[:len(cycles)] and len(cycles)<=9,'cycles/prefix')
    for row in raw['actions']:
        check(type(row['cycle_id']) is int and 1<=row['cycle_id']<=len(cycles),'raw/cycle_present')
    completed=summary['status']=='completed'
    if completed:replay_completed(raw,cycles,targets,board,ck,profile)
    else:audit_partial(raw,cycles,targets,summary,ck,profile)
    outcomes=dict(cycles=len(cycles),cycle_statuses=dict(Counter(c['status'] for c in cycles)),
        reset_crossed=sum(c.get('reset_crossed',False) for c in cycles),reset_confirmed=sum(c.get('reset_confirmed',False) for c in cycles),
        target_statuses=dict(Counter(t['status'] for t in targets)),hit_origins=dict(Counter(t['hit_origin'] for t in targets if 'hit_origin' in t)),
        target_observed_pass=sum(t.get('interval_and_mvm_pass') is True for t in targets),
        per_cell=[dict(wl=wl,bl=bl,targets=[dict(cycle_id=t['cycle_id'],target_q=t['target_q'],status=t['status'],
            hit_origin=t.get('hit_origin'),pre_inside=t.get('pre',{}).get('all_inside'),post_inside=t.get('post',{}).get('all_inside'),
            delayed_inside=t.get('delayed',{}).get('all_inside'),mvm_pass=t.get('mvm_screen',{}).get('passes'),
            interval_and_mvm_pass=t.get('interval_and_mvm_pass'),evidence_q=t.get('evidence_q'))
            for t in targets if (t['wl'],t['bl'])==(wl,bl)]) for wl,bl in CELLS])
    report=dict(audit_status='passed' if not ck.errors else 'failed',checks=ck.count,errors=ck.errors,max_numeric_difference=ck.maxdiff,
        session_id=manifest['session_id'],source=source,acquisition_status=summary['status'],full_protocol_replay=completed,
        auditor_sha256=sha(Path(__file__)),audit_core_sha256=sha(Path(__file__).with_name('interval_audit_core.py')),
        manifest_sha256=sha(session/'manifest.json'),raw_sha256=sha(session/'raw.jsonl'),source_commit=commit,
        protocol_id=plan['protocol_id'],local_source_differences=local_differences,
        private_local_unavailable=missing_private,impact_pairs=len(raw['records']),programming_decisions=raw['decisions'],
        programming_pulse_attempts=raw['pulses'],programming_skipped_current=raw['skips'],mode_counts=raw['modes'],outcomes=outcomes,
        limitations=['Completed acquisition and audit consistency are not scientific qualification.',
            'Only three specified addresses have new interval repeats; no all256 range qualification.',
            'Passing sampled target bands does not establish a continuous range, precision bits or CIFAR accuracy.',
            'Native pulse duration and physical current compliance remain unmeasured.',
            'Partial sessions receive prefix/ledger/evidence checks, not full nine-cycle replay.'])
    out.mkdir(parents=True,exist_ok=False)
    (out/'independent_interval_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    text=['# Независимый аудит повторов интервалов','',f"Сессия `{manifest['session_id']}`, {source}; сбор **{summary['status']}**, аудит **{report['audit_status']}**. {ck.count} проверок, {len(ck.errors)} ошибок.",'',
        f"Actual impacts: {len(raw['records'])}; programming decisions: {raw['decisions']}; реальные попытки импульса: {raw['pulses']}; skipped: {raw['skips']}. Полный replay: {completed}.",'',
        f"Targets: {outcomes['target_statuses']}; наблюдённый interval+MVM pass: {outcomes['target_observed_pass']}. Источник hit: {outcomes['hit_origins']}.",'',
        'Проверены source/result SHA, native conversions, последний mode7 R перед current guard, отдельные skipped decisions, inclusive R bands и происхождение evidence. Для completed воспроизведены все 9 циклов, общий SET budget, все pre/post/delayed samples, MVM-screen, delay и итоги.', '',
        f"Протокол: {plan['protocol_id']}. Остановка hit/overshoot проверена по stop band; подтверждение readbacks — по outer ±1%. Отличия текущих неимпортируемых файлов от сохранённого Git: {len(local_differences)}; детали в JSON. Все импортируемые vendor Python файлы проверены по локальным байтам.", '',
        'Аудитор не создаёт Connector и не вызывает аппаратуру. Passing отдельных точек не подтверждает непрерывный диапазон, все256 ячеек или CIFAR accuracy. Аппаратная длительность импульса неизвестна.', '',
        '```sh',f'python -B audit_cell_range_repeat.py --repo . --session {args.session} --output-dir {args.output_dir}-recheck','```']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('audit_status','checks','errors','impact_pairs','programming_decisions','programming_skipped_current','outcomes')},ensure_ascii=False))
    return 1 if ck.errors else 0


if __name__=='__main__':raise SystemExit(main())
