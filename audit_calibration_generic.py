"""Independent offline audit of completed or partial native discovery. No Connector creation/hardware calls."""
import argparse
from collections import Counter
from configparser import ConfigParser
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys

ENGINE_SHA='5b0abf55138cbf662ea5ced6c290a0381d1204dee8420a2223f534d5279b806d'
ENGINE_V2_SHA='eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8'
PROTOCOL_V1='cell-conductance-discovery-v1'
PROTOCOL_V2='cell-conductance-discovery-v2-gui-write-guard'
GUARD_FIELDS={'programming_decision_id','previous_native_r_ohm','resistance_source_task_id','predicted_current_a','current_limit_a'}
WRITE_GUARD=dict(prediction='requested_voltage_over_last_valid_mode7_native_resistance',
    sign0_limit_a=.04,sign1_limit_source='board.soft_cc',invalid_resistance='global_stop',
    skip_policy='advance_ramp_without_impact_or_state_update')
HISTORICAL_NONIMPORTED_SOURCES={'cell_calibration.py','experiment.py','native_conversions.py',
    'read_all.py','experiment_hardware.py','configs/cell_calibration.json'}
CONVERTER_SHA='28c6495c56afa225501dd6b0dea999215f08b73f30e016ff7ea34d6c5412701c'
PRIVATE_PINS={
 'vendor/MemriBoard/MemriCORE/rp5_fpga_python/rpi_FPGAed.py':'bc717659fa5d4d36d2037eef978bf1cd8dcf0fe92f9bbfad51f1eb1d0d27df9c',
 'vendor/MemriBoard/MemriCORE/rp5_fpga_python/FPGA_SIM.py':'5bcb02c04bc185e332b2ec96acabc2d4303d21724c9c052de540ef92d603b634'}
CELLS=[(wl,bl) for wl in range(8) for bl in range(32)]
TARGETS=[.01,.02,.03,.04,.05]
LEVELS=[0,.025,.05,.075,.1,0]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def jsonl(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.is_file() else []
def stats(values):return dict(n=len(values),mean=mean(values),sd=stdev(values) if len(values)>1 else None,minimum=min(values),maximum=max(values))


def valid_invocation(argv,source):
    return (isinstance(argv,list) and len(argv)==3 and argv[0]=='experiment.py'
        and sorted(argv[1:])==sorted(['--run-'+source,'--calibrate-cells']))


class GuardReplay:
    """Independent v2 decision ledger; mode10 and skips never update mode7 resistance."""
    def __init__(self,check,compare,soft_cc,source):
        self.check,self.compare,self.soft_cc,self.source=check,compare,soft_cc,source
        self.latest={};self.decisions=[];self.skipped=0

    def observe(self,row):
        if row['status']=='ok' and row['command_si']['mode_flag']==7:
            r=row['native_r']
            self.check(type(r) in (int,float) and math.isfinite(r) and r>0,'guard/latest_valid_native_R')
            self.latest[row['wl'],row['bl']]=(r,row['task_id'])

    def decide(self,event,before_record_count):
        skip=event['event']=='skipped';key=(event['wl'],event['bl'])
        task=event['planned_command_si'] if skip else event['command_si']
        did=len(self.decisions)+1;label='guard/'+str(did)
        self.check(GUARD_FIELDS<=set(event),label+'/required_fields')
        self.check(key in self.latest,label+'/previous_read_exists')
        if key not in self.latest:raise ValueError('Guard decision without previous successful mode7')
        r,rid=self.latest[key];sign=task['sign'];v=task['vol']
        self.check(type(v) in (float,int) and math.isfinite(v) and .3<=v<=3 and sign in (0,1),label+'/bounds')
        limit=.04 if sign==0 else self.soft_cc
        predicted=v/r
        for name,value in dict(programming_decision_id=did,previous_native_r_ohm=r,
            resistance_source_task_id=rid,predicted_current_a=predicted,current_limit_a=limit).items():
            self.compare(event.get(name),value,label+'/'+name)
        self.check(math.isfinite(predicted) and (predicted>limit)==skip,label+'/allow_or_skip')
        self.check(event['phase']==('reset' if sign==0 else 'set'),label+'/phase_sign')
        step=event['step']
        self.check(type(step) is int and 0<=step<271 and v==round(.3+.01*step,2),label+'/ramp')
        expected=dict(mode_flag=7,vol=v,pulse_width=.000125 if self.source=='simulator' else 0,sign=sign,wl=key[0],bl=key[1])
        if skip:
            self.check(not ({'task_id','command_si','native_result','raw_adc','native_r','returned_id','q','conductance_siemens','resistance_ohm','status','latency_s'}&set(event)),label+'/no_fabricated_measurement')
            self.check(event['reason']=='predicted_write_current_exceeds_limit',label+'/skip_reason')
            self.skipped+=1
        else:expected['id']=before_record_count+1
        self.compare(task,expected,label+'/exact_command')
        self.decisions.append(dict(wl=key[0],bl=key[1],phase=event['phase'],step=step,
            vol=v,sign=sign,skipped=skip,before_record_count=before_record_count,
            programming_decision_id=did))
        self.check(did<=138752,label+'/decision_budget')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    for arg in ('session','output-dir'):parser.add_argument('--'+arg,required=True)
    parser.add_argument('--source-commit',help='Optional source blob commit; defaults to recorded invocation commit')
    args=parser.parse_args()
    repo=Path(args.repo).resolve()
    def relative_path(value):
        if Path(value).is_absolute() or '..' in Path(value).parts:raise ValueError('Session/output must be repository-relative')
        path=(repo/value).resolve()
        if not path.is_relative_to(repo):raise ValueError('Path escapes repo')
        return path
    run,out=relative_path(args.session),relative_path(args.output_dir)
    if out==run or out.exists():raise ValueError('Use a new audit output folder, separate from immutable source session')
    manifest,summary,plan=(load(run/name) for name in ('manifest.json','summary.json','protocol.json'))
    source=manifest['source'];completed=summary['status']=='completed'
    protocol=plan['protocol_id'];guarded=protocol==PROTOCOL_V2
    source_commit=args.source_commit or manifest['invocation']['git']['commit']
    count=0;errors=[];maxdiff=0.0
    def check(condition,label):
        nonlocal count
        count+=1
        if not condition:errors.append(label)
    def compare(actual,expected,label):
        nonlocal maxdiff
        if isinstance(expected,dict):
            check(isinstance(actual,dict) and set(actual)==set(expected),label+'/keys')
            if isinstance(actual,dict):
                for key in expected:
                    if key in actual:compare(actual[key],expected[key],label+'/'+str(key))
        elif isinstance(expected,list):
            check(isinstance(actual,list) and len(actual)==len(expected),label+'/length')
            if isinstance(actual,list):
                for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,label+'/'+str(i))
        elif type(actual) in (float,int) and type(expected) in (float,int):
            maxdiff=max(maxdiff,abs(actual-expected))
            check(math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),label)
        else:check(actual==expected,label)
    check(protocol in (PROTOCOL_V1,PROTOCOL_V2),'reviewed_protocol')
    check(manifest['source_sha256']['cell_calibration.py']==(ENGINE_V2_SHA if guarded else ENGINE_SHA),'reviewed_engine')
    check(sha(repo/'vendor/MemriBoard/manager/service/converters.py')==CONVERTER_SHA,'converter_pin')
    if errors:raise ValueError('Unreviewed numerical/engine sources')
    verified=[];local_differences=[];private_local_unavailable=[]
    for name,digest in manifest['source_sha256'].items():
        path=(repo/name).resolve()
        if not path.is_relative_to(repo):raise ValueError('Source path escapes repo')
        if name in PRIVATE_PINS:
            check(digest==PRIVATE_PINS[name],'private_source_pin/'+name)
            if path.is_file():check(sha(path)==digest,'source_sha/'+name)
            else:private_local_unavailable.append(name)
        else:
            blob=subprocess.run(['git','-C',str(repo),'show',source_commit+':'+name],capture_output=True,check=True).stdout
            check(hashlib.sha256(blob).hexdigest()==digest,'source_git_blob/'+name)
            local_digest=sha(path) if path.is_file() else None
            if local_digest!=digest and name in HISTORICAL_NONIMPORTED_SOURCES:
                local_differences.append(dict(path=name,recorded_sha256=digest,local_sha256=local_digest,
                    verification='Recorded Git blob; this root source is not imported or executed by auditor'))
            else:check(local_digest==digest,'source_sha/'+name)
        verified.append(name)
    # All vendor Python files potentially imported by the package must be present,
    # fingerprinted and identical, not merely the numerical converter itself.
    for path in (repo/'vendor/MemriBoard').rglob('*.py'):
        name=path.relative_to(repo).as_posix()
        check(name in manifest['source_sha256'],'importable_source_in_manifest/'+name)
    if errors:raise ValueError('Source verification failed before native package import: '+repr(errors))
    for name,digest in manifest['result_sha256'].items():
        path=(run/name).resolve()
        if not path.is_relative_to(run):raise ValueError('Result path escapes session')
        check(sha(path)==digest,'result_sha/'+name)
    check(manifest['status']==summary['status'] and summary['status'] in ('completed','failed'),'status/agreement')
    check(source==summary['source'] and source in ('simulator','hardware'),'source/agreement')
    check(valid_invocation(manifest['invocation']['process_argv'],source),'invocation')
    if source=='hardware':
        check(manifest['invocation']['git']['dirty_tracked'] is False,'hardware/clean_tracked')
        for name,pin in PRIVATE_PINS.items():check(manifest['source_sha256'].get(name)==pin,'hardware/private_required/'+name)
    compare(manifest['invocation'],load(run/'start_manifest.json')['invocation'],'invocation/start_final')
    check(plan['maximum_impacts']==203008 and plan['maximum_programming_pulses']==138752,'bounds/fixed')
    check(plan['geometry']==dict(wl_count=8,bl_count=32,excluded_wl=[]) and plan['cycles']==1,'all256_onecycle')
    check(plan['simulator_pulse_width_s']==.000125 and plan['hardware_pulse_width_s'] is None,'timing_separation')
    check(summary['programming_repeatability_qualified'] is False,'not_repeatability_qualified')
    if guarded:
        compare(plan['programming_guard'],WRITE_GUARD,'protocol/write_guard')
        compare(manifest['programming_guard'],WRITE_GUARD,'manifest/write_guard')
        check(plan['maximum_programming_decisions']==manifest['maximum_programming_decisions']==138752,'bounds/decisions')
        check(manifest['protocol_id']==summary['protocol_id']==protocol,'protocol/agreement')
    sys.dont_write_bytecode=True
    for name in ('.deps','vendor/MemriBoard'):
        if (repo/name).is_dir():sys.path.insert(0,str(repo/name))
    from manager.service import converters as c
    cfg=ConfigParser();cfg.read_string((run/'board.ini').read_text(encoding='utf-8-sig'))
    b={k:cfg.getfloat('board',k) for k in ('gain','res_load','vol_read','res_switches','vol_ref_adc','vol_ref_dac','sum_gain')}
    b.update(adc_bit=cfg.getint('board','adc_bit'),dac_bit=cfg.getint('board','dac_bit'))
    guard=GuardReplay(check,compare,cfg.getfloat('board','soft_cc'),source)
    def voltage(v):
        code=c.convert_volt_to_dac(b['dac_bit'],b['vol_ref_dac'],v)
        return dict(requested_v=v,dac_code=code,native_decoded_v=c.convert_dac_to_volt(b['dac_bit'],b['vol_ref_dac'],code))
    records=[];modes=Counter();phases=Counter();pulses=0;lastmono=-math.inf;lasttime=None
    with (run/'raw.jsonl').open(encoding='utf-8') as stream:
        while request_line:=stream.readline():
            request=json.loads(request_line)
            if request['event']=='skipped':
                check(guarded,'skipped/requires_v2')
                check(request['source']==source and request['session_id']==manifest['session_id'],'skipped/identity')
                stamp=datetime.fromisoformat(request['timestamp_utc'].replace('Z','+00:00'))
                check(stamp.tzinfo is not None and (lasttime is None or stamp>=lasttime),'skipped/timestamp')
                lasttime=stamp
                check(math.isfinite(request['monotonic_time_s']) and request['monotonic_time_s']>=lastmono,'skipped/monotonic')
                lastmono=request['monotonic_time_s']
                check((request['wl'],request['bl']) in CELLS,'skipped/address')
                compare(request['native_voltage_conversion'],voltage(request['planned_command_si']['vol']),'skipped/dac')
                guard.decide(request,len(records))
                continue
            result_line=stream.readline()
            if not result_line:raise ValueError('Unpaired final request')
            result=json.loads(result_line)
            tid=len(records)+1;label='raw/'+str(tid)
            check(request['event']=='request' and result['event']=='result',label+'/pair')
            for key,value in request.items():
                if key!='event':compare(result.get(key),value,label+'/'+key)
            check(result['task_id']==tid and result['source']==source and result['session_id']==manifest['session_id'],label+'/identity')
            stamp=datetime.fromisoformat(result['timestamp_utc'].replace('Z','+00:00'))
            check(stamp.tzinfo is not None and (lasttime is None or stamp>=lasttime),label+'/timestamp')
            lasttime=stamp
            check(math.isfinite(result['monotonic_time_s']) and result['monotonic_time_s']>=lastmono,label+'/monotonic')
            lastmono=result['monotonic_time_s']
            check(result['latency_s']>=0 and math.isfinite(result['latency_s']),label+'/latency')
            check(result['status'] in ('ok','error'),label+'/status')
            task=result['command_si'];mode=task['mode_flag'];vol=task['vol'];key=(result['wl'],result['bl'])
            check(key in CELLS and task['wl']==key[0] and mode in (7,10),label+'/address_mode')
            modes[mode]+=1;phases[result['phase']]+=1
            if mode==7:
                write=vol!=0;pulses+=int(write)
                if guarded and write:guard.decide(request,len(records))
                else:check(not GUARD_FIELDS&set(request),label+'/no_unexpected_guard_fields')
                expected=dict(mode_flag=7,vol=vol,pulse_width=.000125 if write and source=='simulator' else 0,sign=task['sign'],wl=key[0],bl=key[1],id=tid)
                check(type(vol) in (int,float) and (vol==0 or .3<=vol<=3) and task['sign'] in (0,1),label+'/write_bounds')
                compare(result['native_voltage_conversion'],voltage(vol),label+'/dac')
                check(result['simulator_duration_steps']==(125 if write and source=='simulator' else None),label+'/sim_duration')
            else:
                check(not GUARD_FIELDS&set(request),label+'/mvm_no_programming_decision')
                expected=dict(mode_flag=10,vol=vol,wl=key[0],id=0)
                check(len(vol)==32 and all(v in LEVELS for v in vol) and sum(v!=0 for v in vol)<=1,label+'/basis_vector')
                compare(result['native_voltage_conversion'],[voltage(v) for v in vol],label+'/dac')
                check(result['simulator_duration_steps'] is None,label+'/read_duration')
            compare(task,expected,label+'/exact_command')
            check(result['native_adc_samples']==(10 if mode==7 and source=='hardware' else 1) and result['physical_pulse_width_s'] is None,label+'/native_metadata')
            keep={k:result[k] for k in ('task_id','phase','wl','bl')}
            keep.update(mode=mode,vol=vol,sign=task.get('sign'),status=result['status'])
            for k in ('checkpoint_id','repeat','position','nominal_v','step','error'):
                if k in result:keep[k]=result[k]
            native=result.get('native_result')
            shaped=isinstance(native,list) and len(native)==3
            if shaped:
                compare([result.get('native_r'),result.get('returned_id'),result.get('raw_adc')],native,label+'/native_tuple')
                adc,rid=native[2],native[1]
                keep['raw_adc']=adc
                protocol_valid=type(adc) is int and (20 if mode==7 else 0)<=adc<16383 and type(rid) is int and rid==task['id']
                if type(adc) is int:
                    resistance=c.convert_adc_to_res(b['gain'],b['res_load'],b['vol_read'],b['adc_bit'],b['vol_ref_adc'],b['res_switches'],adc)
                    compare(native[0],resistance,label+'/native_R')
                if result['status']=='ok':
                    check(protocol_valid,label+'/valid_adc_and_id')
                    if mode==7:
                        check(resistance>1e-8,label+'/positive_R')
                        for k,v in dict(resistance_ohm=resistance,conductance_siemens=1/resistance,q=b['sum_gain']/resistance,read_voltage_parameter_v=.3).items():compare(result[k],v,label+'/'+k)
                        keep.update(q=b['sum_gain']/resistance,conductance_siemens=1/resistance)
                    else:
                        v=c.convert_adc_to_volt(1,b['adc_bit'],b['vol_ref_adc'],adc)
                        compare(result['output_voltage_v'],v,label+'/output_v')
                        check(v<=.8*b['vol_ref_adc'],label+'/observed_headroom')
                        keep['output_voltage_v']=v
                elif 'output_voltage_v' in result:
                    compare(result['output_voltage_v'],c.convert_adc_to_volt(1,b['adc_bit'],b['vol_ref_adc'],adc),label+'/failed_output_v')
            else:
                check(result['status']=='error',label+'/malformed_response_requires_error')
            if result['status']=='error':
                check(bool(result.get('error')) and not completed,label+'/failed_result_preserved')
                check(not stream.read(1),label+'/no_calls_after_error')
            if guarded:guard.observe(result)
            records.append(keep)
    check(len(records)==manifest['impact_attempts']==summary['impact_attempts']<=203008,'total_impacts')
    check(pulses==manifest['programming_pulse_attempts']==summary['programming_pulse_attempts']<=138752,'total_pulses')
    if guarded:
        check(len(guard.decisions)==manifest['programming_decisions']==summary['programming_decisions']==pulses+guard.skipped<=138752,'total_decisions')
        check(guard.skipped==manifest['programming_skipped_current']==summary['programming_skipped_current'],'total_skips')
    cps=jsonl(run/'checkpoints.jsonl');cells=jsonl(run/'cell_results.jsonl');roles=jsonl(run/'checkpoint_roles.jsonl')
    compare(cells,summary['cells'],'summary/cells')
    compare(jsonl(run/'initial_array.jsonl'),summary['initial_array'],'summary/initial')
    compare(jsonl(run/'final_array.jsonl'),summary['final_array'],'summary/final')
    check([(r['wl'],r['bl']) for r in cells]==CELLS[:len(cells)] and len(cells)<=256,'cells/coverage_prefix')
    if completed:check(len(cells)==256,'completed/cell_count')
    check(len(cps)==summary['checkpoint_count'] and [s['checkpoint_id'] for s in cps]==list(range(1,len(cps)+1)),'checkpoint_ids')
    if completed:
        cursor=cp_cursor=decision_cursor=0;lastwrite=None;expected_roles=[]
        def take(phase,key,**metadata):
            nonlocal cursor,lastwrite
            row=records[cursor];cursor+=1
            check(row['phase']==phase and (row['wl'],row['bl'])==key,'schedule/'+str(cursor))
            for k,v in metadata.items():compare(row.get(k),v,'schedule/'+str(cursor)+'/'+k)
            if row['mode']==7 and row['vol']:lastwrite=row['task_id']
            return row
        def program(phase,key,step,sign):
            nonlocal decision_cursor
            v=round(.3+.01*step,2)
            if guarded:
                decision=guard.decisions[decision_cursor];decision_cursor+=1
                compare(decision,dict(wl=key[0],bl=key[1],phase=phase,step=step,vol=v,sign=sign,
                    skipped=decision['skipped'],before_record_count=cursor,programming_decision_id=decision_cursor),'schedule/programming_decision')
                if decision['skipped']:return None
            return take(phase,key,mode=7,vol=v,sign=sign,step=step)
        def checkpoint(key,role,crossed=()):
            nonlocal cp_cursor
            saved=cps[cp_cursor];cp_cursor+=1;cid=cp_cursor
            check((saved['wl'],saved['bl'])==key and saved['role']==role and saved['last_write_task_id']==lastwrite,'checkpoint/'+str(cid)+'/identity')
            compare(saved['crossed_targets'],list(crossed),'checkpoint/crossings')
            reads=[take('checkpoint_read',key,checkpoint_id=cid,repeat=i,mode=7,vol=0) for i in range(5)]
            compare(saved['reads'],[{k:r[k] for k in ('task_id','raw_adc','q','conductance_siemens')} for r in reads],'checkpoint/read_records')
            for k in ('q','conductance_siemens'):compare(saved[k],stats([r[k] for r in reads]),'checkpoint/readstats/'+k)
            compare(saved['last_read_q'],reads[-1]['q'],'checkpoint/lastread')
            compare(saved['signed_target_errors_q'],{str(q):mean(r['q'] for r in reads)-q for q in crossed},'checkpoint/signederrors')
            g=mean(r['conductance_siemens'] for r in reads);current=.1*g;output=current*b['sum_gain']
            compare(saved['prediction'],dict(current_a=current,output_v=output,uncalibrated=True),'checkpoint/prediction')
            check(g>0 and current<=.005 and output<=4,'checkpoint/modelguard')
            mvms=[]
            for repeat in range(5):
                for position,level in enumerate(LEVELS):
                    vector=[level if i==key[1] else 0 for i in range(32)]
                    mvms.append(take('checkpoint_mvm',key,checkpoint_id=cid,repeat=repeat,position=position,nominal_v=level,mode=10,vol=vector))
            compare(saved['mvm'],[{k:r[k] for k in ('task_id','raw_adc','output_voltage_v','repeat','position','nominal_v')} for r in mvms],'checkpoint/mvm_records')
            transfer=[]
            for position,level in enumerate(LEVELS):
                conversion=voltage(level)
                transfer.append(dict(position=position,nominal_v=level,dac=conversion['dac_code'],native_decoded_v=conversion['native_decoded_v'],
                    adc=stats([r['raw_adc'] for r in mvms if r['position']==position]),output_v=stats([r['output_voltage_v'] for r in mvms if r['position']==position])))
            compare(saved['transfer'],transfer,'checkpoint/transferstats')
            check(saved['status']=='completed','checkpoint/completed')
            return saved
        def reused(state,key,role):
            expected_roles.append(dict(wl=key[0],bl=key[1],role=role,checkpoint_id=state['checkpoint_id'],
                reason='No further nonzero programming command; same checkpoint reused, not a new measured state'))
            return state
        initial={}
        for key,saved in zip(CELLS,summary['initial_array']):
            rows=[take('initial_array_read',key,repeat=i,mode=7,vol=0) for i in range(3)]
            initial[key]=stats([r['conductance_siemens'] for r in rows])
            compare(saved,dict(wl=key[0],bl=key[1],conductance_siemens=initial[key]),'initial/record')
        for key,cell in zip(CELLS,cells):
            lastwrite=None;local=[]
            baseline=checkpoint(key,'local_pre_reset');local.append(baseline);q=baseline['last_read_q'];reset_count=reset_decisions=reset_skips=0
            for step in range(271):
                if q<=.005:break
                row=program('reset',key,step,0);reset_decisions+=1
                if row is None:reset_skips+=1
                else:q=row['q'];reset_count+=1
            check(cell['reset_reached']==(q<=.005),'cell/reset_crossing')
            compare(cell['reset_decision_q'],q,'cell/reset_decision_q')
            reset=reused(local[-1],key,'post_reset') if local[-1]['last_write_task_id']==lastwrite else checkpoint(key,'post_reset')
            if reset is not local[-1]:local.append(reset)
            confirmed=q<=.005 and reset['q']['mean']<=.005 and reset['last_read_q']<=.005
            check(cell['reset_confirmed']==confirmed,'cell/reset_confirmation')
            compare(cell['reset_confirmation_mean_q'],reset['q']['mean'],'cell/reset_mean')
            compare(cell['reset_confirmation_last_q'],reset['last_read_q'],'cell/reset_last')
            crossed=[];set_count=set_decisions=set_skips=0
            if confirmed:
                q=reset['last_read_q']
                for step in range(271):
                    if crossed and crossed[-1]>=.05:break
                    row=program('set',key,step,1);set_decisions+=1
                    if row is None:set_skips+=1;continue
                    q=row['q'];set_count+=1
                    newly=[target for target in TARGETS if target not in crossed and q>=target]
                    if newly:crossed.extend(newly);local.append(checkpoint(key,'set_crossing',newly))
                compare(cell['set_terminal_response_q'],q,'cell/terminal_pulse_q')
            else:
                compare(cell['set_skipped_reason'],'reset_unconfirmed_after_readback' if cell['reset_reached'] else 'reset_endpoint_not_reached','cell/skipped_set')
            terminal=reused(local[-1],key,'terminal') if local[-1]['last_write_task_id']==lastwrite else checkpoint(key,'terminal')
            if terminal is not local[-1]:local.append(terminal)
            compare(cell['reset_pulses'],reset_count,'cell/reset_count');compare(cell['set_pulses'],set_count,'cell/set_count')
            if guarded:
                for name,value in dict(reset_decisions=reset_decisions,set_decisions=set_decisions,
                    reset_skipped_current=reset_skips,set_skipped_current=set_skips).items():compare(cell[name],value,'cell/'+name)
            check(reset_count<=271 and set_count<=271 and len(local)<=7,'cell/bounds')
            compare(cell['crossed_targets'],crossed,'cell/crossedtargets')
            compare(cell['unreached_targets'],[q for q in TARGETS if q not in crossed],'cell/unreachedtargets')
            check(cell['tested_window_upper_reached']==(.05 in crossed),'cell/upper_reached')
            compare(cell['terminal_checkpoint_id'],terminal['checkpoint_id'],'cell/terminal_id')
            compare(cell['terminal_g'],terminal['conductance_siemens'],'cell/terminal_G');compare(cell['terminal_q'],terminal['q'],'cell/terminal_q')
            compare(cell['checkpoints'],[{k:v for k,v in s.items() if k not in ('reads','mvm')} for s in local],'cell/checkpoints')
            check(cell['status']=='discovery_completed','cell/completed_not_qualified')
        for key,cell,saved in zip(CELLS,cells,summary['final_array']):
            rows=[take('final_array_read',key,repeat=i,mode=7,vol=0) for i in range(3)]
            observed=stats([r['conductance_siemens'] for r in rows]);previous=cell['terminal_g']['mean']
            compare(saved,dict(wl=key[0],bl=key[1],conductance_siemens=observed,terminal_mean_siemens=previous,
                fractional_difference=(observed['mean']-previous)/previous,initial_mean_siemens=initial[key]['mean'],
                fractional_difference_from_initial=(observed['mean']-initial[key]['mean'])/initial[key]['mean'],
                interpretation='Observed later difference; not proof of write-disturb cause or retained MVM transfer'),'final/record')
        compare(roles,expected_roles,'dedup_roles')
        check(cursor==len(records) and cp_cursor==len(cps),'schedule/consumed_all')
        if guarded:check(decision_cursor==len(guard.decisions),'schedule/consumed_all_decisions')
    else:
        check(bool(summary.get('error') or summary.get('close_error')),'partial/error_reason')
        check(not summary['programming_repeatability_qualified'],'partial/no_qualification')
        initial_rows=[r for r in records if r['phase']=='initial_array_read']
        expected_initial=[(wl,bl,i) for wl,bl in CELLS for i in range(3)]
        check([(r['wl'],r['bl'],r['repeat']) for r in initial_rows]==expected_initial[:len(initial_rows)],'partial/initial_prefix')
        if any(r['phase']!='initial_array_read' for r in records):check(len(initial_rows)==768,'partial/initial_complete_before_other_io')
        phases_allowed={'initial_array_read','checkpoint_read','checkpoint_mvm','reset','set','final_array_read'}
        check(set(phases)<=phases_allowed,'partial/phases')
        initial_success=[r for r in initial_rows if r['status']=='ok']
        for index,saved in enumerate(summary['initial_array']):
            group=initial_success[index*3:index*3+3]
            check(len(group)==3,'partial/initial_saved_complete')
            if len(group)==3:compare(saved['conductance_siemens'],stats([r['conductance_siemens'] for r in group]),'partial/initial_stats')
        by_cp={cid:[] for cid in range(1,len(cps)+1)}
        for row in records:
            if 'checkpoint_id' in row:
                check(row['checkpoint_id'] in by_cp,'partial/checkpoint_id')
                by_cp.setdefault(row['checkpoint_id'],[]).append(row)
        for state in cps:
            cid=state['checkpoint_id'];group=by_cp[cid]
            expected=[('checkpoint_read',i,None) for i in range(5)]+[('checkpoint_mvm',i,p) for i in range(5) for p in range(6)]
            check([(r['phase'],r['repeat'],r.get('position')) for r in group]==expected[:len(group)],'partial/checkpoint_prefix')
            check(all((r['wl'],r['bl'])==(state['wl'],state['bl']) for r in group),'partial/checkpoint_address')
            reads=[r for r in group if r['phase']=='checkpoint_read' and r['status']=='ok']
            mvms=[r for r in group if r['phase']=='checkpoint_mvm' and r['status']=='ok']
            compare(state['reads'],[{k:r[k] for k in ('task_id','raw_adc','q','conductance_siemens')} for r in reads],'partial/checkpoint_reads')
            compare(state['mvm'],[{k:r[k] for k in ('task_id','raw_adc','output_voltage_v','repeat','position','nominal_v')} for r in mvms],'partial/checkpoint_mvm')
            for r in group:
                if r['phase']=='checkpoint_read':check(r['mode']==7 and r['vol']==0,'partial/checkpoint_read_command')
                else:
                    level=LEVELS[r['position']]
                    compare(r['vol'],[level if i==state['bl'] else 0 for i in range(32)],'partial/basis')
            if len(reads)==5:
                for field in ('q','conductance_siemens'):compare(state[field],stats([r[field] for r in reads]),'partial/checkpoint_stats/'+field)
                g=mean(r['conductance_siemens'] for r in reads)
                compare(state['prediction'],dict(current_a=.1*g,output_v=.1*g*b['sum_gain'],uncalibrated=True),'partial/model')
                if mvms:check(.1*g<=.005 and .1*g*b['sum_gain']<=4,'partial/model_guard_before_mvm')
            if state['status']=='completed':check(len(reads)==5 and len(mvms)==30,'partial/completed_checkpoint_count')
            else:check(bool(state.get('error')),'partial/failed_checkpoint_reason')
        for cell in cells:
            key=(cell['wl'],cell['bl'])
            for operation,sign in (('reset',0),('set',1)):
                rows=[r for r in records if r['phase']==operation and (r['wl'],r['bl'])==key]
                check(len(rows)==cell[operation+'_pulses']<=271,'partial/pulse_ledger')
                steps=rows
                if guarded:
                    steps=[r for r in guard.decisions if r['phase']==operation and (r['wl'],r['bl'])==key]
                    check(len(steps)==cell[operation+'_decisions']<=271,'partial/decision_ledger')
                    check(sum(r['skipped'] for r in steps)==cell[operation+'_skipped_current'],'partial/skip_ledger')
                    check(len(rows)+cell[operation+'_skipped_current']==len(steps),'partial/decision_identity')
                    check([r['step'] for r in rows]==[d['step'] for d in steps if not d['skipped']],'partial/actual_steps')
                check([r['step'] for r in steps]==list(range(len(steps))),'partial/ramp_steps')
                for row in rows:check(row['mode']==7 and row['sign']==sign and row['vol']==round(.3+.01*row['step'],2),'partial/ramp_command')
            if cell['set_pulses']:check(cell.get('reset_confirmed') is True,'partial/set_requires_reset_confirmation')
            states=[s for s in cps if (s['wl'],s['bl'])==key]
            check(len(states)<=7,'partial/checkpoint_budget')
            compare(cell['checkpoints'],[{k:v for k,v in st.items() if k not in ('reads','mvm')} for st in states],'partial/cell_checkpoint_index')
            crossed=[q for st in states for q in st['crossed_targets']]
            check(crossed==sorted(set(crossed)) and all(q in TARGETS for q in crossed),'partial/crossing_dedup')
            compare(cell['crossed_targets'],crossed,'partial/crossings_saved')
        final_rows=[r for r in records if r['phase']=='final_array_read']
        check([(r['wl'],r['bl'],r['repeat']) for r in final_rows]==expected_initial[:len(final_rows)],'partial/final_prefix')
        if final_rows:check(len(cells)==256 and all(c['status']=='discovery_completed' for c in cells),'partial/final_after_discovery')
        for index,saved in enumerate(summary['final_array']):
            group=final_rows[index*3:index*3+3]
            check(len(group)==3 and all(r['status']=='ok' for r in group),'partial/final_saved_complete')
            if len(group)==3 and all(r['status']=='ok' for r in group):compare(saved['conductance_siemens'],stats([r['conductance_siemens'] for r in group]),'partial/final_stats')
    outcomes=dict(reset_reached=sum(c['reset_reached'] for c in cells),reset_confirmed=sum(c.get('reset_confirmed',False) for c in cells),
        set_attempted_cells=sum(c['set_pulses']>0 for c in cells),upper_q_reached=sum(c.get('tested_window_upper_reached',False) for c in cells),
        target_reached_counts={str(q):sum(q in c['crossed_targets'] for c in cells) for q in TARGETS},
        set_skipped_reasons=dict(Counter(c['set_skipped_reason'] for c in cells if 'set_skipped_reason' in c)),
        checkpoint_counts=dict(Counter(len(c['checkpoints']) for c in cells)))
    report=dict(audit_status='failed' if errors else 'passed',checks=count,errors=errors,max_numeric_difference=maxdiff,
        session_id=manifest['session_id'],source=source,acquisition_status=summary['status'],coverage='complete' if completed else 'partial',full_replay_performed=completed,auditor_sha256=sha(Path(__file__)),raw_sha256=sha(run/'raw.jsonl'),
        manifest_sha256=sha(run/'manifest.json'),recorded_invocation=manifest['invocation'],source_blob_commit=source_commit,
        local_source_differences=local_differences,private_local_unavailable=private_local_unavailable,
        verified_source_count=len(verified),provenance_note='Recorded invocation is preserved; source blob comparison is separate from the clean/dirty state at launch.',
        impact_pairs=len(records),programming_pulse_attempts=pulses,mode_counts=dict(modes),phase_counts=dict(phases),
        protocol_id=protocol,programming_guard_verified=guarded,
        programming_decisions=len(guard.decisions) if guarded else pulses,programming_skipped_current=guard.skipped,
        checkpoints=len(cps),checkpoint_reuses=len(roles),scientific_outcomes=outcomes,
        limitations=['Source is '+source+'; completed acquisition is not stable programmable-window qualification.',
            'One-cycle tested q window capped at .05 is not a stable or maximal physical programming range.',
            '125 simulator model steps do not establish the physical native pulse duration.',
            'Checkpoint G precedes MVM; per-state post-MVM G stability is unmeasured.',
            'Earlier observed checkpoints need not be current terminal states or coexist on the array.',
            'Partial runs receive structural/prefix/ledger/conversion checks, not complete protocol replay.'])
    out.mkdir(parents=True,exist_ok=True)
    (out/'independent_calibration_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    text=['# Независимый аудит discovery','',f"Сессия `{manifest['session_id']}`, источник **{source}**, сбор **{summary['status']}**, охват **{report['coverage']}**. Целостность журнала: **{report['audit_status']}**; {count} проверок, {len(errors)} расхождений.",'',
        f"Проверено {len(records)} пар request/result и {pulses} nonzero mode7. Checkpoint {len(cps)}. Полный независимый replay алгоритма: {completed}. Для partial проверены сохранившиеся пары, преобразования, префиксы начального/конечного чтения, checkpoint, пределы и pulse ledger; полного охвата не заявляем.",'',
        f"Протокол `{protocol}`. GUI write guard проверен: {guarded}. Решений записи: {report['programming_decisions']}; пропусков по модельному току: {guard.skipped}. Пропуск не является аппаратной командой и не создаёт ADC/новое измеренное состояние. Для v1 защиту v2 ретроспективно не предполагаем.",'',
        f"RESET crossing: {outcomes['reset_reached']}; RESET confirmed: {outcomes['reset_confirmed']}; SET attempted cells: {outcomes['set_attempted_cells']}; q=.05 crossing: {outcomes['upper_q_reached']}.",'',
        f"Записанный commit `{manifest['invocation']['git']['commit']}`, dirty_tracked={manifest['invocation']['git']['dirty_tracked']}. Source blobs сверены с `{source_commit}`. Неимпортируемых root/config файлов с отличающимися текущими байтами: {len(local_differences)}; они проверены по сохранённому Git blob и перечислены в JSON. Все локальные vendor Python sources проверены до импорта. Приватные драйверы сверены по fixed PIN и локальным байтам при наличии; недоступных локально: {len(private_local_unavailable)}. Приватные файлы не проверяются через Git и не публикуются.",'',
        'Один цикл не устанавливает стабильный диапазон, повторяемость, retention, endurance или MVM нескольких фильтров. .05 — граница исследуемого окна. Hardware duration неизвестна;125 model steps применяются лишь к simulator. Исторический checkpoint не обязательно соответствует текущему состоянию массива.', '',
        'Аудитор не создаёт Connector и не вызывает аппаратуру; обычный импорт пакета используется только для штатных численных converters.', '',
        '```sh',f"python -B audit_calibration_generic.py --repo . --session {args.session} --output-dir {args.output_dir}-recheck --source-commit {source_commit}",'```']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('audit_status','checks','errors','impact_pairs','programming_pulse_attempts','checkpoints','scientific_outcomes','max_numeric_difference')},ensure_ascii=False))
    return 1 if errors else 0


if __name__=='__main__':raise SystemExit(main())
