"""Independent offline JIT-C9 raw replay. No collector/device imports."""
import argparse
from configparser import ConfigParser
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics as st
import subprocess
import tempfile

HELPER_SHA='e7367990b6fc2a00f65e3bb0c4e4e5bf097b9f6b21a00700355edb84e0cbc70f'
if hashlib.sha256(Path(__file__).with_name('audit_cifar_feedback_diagnostic.py').read_bytes()).hexdigest()!=HELPER_SHA:
    raise RuntimeError('Independent raw helper changed')
from audit_cifar_feedback_diagnostic import (CompactChecks,Replay,sha,local,load,raw_lines,
    reset_replay,pair_replay,replay_block,fidelity,ledger,BOARD,PRIVATE,audit_git_status,DIAGNOSTIC_PINS,PINS)
from audit_cifar_feedback import read_raw as native_raw,counts_between,triple
from audit_cifar_memristor import output_voltage

PROTOCOL='cifar-C9-jit-programming-v1'
ROUTE='results/cifar_feedback_remaining_reserves/58ca618b-c37f-454e-be25-db21d1d25f60/tiles/f1_p0_reserve_C/tile.json'
ROUTE_SHA='7b262ed74cfebaa8a95622a01f7bf20165b9443cd9cbea84785355f71aeb2b20'
# Filled only after independent review of the final acquisition/config bytes.
FINAL_PINS={
    'cifar_jit_programming.py':'cd1abd674b032516a78f5f3f9fdce1397375ca766b586770fa926d8509099480',
    'configs/cifar_jit_programming.json':'150c5939b3b9f38224159a16982825646705fb4ef5a1fc20c4f18f3b77eef9a5',
    'experiment.py':'ddf3d1642dfbba35ecdebdde8d3539840cc126f4a41af17502c4e345f897ff83',
}


def require(value,message):
    if not value:raise ValueError(message)


def private_requirements(source):
    # check_sources(simulator) deliberately omits the two hardware-only drivers.
    return PRIVATE if source=='hardware' else {k:v for k,v in PRIVATE.items() if '/MemriCORE/' not in k}


def verify_protocol_copy(plan,git_blob,ck):
    ck.equal(hashlib.sha256(git_blob).hexdigest(),FINAL_PINS['configs/cifar_jit_programming.json'],'protocol/frozen_git_bytes')
    ck.require(plan==json.loads(git_blob),'protocol/exact_JSON_copy')


def validate_plan(plan,route,ck):
    ck.equal(plan,dict(protocol_id=PROTOCOL,cycles=2,pair_order=list(range(9)),sentinel_cell=[3,8],
        retention_delay_s=1.,cutoff_utc='2026-10-16T18:50:00+00:00',maximum_impacts=29934,
        maximum_programming_pulses=3942,maximum_programming_decisions=3942,maximum_pair_decisions=18,
        maximum_impacts_per_decision=1663,reset_signal_limit_adc=2,cifar_acquisition=False,matrix_fit=False,
        target_gain_v_per_weight={'hardware':1.,'simulator':.03},route_source=ROUTE,route_source_sha256=ROUTE_SHA,
        reset_ramp_v=[round(.3+.05*i,3) for i in range(55)],set_ramp_v=[round(.3+.025*i,3) for i in range(109)]),
        'plan/frozen_numeric_contract')
    ck.equal([s['tap_index'] for s in route],list(range(9)),'route/exact_taps')
    ck.require(len({tuple(s[k]) for s in route for k in ('positive_cell','negative_cell')})==18,'route/18_unique_cells')
    ck.require(all(not s['padding'] for s in route),'route/no_padding')


def read_stream(directory,source,sid,ck):
    """Check chunk commitments, then native conversions/guards and auxiliary order."""
    waits=[];sequence=0;pending=False
    with tempfile.TemporaryDirectory(prefix='jit-independent-') as tmp:
        path=Path(tmp)/'native.jsonl'
        with path.open('w',encoding='utf-8') as stream:
            for line in raw_lines(directory/'raw.jsonl'):
                row=json.loads(line);kind=row['event']
                if kind=='retention_wait':
                    require(not pending,'Retention inserted inside request/result pair')
                    ck.equal(row,dict(session_id=sid,requested_delay_s=1.,delay_met=True),'retention/event')
                    ck.require(row['elapsed_monotonic_s']>=1.,'retention/minimum_delay')
                    ck.require(row['finished_monotonic_s']-row['started_monotonic_s']>=row['elapsed_monotonic_s']-1e-9,'retention/clock_consistency')
                    waits.append(dict(after_task_id=sequence,row=row));continue
                require(kind in ('request','result','skipped'),'Unsupported interruption/event: '+kind)
                if kind=='request':require(not pending,'Nested request');pending=True
                elif kind=='result':require(pending,'Orphan result');pending=False;sequence=row['task_id']
                elif kind=='skipped':require(not pending,'Skip inside result pair')
                stream.write(line if line.endswith('\n') else line+'\n')
        require(not pending,'Unpaired terminal request')
        events,counters=native_raw(path,source,ck,sid)
    # This auditor accepts fully captured normal diagnostics, including quiet failures.
    # Any global current stop/interruption remains failed closed, never complete PASS.
    ck.equal(counters['programming_skipped_current'],0,'complete/no_current_stop')
    return events,counters,waits


def snapshot(replay,slot,phase,meta):
    guards=[replay.reads(phase+'_guard',slot[side],1,guard_side=side,**meta)[0]['task_id']
            for side in ('positive_cell','negative_cell')]
    samples=[]
    for repeat in range(5):
        values,rows=replay_block(replay,[slot],[slot],phase,dict(**meta,bracket_repeat=repeat))
        samples.append(dict(row_sums_v=values,signed_voltage_v=values[0]-values[1],measured_rows=rows))
    means=[st.mean(s['row_sums_v'][i] for s in samples) for i in range(2)]
    sds=[st.stdev(s['row_sums_v'][i] for s in samples) for i in range(2)]
    limit=2*output_voltage(1)
    return dict(**meta,phase=phase,tap=slot['tap_index'],positive_cell=slot['positive_cell'],negative_cell=slot['negative_cell'],
        guard_task_ids=guards,samples=samples,row_mean_v=means,row_sd_v=sds,
        signed_mean_v=st.mean(s['signed_voltage_v'] for s in samples),signed_sd_v=st.stdev(s['signed_voltage_v'] for s in samples),
        quiet_limit_v=limit,both_legs_quiet=all(abs(m)<=limit and s<=limit for m,s in zip(means,sds)))


def sentinel(replay,phase,meta):
    cell=[3,8];guard=replay.reads(phase+'_guard',cell,1,**meta)[0]
    return dict(**meta,phase=phase,cell=cell,guard_task_id=guard['task_id'],measurement=triple(replay,cell,phase,**meta))


def replay_session(directory,source,sid,plan,route,ck):
    validate_plan(plan,route,ck)
    events,counts,waits=read_stream(directory,source,sid,ck);r=Replay(events,ck)
    decisions=[];reset_rows=[];reset_records=[];set_rows=[];pair_records=[];sentinels=[];snapshots=[];maps=[]
    gain=1. if source=='hardware' else .03
    for cycle in range(2):
        for index,slot in enumerate(route):
            meta=dict(cycle=cycle,pair_index=index,pair_tap=slot['tap_index'],filter_id=1)
            start=r.position;d=dict(**meta,status='completed')
            for leg,side in (('positive','positive_cell'),('negative','negative_cell')):
                md=dict(**meta,reset_leg=leg);reset=reset_replay(r,slot[side],md,reset_rows)
                reset_records.append(dict(**md,**reset));d[leg+'_reset']=reset
                phase='sentinel_after_'+leg+'_reset';s=sentinel(r,phase,meta);sentinels.append(s);d[phase]=s
            gate=snapshot(r,slot,'fresh_pair_gate',meta);snapshots.append(gate);d['fresh_pair_gate']=gate
            if gate['both_legs_quiet']:
                pair=pair_replay(r,slot,meta,gain,set_rows);pair_records.append(dict(**meta,**pair));d['program']=pair
                s=sentinel(r,'sentinel_after_program',meta);sentinels.append(s);d['sentinel_after_program']=s
                d['outcome']='programmed_diagnostic'
            else:
                d['outcome']='fresh_quiet_gate_failed_no_SET'
                s=snapshot(r,slot,'no_SET_confirmation',meta);snapshots.append(s);d['no_SET_confirmation']=s
            require(len(waits)>len(decisions),'Missing retention event')
            wait=waits[len(decisions)];ck.equal(wait['after_task_id'],events[r.position-1]['task_id'],'retention/exact_order')
            ck.equal(wait['row'],meta,'retention/pair_identity')
            md=dict(**meta,requested_delay_s=1.,elapsed_monotonic_s=wait['row']['elapsed_monotonic_s'])
            s=snapshot(r,slot,'retention',md);snapshots.append(s);d['retention']=s
            observed=[]
            for mi,ms in enumerate(route):
                s=snapshot(r,ms,'after_decision_map',dict(**meta,map_pair_index=mi));snapshots.append(s);observed.append(s)
            observations=[[observed[i]['samples'][k]['signed_voltage_v'] for i in range(9)] for k in range(5)]
            quality=fidelity([s['trained_weight'] for s in route],observations,gain)
            maps.append(dict(**meta,observed=observed,physical_fidelity=quality,
                scope='all current C9 states after this decision; later or gate-failed targets may remain unprogrammed'))
            d['disturbance_map']=dict(phase='after_decision_map',cycle=cycle,pair_index=index,physical_fidelity=quality)
            counted=counts_between(events,start,r.position)
            d.update(impact_attempts=counted['impact_attempts'],programming_pulse_attempts=counted['programming_pulse_attempts'])
            ck.require(d['impact_attempts']<=1663 and d['programming_pulse_attempts']<=219,'budget/per_decision')
            decisions.append(d)
    ck.equal(r.position,len(events),'raw/exact_consumption');ck.equal(len(waits),18,'retention/exact18')
    for name,expected in [('reset_candidates',reset_records),('reset_feedback',reset_rows),('pair_programming',pair_records),
                          ('set_feedback',set_rows),('sentinel',sentinels),('paired_snapshots',snapshots),
                          ('disturbance_maps',maps),('decisions',decisions)]:
        ck.equal(ledger(directory/(name+'.jsonl')),expected,'ledger/'+name)
    final=[d['disturbance_map']['physical_fidelity'] for d in decisions if d['pair_index']==8]
    expected=dict(status='completed',technical_completed=True,cifar_acquisition=False,decisions=decisions,route=route,
        final_state_fidelity_pass=all(q['physical_fidelity_pass'] for q in final),
        programming_fidelity_pass=all(q['physical_fidelity_pass'] for q in final) and len(pair_records)==18,
        cycle_final_physical_fidelity=final,
        all_fresh_gates_passed=all(d['fresh_pair_gate']['both_legs_quiet'] for d in decisions),
        programming_repeatability_qualified=False,quiet_gate_failures=sum(d['outcome']=='fresh_quiet_gate_failed_no_SET' for d in decisions),**counts)
    ck.equal(load(directory/'summary.json'),expected,'summary/independent_replay')
    for key,maximum in [('impact_attempts',29934),('programming_pulse_attempts',3942),('programming_decisions',3942)]:
        ck.require(counts[key]<=maximum,'budget/'+key)
    return dict(technical_completed=True,cifar_acquisition=False,unique_trained_taps=9,physical_cells=18,cycles=2,
        counters=counts,quiet_gate_failures=expected['quiet_gate_failures'],programmed_pair_decisions=len(pair_records),
        cycle_final_physical_fidelity=final,final_state_fidelity_pass=expected['final_state_fidelity_pass'],
        programming_fidelity_pass=expected['programming_fidelity_pass'],
        programming_repeatability_qualified=False,decisions=[{k:v for k,v in d.items() if k in ('cycle','pair_index','pair_tap','outcome','impact_attempts','programming_pulse_attempts')} for d in decisions])


def verify_session(repo,directory,ck):
    require(bool(FINAL_PINS),'Auditor source review is not frozen')
    m=load(directory/'manifest.json');plan=load(directory/'protocol.json');route=load(directory/'route.json')
    require(m['source'] in ('hardware','simulator'),'Unknown backend')
    require(m['status']=='completed' and m['technical_completed'] is True,'Incomplete acquisition; partial replay not supported')
    provenance=audit_git_status(m['invocation']['git'],m['source_sha256'],ck)
    for name,digest in m['result_sha256'].items():ck.equal(sha(local(directory,name)),digest,'result/hash/'+name)
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name!='manifest.json'}
    ck.equal(sorted(actual),sorted(m['result_sha256']),'result/exact_files')
    for name,digest in m['source_sha256'].items():
        if name in PRIVATE:ck.equal(digest,PRIVATE[name],'source/private/'+name)
        else:
            proc=subprocess.run(['git','-C',str(repo),'show',m['invocation']['git']['commit']+':'+name],capture_output=True)
            ck.equal(proc.returncode,0,'source/git_exists/'+name);ck.equal(hashlib.sha256(proc.stdout).hexdigest(),digest,'source/blob/'+name)
    reviewed={**PINS,**DIAGNOSTIC_PINS,**private_requirements(m['source']),
        'experiments/unn_rev3_20261014/paths.py':'2c841b4a446a767595c68aad243abd966cee94ffd91f74c3c2823998fdc9ec1c',
        ROUTE:ROUTE_SHA,**FINAL_PINS}
    for name,digest in reviewed.items():ck.equal(m['source_sha256'].get(name),digest,'source/frozen/'+name)
    ck.equal(sha(local(repo,ROUTE)),ROUTE_SHA,'route/pinned_origin')
    ck.equal(route,load(local(repo,ROUTE))['effective_route'],'route/exact_original')
    config_blob=subprocess.run(['git','-C',str(repo),'show',m['invocation']['git']['commit']+':configs/cifar_jit_programming.json'],capture_output=True,check=True).stdout
    verify_protocol_copy(plan,config_blob,ck)
    board=ConfigParser();board.read(directory/'board.ini',encoding='utf-8')
    for k,v in BOARD.items():ck.equal(board.getfloat('board',k),v,'INI/'+k)
    start=load(directory/'start_manifest.json')
    for k in ('session_id','source','source_sha256','protocol_id','invocation','cutoff_utc'):ck.equal(m[k],start[k],'start/'+k)
    ck.equal(m,dict(protocol_id=PROTOCOL,hardware_access=m['source']=='hardware',physical_pulse_width_s=None,
        maximum_impacts=29934,maximum_programming_pulses=3942,maximum_programming_decisions=3942,pair_decisions=18,
        cutoff_utc='2026-10-16T18:50:00+00:00',cifar_acquisition=False),'manifest/contract')
    ck.require(datetime.fromisoformat(m['started_utc'])<datetime.fromisoformat(m['cutoff_utc']),'deadline/start_before_cutoff')
    ck.require(datetime.fromisoformat(m['finished_utc'])<datetime.fromisoformat(m['cutoff_utc']),'deadline/closed_before_cutoff')
    return m,plan,route,provenance


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.')
    p.add_argument('--session',required=True);p.add_argument('--output',required=True);args=p.parse_args(argv)
    repo=Path(args.repo).resolve();directory=local(repo,args.session);output=local(repo,args.output)
    require(not output.is_relative_to(directory),'Output would modify source');ck=CompactChecks();report={}
    try:
        m,plan,route,provenance=verify_session(repo,directory,ck)
        report=replay_session(directory,m['source'],m['session_id'],plan,route,ck)
        ck.equal(m,{k:report['counters'][k] for k in ('impact_attempts','programming_pulse_attempts','programming_decisions')},'manifest/replayed_counts')
        for key in ('final_state_fidelity_pass','programming_fidelity_pass'):ck.equal(m[key],report[key],'manifest/'+key)
        report.update(session_id=m['session_id'],source=m['source'],manifest_sha256=sha(directory/'manifest.json'),provenance=provenance)
    except Exception as exc:ck.require(False,type(exc).__name__+': '+str(exc))
    report.update(audit_status='passed' if not ck.errors else 'failed',checks=ck.count,errors=ck.errors,
        auditor_sha256=sha(Path(__file__)),helper_sha256=HELPER_SHA,hardware_io=False,
        limitations=['PASS validates capture/protocol, not physical programming accuracy or all114 cells.',
                    'Interrupted/current-stop/transport sessions fail closed; this auditor does not certify partial acquisition.',
                    'Two fixed-order cycles are not randomized causal controls.'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['audit_status'],checks=ck.count,errors=ck.errors[:15])))
    return int(bool(ck.errors))


if __name__=='__main__':raise SystemExit(main())
