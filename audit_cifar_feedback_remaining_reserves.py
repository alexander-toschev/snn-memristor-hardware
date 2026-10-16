"""Independent B/C raw auditor draft. No collectors, drivers or backends imported.
Synthetic component replay does not establish native acquisition/parent provenance.
"""
import argparse
import copy
import hashlib
import json
import sys
import subprocess
from pathlib import Path
from collections import Counter
from configparser import ConfigParser

A_SHA='372f48a05abc652cda0f2c9d5891660a9c23429f53d18ea84749c587f5e10459'
DRAFT_RUNTIME_SHA='89764591d65e3ffb082fe7094f353cd2b2ccd5fff362022d78fc31e5e1f95401'
DRAFT_PLAN_SHA='87d2656587a5f1484243019d82434b8b89e25b81ebfd92818883a0188d59461b'
# Filled only after final runtime/config/parent review. Actual-session audit fails closed meanwhile.
FINAL_PINS={}
FINAL_PROTOCOL=None
COUNTERS=('impact_attempts','programming_pulse_attempts','programming_decisions','programming_skipped_current')
SPECS={
 'B':dict(fid=2,n=11,extra=[[5,15]],positive={0:[4,7]},negative={0:[5,6],1:[5,15]},calls=21421,writes=2794),
 'C':dict(fid=1,n=9,extra=[[3,12],[3,13],[3,31]],positive={0:[2,3]},negative={0:[3,12],1:[3,13],2:[3,31]},calls=18957,writes=2466)}


def initialize(repo):
    global A
    if hashlib.sha256((repo/'audit_cifar_feedback_reserve.py').read_bytes()).hexdigest()!=A_SHA:
        raise ValueError('Independent A helper changed')
    sys.path.insert(0,str(repo))
    import audit_cifar_feedback_reserve as helper
    A=helper
    for name in ('CompactChecks','require','local','load','sha','verify_inputs','pools_for','read_raw','Replay',
                 'reset_replay','pair_replay','ledger','multiply','fixed_calibration','fixed_effective_weights',
                 'replay_block','fixed_decode','fidelity','audit_adaptive_matrix','measurement_tail',
                 'aggregate_microtiles','coverage','normalized_result','ordered','verify_wrapper_coverage',
                 'PINS','DIAGNOSTIC_PINS','ADAPTIVE_PINS','PRIVATE','BOARD','audit_git_status'):
        globals()[name]=getattr(helper,name)


def block_pools(parent,stage):
    s=SPECS[stage];active=[x for x in parent['slots'] if not x['padding']]
    require(parent['tile_id']==0 and [x['tap_index'] for x in active]==list(range(s['n'])),'Unexpected block taps')
    result=pools_for(parent)
    require(all(c not in result['negative_cell'] for c in s['extra']),'Duplicate extra candidate')
    result['negative_cell']+=copy.deepcopy(s['extra'])
    require(len(result['positive_cell'])==s['n']+3 and len(result['negative_cell'])==s['n']+3+len(s['extra']),'Pool bound')
    return result


def allocation(parent,stage,eligible):
    spec=SPECS[stage];pools=block_pools(parent,stage);n=spec['n']
    for side,pool in pools.items():
        require(len(set(map(tuple,eligible[side])))==len(eligible[side]),'Duplicate eligible')
        require(eligible[side]==[c for c in pool if c in eligible[side]],'Unknown/reordered eligible')
    capacity=all(len(x)>=n for x in eligible.values());reservation={};route={};dispositions=[]
    for short in ('positive','negative'):
        side=short+'_cell';reserved=[None]*n
        for tap,cell in spec[short].items():
            quiet=cell in eligible[side]
            if quiet:reserved[tap]=cell
            dispositions.append(dict(cell=cell,tap=tap,quiet=quiet,status=(
                'screened_quiet_selected_for_route' if capacity else 'quiet_but_route_capacity_abort') if quiet else 'screened_not_quiet',actual_cifar_assigned=False))
        reservation[side]=copy.deepcopy(reserved)
        remaining=iter(c for c in eligible[side] if c not in reserved)
        route[side]=[c if c is not None else next(remaining) for c in reserved] if capacity else []
    return dict(capacity_pass=capacity,route=route,preferred_disposition=dispositions,
        allocation_before_fill=reservation,quiet_counts={k:len(v) for k,v in eligible.items()},may_issue_SET=capacity)


def check_dispositions(saved,stage,candidates,active,pairs,complete,capacity_failed,ck):
    expected=[];bycell={tuple(c['cell']):c for c in candidates};assigned={tuple(s[k]) for s in active for k in ('positive_cell','negative_cell')}
    for side in ('positive','negative'):
        for tap,cell in SPECS[stage][side].items():
            candidate=bycell.get(tuple(cell));chosen=tuple(cell) in assigned
            if candidate is None:status='technical_unassessed'
            elif not candidate['eligible']:status='screened_not_quiet'
            elif capacity_failed:status='quiet_but_route_capacity_abort'
            elif chosen:status='quiet_actual_CIFAR_assigned' if complete else 'quiet_selected_acquisition_incomplete'
            elif active:status='quiet_not_selected'
            else:status='quiet_route_unassessed'
            wrote=any(p['programming_pulse_attempts']>0 and p[p['active_set_side']]==cell for p in pairs)
            expected.append(dict(cell=cell,tap=tap,status=status,quiet=candidate['eligible'] if candidate else None,
                screening_complete=candidate is not None,route_assigned=chosen,cifar_complete=complete,
                technical_stop=saved.get('error'),actual_cifar_assigned=complete and chosen,
                SET_attempted=wrote if complete else False if capacity_failed else None))
    ck.equal(saved['preferred_disposition'],expected,'preferred/raw_derived_disposition')
    return expected


def audit_block(directory,filt,parent,fixture,plan,source,ck,session_id,stage):
    pending=[s for s in parent['slots'] if not s['padding']];micro=0
    saved=load(directory/'tile.json');fid=filt['filter_id'];tid=2*parent['tile_id']+micro;meta=dict(filter_id=fid,tile_id=tid,parent_tile_id=parent['tile_id'],microtile_id=micro,reserve_block=stage)
    active=copy.deepcopy(pending);pools=block_pools(parent,stage)
    ck.equal(saved,meta,'microtile/saved_identity')
    ck.equal(saved['pending_taps_before'],[s['tap_index'] for s in pending],'microtile/pending_prefix')
    ck.equal(saved['original_chunk_taps'],[s['tap_index'] for s in parent['slots'] if not s['padding']],'microtile/original_chunk')
    events,counters=read_raw(directory/'raw.jsonl',source,ck,session_id);replay=Replay(events,ck)
    times=[e['monotonic_time_s'] for e in events if 'monotonic_time_s' in e]
    time_bounds=[times[0],times[-1]]
    candidates=[];reset_rows=[];candidate_ledger=[];chosen={}
    for side,cells in pools.items():
        qualifying=[]
        for index,cell in enumerate(cells):
            md=dict(**meta,side=side,candidate_index=index);candidate=reset_replay(replay,cell,md,reset_rows)
            candidates.append(dict(side=side,candidate_index=index,**candidate));candidate_ledger.append(dict(**md,**candidate))
            if candidate['eligible']:qualifying.append(cell)
        chosen[side]=qualifying
    ck.equal(saved['reset_candidates'],candidates,'reset/all_candidates')
    ck.equal(ledger(directory/'reset_candidates.jsonl'),candidate_ledger,'reset/candidate_ledger')
    ck.equal(ledger(directory/'reset_feedback.jsonl'),reset_rows,'reset/feedback_ledger')
    decision=allocation(parent,stage,chosen)
    ck.equal(saved['route_decision'],decision,'route/independent_decision')
    if not decision['capacity_pass']:
        ck.equal(saved['status'],'failed','reserve/gate_failure_status')
        ck.equal(saved['technical_completed'],False,'block/technical_incomplete')
        ck.equal(saved['stop_reason'],'whole_route_capacity_failed_before_SET','reserve/gate_failure_reason')
        ck.equal(replay.position,len(events),'reserve/no_SET_or_CIFAR_after_gate_failure')
        ck.equal(saved['pairs'],[],'reserve/no_pairs_on_gate_failure');ck.equal(saved['partials'],[],'reserve/no_partials_on_gate_failure')
        ck.equal(saved,counters,'reserve/failed_counters')
        ck.require(not (directory/'frozen_route.json').exists(),'reserve/no_frozen_route_on_failure')
        return dict(**meta,technical_completed=False,physical_weight_quality_pass=False,counters=counters,
                    reset_candidates=candidates,partials=[],pairs=[],acquisition_stop='whole_route_capacity_failed_before_SET',
                    preferred_disposition=check_dispositions(saved,stage,candidates,[],[],False,True,ck),gate=decision,raw_time_bounds=time_bounds)
    chosen=decision['route']
    ck.equal(saved['assigned_taps'],[s['tap_index'] for s in active],'reserve/exact13_taps')
    for i,slot in enumerate(active):
        for side in chosen:slot[side]=chosen[side][i]
    ck.equal(saved['effective_route'],active,'routing/deterministic_weight_independent')
    route=load(directory/'frozen_route.json');freeze=events[replay.position];replay.position+=1
    ck.equal(route,dict(candidate_pools=pools,slots=active,allocation_before_fill=decision['allocation_before_fill'],preferred_disposition=decision['preferred_disposition'],mapping_uses_trained_weight_values=False,
                        frozen_before_task_id=events[replay.position]['task_id']),'routing/frozen_values')
    ck.equal(freeze,dict(event='route_frozen',session_id=session_id,**meta,sha256=sha(directory/'frozen_route.json'),
                        before_task_id=route['frozen_before_task_id']),'routing/freeze_commitment')
    gain=1. if source=='hardware' else .03;pairs=[];set_rows=[];pair_ledger=[];programming=[]
    for slot in active:
        md=dict(**meta,pair_tap=slot['tap_index']);pair=pair_replay(replay,slot,md,gain,set_rows)
        pairs.append(pair);pair_ledger.append(dict(**md,**pair))
        for side in ('positive_cell','negative_cell'):
            programming.append(dict(cell=slot[side],status='completed',tap=slot['tap_index'],side=side,
                feedback_pair_stop=pair['stop'],selected_SET_side=side==pair['active_set_side'],participated_in_SET=side==pair['active_set_side'] and pair['programming_pulse_attempts']>0,reset_eligible=True))
    ck.equal(saved['pairs'],pairs,'pairs/results');ck.equal(saved['programming'],programming,'pairs/assignments')
    ck.equal(ledger(directory/'pair_programming.jsonl'),pair_ledger,'pairs/ledger')
    ck.equal(ledger(directory/'set_feedback.jsonl'),set_rows,'pairs/feedback_ledger')
    wmax=max(abs(s['trained_weight']) for t in filt['tiles'] for s in t['slots'] if not s['padding'])

    prereads=[]
    for slot in active:
        for side in ('positive_cell','negative_cell'):
            prereads.append(replay.reads('tile_pre_readback',slot[side],3,tap=slot['tap_index'],side=side,**meta))
    ck.equal(saved['pre_readback'],[[{k:v for k,v in r.items() if k not in ('event','latency_s')} for r in group] for group in prereads],'tile/pre_readback')
    def anchors(phase):
        records=[]
        for repeat in range(5):
            pair=[]
            for side in ('positive_cell','negative_cell'):
                z=replay.mvm(phase,active[0][side],False,repeat=repeat,side=side,role='zero',**meta)
                singles=[replay.mvm(phase,s[side],True,repeat=repeat,side=side,role='single',tap=s['tap_index'],**meta) for s in active]
                pair.append(dict(zero=z,singles=singles))
            records.append(pair)
        return records
    calanchors=anchors('calibration');ck.equal(load(directory/'calibration_anchors.json'),calanchors,'tile/anchors')
    correction=audit_adaptive_matrix(directory,replay,active,calanchors,meta,ck)
    ck.equal(saved['matrix'],correction,'tile/matrix',1e-9)
    differences=[]
    for plus,minus in calanchors:
        corrected=[multiply(correction['matrix'],[p['voltage_v']-plus['zero']['voltage_v'],m['voltage_v']-minus['zero']['voltage_v']])
                   for p,m in zip(plus['singles'],minus['singles'])]
        differences.append([pair[0]-pair[1] for pair in corrected])
    cal=fixed_calibration([s['trained_weight'] for s in active],differences,gain,plan['alpha_absolute_floor_adc'])
    ck.equal(cal['effective_weights'],fixed_effective_weights(correction['matrix'],calanchors,gain),'calibration/independent_fixed_units')
    freeze=events[replay.position];replay.position+=1
    ck.equal(freeze,dict(event='calibration_frozen',**meta,sha256=sha(directory/'frozen_calibration.json'),
                        matrix_sha256=sha(directory/'frozen_matrix.json'),
                        before_task_id=events[replay.position]['task_id']),'tile/freeze_commitment')
    cal.update(frozen_before_task_id=events[replay.position]['task_id'],trained_weights=[s['trained_weight'] for s in active],wmax_per_filter=wmax)
    ck.require(set(saved['calibration'])==set(cal),'tile/calibration_schema_supported')
    ck.equal(saved['calibration'],cal,'tile/calibration',1e-9);ck.equal(load(directory/'frozen_calibration.json'),cal,'tile/frozen_calibration',1e-9)
    partials=[]
    for ci,case in enumerate(fixture['cases']):
        for repeat in range(5):
            for t,spikes in enumerate(case['spikes_t12x75']):
                selected=[s for s in active if spikes[s['tap_index']]]
                event=dict(**meta,case=ci,original_test_index=case['original_test_index'],repeat=repeat,t=t)
                values,rows=replay_block(replay,active,selected,'cifar',event)
                corrected=multiply(correction['matrix'],values);difference=corrected[0]-corrected[1]
                partials.append(dict(**event,active_taps=[s['tap_index'] for s in selected],measured_rows=rows,
                                     raw_row_sums_v=values,corrected_row_sums_v=corrected,raw_signed_voltage_v=values[0]-values[1],
                                     signed_voltage_v=difference,trained_unit_contribution=fixed_decode(corrected,gain)))
    ck.equal(saved['partials'],partials,'tile/partials',1e-9)
    ck.equal([json.loads(line) for line in (directory/'partials.jsonl').read_text(encoding='utf-8').splitlines()],partials,'tile/partial_ledger',1e-9)
    finalanchors=anchors('final_anchors')
    ck.equal(saved['final_anchors'],finalanchors,'tile/final_anchors')
    postreads=[]
    for slot in active:
        for side in ('positive_cell','negative_cell'):
            postreads.append(replay.reads('tile_post_readback',slot[side],3,tap=slot['tap_index'],side=side,**meta))
    ck.equal(saved['post_readback'],[[{k:v for k,v in r.items() if k not in ('event','latency_s')} for r in group] for group in postreads],'tile/post_readback')
    ck.equal(replay.position,len(events),'tile/no_unconsumed_events')
    ck.equal(saved,counters,'tile/counters');ck.equal(saved['status'],'completed','tile/completed')
    ck.equal(saved['technical_completed'],True,'block/technical_complete')
    def raw_differences(samples):
        return [[(row[0]['singles'][i]['voltage_v']-row[0]['zero']['voltage_v'])-(row[1]['singles'][i]['voltage_v']-row[1]['zero']['voltage_v']) for i in range(len(active))] for row in samples]
    weights=[s['trained_weight'] for s in active]
    before=fidelity(weights,raw_differences(calanchors),gain);after=fidelity(weights,raw_differences(finalanchors),gain)
    ck.equal(saved['physical_fidelity'],before,'fidelity/fixed_gain_after_all_writes')
    ck.equal(saved['final_physical_fidelity'],after,'fidelity/post_inference')
    eligible=before['physical_fidelity_pass'] and after['physical_fidelity_pass']
    ck.equal(saved['physical_weight_quality_pass'],eligible,'fidelity/reported_quality_not_collection_gate')
    n=len(active);candidate_count=sum(len(cells) for cells in pools.values())
    ck.require(counters['programming_decisions']<=55*candidate_count+109*n,'feedback/decision_bound')
    ck.require(counters['programming_pulse_attempts']<=counters['programming_decisions'],'feedback/pulse_bound')
    return dict(**meta,technical_completed=True,physical_weight_quality_pass=eligible,calibration=cal,matrix=correction,
        reset_candidates=candidates,pairs=pairs,effective_route=active,physical_fidelity=before,final_physical_fidelity=after,partials=partials,counters=counters,
        candidate_eligible_counts={side:sum(c['eligible'] for c in candidates if c['side']==side) for side in pools},
        pair_stop_counts=dict(Counter(p['stop'] for p in pairs)),gate=decision,preferred_disposition=check_dispositions(saved,stage,candidates,active,pairs,True,False,ck),raw_time_bounds=time_bounds)


def block_budget(parent,stage,fixture):
    active=[s for s in parent['slots'] if not s['padding']]
    c=sum(map(len,block_pools(parent,stage).values()));n=len(active)
    return dict(maximum_impacts=244*c+811*n+measurement_tail(active,fixture),maximum_writes=55*c+109*n)


def replay_blocks(directory,fixture,routing,plan,source,session_id,ck):
    results=[];totals={k:0 for k in COUNTERS};capacity_failures=[]
    for stage in ('B','C'):
        spec=SPECS[stage];folder=directory/'tiles'/f"f{spec['fid']}_p0_reserve_{stage}"
        require(folder.is_dir(),'Missing planned block '+stage)
        parent=routing['filters'][spec['fid']]['tiles'][0]
        bound=block_budget(parent,stage,fixture)
        ck.equal(bound,dict(maximum_impacts=spec['calls'],maximum_writes=spec['writes']),'budget/'+stage)
        r=audit_block(folder,routing['filters'][spec['fid']],parent,fixture,plan,source,ck,session_id,stage)
        ck.require(r['counters']['impact_attempts']<=spec['calls'],'block/impact_bound')
        ck.require(r['counters']['programming_decisions']<=spec['writes'],'block/decision_bound')
        for k in COUNTERS:totals[k]+=r['counters'][k]
        if not r['technical_completed']:
            require(r['acquisition_stop']=='whole_route_capacity_failed_before_SET','Technical failure must stop all blocks')
            capacity_failures.append(dict(block=stage,reason='whole_route_capacity_failed_before_SET'))
        results.append(r)
    ck.require(results[0]['raw_time_bounds'][1]<=results[1]['raw_time_bounds'][0],'control/B_before_C_once')
    ck.require(totals['impact_attempts']<=40378 and totals['programming_decisions']<=5260,'global/budget')
    if capacity_failures:
        ck.equal(load(directory/'capacity_failures.json'),capacity_failures,'control/capacity_only_continuation')
        ck.require(not (directory/'summary.json').exists(),'control/no_full_summary_after_capacity_failure')
    else:ck.require(not (directory/'capacity_failures.json').exists(),'control/no_invented_failure')
    require({p.name for p in (directory/'tiles').iterdir() if p.is_dir()}=={'f2_p0_reserve_B','f1_p0_reserve_C'},'Unexpected additional block/retry')
    return results,totals,capacity_failures


def verify_files(directory,hashes,ck):
    for name,digest in hashes.items():ck.equal(sha(local(directory,name)),digest,'result_sha/'+name)
    for path in (directory/'tiles').rglob('*'):
        if path.is_file():require(path.relative_to(directory).as_posix() in hashes,'Unbound tile artifact')


def imported280(originals,parent_session,wrapper_sha,routing):
    verify_wrapper_coverage(originals,routing)
    out=[];removed=[]
    for index,w in enumerate(originals):
        if (w['filter_id'],w['parent_tile_id']) in ((2,0),(1,0)):
            removed.extend((w['filter_id'],s['tap_index']) for s in w['result']['effective_route']);continue
        origin=dict(session=parent_session,path='completed_microtiles.json',wrapper_index=index,sha256=wrapper_sha,
            reused=True,new_measurement=False,acquisition_tile_id=w['result']['tile_id'],ancestry_original_origin=w['origin'])
        out.append({**w,'origin':origin})
    require(len(removed)==20 and set(removed)=={(f,t) for f,n in ((2,11),(1,9)) for t in range(n)},'Exclusion must be exactly B11+C9')
    require(sum(len(w['result']['effective_route']) for w in out)==280,'Not exactly280 imported')
    return out


def bound_evidence(repo,binding,source,complete,ck):
    folder=local(repo,binding['session']);mp=folder/'manifest.json';ap=local(repo,binding['audit_path'])
    require(sha(mp)==binding['manifest_sha256'] and sha(ap)==binding['audit_sha256'],'Bound parent/audit SHA changed')
    m=load(mp);a=load(ap)
    require(a['audit_status']=='passed' and not a['errors'],'Parent independent audit did not pass')
    require(m['source']==a['source']==source and m['session_id']==a['session_id'],'Parent source/session mismatch')
    require(a['manifest_sha256']==sha(mp),'Parent audit does not bind manifest')
    require(bool(a['acquisition_complete'])==complete and (m['status']=='completed')==complete,'Parent acquisition status mismatch')
    require(a['new_counters']==m['new_counters'],'Parent counters differ')
    if complete:require(a['full_time_outputs']==2400,'Parent outputs incomplete')
    verify_files(folder,m['result_sha256'],ck)
    expected=A_SHA if m['protocol_id']==A.PROTOCOL else A.ADAPTIVE_AUDITOR_SHA
    require(a['auditor_sha256']==expected,'Unreviewed parent auditor')
    return folder,m,a


def historical_totals(session,parent,failed=None):
    key='cumulative_impacts_including_all_ancestry' if 'cumulative_impacts_including_all_ancestry' in parent else 'cumulative_impacts_including_all_parent_attempts'
    impacts=parent[key];pulses=parent[key.replace('impacts','pulses')]
    records=[dict(session=session,impact_attempts=impacts,programming_pulse_attempts=pulses,
        accounting_scope='parent_with_all_ancestry_already_included')]
    if failed:
        name,m=failed
        require(name!=session and m['status']!='completed' and m['parent_import']['session']==session,'Failed-A is not distinct child of baseline')
        n=m['new_counters'];impacts+=n['impact_attempts'];pulses+=n['programming_pulse_attempts']
        records.append(dict(session=name,impact_attempts=n['impact_attempts'],programming_pulse_attempts=n['programming_pulse_attempts'],accounting_scope='failed_A_new_attempts_only'))
    return dict(parent_cumulative_impacts=impacts,parent_cumulative_pulses=pulses,historical_attempt_ledger=records)


def verify_parent(repo,metadata,source,plan,fixture,routing,ck):
    binding=plan['parent_bindings'][source]
    require(binding and binding['source']==source,'Unbound/cross-source parent')
    folder,parent,audit=bound_evidence(repo,binding,source,True,ck);failed=None
    if binding['choice']=='completed_A':
        require(not binding.get('failed_A') and parent['protocol_id']==A.PROTOCOL,'Invalid completed-A choice')
        pm,pp,_=A.verify_reserve_manifest(repo,folder,ck)
        A.verify_reserve_parent(repo,pm['parent_import'],source,pp,fixture,routing,ck)
        proof=pm['parent_import']
    elif binding['choice']=='baseline_with_failed_A':
        require(parent['protocol_id']=='cifar-feedback-adaptive-two-microtiles-v1','Expected adaptive baseline')
        fb=binding['failed_A'];ff,fm,fa=bound_evidence(repo,fb,source,False,ck)
        require(fm['protocol_id']==A.PROTOCOL,'Expected failed-A protocol')
        _,fp,_=A.verify_reserve_manifest(repo,ff,ck)
        require(fm['parent_import']['session']==binding['session'],'Failed-A has different baseline')
        A.verify_reserve_parent(repo,fm['parent_import'],source,fp,fixture,routing,ck)
        proof=fm['parent_import'];failed=(fb['session'],fm)
    else:raise ValueError('Unknown parent choice; fallback forbidden')
    originals=load(folder/'completed_microtiles.json')
    chunks=imported280(originals,binding['session'],parent['result_sha256']['completed_microtiles.json'],routing)
    # Check nested immediate origins, while pinned parent audits bind all older ancestry.
    for w in originals:
        o=w['origin'];path=local(local(repo,o['session']),o['path'])
        require(sha(path)==o['sha256'],'Imported wrapper origin SHA changed')
        raw=load(path)
        expected=raw[o['wrapper_index']]['result'] if 'wrapper_index' in o else raw
        require(expected==w['result'],'Imported wrapper bytes/result differ')
    totals=historical_totals(binding['session'],parent,failed)
    expected={**binding,'result_sha256':parent['result_sha256'],'imported_trained_coefficients':280,
        'excluded_blocks':[dict(filter_id=2,parent_tile_id=0,taps=list(range(11))),dict(filter_id=1,parent_tile_id=0,taps=list(range(9)))],
        'verified_ancestry':proof,**totals}
    require(metadata==expected,'Unsupported/altered parent metadata')
    return chunks,totals


def verify_manifest_contract(directory,m,ck):
    required={'protocol.json','board.ini','start_manifest.json','imported_microtiles.json'}
    roots={'completed_microtiles.json','summary.json','capacity_failures.json'}
    if m['status']=='completed':required.update({'completed_microtiles.json','summary.json'})
    required.update(name for name in roots if (directory/name).exists())
    for name in sorted(required):
        require(name in m['result_sha256'],'Unbound required file '+name)
    ck.equal(m,dict(maximum_impacts=40378,maximum_programming_pulses=5260,
        maximum_programming_decisions=5260,new_trained_coefficients=20,
        imported_trained_coefficients=280),'manifest/independent_declarations')


def verify_manifest(repo,directory,ck):
    require(FINAL_PINS and FINAL_PROTOCOL,'Final B/C source/parent review not frozen; actual-session audit disabled')
    m=load(directory/'manifest.json');git=m['invocation']['git'];provenance=audit_git_status(git,m['source_sha256'],ck)
    require(m['source'] in ('hardware','simulator'),'Unsupported backend')
    verify_files(directory,m['result_sha256'],ck)
    verify_manifest_contract(directory,m,ck)
    for name,digest in m['source_sha256'].items():
        p=local(repo,name)
        if name in PRIVATE:
            ck.equal(digest,PRIVATE[name],'source/private_pin/'+name)
            if p.exists():ck.equal(sha(p),digest,'source/private_local/'+name)
        else:
            proc=subprocess.run(['git','-C',str(repo),'show',git['commit']+':'+name],capture_output=True)
            ck.equal(proc.returncode,0,'source/git_exists/'+name)
            ck.equal(hashlib.sha256(proc.stdout).hexdigest(),digest,'source/git_blob/'+name)
    pins={**PINS,**DIAGNOSTIC_PINS,**ADAPTIVE_PINS,**A.RESERVE_PINS,**FINAL_PINS}
    for name,digest in pins.items():ck.equal(m['source_sha256'].get(name),digest,'source/frozen/'+name)
    plan=load(directory/'protocol.json')
    raw=subprocess.run(['git','-C',str(repo),'show',git['commit']+':configs/cifar_feedback_remaining_reserves.json'],capture_output=True,check=True).stdout
    require(plan==json.loads(raw),'Protocol differs from recorded Git config')
    require(plan['protocol_id']==m['protocol_id']==FINAL_PROTOCOL and not plan['draft'] and plan['native_execution_enabled'],'Draft/unknown protocol')
    ck.equal(plan,dict(maximum_impacts=40378,maximum_programming_pulses=5260,maximum_programming_decisions=5260,
        maximum_new_microtiles=2,automatic_retry=False,automatic_parent_fallback=False,target_gain_v_per_weight={'hardware':1.,'simulator':.03}),'protocol/contract')
    ck.equal(m['hardware_access'],m['source']=='hardware','manifest/hardware_access')
    ck.equal(m['fixed_decoder_gain_v_per_weight'],1. if m['source']=='hardware' else .03,'manifest/fixed_gain')
    board=ConfigParser();board.read(directory/'board.ini',encoding='utf-8')
    for k,v in BOARD.items():ck.equal(board.getfloat('board',k),v,'INI/'+k)
    start=load(directory/'start_manifest.json')
    for k in ('session_id','source','source_sha256','protocol_id','invocation','parent_import'):ck.equal(m[k],start[k],'manifest/start/'+k)
    return m,plan,provenance


def new_wrapper(repo,directory,stage):
    fid=SPECS[stage]['fid'];path=f'tiles/f{fid}_p0_reserve_{stage}/tile.json'
    return dict(filter_id=fid,parent_tile_id=0,microtile_id=0,result=load(directory/path),
        origin=dict(session=directory.relative_to(repo).as_posix(),path=path,sha256=sha(directory/path),reused=False,new_measurement=True,acquisition_tile_id=0))


def audit_session(repo,directory,fixture,routing,ck):
    m,plan,provenance=verify_manifest(repo,directory,ck)
    chunks,parent=verify_parent(repo,m['parent_import'],m['source'],plan,fixture,routing,ck)
    require(load(directory/'imported_microtiles.json')==chunks,'Imported280 wrappers changed')
    fresh,totals,failures=replay_blocks(directory,fixture,routing,plan,m['source'],m['session_id'],ck)
    ck.equal(m['new_counters'],totals,'manifest/new_counters')
    ck.equal(m['new_attempt_address_coverage'],coverage(fresh),'manifest/all_attempt_coverage')
    for noun,counter in (('impacts','impact_attempts'),('pulses','programming_pulse_attempts')):
        ck.equal(m['cumulative_'+noun+'_including_all_ancestry'],parent['parent_cumulative_'+noun]+totals[counter],'history/count_once/'+noun)
    expected=chunks+[new_wrapper(repo,directory,s) for s,r in zip(('B','C'),fresh) if r['technical_completed']]
    if any(r['technical_completed'] for r in fresh):
        require(load(directory/'completed_microtiles.json')==expected,'Completed wrapper checkpoint differs')
    else:require(not (directory/'completed_microtiles.json').exists(),'False completed checkpoint')
    complete=all(r['technical_completed'] for r in fresh);trajectories=[];quality=[]
    if complete:
        verify_wrapper_coverage(expected,routing)
        results=[normalized_result(w) for w in ordered(expected)]
        trajectories,quality=aggregate_microtiles(results,fixture,routing,ck)
        saved=load(directory/'summary.json')
        ck.equal(saved['trajectories'],trajectories,'summary/independent_sum75_BN_LIF',1e-8)
        ck.equal(saved,dict(status='completed',technical_completed=True,acquisition_complete=True,
            decoded_conv_bn_lif_complete=True,new_counters=totals,parent_import=m['parent_import'],
            physical_weight_quality_pass=all(r['physical_weight_quality_pass'] for r in results),
            new_trained_coefficients=20,imported_trained_coefficients=280,fitted_alpha_used_for_decoding=False,
            diagnostic_acquisition_not_accuracy_qualification=True,address_coverage=coverage(results)),'summary/flags')
        ck.equal(m['status'],'completed','manifest/completed');ck.equal(m['acquisition_complete'],True,'manifest/acquisition')
        ck.equal(len(trajectories)*12,2400,'summary/exact2400')
    else:
        ck.equal(m['status'],'failed','manifest/incomplete')
        ck.require(not m.get('acquisition_complete',False) and not (directory/'summary.json').exists(),'incomplete/no_old_BC_fallback')
    return dict(session_id=m['session_id'],source=m['source'],protocol_id=m['protocol_id'],manifest_sha256=sha(directory/'manifest.json'),
        acquisition_complete=complete,new_counters=totals,full_time_outputs=len(trajectories)*12,
        new_partial_points=sum(len(r['partials']) for r in fresh),parent_accounting=parent,provenance=provenance,
        capacity_failures=failures,quality=quality,blocks=[compact(r) for r in fresh])


def compact(r):
    return {k:v for k,v in r.items() if k not in ('reset_candidates','pairs','partials','matrix')}


def audit_synthetic(directory,fixture,routing,plan,ck):
    marker=load(directory/'FIXTURE.json')
    require(marker['synthetic'] is True and marker['native_simulator'] is False and marker['hardware'] is False
        and marker['parent_provenance_verified'] is False and marker['full300_result'] is False,'Fixture must be explicitly synthetic')
    ck.equal(marker['source_sha256']['cifar_feedback_remaining_reserves.py'],DRAFT_RUNTIME_SHA,'synthetic/runtime_snapshot')
    ck.equal(marker['source_sha256']['reserve_remaining_plan.py'],DRAFT_PLAN_SHA,'synthetic/plan_snapshot')
    verify_files(directory,marker['result_sha256'],ck)
    with (directory/'tiles/f2_p0_reserve_B/raw.jsonl').open(encoding='utf-8') as stream:first=json.loads(next(stream))
    require(first['source']=='simulator' and first['session_id'].startswith('SYNTHETIC_'),'Synthetic source/id mismatch')
    results,totals,failures=replay_blocks(directory,fixture,routing,plan,'simulator',first['session_id'],ck)
    ck.equal(totals,marker['counters'],'synthetic/all_command_counts')
    ck.equal(sum(len(r.get('effective_route',[])) for r in results),marker['completed_new_coefficients'],'synthetic/coefficient_count')
    return dict(synthetic_fixture=True,native_acquisition=False,parent_provenance_verified=False,full300_result=False,
        session_id=first['session_id'],new_counters=totals,new_partial_points=sum(len(r['partials']) for r in results),
        acquisition_complete=not failures,capacity_failures=failures,blocks=[compact(r) for r in results])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.')
    p.add_argument('--session',required=True);p.add_argument('--output',required=True)
    p.add_argument('--synthetic-fixture',action='store_true');p.add_argument('--synthetic-plan')
    args=p.parse_args(argv);repo=Path(args.repo).resolve()
    helper_repo=repo if (repo/'audit_cifar_feedback_reserve.py').exists() else repo/'.tmp/hardware-public'
    initialize(helper_repo)
    directory=local(repo,args.session);output=local(repo,args.output)
    require(not output.is_relative_to(directory),'Output cannot modify source data')
    ck=CompactChecks();report={}
    try:
        fixture,routing,golden=verify_inputs(helper_repo,ck)
        if args.synthetic_fixture:
            require(args.synthetic_plan is not None,'Explicit synthetic plan required')
            report=audit_synthetic(directory,fixture,routing,load(local(repo,args.synthetic_plan)),ck)
        else:report=audit_session(repo,directory,fixture,routing,ck)
        report['golden_validation']=golden
    except Exception as exc:ck.require(False,'exception/'+type(exc).__name__+': '+str(exc))
    report.update(audit_status='passed' if not ck.errors else 'failed',checks=ck.count,errors=ck.errors,
        auditor_sha256=sha(Path(__file__)),independent_A_helper_sha256=A_SHA,hardware_io=False,
        limitations=['Consistency PASS is not accuracy or programmability qualification.',
            'New20 and retained280 are separated; no claim all weights coexist.',
            'Actual-session validation is disabled until final source/config/parent pins are reviewed.',
            'Scientific route-capacity failures are replayed fully. Transport/interrupted raw currently fails audit closed; it is never accepted as completed.'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (output/'REPORT.md').write_text('# Independent B/C audit\n\n'+f"{report['audit_status']}; {ck.count} checks; {len(ck.errors)} errors.\n\n"+
        f"Synthetic: {report.get('synthetic_fixture',False)}; acquisition complete: {report.get('acquisition_complete')}; fresh partials: {report.get('new_partial_points')}.\n",encoding='utf-8')
    print(json.dumps(dict(status=report['audit_status'],checks=ck.count,errors=ck.errors[:12],output=args.output)))
    return int(bool(ck.errors))


if __name__=='__main__':raise SystemExit(main())
