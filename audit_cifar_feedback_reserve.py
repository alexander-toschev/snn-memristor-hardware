"""Independent reserve raw replay and immutable 287+13 CIFAR reconstruction.

Only independent auditors are imported; no acquisition, analyzer or device code.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from collections import Counter
from configparser import ConfigParser

ADAPTIVE_AUDITOR_SHA='09d8ac17e57e8a2df12c26d12fe187c52ceeb25f007ed93baeebe67262b4c92f'
if hashlib.sha256(Path(__file__).with_name('audit_cifar_feedback_adaptive.py').read_bytes()).hexdigest()!=ADAPTIVE_AUDITOR_SHA:
    raise RuntimeError('Frozen independent adaptive helper changed')
from audit_cifar_feedback_adaptive import (CompactChecks,require,local,load,sha,verify_inputs,
    verify_adaptive_manifest,verify_manifest,pools_for,read_raw,Replay,reset_replay,pair_replay,
    ledger,multiply,fixed_calibration,fixed_effective_weights,replay_block,fixed_decode,fidelity,
    audit_adaptive_matrix,measurement_tail,aggregate_microtiles,coverage,DIAGNOSTIC_PINS,PINS,
    PRIVATE,BOARD,ADAPTIVE_PINS,audit_git_status,trained_slots,DIAGNOSTIC_AUDITOR_SHA)

PROTOCOL='cifar-feedback-reserve-wl7-bl15-v1'
RESERVE_PINS={
    'cifar_feedback_reserve.py':'a4e3a822ede1460c905435000912dc91b851b26c8d0608006ceba12708d3f7c9',
    'configs/cifar_feedback_reserve.json':'d042e27faf56a0334f7669689255ea45ad1d43cface8a3fc93ffdf42f4415b9c',
    'experiment.py':'718cb130a40c4e17b20041c503bb7f8557fd1d1f96abccdfe5546be0ee4e7a37'}
COUNTERS=('impact_attempts','programming_pulse_attempts','programming_decisions','programming_skipped_current')


def reserve_pools(parent):
    require(parent['tile_id']==4,'Wrong reserve parent')
    active=[s for s in parent['slots'] if not s['padding']]
    require([s['tap_index'] for s in active]==list(range(52,65)),'Wrong reserve taps')
    pools=pools_for(parent)
    require(len(pools['positive_cell'])==16 and len(pools['negative_cell'])==16,'Original pool size changed')
    require(all(c[0]==6 for c in pools['positive_cell']) and all(c[0]==7 for c in pools['negative_cell']),'Wrong reserve WL pair')
    require([7,15] not in pools['negative_cell'],'Reserve already in original pool')
    return dict(positive_cell=pools['positive_cell'],negative_cell=pools['negative_cell']+[[7,15]])


def gate_diagnostics(eligible):
    return dict(reserved_negative_quiet=[7,15] in eligible['negative_cell'],
        original_negative_quiet=sum(c!=[7,15] for c in eligible['negative_cell']),positive_quiet=len(eligible['positive_cell']))


def reserve_route(pools,eligible):
    for side in pools:
        require(len({tuple(c) for c in eligible[side]})==len(eligible[side]),'Duplicate eligible address')
        require(all(c in pools[side] for c in eligible[side]),'Eligible address outside reviewed pool')
        require(eligible[side]==[c for c in pools[side] if c in eligible[side]],'Eligibility order changed')
    g=gate_diagnostics(eligible)
    if not g['reserved_negative_quiet'] or g['original_negative_quiet']<12 or g['positive_quiet']<13:return None
    ordered=[]
    for c in [[6,2],[6,11]]+pools['positive_cell']:
        if c in eligible['positive_cell'] and c not in ordered:ordered.append(c)
    return dict(positive_cell=ordered[:13],negative_cell=[c for c in eligible['negative_cell'] if c!=[7,15]][:12]+[[7,15]])


def reserve_budget(routing,fixture):
    parent=routing['filters'][3]['tiles'][4];pool=reserve_pools(parent)
    active=[s for s in parent['slots'] if not s['padding']];n=len(active);c=sum(map(len,pool.values()))
    return dict(maximum_impacts=244*c+811*n+measurement_tail(active,fixture),
        maximum_programming_pulses=55*c+109*n,maximum_programming_decisions=55*c+109*n,
        reset_candidate_count=c,new_trained_coefficients=n,imported_trained_coefficients=300-n)


def audit_reserve_tile(directory,filt,parent,fixture,plan,source,ck,session_id):
    pending=[s for s in parent['slots'] if not s['padding']];micro=0
    saved=load(directory/'tile.json');fid=filt['filter_id'];tid=2*parent['tile_id']+micro;meta=dict(filter_id=fid,tile_id=tid,parent_tile_id=parent['tile_id'],microtile_id=micro)
    active=copy.deepcopy(pending);pools=reserve_pools(parent)
    ck.equal(saved,meta,'microtile/saved_identity')
    ck.equal(saved['pending_taps_before'],[s['tap_index'] for s in pending],'microtile/pending_prefix')
    ck.equal(saved['original_chunk_taps'],[s['tap_index'] for s in parent['slots'] if not s['padding']],'microtile/original_chunk')
    events,counters=read_raw(directory/'raw.jsonl',source,ck,session_id);replay=Replay(events,ck)
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
    ck.equal(saved['quiet_candidate_counts'],{side:len(cells) for side,cells in chosen.items()},'reserve/fresh_quiet_counts')
    decision=reserve_route(pools,chosen)
    if decision is None:
        ck.equal(saved['status'],'failed','reserve/gate_failure_status')
        ck.equal(saved['stop_reason'],'reserve_route_gate_failed_before_SET','reserve/gate_failure_reason')
        ck.equal(replay.position,len(events),'reserve/no_SET_or_CIFAR_after_gate_failure')
        ck.equal(saved['pairs'],[],'reserve/no_pairs_on_gate_failure');ck.equal(saved['partials'],[],'reserve/no_partials_on_gate_failure')
        ck.equal(saved,counters,'reserve/failed_counters')
        ck.require(not (directory/'frozen_route.json').exists(),'reserve/no_frozen_route_on_failure')
        return dict(**meta,technical_completed=False,physical_weight_quality_pass=False,counters=counters,
                    reset_candidates=candidates,partials=[],pairs=[],acquisition_stop='reserve_route_gate_failed_before_SET',
                    gate=gate_diagnostics(chosen))
    chosen=decision
    ck.equal(saved['assigned_taps'],[s['tap_index'] for s in active],'reserve/exact13_taps')
    for i,slot in enumerate(active):
        for side in chosen:slot[side]=chosen[side][i]
    ck.equal(saved['effective_route'],active,'routing/deterministic_weight_independent')
    route=load(directory/'frozen_route.json');freeze=events[replay.position];replay.position+=1
    ck.equal(route,dict(candidate_pools=pools,slots=active,mapping_uses_trained_weight_values=False,
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
        pair_stop_counts=dict(Counter(p['stop'] for p in pairs)),gate=gate_diagnostics({side:[c['cell'] for c in candidates if c['side']==side and c['eligible']] for side in pools}))

def ordered(wrappers):
    return sorted(wrappers,key=lambda w:(w['filter_id'],w['parent_tile_id'],w['microtile_id']))


def verify_wrapper_coverage(wrappers,routing,excluded=False):
    targets=trained_slots(routing);seen=[];identities=set()
    for w in wrappers:
        identity=(w['filter_id'],w['parent_tile_id'],w['microtile_id'])
        require(identity not in identities,'Duplicate logical wrapper');identities.add(identity)
        r=w['result'];require(r['technical_completed'],'Incomplete imported/result wrapper')
        fid=w['filter_id'];require(r['filter_id']==fid,'Wrapper filter mismatch')
        parent=next((t for f in routing['filters'] if f['filter_id']==fid for t in f['tiles'] if t['tile_id']==w['parent_tile_id']),None)
        require(parent is not None and w['microtile_id'] in (0,1),'Invalid wrapper parent/microtile')
        allowed={s['tap_index'] for s in parent['slots'] if not s['padding']}
        for s in r['effective_route']:
            tap=s['tap_index'];require(tap in allowed and s['trained_weight']==targets[fid][tap],'Changed imported tap/weight')
            seen.append((fid,tap))
        points=[(p['case'],p['repeat'],p['t']) for p in r['partials']]
        require(len(points)==600 and set(points)=={(c,repeat,t) for c in range(10) for repeat in range(5) for t in range(12)},'Incomplete imported partial ledger')
    expected={(f,t) for f in range(4) for t in range(75)}
    if excluded:expected-={(3,t) for t in range(52,65)}
    require(len(seen)==len(expected) and set(seen)==expected,'Coefficient replacement coverage is not exact')
    return dict(coefficients=len(seen),by_filter={str(f):sum(fid==f for fid,t in seen) for f in range(4)},measured_microtiles=len(wrappers))


def selected_imports(originals,parent_session,parent_wrappers_sha,routing):
    verify_wrapper_coverage(originals,routing)
    chunks=[];removed=[]
    for index,w in enumerate(originals):
        if (w['filter_id'],w['parent_tile_id'])==(3,4):
            removed.extend(s['tap_index'] for s in w['result']['effective_route']);continue
        origin=dict(session=parent_session,path='completed_microtiles.json',wrapper_index=index,sha256=parent_wrappers_sha,
            reused=True,new_measurement=False,acquisition_tile_id=w['result']['tile_id'],ancestry_original_origin=w['origin'])
        chunks.append({**w,'origin':origin})
    require(sorted(removed)==list(range(52,65)),'Must exclude every old target microtile, exactly 13 taps')
    verify_wrapper_coverage(chunks,routing,True)
    return chunks


def sum_counter(wrappers):
    return {key:sum(w['result'][key] for w in wrappers) for key in COUNTERS}


def verify_ancestry42(repo,metadata,source,parent_plan,ck):
    binding=parent_plan['parent_bindings'][source]
    require(all(metadata.get(k)==v for k,v in binding.items()),'Ancestral 42 binding changed')
    folder=local(repo,binding['session']);ap=local(repo,binding['audit_path'])
    require(sha(folder/'manifest.json')==binding['manifest_sha256'] and sha(ap)==binding['audit_sha256'],'Ancestral manifest/audit hash changed')
    report=load(ap)
    require(report['audit_status']=='passed' and report['errors']==[] and report['auditor_sha256']==DIAGNOSTIC_AUDITOR_SHA,'Unreviewed ancestral audit')
    manifest,plan,provenance=verify_manifest(repo,folder,ck) # every original raw/result byte commitment
    require(report['source']==manifest['source']==source and report['session_id']==manifest['session_id'],'Ancestral backend/identity changed')
    require(report['manifest_sha256']==sha(folder/'manifest.json'),'Ancestral audit binding changed')
    require(metadata['result_sha256']==manifest['result_sha256'],'Ancestral result dictionary changed')
    ck.equal(metadata,dict(imported_prefix=['f0_t0','f0_t1','f0_t2'],imported_trained_coefficients=42,
        failed_parent_tile_contributes=False,other_parent_tiles_ignored=True,
        parent_total_impacts=report['counters']['impact_attempts'],parent_total_pulses=report['counters']['programming_pulse_attempts']),
        'ancestry42/metadata')
    return dict(session=binding['session'],manifest_sha256=binding['manifest_sha256'],audit_sha256=binding['audit_sha256'],
        acquisition_status=manifest['status'],all_attempt_counters=report['counters'],provenance=provenance)


def verify_reserve_parent(repo,metadata,source,plan,fixture,routing,ck):
    binding=plan['parent_bindings'][source]
    require(all(metadata.get(k)==v for k,v in binding.items()),'Reserve parent binding differs')
    folder=local(repo,binding['session']);mp=folder/'manifest.json';ap=local(repo,binding['audit_path'])
    require(sha(mp)==binding['manifest_sha256'] and sha(ap)==binding['audit_sha256'],'Reserve parent manifest/audit changed')
    audit=load(ap)
    require(audit['audit_status']=='passed' and audit['errors']==[] and audit['auditor_sha256']==ADAPTIVE_AUDITOR_SHA,'Reserve parent lacks reviewed audit PASS')
    parent,parent_plan,provenance=verify_adaptive_manifest(repo,folder,ck)
    require(parent['status']=='completed' and audit['acquisition_complete'] and audit['full_time_outputs']==2400,'Parent incomplete')
    require(audit['manifest_sha256']==binding['manifest_sha256'] and audit['session_id']==parent['session_id'],'Parent audit identity differs')
    require(audit['source']==parent['source']==source,'Cross-backend parent reuse')
    require(metadata['result_sha256']==parent['result_sha256'],'Parent complete result dictionary changed')
    require(audit['new_counters']==parent['new_counters'],'Parent new counters differ from audit')
    ancestry=verify_ancestry42(repo,parent['parent_import'],source,parent_plan,ck)
    require(metadata['ancestry42']==parent['parent_import'],'Copied ancestral42 metadata differs')
    ck.equal(parent['cumulative_impacts_including_all_parent_attempts'],parent['new_counters']['impact_attempts']+ancestry['all_attempt_counters']['impact_attempts'],'parent/no_ancestry_double_count_impacts')
    ck.equal(parent['cumulative_pulses_including_all_parent_attempts'],parent['new_counters']['programming_pulse_attempts']+ancestry['all_attempt_counters']['programming_pulse_attempts'],'parent/no_ancestry_double_count_pulses')
    originals=load(folder/'completed_microtiles.json')
    for w in originals:
        o=w['origin'];origin_folder=local(repo,o['session']);path=local(origin_folder,o['path'])
        require(sha(path)==o['sha256'] and load(path)==w['result'],'Original parent wrapper result changed')
        # Imported first42 must point into the reviewed original ancestry; other wrappers into parent itself.
        expected_session=parent['parent_import']['session'] if o['reused'] else binding['session']
        require(o['session']==expected_session,'Unexpected extra ancestry')
    chunks=selected_imports(originals,binding['session'],parent['result_sha256']['completed_microtiles.json'],routing)
    expected={**binding,'result_sha256':parent['result_sha256'],'imported_trained_coefficients':287,
        'excluded_filter_id':3,'excluded_parent_tile_id':4,'excluded_taps':list(range(52,65)),
        'parent_new_counters':parent['new_counters'],'ancestry42':parent['parent_import'],
        'parent_cumulative_impacts':parent['cumulative_impacts_including_all_parent_attempts'],
        'parent_cumulative_pulses':parent['cumulative_pulses_including_all_parent_attempts']}
    require(metadata==expected,'Parent import metadata has unsupported/missing fields')
    return chunks,dict(session=binding['session'],manifest_sha256=binding['manifest_sha256'],audit_sha256=binding['audit_sha256'],
        parent_new_counters=parent['new_counters'],ancestry42=ancestry,
        parent_cumulative_impacts=expected['parent_cumulative_impacts'],parent_cumulative_pulses=expected['parent_cumulative_pulses'],
        imported_contributing_counters=sum_counter(chunks),imported_coefficients=287,imported_microtiles=len(chunks),
        excluded_coefficients=13,excluded_microtiles=len(originals)-len(chunks),provenance=provenance,
        raw_replay_policy='Earlier complete raw replay is trusted only through pinned audit and every immutable file hash; new reserve raw is fully replayed here.')


def verify_reserve_manifest(repo,directory,ck):
    require(bool(RESERVE_PINS),'Reserve runtime review is not frozen')
    m=load(directory/'manifest.json');git=m['invocation']['git'];provenance=audit_git_status(git,m['source_sha256'],ck)
    require(m['source'] in ('hardware','simulator'),'Invalid source')
    ck.equal(m['hardware_access'],m['source']=='hardware','manifest/hardware_access')
    for name,digest in m['result_sha256'].items():ck.equal(sha(local(directory,name)),digest,'result_sha/'+name)
    for required in ('protocol.json','board.ini','start_manifest.json','imported_microtiles.json'):
        require(required in m['result_sha256'],'Required artifact not bound: '+required)
    if m['status']=='completed':
        for name in ('completed_microtiles.json','summary.json'):require(name in m['result_sha256'],'Completed artifact not bound: '+name)
    for name,digest in m['source_sha256'].items():
        p=local(repo,name)
        if name in PRIVATE:
            ck.equal(digest,PRIVATE[name],'source/private_pin/'+name)
            if p.exists():ck.equal(sha(p),digest,'source/private_local/'+name)
        else:
            proc=subprocess.run(['git','-C',str(repo),'show',git['commit']+':'+name],capture_output=True)
            ck.equal(proc.returncode,0,'source/git_exists/'+name)
            ck.equal(hashlib.sha256(proc.stdout).hexdigest(),digest,'source/git_blob/'+name)
    for name,digest in {**PINS,**DIAGNOSTIC_PINS,**ADAPTIVE_PINS,**RESERVE_PINS}.items():
        ck.equal(m['source_sha256'].get(name),digest,'source/frozen_pin/'+name)
    plan=load(directory/'protocol.json')
    raw=subprocess.run(['git','-C',str(repo),'show',git['commit']+':configs/cifar_feedback_reserve.json'],capture_output=True,check=True).stdout
    require(plan==json.loads(raw),'Protocol snapshot differs from recorded Git config')
    ck.equal(m['protocol_id'],PROTOCOL,'protocol/manifest');ck.equal(plan['protocol_id'],PROTOCOL,'protocol/snapshot')
    ck.equal(plan,dict(maximum_impacts=24323,maximum_programming_pulses=3232,maximum_programming_decisions=3232,
        target_filter_id=3,target_parent_tile_id=4,target_taps=list(range(52,65)),reserved_negative_cell=[7,15],reserved_tap=64,
        positive_priority=[[6,2],[6,11]],maximum_new_microtiles=1,automatic_retry=False,target_gain_v_per_weight={'hardware':1.,'simulator':.03}),'protocol/independent_contract')
    ck.equal(m['fixed_decoder_gain_v_per_weight'],1. if m['source']=='hardware' else .03,'manifest/fixed_gain')
    board=ConfigParser();board.read(directory/'board.ini',encoding='utf-8')
    for k,v in BOARD.items():ck.equal(board.getfloat('board',k),v,'INI/'+k)
    start=load(directory/'start_manifest.json')
    for k in ('session_id','source','source_sha256','protocol_id','invocation','parent_import'):ck.equal(m[k],start[k],'manifest/start/'+k)
    return m,plan,provenance


def normalized_result(wrapper):
    r=wrapper['result'];return {**r,'parent_tile_id':wrapper['parent_tile_id'],'microtile_id':wrapper['microtile_id'],
        'counters':{key:r[key] for key in COUNTERS}}


def audit_session(repo,directory,fixture,routing,ck):
    m,plan,provenance=verify_reserve_manifest(repo,directory,ck);budget=reserve_budget(routing,fixture)
    ck.equal(budget,dict(maximum_impacts=24323,maximum_programming_pulses=3232,maximum_programming_decisions=3232,
        reset_candidate_count=33,new_trained_coefficients=13,imported_trained_coefficients=287),'budget/derived')
    ck.equal(m,budget,'manifest/derived_budget')
    chunks,parent=verify_reserve_parent(repo,m['parent_import'],m['source'],plan,fixture,routing,ck)
    require(load(directory/'imported_microtiles.json')==chunks,'Imported wrappers differ from exact immutable 287 selection')
    folder=directory/'tiles/f3_p4_reserve'
    actual={p.name for p in (directory/'tiles').iterdir() if p.is_dir()}
    require(actual=={'f3_p4_reserve'},'Extra or missing reserve acquisition folder')
    for p in folder.iterdir():
        if p.is_file():require('tiles/f3_p4_reserve/'+p.name in m['result_sha256'],'Unbound reserve artifact')
    fresh=audit_reserve_tile(folder,routing['filters'][3],routing['filters'][3]['tiles'][4],fixture,plan,m['source'],ck,m['session_id'])
    counts=fresh['counters'];ck.equal(m['new_counters'],counts,'manifest/new_counters')
    ck.require(counts['impact_attempts']<=24323,'budget/impacts');ck.require(counts['programming_decisions']<=3232,'budget/decisions')
    ck.require(counts['programming_pulse_attempts']<=counts['programming_decisions'],'budget/writes')
    ck.equal(m['new_attempt_address_coverage'],coverage([fresh]),'manifest/fresh_coverage')
    ck.equal(m['cumulative_impacts_including_all_ancestry'],counts['impact_attempts']+parent['parent_cumulative_impacts'],'manifest/ancestry_counted_once_impacts')
    ck.equal(m['cumulative_pulses_including_all_ancestry'],counts['programming_pulse_attempts']+parent['parent_cumulative_pulses'],'manifest/ancestry_counted_once_pulses')
    trajectories=[];quality=[];complete=fresh['technical_completed'];fullcoverage=None;physical=False
    if complete:
        path='tiles/f3_p4_reserve/tile.json';newwrapper=dict(filter_id=3,parent_tile_id=4,microtile_id=0,
            result=load(folder/'tile.json'),origin=dict(session=directory.relative_to(repo).as_posix(),path=path,
            sha256=m['result_sha256'][path],reused=False,new_measurement=True,acquisition_tile_id=8))
        expected=chunks+[newwrapper];require(load(directory/'completed_microtiles.json')==expected,'Completed wrappers changed/duplicated/retained excluded taps')
        fullcoverage=verify_wrapper_coverage(expected,routing)
        results=[normalized_result(w) if w['origin']['reused'] else fresh for w in ordered(expected)]
        trajectories,quality=aggregate_microtiles(results,fixture,routing,ck)
        physical=all(r['physical_weight_quality_pass'] for r in results)
        saved=load(directory/'summary.json');ck.equal(saved['trajectories'],trajectories,'summary/raw_new_plus_immutable287_sum75_BN_LIF',1e-8)
        ck.equal(saved,dict(status='completed',technical_completed=True,acquisition_complete=True,
            decoded_conv_bn_lif_complete=True,physical_weight_quality_pass=physical,fresh_weight_quality_pass=fresh['physical_weight_quality_pass'],
            parent_import=m['parent_import'],fixed_decoder_gain_v_per_weight=1. if m['source']=='hardware' else .03,
            fitted_alpha_used_for_decoding=False,diagnostic_acquisition_not_accuracy_qualification=True,
            actual_branch_uses_software_dot_product=False,full_network_accuracy_claim=False,
            new_trained_coefficients=13,imported_trained_coefficients=287,
            address_coverage=coverage(results),new_attempt_address_coverage=coverage([fresh])),'summary/flags_coverage')
        ck.equal(len(trajectories)*12,2400,'summary/2400_outputs');ck.equal(len(fresh['partials']),600,'summary/600_fresh_partials')
        ck.equal(m['status'],'completed','manifest/complete');ck.equal(m['acquisition_complete'],True,'manifest/acquisition_complete')
    else:
        ck.equal(m['status'],'failed','manifest/gate_failed')
        ck.require(not (directory/'summary.json').exists() and not (directory/'completed_microtiles.json').exists(),'gate/no_old13_fallback_or_false_complete')
    compact={k:v for k,v in fresh.items() if k not in ('partials','pairs','reset_candidates','matrix')}
    if 'matrix' in fresh:compact['matrix']={k:fresh['matrix'][k] for k in ('matrix','selected_model','selection_pass','reference_design_rank')}
    return dict(session_id=m['session_id'],source=m['source'],protocol_id=PROTOCOL,manifest_sha256=sha(directory/'manifest.json'),
        acquisition_status=m['status'],acquisition_complete=complete,new_counters=counts,parent_import_audit=parent,
        provenance=provenance,full_time_outputs=len(trajectories)*12,new_partial_points=len(fresh['partials']),
        physical_weight_quality_pass=physical,fresh_weight_quality_pass=fresh['physical_weight_quality_pass'],
        full_coefficient_coverage=fullcoverage,fresh_tile=compact,quality=quality,
        cumulative_impacts_including_all_ancestry=m['cumulative_impacts_including_all_ancestry'],
        cumulative_pulses_including_all_ancestry=m['cumulative_pulses_including_all_ancestry'],
        limitations=['PASS is raw/data consistency, not exact weight programming or CIFAR classification accuracy.',
            'Only 13 coefficients are freshly measured. The remaining 287 preserve their historical acquisition states and times.',
            'The 13 assigned pairs coexist within this microtile, but mode10 acquisition is sequential one-hot, not parallel analog MVM.',
            'Assigned reserve is not necessarily SET-written; polarity follows actual D0 residual, never an address-coverage counter.',
            'All parent/ancestor attempts are counted once, separately from retained contributing measurements.',
            'Fixed HW=1/SIM=0.03 gains are provisional. Fitted alpha is diagnostic only.'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.')
    for name in ('session','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args(argv);repo=Path(args.repo).resolve();directory=local(repo,args.session);output=local(repo,args.output)
    require(not output.is_relative_to(directory),'Output may not modify acquired data')
    ck=CompactChecks();report={}
    try:
        fixture,routing,golden=verify_inputs(repo,ck);report=audit_session(repo,directory,fixture,routing,ck);report['golden_validation']=golden
    except Exception as exc:ck.require(False,'exception/'+type(exc).__name__+': '+str(exc))
    report.update(audit_status='passed' if not ck.errors else 'failed',checks=ck.count,errors=ck.errors,
        auditor_sha256=sha(Path(__file__)),independent_adaptive_helper_sha256=ADAPTIVE_AUDITOR_SHA,hardware_io=False)
    output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (output/'REPORT.md').write_text('# Independent reserve audit\n\n'+f"{report['audit_status']}; {ck.count} checks; {len(ck.errors)} errors.\n\n"+
        f"Acquisition complete: {report.get('acquisition_complete')}; fresh weight quality: {report.get('fresh_weight_quality_pass')}; combined weight quality: {report.get('physical_weight_quality_pass')}.\n\n"+
        'The fresh reserve route is reconstructed from all 33 RESET confirmations before SET. Exactly13 new coefficients replace the old target chunk; immutable287 retain historical provenance. Raw ADC, frozen C and fixed decoder precede sum75, trained BN and digital LIF. Consistency PASS is separate from accuracy.\n',encoding='utf-8')
    print(json.dumps(dict(status=report['audit_status'],checks=ck.count,errors=ck.errors[:15],output=args.output)))
    return int(bool(ck.errors))


if __name__=='__main__':raise SystemExit(main())
