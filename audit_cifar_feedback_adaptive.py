"""Independent raw replay of adaptive CIFAR; no collector or device imports."""
from collections import Counter
import copy
import hashlib
from pathlib import Path
import json
import math
import statistics
import argparse
import subprocess
from configparser import ConfigParser

DIAGNOSTIC_AUDITOR_SHA='e7367990b6fc2a00f65e3bb0c4e4e5bf097b9f6b21a00700355edb84e0cbc70f'
if hashlib.sha256(Path(__file__).with_name('audit_cifar_feedback_diagnostic.py').read_bytes()).hexdigest()!=DIAGNOSTIC_AUDITOR_SHA:
    raise RuntimeError('Frozen independent diagnostic helper changed')
from audit_cifar_feedback_diagnostic import (
    CompactChecks,local,load,sha,verify_inputs,verify_manifest,audit_diagnostic_tile,aggregate_independent,
    pools_for,read_raw,Replay,reset_replay,pair_replay,ledger,multiply,fixed_calibration,
    fixed_effective_weights,replay_block,fixed_decode,fidelity,DIAGNOSTIC_PINS,PINS,PRIVATE,BOARD,audit_git_status,coverage)
from audit_cifar_memristor import matrix_select


def require(condition,message):
    if not condition:raise ValueError(message)


def trained_slots(routing):
    targets={}
    for filt in routing['filters']:
        fid=filt['filter_id'];weights={}
        for tile in filt['tiles']:
            for slot in tile['slots']:
                if slot['padding']:continue
                tap=slot['tap_index']
                require(tap not in weights,'Duplicate trained coefficient')
                weights[tap]=slot['trained_weight']
        require(set(weights)==set(range(75)),'Trained filter must contain exactly 75 coefficients')
        require(fid not in targets,'Duplicate trained filter')
        targets[fid]=weights
    require(set(targets)==set(range(4)),'Expected filters 0..3')
    return targets


def audit_coefficient_coverage(results,routing,complete):
    """Variable groups may partition taps; never duplicate, omit or replace them."""
    targets=trained_slots(routing);seen=Counter();ordered={f:[] for f in range(4)};identities=set()
    for result in results:
        if not result['technical_completed']:continue
        fid=result['filter_id'];identity=(fid,result['parent_tile_id'],result['microtile_id'])
        require(result['microtile_id'] in (0,1),'Invalid microtile index')
        parent=next((t for f in routing['filters'] if f['filter_id']==fid for t in f['tiles'] if t['tile_id']==result['parent_tile_id']),None)
        require(parent is not None,'Unknown original chunk')
        allowed={s['tap_index'] for s in parent['slots'] if not s['padding']}
        require(identity not in identities,'Duplicate measured tile identity');identities.add(identity)
        require(fid in targets,'Unknown filter')
        slots=result['effective_route'];require(bool(slots),'Empty measured microtile')
        positive=[];negative=[]
        for slot in slots:
            tap=slot['tap_index'];require(type(tap) is int and tap in targets[fid],'Invalid tap')
            require(tap in allowed,'Tap assigned to wrong original chunk')
            require(slot['trained_weight']==targets[fid][tap],'Trained weight was replaced')
            require(not slot.get('padding',False),'Padding cannot become a trained coefficient')
            seen[fid,tap]+=1;ordered[fid].append(tap)
            positive.append(tuple(slot['positive_cell']));negative.append(tuple(slot['negative_cell']))
        require(len(set(positive))==len(positive) and len(set(negative))==len(negative),'Address reused within microtile')
        require(set(positive).isdisjoint(negative),'Same physical address on both legs')
    require(all(n==1 for n in seen.values()),'Coefficient was measured/reused twice')
    for fid,taps in ordered.items():
        require(taps==list(range(len(taps))),'Measured taps are not an ordered prefix')
        if complete:require(taps==list(range(75)),'Completed filter omits coefficients')
    return dict(coefficients=len(seen),by_filter={str(f):len(taps) for f,taps in ordered.items()},
        measured_tiles=len(identities),exact_300_coverage=len(seen)==300)


def measurement_tail(active,fixture):
    """Independent count of readbacks, anchors, matrix probes and fresh CIFAR."""
    n=len(active);require(n>0,'Empty tail')
    masks=[[i] for i in range(n)]+[list(range(n)),list(range(0,n,2)),list(range(1,n,2))]
    masks += [[]]+[[i] for i in range(n)]+[[i,i+1] for i in range(0,n-1,2)]
    total=12*n+20*(n+1)+sum(3*(4+2*max(1,len(mask))) for mask in masks)
    for case in fixture['cases']:
        for spikes in case['spikes_t12x75']:
            total+=5*(4+2*max(1,int(sum(spikes[s['tap_index']] for s in active))))
    return total


def adaptive_budget(routing,fixture,import_first_three=False):
    chunks=[(f,t) for f in routing['filters'] for t in f['tiles']]
    if import_first_three:
        require([(f['filter_id'],t['tile_id']) for f,t in chunks[:3]]==[(0,0),(0,1),(0,2)],'Unexpected prefix identity')
        require(sum(sum(not s['padding'] for s in t['slots']) for _,t in chunks[:3])==42,'Prefix is not exactly 42 taps')
        chunks=chunks[3:]
    impacts=decisions=0
    for _,tile in chunks:
        active=[s for s in tile['slots'] if not s['padding']];n=len(active)
        tail=max([measurement_tail(active,fixture)]+
            [measurement_tail(active[:k],fixture)+measurement_tail(active[k:],fixture) for k in range(1,n)])
        # Two fresh ORIGINAL candidate pools, even when the remaining tail is small.
        impacts+=2*244*(2*n+6)+811*n+tail
        decisions+=2*55*(2*n+6)+109*n
    return dict(maximum_impacts=impacts,maximum_programming_pulses=decisions,
        maximum_programming_decisions=decisions,maximum_new_microtiles=2*len(chunks),
        new_trained_coefficients=258 if import_first_three else 300)


def verify_parent_prefix(repo,parent_session,parent_audit,expected_manifest_sha,expected_audit_sha,
                         expected_result_sha,fixture,routing,ck):
    """Replay the three reused tiles; all other attempts remain in parent totals."""
    directory=local(repo,parent_session);report_path=local(repo,parent_audit)
    require(sha(directory/'manifest.json')==expected_manifest_sha,'Parent manifest commitment mismatch')
    require(sha(report_path)==expected_audit_sha,'Parent audit commitment mismatch')
    report=load(report_path)
    require(report['audit_status']=='passed' and report['errors']==[],'Parent independent consistency audit failed')
    require(report['auditor_sha256']==DIAGNOSTIC_AUDITOR_SHA,'Unreviewed parent auditor')
    require(report['manifest_sha256']==expected_manifest_sha,'Parent audit/manifest binding mismatch')
    # The frozen verifier checks every result SHA and recorded Git source blob,
    # while preserving failed status and permitting no changes to old protocol.
    manifest,plan,provenance=verify_manifest(repo,directory,ck)
    require(manifest['result_sha256']==expected_result_sha,'Parent result dictionary differs')
    ck.equal(manifest['result_sha256'],expected_result_sha,'parent/all_result_commitments')
    ck.equal(report['session_id'],manifest['session_id'],'parent/audit_session')
    ck.equal(report['source'],manifest['source'],'parent/audit_backend')
    selected=[(f,t) for f in routing['filters'] for t in f['tiles']][:3]
    ck.equal([(f['filter_id'],t['tile_id']) for f,t in selected],[(0,0),(0,1),(0,2)],'parent/exact_first_three')
    results=[]
    for filt,tile in selected:
        folder=directory/f"tiles/f{filt['filter_id']}_t{tile['tile_id']}"
        result=audit_diagnostic_tile(folder,filt,tile,fixture,plan,manifest['source'],ck,manifest['session_id'])
        require(result['technical_completed'],'Imported parent tile was incomplete')
        # Original tile_id and every acquired partial remain unchanged.
        result.update(parent_tile_id=tile['tile_id'],microtile_id=0,
            imported_origin=dict(session=parent_session,session_id=manifest['session_id'],
                tile=f"tiles/f{filt['filter_id']}_t{tile['tile_id']}",manifest_sha256=expected_manifest_sha))
        results.append(result)
    coverage=audit_coefficient_coverage(results,routing,False)
    require(coverage['coefficients']==42 and coverage['by_filter']['0']==42,'Parent prefix must contain exactly 42 taps')
    selected_counts={key:sum(r['counters'][key] for r in results) for key in report['counters']}
    return results,dict(session=parent_session,session_id=manifest['session_id'],source=manifest['source'],
        acquisition_status=manifest['status'],manifest_sha256=expected_manifest_sha,audit_sha256=expected_audit_sha,
        imported_coefficients=42,imported_partial_points=1800,imported_tiles=3,
        selected_tile_counters=selected_counts,all_parent_attempt_counters=report['counters'],
        unimported_attempts_not_erased=True,provenance=provenance)


def aggregate_microtiles(results,fixture,routing,ck):
    """Normalize only in-memory grouping labels after exact coefficient coverage."""
    audit_coefficient_coverage(results,routing,True)
    normalized=[];virtual={'filters':[]}
    for fid in range(4):
        measured=[r for r in results if r['filter_id']==fid and r['technical_completed']]
        tiles=[]
        for index,result in enumerate(measured):
            # No mutation of imported/raw metadata or saved source artifacts.
            record={**result,'tile_id':index};normalized.append(record)
            slots=copy.deepcopy(result['effective_route'])
            for slot in slots:slot['padding']=False
            tiles.append(dict(tile_id=index,slots=slots))
        virtual['filters'].append(dict(filter_id=fid,tiles=tiles))
    return aggregate_independent(normalized,virtual,fixture,ck)


def reference_rank(rows):
    """Two-column singular rank from stable pairwise minors, not collector SVD."""
    a=math.fsum(r[0]*r[0] for r in rows);b=math.fsum(r[0]*r[1] for r in rows);d=math.fsum(r[1]*r[1] for r in rows)
    lam=(a+d+math.hypot(a-d,2*b))/2
    determinant=math.fsum((x[0]*y[1]-x[1]*y[0])**2 for i,x in enumerate(rows) for y in rows[i+1:])
    high=math.sqrt(lam);low=math.sqrt(max(0.,determinant/lam)) if lam else 0.
    tolerance=high*max(len(rows),2)*2**-52
    return int(high>tolerance)+int(low>tolerance)


def adaptive_matrix_select(train,selection):
    correction=matrix_select(train,selection)
    rank=reference_rank([r['reference_v'] for r in train])
    for candidate in correction['candidates']:
        if candidate['name'].startswith('full_') and rank<2:
            candidate.update(eligible=False,selection_pass=False,ineligible_reason='reference_design_rank_below_two')
    eligible=[c for c in correction['candidates'] if c['eligible']]
    chosen=min(eligible,key=lambda c:(c['selection_score'],c['name']!='identity',c['name'])) if eligible else correction['candidates'][0]
    correction.update(selected_model=chosen['name'],matrix=chosen['matrix'],selection_pass=chosen['selection_pass'],
        identity_fallback=not eligible,reference_design_rank=rank,full_matrix_identifiable_from_reference=rank==2)
    return correction


def audit_adaptive_matrix(directory,replay,active,anchors,meta,ck):
    n=len(active);h=[[statistics.mean(a[side]['singles'][i]['voltage_v']-a[side]['zero']['voltage_v'] for a in anchors)
                      for i in range(n)] for side in range(2)]
    masks={'train':[[i] for i in range(n)]+[list(range(n)),list(range(0,n,2)),list(range(1,n,2))],
           'selection':[[]]+[[i] for i in range(n)]+[[i,i+1] for i in range(0,n-1,2)]}
    phases={};ledger=[]
    for phase,groups in masks.items():
        observations=[]
        for probe_id,mask in enumerate(groups):
            measurements=[]
            for repeat in range(3):
                values,rows=replay_block(replay,active,[active[i] for i in mask],'matrix_'+phase,
                    dict(**meta,probe_id=probe_id,repeat=repeat,mask_slots=mask))
                measurements.append(values)
                ledger.append(dict(phase=phase,probe_id=probe_id,repeat=repeat,mask_slots=mask,
                                   measured_rows=rows,measured_voltage_v=values))
            observations.append(dict(probe_id=probe_id,mask_slots=mask,measured_repeats_v=measurements,
                measured_mean_v=[statistics.mean(v[i] for v in measurements) for i in range(2)],
                reference_v=[math.fsum(h[i][j] for j in mask) for i in range(2)]))
        phases[phase]=observations
    ck.equal([json.loads(line) for line in (directory/'matrix_probes.jsonl').read_text(encoding='utf-8').splitlines()],ledger,'matrix/probe_ledger')
    correction=adaptive_matrix_select(phases['train'],phases['selection'])
    correction.update(reference_h_v=h,observations=phases,
        max_singleton_change_v=max(abs(multiply(correction['matrix'],[h[0][i],h[1][i]])[j]-h[j][i]) for i in range(n) for j in range(2)))
    ck.equal(load(directory/'frozen_matrix.json'),correction,'matrix/independent_fit_selection',1e-9)
    return correction


def audit_microtile(directory,filt,parent,pending,micro,fixture,plan,source,ck,session_id):
    saved=load(directory/'tile.json');fid=filt['filter_id'];tid=2*parent['tile_id']+micro;meta=dict(filter_id=fid,tile_id=tid,parent_tile_id=parent['tile_id'],microtile_id=micro)
    active=copy.deepcopy(pending);pools=pools_for(parent)
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
    k=min(len(active),*(len(cells) for cells in chosen.values()))
    if k==0:
        ck.equal(saved['status'],'failed','reset/insufficient_status')
        ck.equal(saved['stop_reason'],'zero_quiet_pair_capacity','reset/insufficient_reason')
        ck.equal(replay.position,len(events),'reset/no_measurement_after_insufficient_candidates')
        return dict(**meta,technical_completed=False,physical_weight_quality_pass=False,counters=counters,
                    reset_candidates=candidates,partials=[],acquisition_stop='zero_quiet_pair_capacity')
    active=active[:k];chosen={side:cells[:k] for side,cells in chosen.items()}
    ck.equal(saved['assigned_taps'],[s['tap_index'] for s in active],'microtile/first_k_pending')
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
        pair_stop_counts=dict(Counter(p['stop'] for p in pairs)))



PROTOCOL='cifar-feedback-adaptive-two-microtiles-v1'
# Filled only after the independent runtime review is frozen.
ADAPTIVE_PINS={
    'cifar_feedback_adaptive.py':'a46eb7b94891d1adf48d10f126227e10ecfa696aba61a98f293199a8dcb9f2ba',
    'configs/cifar_feedback_adaptive.json':'c788a91a1dc51e4a9f6070354c9597249a45b4770decb07847030f683ef09663',
    'experiment.py':'83fd05070e09de9f296d143803be44c0ac07f0366fdffee2a6edd62f4b3f6e71'}


def verify_adaptive_manifest(repo,directory,ck):
    require(bool(ADAPTIVE_PINS),'Adaptive source review not frozen')
    m=load(directory/'manifest.json');git=m['invocation']['git']
    provenance=audit_git_status(git,m['source_sha256'],ck)
    ck.require(m['source'] in ('hardware','simulator'),'manifest/source')
    ck.equal(m['hardware_access'],m['source']=='hardware','manifest/hardware_access')
    for name,digest in m['result_sha256'].items():
        ck.equal(sha(local(directory,name)),digest,'result_sha/'+name)
    for required in ('protocol.json','board.ini','start_manifest.json','completed_microtiles.json'):
        require(required in m['result_sha256'],'Unbound required artifact: '+required)
    if m['status']=='completed':require('summary.json' in m['result_sha256'],'Unbound summary')
    for name,digest in m['source_sha256'].items():
        path=local(repo,name)
        if name in PRIVATE:
            ck.equal(digest,PRIVATE[name],'source/private_pin/'+name)
            if path.exists():ck.equal(sha(path),digest,'source/private_local/'+name)
        else:
            proc=subprocess.run(['git','-C',str(repo),'show',git['commit']+':'+name],capture_output=True)
            ck.equal(proc.returncode,0,'source/git_exists/'+name)
            ck.equal(hashlib.sha256(proc.stdout).hexdigest(),digest,'source/git_blob/'+name)
    for name,digest in {**PINS,**DIAGNOSTIC_PINS,**ADAPTIVE_PINS}.items():
        ck.equal(m['source_sha256'].get(name),digest,'source/reviewed_pin/'+name)
    plan=load(directory/'protocol.json')
    config=subprocess.run(['git','-C',str(repo),'show',git['commit']+':configs/cifar_feedback_adaptive.json'],capture_output=True,check=True).stdout
    require(plan==json.loads(config),'Protocol snapshot differs from committed configuration')
    ck.equal(m['protocol_id'],PROTOCOL,'protocol/manifest');ck.equal(plan['protocol_id'],PROTOCOL,'protocol/id')
    ck.equal(plan,dict(maximum_impacts=868466,maximum_programming_pulses=117180,
        maximum_programming_decisions=117180,maximum_microtiles_per_original_chunk=2,
        prefix42_remaining_maximum_impacts=756104,prefix42_remaining_maximum_programming_pulses=101382,
        full_matrix_requires_reference_rank_two=True,target_gain_v_per_weight={'hardware':1.,'simulator':.03}),
        'protocol/reviewed_contract')
    ck.equal(m['fixed_decoder_gain_v_per_weight'],1. if m['source']=='hardware' else .03,'manifest/fixed_gain')
    board=ConfigParser();board.read(directory/'board.ini',encoding='utf-8')
    for key,value in BOARD.items():ck.equal(board.getfloat('board',key),value,'INI/'+key)
    start=load(directory/'start_manifest.json')
    for key in ('session_id','source_sha256','protocol_id','source','invocation','parent_import'):
        ck.equal(m[key],start[key],'manifest/start_consistency/'+key)
    return m,plan,provenance


def validate_wrapper(wrapper,result,session,path,digest,reused,ck):
    ck.equal(wrapper,dict(filter_id=result['filter_id'],parent_tile_id=result['parent_tile_id'],
        microtile_id=result['microtile_id'],origin=dict(session=session,path=path,sha256=digest,
        reused=reused,new_measurement=not reused,acquisition_tile_id=result['tile_id'])), 'wrapper/origin')


def audit_session(repo,directory,fixture,routing,ck):
    m,plan,provenance=verify_adaptive_manifest(repo,directory,ck)
    imported=m.get('parent_import');results=[];historical=None;wrappers=load(directory/'completed_microtiles.json')
    allparents=[(f,t) for f in routing['filters'] for t in f['tiles']]
    if imported:
        binding=plan['parent_bindings'][m['source']]
        ck.equal(imported,binding,'parent/reviewed_binding')
        results,historical=verify_parent_prefix(repo,binding['session'],binding['audit_path'],
            binding['manifest_sha256'],binding['audit_sha256'],imported['result_sha256'],fixture,routing,ck)
        ck.equal(historical['source'],m['source'],'parent/same_backend')
        ck.equal(imported,dict(imported_prefix=['f0_t0','f0_t1','f0_t2'],imported_trained_coefficients=42,
            failed_parent_tile_contributes=False,other_parent_tiles_ignored=True,
            parent_total_impacts=historical['all_parent_attempt_counters']['impact_attempts'],
            parent_total_pulses=historical['all_parent_attempt_counters']['programming_pulse_attempts']), 'parent/accounting')
        for index,r in enumerate(results):
            path=f"tiles/f0_t{index}/tile.json"
            validate_wrapper(wrappers[index],r,binding['session'],path,imported['result_sha256'][path],True,ck)
            require(wrappers[index]['result']==load(local(repo,binding['session'])/path),'Imported result bytes/content altered')
    budget=adaptive_budget(routing,fixture,bool(imported))
    ck.equal(m,dict(maximum_impacts=budget['maximum_impacts'],maximum_programming_pulses=budget['maximum_programming_pulses'],
        maximum_microtiles=budget['maximum_new_microtiles'],trained_coefficients=budget['new_trained_coefficients']), 'manifest/budget')
    new=[];visited=set();stopped=False;position=len(results)
    for filt,parent in allparents[3 if imported else 0:]:
        pending=[s for s in parent['slots'] if not s['padding']]
        for micro in range(2):
            name=f"f{filt['filter_id']}_p{parent['tile_id']}_m{micro}";folder=directory/'tiles'/name
            if not folder.exists():
                stopped=True;break
            visited.add(name)
            for artifact in folder.iterdir():
                if artifact.is_file():ck.require('tiles/'+name+'/'+artifact.name in m['result_sha256'],'manifest/bound_microtile_artifact')
            r=audit_microtile(folder,filt,parent,pending,micro,fixture,plan,m['source'],ck,m['session_id']);new.append(r)
            if not r['technical_completed']:
                stopped=True;break
            results.append(r);path='tiles/'+name+'/tile.json'
            require(position<len(wrappers),'Completed microtile absent from ledger')
            validate_wrapper(wrappers[position],r,directory.relative_to(repo).as_posix(),path,m['result_sha256'][path],False,ck)
            require(wrappers[position]['result']==load(folder/'tile.json'),'Completed result differs from journal wrapper')
            position+=1;pending=pending[len(r['effective_route']):]
            if not pending:break
        if stopped or pending:break
    actual={p.name for p in (directory/'tiles').iterdir() if p.is_dir()} if (directory/'tiles').exists() else set()
    require(visited==actual,'Unexpected/unreplayed microtiles')
    require(position==len(wrappers),'Extra completed microtile wrappers')
    complete=sum(len(r['effective_route']) for r in results)==300
    coefficient_coverage=audit_coefficient_coverage(results,routing,complete)
    counts={k:sum(r['counters'][k] for r in new) for k in ('impact_attempts','programming_pulse_attempts','programming_decisions','programming_skipped_current')}
    ck.equal(m['new_counters'],counts,'manifest/new_counters')
    ck.equal(m['new_attempted_microtiles'],len(new),'manifest/new_attempted_microtiles')
    ck.equal(m['microtiles_completed'],len(results),'manifest/completed_microtiles')
    ck.equal(m['new_attempt_address_coverage'],coverage(new),'manifest/new_attempt_address_coverage')
    prior=historical['all_parent_attempt_counters'] if historical else dict(impact_attempts=0,programming_pulse_attempts=0)
    for field,key in (('cumulative_impacts_including_all_parent_attempts','impact_attempts'),('cumulative_pulses_including_all_parent_attempts','programming_pulse_attempts')):
        ck.equal(m[field],counts[key]+prior[key],'manifest/'+field)
    ck.require(counts['impact_attempts']<=budget['maximum_impacts'],'budget/impacts')
    ck.require(counts['programming_decisions']<=budget['maximum_programming_decisions'],'budget/decisions')
    ck.require(counts['programming_pulse_attempts']<=counts['programming_decisions'],'budget/pulses')
    trajectories=[];quality=[];physical=complete and all(r['physical_weight_quality_pass'] for r in results)
    if complete:
        trajectories,quality=aggregate_microtiles(results,fixture,routing,ck);saved=load(directory/'summary.json')
        ck.equal(saved['trajectories'],trajectories,'summary/all75_sum_then_BN_LIF_two_refs',1e-8)
        ck.equal(saved,dict(status='completed',technical_completed=True,acquisition_complete=True,
            decoded_conv_bn_lif_complete=True,physical_weight_quality_pass=physical,new_counters=counts,
            parent_import=imported,microtiles_completed=len(results),fixed_decoder_gain_v_per_weight=1. if m['source']=='hardware' else .03,
            fitted_alpha_used_for_decoding=False,actual_branch_uses_software_dot_product=False,
            full_network_accuracy_claim=False,address_coverage=coverage(results)), 'summary/flags_coverage')
        ck.equal(len(trajectories)*12,2400,'summary/2400_points')
        ck.equal(m['status'],'completed','manifest/completed')
    else:ck.equal(m['status'],'failed','manifest/partial_preserved')
    compact=[]
    for r in new:
        row={k:v for k,v in r.items() if k not in ('partials','reset_candidates','pairs','matrix')}
        if 'matrix' in r:row['matrix']={k:r['matrix'][k] for k in ('matrix','selected_model','selection_pass','reference_design_rank')}
        compact.append(row)
    return dict(session_id=m['session_id'],source=m['source'],protocol_id=PROTOCOL,manifest_sha256=sha(directory/'manifest.json'),
        acquisition_complete=complete,acquisition_status=m['status'],physical_weight_quality_pass=physical,
        new_counters=counts,parent_import_audit=historical,provenance=provenance,coefficient_coverage=coefficient_coverage,
        partial_points=sum(len(r['partials']) for r in results),full_time_outputs=len(trajectories)*12,
        new_microtiles=compact,quality=quality,address_coverage=coverage(results),
        limitations=['Audit PASS establishes consistency, not weight fidelity or network accuracy.',
            'Imported measurements retain their historical time and original acquisition identity.',
            'All parent attempts, including the failed unimported chunk, are counted separately.',
            'Fixed gains HW=1 and SIM=0.03 are provisional; fitted alpha never replaces them.',
            'Sequential microtiles do not establish 300 simultaneously stored weights.'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',default='.')
    parser.add_argument('--session',required=True);parser.add_argument('--output',required=True);args=parser.parse_args(argv)
    repo=Path(args.repo).resolve();directory=local(repo,args.session);output=local(repo,args.output)
    require(output!=directory and not output.is_relative_to(directory),'Audit must not modify acquired data')
    ck=CompactChecks();report={}
    try:
        fixture,routing,golden=verify_inputs(repo,ck);report=audit_session(repo,directory,fixture,routing,ck)
        report['golden_validation']=golden
    except Exception as exc:ck.require(False,'exception/'+type(exc).__name__+': '+str(exc))
    report.update(audit_status='passed' if not ck.errors else 'failed',checks=ck.count,errors=ck.errors,
        auditor_sha256=sha(Path(__file__)),independent_diagnostic_helper_sha256=DIAGNOSTIC_AUDITOR_SHA,hardware_io=False)
    output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (output/'REPORT.md').write_text('# Independent adaptive CIFAR audit\n\n'+
        f"{report['audit_status']}; {ck.count} checks; {len(ck.errors)} errors.\n\n"+
        f"Acquisition complete: {report.get('acquisition_complete')}; physical weight quality: {report.get('physical_weight_quality_pass')}.\n\n"+
        'Raw replay verifies deterministic routing, independent calibration, fixed-gain actual contributions and all-75 summation before trained BN/LIF. Imported measurements remain historical. Consistency PASS is not an accuracy qualification.\n',encoding='utf-8')
    print(json.dumps(dict(status=report['audit_status'],checks=ck.count,errors=ck.errors[:15],output=args.output)))
    return int(bool(ck.errors))


if __name__=='__main__':raise SystemExit(main())
