"""Offline endpoint screening, not physical capacity/range qualification."""
import argparse,json,hashlib,math,statistics as st
from pathlib import Path
from collections import defaultdict,Counter

HISTORY='results/cifar_feedback_remaining_reserves/58ca618b-c37f-454e-be25-db21d1d25f60'
HISTORY_MANIFEST='5fb5041cbdba148ff0a02182512a98ca0784fed074e2830d6fc06b230f933171'
JIT='results/cifar_jit_programming/ceedcbde-8443-4518-ae4e-8d901f2b3ddd'
JIT_MANIFEST='361f0f8759941e481281f2847eb9823fc4bb5db3c4fbcbc402007897c2e227aa'
ADC=.00031
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def require(v,s):
    if not v:raise ValueError(s)
def path(root,s):
    p=(root/s).resolve();require(not Path(s).is_absolute() and p.is_relative_to(root),'Path escape');return p
def channels(rows):
    return [sum(x['voltage_v'] for x in r['signals'])-len(r['signals'])*(r['zero_before']['voltage_v']+r['zero_after']['voltage_v'])/2 for r in rows]
def bound(root,relative,pin):
    d=path(root,relative);require(sha(d/'manifest.json')==pin,'Manifest changed');m=load(d/'manifest.json')
    for name,digest in m['result_sha256'].items():require(sha(path(d,name))==digest,'Result changed '+name)
    return d,m

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.repo).resolve();hd,hm=bound(root,HISTORY,HISTORY_MANIFEST);jd,jm=bound(root,JIT,JIT_MANIFEST)
    wrappers=load(hd/'completed_microtiles.json');observed=defaultdict(list);union=set();targets=[];provenance={}
    for w in wrappers:
        r=w['result'];origin=w['origin']
        while 'ancestry_original_origin' in origin:origin=origin['ancestry_original_origin']
        fp=path(root,origin['session']+'/'+origin['path']);require(sha(fp)==origin['sha256'],'Historical leaf changed')
        provenance[fp.relative_to(root).as_posix()]=origin['sha256']
        for pair,slot in zip(r['pairs'],r['effective_route']):
            require(pair['tap']==slot['tap_index'],'Pair/tap mismatch')
            for side in ('positive_cell','negative_cell'):union.add(tuple(slot[side]))
            targets.append(dict(filter_id=w['filter_id'],tap=slot['tap_index'],trained_weight=slot['trained_weight'],
                historical_active_cell=pair[pair['active_set_side']]))
            initial=channels(pair['initial_mvm'])
            if max(map(abs,initial))>2*ADC or not pair['programming_pulse_attempts']:continue
            side=pair['active_set_side'];idx=0 if side=='positive_cell' else 1
            levels=[channels(f['measured_rows'])[idx] for f in pair['post_feedback']]
            observed[tuple(pair[side])].append(dict(source='historical',filter_id=w['filter_id'],tap=pair['tap'],
                source_path=fp.relative_to(root).as_posix(),source_sha256=origin['sha256'],
                initial_channels_v=initial,mean_v=st.mean(levels),sample_sd_v=st.stdev(levels),
                SET_count=pair['programming_pulse_attempts'],stop=pair['stop'],
                readback_task_ids=[x['measured_rows'][idx]['signals'][0]['task_id'] for x in pair['post_feedback']]))
    historical_observed={c:list(v) for c,v in observed.items()}
    recent_fail=defaultdict(list);recent_gate=defaultdict(list)
    js=load(jd/'summary.json')
    for decision in js['decisions']:
        gate=decision['fresh_pair_gate']
        for leg,side in enumerate(('positive_cell','negative_cell')):
            record=dict(cycle=decision['cycle'],tap=decision['pair_tap'],mean_v=gate['row_mean_v'][leg],sample_sd_v=gate['row_sd_v'][leg])
            recent_gate[tuple(gate[side])].append(record)
            if abs(record['mean_v'])>2*ADC or record['sample_sd_v']>2*ADC:recent_fail[tuple(gate[side])].append(record)
        pair=decision.get('program')
        if pair and pair['programming_pulse_attempts']:
            initial=channels(pair['initial_mvm'])
            if max(map(abs,initial))>2*ADC:continue
            side=pair['active_set_side'];idx=0 if side=='positive_cell' else 1
            levels=[channels(f['measured_rows'])[idx] for f in pair['post_feedback']]
            observed[tuple(pair[side])].append(dict(source='JIT',cycle=decision['cycle'],filter_id=1,tap=pair['tap'],
                source_path=JIT+'/summary.json',source_sha256=jm['result_sha256']['summary.json'],initial_channels_v=initial,
                mean_v=st.mean(levels),sample_sd_v=st.stdev(levels),SET_count=pair['programming_pulse_attempts'],stop=pair['stop'],
                readback_task_ids=[x['measured_rows'][idx]['signals'][0]['task_id'] for x in pair['post_feedback']]))
    rows=[]
    for cell in sorted(union):
        evidence=observed[cell];maximum=max([x['mean_v'] for x in evidence] or [0.])
        lower=max([x['mean_v']-2*x['sample_sd_v']-2*ADC for x in evidence] or [0.])
        rows.append(dict(cell=list(cell),historically_assigned=True,historical_clean_SET_observations=len(historical_observed.get(cell,[])),
            all_clean_SET_observations=len(evidence),maximum_observed_mean_v=maximum,
            screening_amplitude_v=max(0.,lower),screening_formula='max_observations(mean - 2 sampleSD - 2ADC), clipped at0; not a confidence bound or continuous-range guarantee',
            active_candidate_allowed=maximum>2*ADC and lower>0 and not recent_fail[cell],
            active_exclusion_reasons=(['no clean-start SET observation'] if not evidence else ['no observed response above2ADC'] if maximum<=2*ADC else [])+
                (['recent JIT combined quiet gate failure; fresh confirmation required'] if recent_fail[cell] else []),
            inactive_candidate_allowed_only_after_fresh_joint_quiet=True,recent_JIT_gate_observations=recent_gate[cell],
            recent_JIT_gate_failures=recent_fail[cell],observations=evidence))
    require(len(union)==114 and len(targets)==300,'Unexpected coverage')
    mx={c:max(x['mean_v'] for x in v) for c,v in historical_observed.items()};wrms=math.sqrt(st.mean(t['trained_weight']**2 for t in targets))
    screening=[]
    for gain in (1.,.65,.5,.25,.1,.03):
        screening.append(dict(gain=gain,one_ADC_weight=ADC/gain,ADC_weight_NRMS=ADC/gain/wrms,
            targets_le_2ADC=sum(abs(t['trained_weight'])*gain<=2*ADC for t in targets),
            historical_current_active_endpoint_covers=sum(mx.get(tuple(t['historical_active_cell']),-1)>=gain*abs(t['trained_weight']) for t in targets),
            unknown_historical_active=sum(tuple(t['historical_active_cell']) not in mx for t in targets)))
    out=dict(schema='physical_endpoint_screening_v1',hardware_io=False,qualification=False,
        hardware_gain=.25,simulator_gain=.03,source_gains_are_not_physical_equivalence=True,
        source_manifest_sha256={HISTORY:HISTORY_MANIFEST,JIT:JIT_MANIFEST},historical_leaf_sha256=provenance,
        historical_clean_active_cells=len(historical_observed),historical_clean_above_2ADC=sum(v>2*ADC for v in mx.values()),
        historical_assigned_cells=114,fresh_JIT_cells=18,active_candidate_counts_by_WL={str(w):sum(r['active_candidate_allowed'] and r['cell'][0]==w for r in rows) for w in range(8)},
        gains=screening,targets=sorted(targets,key=lambda t:(t['filter_id'],t['tap'])),cells=rows,
        policy=dict(quiet_limit_v=.00062,max_reset_pulses_per_leg=55,reset_budget_never_restarts=True,
            maximum_candidate_pair_attempts_per_logical_tap=2,
            maximum_joint_repair_rounds=2,
            candidate_order='predeclare from target sign and observed amplitude; prefer globally unused compatible addresses, then smallest adequate screening amplitude, then WL/BL; inactive prefer unused historical zero-response addresses subject to joint quiet',
            uniqueness='Each physical address appears at most once in an acquired microtile. Fallback may not steal an already assigned address.',
            active_fit='abs(trained_weight)*backend_gain <= screening_amplitude_v is a historical hardware screen, not a forecast of successful programming; SIM has distinct physical model.',
            failure='If2 predeclared candidate attempts or remaining55RESET/109SET exhausted, save explicit failure; no hidden CPU/ancestry coefficient.',
            interpretability='Do not call zero-response cells dead or endpoint maxima physical bounds; no per-weight gain.'))
    dest=path(root,args.output);require(not dest.exists(),'Refuse overwrite');dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:out[k] for k in ('historical_clean_active_cells','historical_clean_above_2ADC','active_candidate_counts_by_WL','gains')},indent=2))

if __name__=='__main__':main()
