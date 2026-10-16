"""Pure offline scientific comparison/map of an audited JIT-C9 session."""
import argparse
from collections import Counter
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics as st

AUDITOR='266faf7a06f2dac6dcaa2d2418426d3b6de2ba3de7ea23cdee22f3cb8399fac4'
HISTORY='results/cifar_feedback_remaining_reserves/58ca618b-c37f-454e-be25-db21d1d25f60'
TILE='tiles/f1_p0_reserve_C/tile.json'
TILE_SHA='7b262ed74cfebaa8a95622a01f7bf20165b9443cd9cbea84785355f71aeb2b20'
SIM='results/cifar_jit_programming/6ebac641-2151-42ba-8b1e-48de43f282e1'
SIM_AUDIT='results/cifar_feedback_reviews/jit_simulator_20261016_v2/audit.json'
COLORS={'historical':'#cbd5e1','unobserved':'#f8fafc','gate_failed':'#e87979',
    'pair_target_missed':'#f7ba63','pair_changed_later':'#c4a1ed','pair_band_observed':'#62bfd2'}


def require(v,s):
    if not v:raise ValueError(s)


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def ledger(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rms(v):return math.sqrt(st.mean(x*x for x in v))
def local(root,v):
    p=(root/v).resolve();require(not Path(v).is_absolute() and p.is_relative_to(root),'Outside repository');return p


def channels(rows):
    return [sum(v['voltage_v'] for v in row['signals'])-len(row['signals'])*(row['zero_before']['voltage_v']+row['zero_after']['voltage_v'])/2 for row in rows]


def verify(repo,session,audit):
    d=local(repo,session);a=load(local(repo,audit));m=load(d/'manifest.json')
    require(a['audit_status']=='passed' and not a['errors'] and a['auditor_sha256']==AUDITOR,'Unverified independent audit')
    require(a['manifest_sha256']==sha(d/'manifest.json') and a['session_id']==m['session_id'] and a['source']==m['source'],'Audit binding mismatch')
    require(m['status']=='completed' and m['technical_completed'],'Not complete; partial study needs separate analysis')
    for n,h in m['result_sha256'].items():require(sha(local(d,n))==h,'Changed result '+n)
    return d,m,a


def compact(d,m):
    s=load(d/'summary.json');maps=ledger(d/'disturbance_maps.jsonl');sentinels=ledger(d/'sentinel.jsonl')
    gain=1. if m['source']=='hardware' else .03;route=s['route'];weights=[x['trained_weight'] for x in route]
    require(len(route)==9 and len(maps)==18 and len(s['decisions'])==18,'Incomplete C9')
    cycles=[];pairs=[];cells=[]
    for cycle in range(2):
        ds=[x for x in s['decisions'] if x['cycle']==cycle];ms=[x for x in maps if x['cycle']==cycle]
        final=ms[-1]['observed'];own_values=[];initial_values=[];retention_values=[]
        for i,(slot,decision) in enumerate(zip(route,ds)):
            program=decision.get('program');gate=decision['fresh_pair_gate'];ret=decision['retention']
            confirmation=program['post_feedback'] if program else decision['no_SET_confirmation']['samples']
            ownchannels=[st.mean(channels(r['measured_rows'])[leg] for r in confirmation) for leg in range(2)]
            initialchannels=channels(program['initial_mvm']) if program else gate['row_mean_v']
            own=ownchannels[0]-ownchannels[1];initial=program['initial_signed_v'] if program else gate['signed_mean_v']
            target=gain*slot['trained_weight'];tol=max(.000155,.05*abs(target));initial_values.append(initial)
            own_values.append(own);retention_values.append(ret['signed_mean_v'])
            later=[dict(after_pair=j,signed_delta_v=ms[j]['observed'][i]['signed_mean_v']-ms[i]['observed'][i]['signed_mean_v'],
                channel_delta_v=[ms[j]['observed'][i]['row_mean_v'][leg]-ms[i]['observed'][i]['row_mean_v'][leg] for leg in range(2)]) for j in range(i+1,9)]
            stages=[own,ret['signed_mean_v'],final[i]['signed_mean_v']]
            pair_band=program is not None and all(abs(x-target)<=tol for x in stages)
            category=('gate_failed' if not gate['both_legs_quiet'] else 'pair_target_missed' if abs(own-target)>tol
                else 'pair_changed_later' if not pair_band else 'pair_band_observed')
            pairs.append(dict(cycle=cycle,tap=i,target_v=target,stop_tolerance_v=tol,gate_pass=gate['both_legs_quiet'],
                initial_v=initial,own_v=own,retention_v=stages[1],final_v=stages[2],
                own_error_v=own-target,final_error_v=stages[2]-target,
                program_stop=program['stop'] if program else 'no_SET_gate_failed',
                SET_count=program['programming_pulse_attempts'] if program else 0,
                selected_SET_side=program['active_set_side'] if program else None,
                observed_pair_stop_band_all_three_stages=pair_band,category=category,later_changes=later))
            for leg,side in enumerate(('positive_cell','negative_cell')):
                cells.append(dict(cycle=cycle,tap=i,side=side,cell=slot[side],pair_category=category,
                    independently_reset_mean_v=decision[('positive' if leg==0 else 'negative')+'_reset']['baseline_mvm_mean_v'],
                    independently_reset_sd_v=decision[('positive' if leg==0 else 'negative')+'_reset']['baseline_mvm_sd_v'],
                    combined_gate_mean_v=gate['row_mean_v'][leg],combined_gate_sd_v=gate['row_sd_v'][leg],
                    combined_leg_quiet=abs(gate['row_mean_v'][leg])<=.00062 and gate['row_sd_v'][leg]<=.00062,
                    initial_program_channel_v=initialchannels[leg],gate_to_initial_channel_delta_v=initialchannels[leg]-gate['row_mean_v'][leg],
                    own_mean_v=ownchannels[leg],retention_mean_v=ret['row_mean_v'][leg],retention_sd_v=ret['row_sd_v'][leg],
                    final_mean_v=final[i]['row_mean_v'][leg],final_sd_v=final[i]['row_sd_v'][leg],
                    SET_count=program['programming_pulse_attempts'] if program and program['active_set_side']==side else 0,
                    later_max_abs_change_v=max([abs(x['channel_delta_v'][leg]) for x in later] or [0.]),
                    reason_scope='Observed channels/pair outcome; not an isolated cell conductance range or physical failure diagnosis'))
        final_values=[x['signed_mean_v'] for x in final]
        cycles.append(dict(cycle=cycle,own_nrmse=rms([v/gain-w for v,w in zip(own_values,weights)])/rms(weights),
            retention_nrmse=rms([v/gain-w for v,w in zip(retention_values,weights)])/rms(weights),
            final_nrmse=rms([v/gain-w for v,w in zip(final_values,weights)])/rms(weights),
            own_v=own_values,initial_v=initial_values,retention_v=retention_values,final_v=final_values,
            quiet_gate_failures=sum(not x['fresh_pair_gate']['both_legs_quiet'] for x in ds),
            stop_counts=dict(Counter(x['program']['stop'] if 'program' in x else 'no_SET_gate_failed' for x in ds)),
            impulses=sum(x['impact_attempts'] for x in ds),writes=sum(x['programming_pulse_attempts'] for x in ds)))
    return dict(source=m['source'],gain=gain,route=route,cycles=cycles,pairs=pairs,cells=cells,
        final_cycle_difference_v=[b-a for a,b in zip(cycles[0]['final_v'],cycles[1]['final_v'])],
        counters={k:s[k] for k in ('impact_attempts','programming_pulse_attempts','programming_decisions','programming_skipped_current')},
        quality={k:s[k] for k in ('final_state_fidelity_pass','programming_fidelity_pass','programming_repeatability_qualified')},
        sentinel_rows=sentinels,scope='Compact scientific arithmetic bound to independent raw audit; no new raw replay')


def historical(repo,route):
    d=local(repo,HISTORY);m=load(d/'manifest.json');p=d/TILE
    require(sha(d/'manifest.json')=='5fb5041cbdba148ff0a02182512a98ca0784fed074e2830d6fc06b230f933171','Historical manifest changed')
    require(sha(p)==TILE_SHA and m['result_sha256'][TILE]==TILE_SHA,'Historical C9 changed')
    t=load(p);require(t['effective_route']==route,'Historical route is not matched')
    wrappers=load(d/'completed_microtiles.json')
    require(sha(d/'completed_microtiles.json')==m['result_sha256']['completed_microtiles.json'],'Historical union changed')
    union=sorted({tuple(s[side]) for w in wrappers for s in w['result']['effective_route'] for side in ('positive_cell','negative_cell')})
    require(len(union)==114,'Unexpected historical union')
    weights=[r['trained_weight'] for r in route]
    own=[p['post_mean_signed_v'] for p in t['pairs']];final=t['physical_fidelity']['raw_measured_weight']
    return dict(session=HISTORY,source='hardware',tile_sha256=TILE_SHA,union=[list(x) for x in union],
        initial_v=[p['initial_signed_v'] for p in t['pairs']],own_v=own,final_v=final,
        own_nrmse=rms([a-b for a,b in zip(own,weights)])/rms(weights),
        final_nrmse=rms([a-b for a,b in zip(final,weights)])/rms(weights),
        pairs=[dict(tap=p['tap'],stop=p['stop'],SET_count=p['programming_pulse_attempts'],
                    initial_channels_v=channels(p['initial_mvm'])) for p in t['pairs']],
        comparison_scope='Same targets and addresses, historical temporal comparison; not randomized matched control')


def svg(report):
    hist={tuple(x) for x in report['history']['union']};fresh={tuple(s[k]):i for i,s in enumerate(report['current']['route']) for k in ('positive_cell','negative_cell')}
    cats={}
    priority=['gate_failed','pair_target_missed','pair_changed_later','pair_band_observed']
    for cell,tap in fresh.items():
        observed={p['category'] for p in report['current']['pairs'] if p['tap']==tap}
        cats[cell]=next(c for c in priority if c in observed)
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="710" viewBox="0 0 1500 710">',
        '<rect width="1500" height="710" fill="white"/>',
        '<g font-family="Arial,sans-serif" fill="#172033">',
        '<text x="35" y="37" font-size="24">JIT-C9: 18 fresh addresses / 114 historical assigned</text>',
        '<text x="35" y="65" font-size="16">'+escape(report['current']['source'])+' | Pair observations in two cycles; no cell-wide qualification</text>']
    ox=86;oy=110;cw=42;ch=43
    for b in range(32):parts.append(f'<text x="{ox+b*cw+18}" y="98" text-anchor="middle" font-size="12">{b}</text>')
    parts.append('<text x="40" y="97" font-size="12">WL / BL</text>')
    for w in range(8):
        parts.append(f'<text x="40" y="{oy+w*ch+25}" font-size="16">{w}</text>')
        for b in range(32):
            cell=(w,b);cat=cats.get(cell,'historical' if cell in hist else 'unobserved');label=str(fresh[cell]) if cell in fresh else ''
            parts.append(f'<rect x="{ox+b*cw}" y="{oy+w*ch}" width="39" height="40" fill="{COLORS[cat]}" stroke="#64748b" stroke-width="{2 if cell in fresh else .4}"><title>WL{w}/BL{b}: {cat}; tap {label}</title></rect>')
            if label:parts.append(f'<text x="{ox+b*cw+19}" y="{oy+w*ch+25}" text-anchor="middle" font-size="15">{label}</text>')
    legends=[('gate_failed','Fresh combined quiet gate failed in at least one cycle'),('pair_target_missed','Pair missed stop band immediately after own SET in at least one cycle'),
        ('pair_changed_later','Own pair met stop band, retention/final did not'),('pair_band_observed','Pair mean within stop band at own/retention/final in both cycles; not qualification'),
        ('historical','Historical assigned; no fresh JIT-C9 observation'),('unobserved','Outside historical assigned union; no JIT evidence')]
    for i,(cat,label) in enumerate(legends):
        y=486+i*29;parts.append(f'<rect x="40" y="{y-16}" width="20" height="20" fill="{COLORS[cat]}" stroke="#64748b"/><text x="72" y="{y}" font-size="15">{escape(label)}</text>')
    parts.append('<text x="40" y="682" font-size="14">Numbers inside fresh cells are logical tap indices. Both legs share pair-outcome color; see per-cell measured evidence in JSON.</text></g></svg>')
    return '\n'.join(parts)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--session',required=True)
    p.add_argument('--audit',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    repo=Path(args.repo).resolve();d,m,a=verify(repo,args.session,args.audit);current=compact(d,m);history=historical(repo,current['route'])
    sd,sm,sa=verify(repo,SIM,SIM_AUDIT);sim=compact(sd,sm)
    sim.update(session=SIM,manifest_sha256=sha(sd/'manifest.json'),audit=SIM_AUDIT,audit_sha256=sha(local(repo,SIM_AUDIT)))
    report=dict(session=args.session,manifest_sha256=sha(d/'manifest.json'),audit=args.audit,audit_sha256=sha(local(repo,args.audit)),
        script_sha256=sha(Path(__file__)),hardware_io=False,current=current,history=history,
        native_SIM_comparison=sim,comparison_same_backend=m['source']=='hardware',
        historical_assigned=114,fresh_addresses=18,historical_without_fresh_C9=96,
        caveats=['Same order twice is not randomized causal evidence.','Quiet is not programmability.',
                 'Pair stop-band color is not independent cell qualification.','All300/Conv/BN/LIF are absent from this pilot.'])
    out=local(repo,args.output);out.mkdir(parents=True,exist_ok=True)
    for n in ('JIT_SCIENCE.json','JIT_FRESH_VS_HISTORICAL.svg'):require(not (out/n).exists(),'Refuse overwrite')
    (out/'JIT_SCIENCE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'JIT_FRESH_VS_HISTORICAL.svg').write_text(svg(report),encoding='utf-8')
    print(json.dumps(dict(output=args.output,cycles=current['cycles'],history={k:history[k] for k in ('own_nrmse','final_nrmse')},quality=current['quality']),indent=2))


if __name__=='__main__':main()
