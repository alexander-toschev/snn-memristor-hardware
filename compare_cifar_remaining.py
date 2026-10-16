"""Offline final finite-reserve comparison; draft pending actual B/C evidence.

Pinned pure comparison code only. No collector/auditor/driver imports or Git calls.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
from types import SimpleNamespace

A_HELPER_SHA='4c86e24491a28c3f018902c85003b493c8eac54c9db100b41c75dfa26b72cfb4'
AUDITOR_SHA='4f2f73eb96fd286aabde4e0513b7aac2ef96e57ef873ce36573a73a29a3b0525'
PROTOCOL='cifar-feedback-remaining-reserves-v1'
ALL={(f,t) for f in range(4) for t in range(75)}
BC={(f,t) for f,n in ((2,11),(1,9)) for t in range(n)}
A13={(3,t) for t in range(52,65)}
RESERVES={
 'A':[(6,2),(6,11),(7,15)],
 'B':[(4,7),(5,6),(5,15)],
 'C':[(2,3),(3,12),(3,13),(3,31)]}
STAGES={'B':(2,11),'C':(1,9)}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def pure_helpers(repo):
    path=repo/'compare_cifar_reserve.py'
    require(sha(path)==A_HELPER_SHA,'Reviewed A comparator changed')
    ns={'__name__':'_reviewed_A_comparison','__file__':str(path)}
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),ns)
    a=SimpleNamespace(**ns)
    return a,a.helper(repo)


def keys(wrappers,a):return set(a.indexed_weights(wrappers))


def partition(wrappers,a):
    """Disjoint provenance classes; partial coverage is never padded to300."""
    groups={'original267':[],'reserve_A13':[],'reserve_BC20':[]}
    for w in wrappers:
        kk=keys([w],a)
        matches=[name for name,domain in [('original267',ALL-BC-A13),('reserve_A13',A13),('reserve_BC20',BC)] if kk<=domain]
        require(len(matches)==1 and kk,'Mixed/empty provenance chunk')
        groups[matches[0]].append(w)
    allkeys=[k for group in groups.values() for w in group for k in keys([w],a)]
    require(len(allkeys)==len(set(allkeys)),'Duplicate coefficient across provenance classes')
    return groups


def dispositions(stage,tile,complete):
    candidates={tuple(c['cell']):c for c in tile['reset_candidates']}
    require(len(candidates)==len(tile['reset_candidates']),'Duplicate RESET address')
    route=tile.get('effective_route',[])
    records=[]
    for cell in RESERVES[stage]:
        candidate=candidates.get(cell)
        slots=[s for s in route if list(cell) in (s['positive_cell'],s['negative_cell'])]
        if candidate is None:status='technical_unassessed'
        elif not candidate['eligible']:status='not_freshly_quiet'
        elif complete and slots:status='freshly_quiet_assigned_and_measured'
        elif not complete:status='freshly_quiet_route_incomplete'
        else:status='freshly_quiet_not_assigned'
        count=sum(p['programming_pulse_attempts'] for p in tile.get('pairs',[])
                  if p.get('active_set_side') in ('positive_cell','negative_cell') and p[p['active_set_side']]==list(cell))
        records.append(dict(stage=stage,cell=list(cell),status=status,assigned_taps=[s['tap_index'] for s in slots],
            actual_cifar_measured=bool(complete and slots),SET_pulses=count,
            exact_weight_claim=False,screen=None if candidate is None else {k:candidate.get(k) for k in
            ('eligible','reset_stop','baseline_mvm_mean_v','baseline_mvm_sd_v','mode10_quiet_limit_v')}))
    return records


def sum_counts(wrappers,h):
    return {k:sum(w['result'][k] for w in wrappers) for k in h.COUNTERS}


def verify(repo,session,audit_path,source,indices,a,h):
    folder=a.local(repo,session);m=a.load(folder/'manifest.json');ap=a.local(repo,audit_path);audit=a.load(ap)
    require(audit['audit_status']=='passed' and not audit['errors'] and audit['checks']>0,'B/C audit not PASS')
    require(audit['auditor_sha256']==AUDITOR_SHA and not audit.get('synthetic_fixture',False),'Unreviewed/synthetic audit')
    require(audit['manifest_sha256']==sha(folder/'manifest.json'),'Audit manifest binding mismatch')
    require(m['protocol_id']==audit['protocol_id']==PROTOCOL,'Unknown protocol')
    require(m['source']==audit['source']==source and m['session_id']==audit['session_id'],'Backend/session identity mismatch')
    require(m['hardware_access'] is (source=='hardware'),'Hardware label mismatch')
    gain=1.0 if source=='hardware' else .03
    require(m['fixed_decoder_gain_v_per_weight']==gain,'Fixed gain mismatch')
    require(m['source_sha256'][h.FIXTURE]==h.FIXTURE_SHA,'Fixture mismatch')
    complete=audit['acquisition_complete'];require(type(complete) is bool,'Invalid completion flag')
    require(m['status']==('completed' if complete else 'failed'),'Status inconsistent')
    if complete:require(m.get('acquisition_complete') is True,'Completion requires explicit true')
    else:
        require('acquisition_complete' not in m or m['acquisition_complete'] is False,'Failed completion flag inconsistent')
        require(not (folder/'summary.json').exists(),'Partial route contains full summary')
    new=h.counters(m['new_counters']);require(new==audit['new_counters'],'New counters inconsistent')
    plan=a.bound_json(folder,m,'protocol.json');parent=m['parent_import']
    binding=plan['parent_bindings'][source]
    require(all(parent.get(k)==v for k,v in binding.items()),'Parent differs from frozen protocol')
    require(binding['choice']=='completed_A' and not binding.get('failed_A'),'This reviewed final comparator requires actual completed A; fallback needs separate review')
    parent_folder=a.local(repo,parent['session']);pm=a.load(parent_folder/'manifest.json')
    require(sha(parent_folder/'manifest.json')==parent['manifest_sha256'],'Parent manifest changed')
    require(sha(a.local(repo,parent['audit_path']))==parent['audit_sha256'],'Parent audit changed')
    verified_a=a.verify_session(repo,parent['session'],parent['audit_path'],source,indices,h)
    require(parent['result_sha256']==pm['result_sha256'] and parent['verified_ancestry']==pm['parent_import'],'Ancestry metadata changed')
    require(parent['imported_trained_coefficients']==280,'Expected280 imported')
    originals=a.bound_json(parent_folder,pm,'completed_microtiles.json')
    imported=a.bound_json(folder,m,'imported_microtiles.json')
    require(keys(imported,a)==ALL-BC,'Imported280 not exact complement ofB/C')
    for w in imported:
        o=w['origin'];i=o['wrapper_index'];old=originals[i]
        require(o['session']==parent['session'] and o['path']=='completed_microtiles.json' and
                o['sha256']==pm['result_sha256']['completed_microtiles.json'],'Wrapper origin binding changed')
        require(o['ancestry_original_origin']==old['origin'] and w['result']==old['result'],'Historical result/origin changed')
        require(all(w[k]==old[k] for k in ('filter_id','parent_tile_id','microtile_id')),'Historical wrapper identity changed')
    account=audit['parent_accounting']
    for noun,counter in [('impacts','impact_attempts'),('pulses','programming_pulse_attempts')]:
        parent_total=pm['cumulative_'+noun+'_including_all_ancestry']
        require(parent['parent_cumulative_'+noun]==account['parent_cumulative_'+noun]==parent_total,'Parent cumulative mismatch')
        require(m['cumulative_'+noun+'_including_all_ancestry']==parent_total+new[counter],'Ancestry double count/omission')
    require(account['historical_attempt_ledger']==parent['historical_attempt_ledger'],'Attempt ledger mismatch')
    require(len(account['historical_attempt_ledger'])==1 and account['historical_attempt_ledger'][0]['session']==parent['session'],'Unexpected attempt ancestry')
    wrappers=list(imported);blocks={};reserve_records=[];fresh=[]
    require(len(audit['blocks'])==2,'Both bounded block attempts must be audited')
    for stage,(fid,n) in STAGES.items():
        path=f'tiles/f{fid}_p0_reserve_{stage}/tile.json';tile=a.bound_json(folder,m,path)
        match=[b for b in audit['blocks'] if b['filter_id']==fid]
        require(len(match)==1,'Missing/duplicate block audit');b=match[0]
        done=tile.get('technical_completed',False)
        require(type(done) is bool and done is b['technical_completed'],'Block completion differs')
        require({k:tile[k] for k in h.COUNTERS}==b['counters'],'Block counters differ')
        records=dispositions(stage,tile,done);reserve_records.extend(records)
        block=dict(acquisition_complete=done,counters=b['counters'],gate=b.get('gate'),
                   reserve_dispositions=records,physical_weight_quality_pass=tile.get('physical_weight_quality_pass',False))
        if done:
            w=dict(filter_id=fid,parent_tile_id=0,microtile_id=0,result=tile,
                   origin=dict(session=session,path=path,sha256=m['result_sha256'][path],reused=False,new_measurement=True,acquisition_tile_id=0))
            require(keys([w],a)=={(fid,t) for t in range(n)},'Incorrect fresh block taps')
            fresh.append(w);wrappers.append(w)
            partial=a.partial_metrics(tile,h);partial['scope']=f'fresh_{stage}_{n}_only_before_BN_LIF'
            block.update(weights=a.weight_stats([w],h),partials=partial,physical_fidelity=tile['physical_fidelity'],
                final_physical_fidelity=tile['final_physical_fidelity'],pair_stop_counts=dict(Counter(p['stop'] for p in tile['pairs'])),
                matrix={k:tile['matrix'].get(k) for k in ('matrix','selected_model','selection_pass','reference_design_rank')})
        else:require(not tile['partials'] and not tile['pairs'],'Failed route contains measurements/SET pairs')
        blocks[stage]=block
    for k in h.COUNTERS:require(sum(b['counters'][k] for b in blocks.values())==new[k],'Block total differs')
    require(complete==all(b['acquisition_complete'] for b in blocks.values()),'Full completion differs from blocks')
    if fresh:require(a.bound_json(folder,m,'completed_microtiles.json')==wrappers,'Completed checkpoint changed/fallback retained')
    else:require(not (folder/'completed_microtiles.json').exists(),'No completed new block but checkpoint present')
    groups=partition(wrappers,a);require(keys(groups['original267'],a)==ALL-BC-A13 and keys(groups['reserve_A13'],a)==A13,'A/original split lost provenance')
    for row in verified_a['provenance']['reserve_dispositions']:
        reserve_records.insert(0,dict(stage='A',cell=row['cell'],status=row['disposition'],assigned_taps=row['assigned_taps'],
            actual_cifar_measured=row['disposition']=='freshly_quiet_assigned_and_measured',SET_pulses=row['set_pulses'],
            exact_weight_claim=False,screen=row['reset_evidence']))
    require(len(reserve_records)==10 and len({tuple(r['cell']) for r in reserve_records})==10,'Expected ten fixed reserves')
    union=a.address_sets(wrappers);old_union=a.address_sets(originals)
    now={tuple(c) for c in union['assigned']['cells']};before={tuple(c) for c in old_union['assigned']['cells']}
    declared_new=len(keys(fresh,a)) if fresh else 0
    require(audit['new_partial_points']==600*len(fresh) and audit['full_time_outputs']==(2400 if complete else 0),'Audit output completeness differs')
    p=dict(session=session,session_id=m['session_id'],source=source,source_commit=m['invocation']['git']['commit'],
        manifest_sha256=sha(folder/'manifest.json'),audit_path=audit_path,audit_sha256=sha(ap),auditor_sha256=AUDITOR_SHA,
        result_sha256=m['result_sha256'],fixed_gain=gain,acquisition_complete=complete,new_counters=new,
        parent_A=verified_a['provenance'],historical_attempt_ledger=account['historical_attempt_ledger'],
        cumulative_impacts=m['cumulative_impacts_including_all_ancestry'],cumulative_pulses=m['cumulative_pulses_including_all_ancestry'],
        new_BC_coefficients=declared_new,historical_A_coefficients=13,original_coefficients=267,
        covered_coefficients=len(keys(wrappers,a)),full_time_outputs=2400 if complete else 0,
        blocks=blocks,ten_reserve_dispositions=sorted(reserve_records,key=lambda r:(r['stage'],r['cell'])),
        measured_union_scope='retained280 plus completed new blocks; not final300 if incomplete',contributing_address_union=union,
        assigned_added_vs_A=[list(c) for c in sorted(now-before)],assigned_displaced_vs_A=[list(c) for c in sorted(before-now)],
        contributing_counters=sum_counts(wrappers,h),capacity_failures=audit['capacity_failures'],
        weight_metrics={name:a.weight_stats(rows,h) if rows else None for name,rows in groups.items()})
    if not complete:return dict(provenance=p)
    require(keys(wrappers,a)==ALL and declared_new==20,'False300 completion')
    summary=a.bound_json(folder,m,'summary.json')
    require(summary['parent_import']==parent and summary['new_counters']==new,'Summary history/counts differ')
    require(summary['new_trained_coefficients']==20 and summary['imported_trained_coefficients']==280,'Summary contribution counts differ')
    # B/C summary does not repeat the gain; manifest and every contributing calibration do.
    require(summary.get('fixed_decoder_gain_v_per_weight',gain)==gain,'Summary fixed gain changed')
    for w in wrappers:
        cal=w['result']['calibration']
        require(cal['decoder_gain_v_per_weight']==gain and cal['decoder_gain_is_fixed_before_acquisition'] is True,'Contributing fixed gain changed')
    actual,effective=h.indexed_summary({**summary,'microtiles_completed':len(wrappers)},indices)
    p.update(summary_sha256=sha(folder/'summary.json'),physical_weight_quality_pass=summary['physical_weight_quality_pass'],
             final33_weight_metrics=a.weight_stats(groups['reserve_A13']+groups['reserve_BC20'],h),all300_weight_metrics=a.weight_stats(wrappers,h))
    return dict(actual=actual,effective=effective,provenance=p)


def compare(sim,hw,trained,h):
    complete=sim['provenance']['acquisition_complete'] and hw['provenance']['acquisition_complete']
    result=dict(schema='final-finite-reserve-comparison-draft-v1',hardware_io=False,complete=complete,
        sessions=dict(simulator=sim['provenance'],hardware=hw['provenance']),
        comparisons=h.build_comparison(sim,hw,trained) if complete else None,
        interpretation='20 fresh B/C +13 prior A +267 original; acquisition, weight fidelity and finite reserve coverage are separate.')
    result['fresh_block_comparison']={}
    for stage in STAGES:
        s=sim['provenance']['blocks'][stage];w=hw['provenance']['blocks'][stage]
        if s['acquisition_complete'] and w['acquisition_complete']:
            ss=s['partials']['aligned'];ww=w['partials']['aligned']
            require([(x['case'],x['repeat'],x['t']) for x in ss]==[(x['case'],x['repeat'],x['t']) for x in ww],'Fresh partial keys differ')
            result['fresh_block_comparison'][stage]=h.error_stats([x['actual'] for x in ww],[x['actual'] for x in ss])
        else:result['fresh_block_comparison'][stage]=None
    return result


def svg_map(p):
    """Simple scientific status map, not a conductance image or final-state claim."""
    sets={k:{tuple(c) for c in v['cells']} for k,v in p['contributing_address_union'].items()}
    label='complete300 contributing union' if p['acquisition_complete'] else 'PARTIAL measured union; missing blocks not restored'
    lines=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="400" viewBox="0 0 1120 400">',
           '<rect width="1120" height="400" fill="white"/>',
           '<g font-family="sans-serif" font-size="12" fill="#111">',
           f'<text x="20" y="25">{html.escape(p["source"]+": "+label)}</text>']
    for bl in range(32):lines.append(f'<text x="{72+bl*32}" y="60">{bl}</text>')
    for wl in range(8):
        lines.append(f'<text x="12" y="{87+wl*32}">WL{wl}</text>')
        for bl in range(32):
            c=(wl,bl)
            state='assigned' if c in sets['assigned'] else 'quiet unused' if c in sets['reset_eligible'] else 'screened not quiet' if c in sets['reset_probed'] else 'outside contributing screen'
            color={'assigned':'#43a572','quiet unused':'#e7ba54','screened not quiet':'#cf7070','outside contributing screen':'#dddddd'}[state]
            lines.append(f'<rect x="{65+bl*32}" y="{70+wl*32}" width="29" height="29" fill="{color}" stroke="white"><title>WL{wl} BL{bl}: {state}</title></rect>')
    lines+=['<text x="20" y="350">Green: assigned. Amber: quiet unused. Red: screened, not quiet. Grey: outside contributing screen.</text>',
            '<text x="20" y="375">Union across time; not final conductance, accurate weights, or simultaneous storage.</text>','</g></svg>']
    return '\n'.join(lines)


def markdown(report):
    lines=['# Final finite-reserve evidence: B/C after A','',report['interpretation'],'',
           'Complete HW/SIM comparison: '+str(report['complete'])+'.','',
           '| Source | Fresh B/C | Prior A | Original | Covered | Outputs | Assigned union | New impacts |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for source,p in report['sessions'].items():
        lines.append(f"| {source} | {p['new_BC_coefficients']} | 13 | 267 | {p['covered_coefficients']} | {p['full_time_outputs']} | {p['contributing_address_union']['assigned']['count']} | {p['new_counters']['impact_attempts']} |")
    lines+=['','| Source | Stage | Cell | Fresh disposition | SET pulses |','|---|---|---|---|---:|']
    for source,p in report['sessions'].items():
        for r in p['ten_reserve_dispositions']:lines.append(f"| {source} | {r['stage']} | {r['cell']} | {r['status']} | {r['SET_pulses']} |")
    if report['complete']:
        lines+=['','| Comparison | Conv RMSE | Spike differences /2400 |','|---|---:|---:|']
        for name,rows in report['comparisons'].items():
            if name!='repeat_variation':lines.append(f"| {name} | {rows['overall']['numeric']['conv_sum']['rmse']:.9g} | {rows['overall']['spike_mismatches']} |")
    else:lines+=['','Partial routes have no full300/2400 derived output. Excluded B/C originals are not reinstated. Completed-block contributions and all attempted commands remain explicit.']
    lines+=['','See comparison.json for provenance, fixed gains, bound audits, ancestry, per-filter BN/LIF, fresh block metrics and replaced-cell accounting. Assigned/SET/accurate are distinct. No classification accuracy or absolute optimum is claimed. Maps show only contributing unions across time; no final conductance is inferred.','']
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.')
    for name in ('sim-session','hardware-session','sim-audit','hardware-audit','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args(argv);repo=Path(args.repo).resolve();a,h=pure_helpers(repo);out=a.local(repo,args.output)
    require(not out.exists(),'Output directory must be new')
    fpath=repo/h.FIXTURE;require(sha(fpath)==h.FIXTURE_SHA,'Fixture changed');fixture=a.load(fpath)
    indices=[c['original_test_index'] for c in fixture['cases']]
    sim=verify(repo,args.sim_session,args.sim_audit,'simulator',indices,a,h)
    hw=verify(repo,args.hardware_session,args.hardware_audit,'hardware',indices,a,h)
    for data in (sim,hw):
        p=data['provenance'];ancestors=[p['session'],p['parent_A']['session'],p['parent_A']['parent']['session']]
        nested=p['parent_A']['parent']['parent_import']
        if nested:ancestors.append(nested['session'])
        require(not any(out.is_relative_to(a.local(repo,s)) for s in ancestors),'Output would modify measured ancestry')
    report=compare(sim,hw,h.trained_reference(fixture),h)
    report.update(script_sha256=sha(Path(__file__)),A_comparator_sha256=A_HELPER_SHA,BC_auditor_sha256=AUDITOR_SHA)
    out.mkdir(parents=True,exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (out/'REPORT.md').write_text(markdown(report),encoding='utf-8')
    for source,p in report['sessions'].items():(out/(source+'_contributing_map.svg')).write_text(svg_map(p),encoding='utf-8')
    print(json.dumps(dict(status='completed' if report['complete'] else 'partial',output=args.output)))


if __name__=='__main__':main()
