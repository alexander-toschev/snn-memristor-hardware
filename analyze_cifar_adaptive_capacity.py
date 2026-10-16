"""Offline address coverage/capacity description from independently audited adaptive data."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import statistics as st
from collections import Counter

AUDITOR_SHA='09d8ac17e57e8a2df12c26d12fe187c52ceeb25f007ed93baeebe67262b4c92f'
PROTOCOL='cifar-feedback-adaptive-two-microtiles-v1'
ROUTING='weights/candidate_cifar75_routing.json'
ROUTING_SHA='bbe71222138fd28ef928dcca22b319abc1223ec87b6680edd56b03c63dac0245'
COLORS={'outside_pool':'#d9dde3','probed_not_quiet':'#de806d','quiet_not_selected':'#e5bf57','selected_no_SET':'#70a5d7','SET_attempted':'#57a982'}
LABELS={'outside_pool':'Вне фиксированного пула','probed_not_quiet':'Проверена, quiet не подтверждён','quiet_not_selected':'Quiet подтверждён, не выбрана','selected_no_SET':'Выбрана, SET не выполнялся','SET_attempted':'Была попытка SET'}


def require(ok,message):
    if not ok:raise ValueError(message)


def local(root,name):
    require(not Path(name).is_absolute(),'Paths must be repository-relative');p=(root/name).resolve()
    require(p.is_relative_to(root),'Path escapes repository');return p


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def original_pools(parent):
    active=[s for s in parent['slots'] if not s['padding']];out={}
    for side in ('positive_cell','negative_cell'):
        rows={s[side][0] for s in active};require(len(rows)==1,'Mixed row in original parent')
        used=[s[side][1] for s in active]
        spare=[b for b in list(range(16))+[31] if b not in used][:3]
        out[side]=[[next(iter(rows)),b] for b in used+spare]
    return out


def category(record):
    if record['SET_attempts']>0:return 'SET_attempted'
    if record['assignments']>0:return 'selected_no_SET'
    if record['quiet_confirmations']>0:return 'quiet_not_selected'
    if record['reset_probes']>0:return 'probed_not_quiet'
    return 'outside_pool'


def analyze(wrappers,routing):
    addresses={(w,b):dict(wl=w,bl=b,in_original_pool=False,reset_probes=0,quiet_confirmations=0,quiet_failures=0,
        assignments=0,SET_attempts=0,SET_pulses=0,new_reset_probes=0,imported_reset_probes=0,
        new_quiet_confirmations=0,imported_quiet_confirmations=0,new_SET_attempts=0,imported_SET_attempts=0,
        new_SET_pulses=0,imported_SET_pulses=0,
        new_assignments=0,imported_assignments=0,evidence=[],exclusion_reasons=[]) for w in range(8) for b in range(32)}
    parents={(f['filter_id'],t['tile_id']):t for f in routing['filters'] for t in f['tiles']}
    for parent in parents.values():
        for pool in original_pools(parent).values():
            for cell in pool:addresses[tuple(cell)]['in_original_pool']=True
    microtiles=[];seen_taps=[];pending_by_parent={};parent_micro=Counter()
    for wrapper in wrappers:
        fid,pid,mid=(wrapper[k] for k in ('filter_id','parent_tile_id','microtile_id'));r=wrapper['result'];origin=wrapper['origin']
        historical=origin['reused'];scope='imported' if historical else 'new';parent=parents[fid,pid]
        original=[s for s in parent['slots'] if not s['padding']];pools=original_pools(parent)
        pending=pending_by_parent.get((fid,pid),[s['tap_index'] for s in original])
        require(mid==parent_micro[fid,pid] and mid<2,'Invalid microtile sequence')
        require(r['technical_completed'],'Only completed contributing microtiles are supported')
        if not historical:require(r['pending_taps_before']==pending,'Pending prefix differs')
        selected={tuple(s[side]) for s in r['effective_route'] for side in ('positive_cell','negative_cell')}
        eligible={side:[] for side in pools};observed={side:[] for side in pools}
        for c in r['reset_candidates']:
            cell=tuple(c['cell']);side=c['side'];observed[side].append(list(cell));values=[v['voltage_v'] for v in c['mvm_confirmation']]
            mean=st.mean(values);sd=st.stdev(values);limit=c['mode10_quiet_limit_v']
            quiet=c['reset_crossed'] and abs(mean)<=limit and sd<=limit
            require(quiet==c['eligible'],'Stored quiet flag differs from numeric criterion')
            if quiet:eligible[side].append(list(cell))
            a=addresses[cell];a['reset_probes']+=1;a[scope+'_reset_probes']+=1
            a['quiet_confirmations']+=int(quiet);a['quiet_failures']+=int(not quiet)
            a[scope+'_quiet_confirmations']+=int(quiet)
            reason=[]
            if not c['reset_crossed']:reason.append('RESET_did_not_cross_quiet_threshold')
            if abs(mean)>limit:reason.append('confirmation_abs_mean_exceeds_quiet_limit')
            if sd>limit:reason.append('confirmation_sample_SD_exceeds_quiet_limit')
            if quiet and cell not in selected:reason.append('quiet_but_not_in_deterministic_first_k_assignment')
            a['exclusion_reasons'].extend(reason)
            a['evidence'].append(dict(filter_id=fid,parent_tile_id=pid,microtile_id=mid,origin=scope,source=origin,
                quiet=quiet,selected=cell in selected,confirmation_mean_v=mean,confirmation_sample_sd_v=sd,
                quiet_limit_v=limit,reset_stop=c['reset_stop'],reasons=reason))
        require(observed==pools,'Fresh candidate scan does not cover complete original pools')
        pcount,ncount=(len(eligible[s]) for s in ('positive_cell','negative_cell'));expected=min(len(pending),pcount,ncount)
        active=r['effective_route'];k=len(active)
        require(k==expected and [s['tap_index'] for s in active]==pending[:k],'Incorrect deterministic capacity assignment')
        for i,s in enumerate(active):
            seen_taps.append((fid,s['tap_index']))
            for side in pools:
                require(s[side]==eligible[side][i],'Assignment did not use first eligible cell')
                a=addresses[tuple(s[side])];a['assignments']+=1;a[scope+'_assignments']+=1
        for pair in r['pairs']:
            if pair['programming_pulse_attempts']:
                a=addresses[tuple(pair[pair['active_set_side']])];a['SET_attempts']+=1;a['SET_pulses']+=pair['programming_pulse_attempts']
                a[scope+'_SET_attempts']+=1;a[scope+'_SET_pulses']+=pair['programming_pulse_attempts']
        remaining=pending[k:];pending_by_parent[fid,pid]=remaining;parent_micro[fid,pid]+=1
        limiting=[name for name,n in [('pending_taps',len(pending)),('positive_quiet',pcount),('negative_quiet',ncount)] if n==expected]
        rowp=pools['positive_cell'][0][0];rown=pools['negative_cell'][0][0]
        microtiles.append(dict(filter_id=fid,parent_tile_id=pid,microtile_id=mid,origin=scope,positive_WL=rowp,negative_WL=rown,
            pending_before=len(pending),quiet_positive=pcount,quiet_negative=ncount,k_expected=expected,k_used=k,pending_after=len(remaining),
            unused_quiet_positive=pcount-k,unused_quiet_negative=ncount-k,limiting_factors=limiting,
            WL3_conditional_gain_if_three_unmeasured_alternatives_were_quiet=(min(len(pending),pcount,ncount+3)-k if rown==3 else None)))
    require(len(seen_taps)==300 and set(seen_taps)=={(f,t) for f in range(4) for t in range(75)},'Not exact 300 contributing coefficients')
    for a in addresses.values():
        a['exclusion_reasons']=sorted(set(a['exclusion_reasons']))
        if not a['in_original_pool']:a['exclusion_reasons'].append('outside_frozen_candidate_pool_not_qualified_in_this_experiment')
        if a['quiet_confirmations'] and a['quiet_failures']:a['exclusion_reasons'].append('quiet_status_varied_between_sweeps')
        a['category']=category(a)
    rows=[]
    for wl in range(8):
        cells=[a for a in addresses.values() if a['wl']==wl]
        rows.append(dict(wl=wl,pool=sum(a['in_original_pool'] for a in cells),
            **{name:sum(a[field]>0 for a in cells) for name,field in [('reset_probed','reset_probes'),('ever_quiet','quiet_confirmations'),('ever_assigned','assignments'),('SET_attempted','SET_attempts')]},
            by_origin={scope:{name:sum(a[scope+'_'+field]>0 for a in cells) for name,field in [('reset_probed','reset_probes'),('ever_quiet','quiet_confirmations'),('ever_assigned','assignments'),('SET_attempted','SET_attempts')]} for scope in ('new','imported')},
            probe_events=sum(a['reset_probes'] for a in cells),quiet_failure_events=sum(a['quiet_failures'] for a in cells)))
    pairs=[]
    for wp,wn in ((0,1),(2,3),(4,5),(6,7)):
        records=[r for r in microtiles if (r['positive_WL'],r['negative_WL'])==(wp,wn)]
        pairs.append(dict(positive_WL=wp,negative_WL=wn,contributing_microtiles=len(records),split_parent_chunks=sum(n==2 for (f,p),n in parent_micro.items() if any(r['filter_id']==f and r['parent_tile_id']==p for r in records)),
            new_microtiles=sum(r['origin']=='new' for r in records),imported_microtiles=sum(r['origin']=='imported' for r in records),
            maximum_observed_assigned_coefficients_in_one_microtile=max(r['k_used'] for r in records),
            min_ever_quiet_union_counts_not_simultaneous=min(rows[wp]['ever_quiet'],rows[wn]['ever_quiet']),
            capacity_limited_microtiles=sum(r['pending_after']>0 for r in records)))
    return dict(addresses=list(addresses.values()),rows=rows,pairs=pairs,microtiles=microtiles,
        contributing_microtiles=len(wrappers),imported_microtiles=sum(w['origin']['reused'] for w in wrappers),
        split_parent_chunks=sum(n==2 for n in parent_micro.values()),category_counts=dict(Counter(a['category'] for a in addresses.values())),
        WL3_alternatives=[addresses[3,b] for b in (12,13,31)])


def svg(report):
    counts=report['category_counts'];width=1310;height=520
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="#fff"/>','<g font-family="Arial,sans-serif" fill="#172331">',
         '<text x="24" y="30" font-size="20">Карта вкладов завершённых new + imported microtiles</text>',
         '<text x="24" y="55" font-size="14">Failed/unimported attempts не включены; их команды учтены отдельно. Это не карта финальных проводимостей.</text>']
    for bl in range(32):out.append(f'<text x="{84+37*bl}" y="90" font-size="12" text-anchor="middle">{bl}</text>')
    for wl in range(8):out.append(f'<text x="18" y="{120+36*wl}" font-size="13">WL{wl}</text>')
    for a in report['addresses']:
        x=67+37*a['bl'];y=100+36*a['wl'];title=html.escape(f"WL{a['wl']} BL{a['bl']}: {LABELS[a['category']]}; probes={a['reset_probes']}, quiet={a['quiet_confirmations']}, selected={a['assignments']}, SET pulses={a['SET_pulses']}; "+'; '.join(a['exclusion_reasons']))
        out.append(f'<rect x="{x}" y="{y}" width="33" height="30" rx="3" fill="{COLORS[a["category"]]}" stroke="#fff"><title>{title}</title></rect>')
        if a['imported_reset_probes'] and not a['new_reset_probes']:out.append(f'<text x="{x+16}" y="{y+20}" text-anchor="middle" font-size="11">H</text>')
    for i,(name,color) in enumerate(COLORS.items()):
        y=410+(i//3)*35;x=24+(i%3)*420
        out.append(f'<rect x="{x}" y="{y-14}" width="18" height="18" fill="{color}"/><text x="{x+25}" y="{y}" font-size="13">{LABELS[name]}: {counts.get(name,0)}</text>')
    out.append('<text x="24" y="500" font-size="12">Приоритет цвета: SET → выбрана → когда-либо quiet → проверена → вне пула. История вкладов и причины — в CSV/JSON.</text></g></svg>')
    return '\n'.join(out)+'\n'


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.')
    for name in ('session','audit','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args(argv);repo=Path(args.repo).resolve();session=local(repo,args.session);out=local(repo,args.output)
    require(not out.exists() and not out.is_relative_to(session),'Output must be a new directory outside acquired session')
    m=load(session/'manifest.json');audit_path=local(repo,args.audit);audit=load(audit_path)
    require(m['status']=='completed' and m['protocol_id']==PROTOCOL,'Expected completed adaptive session')
    require(audit['audit_status']=='passed' and not audit['errors'] and audit['auditor_sha256']==AUDITOR_SHA,'Reviewed independent audit PASS required')
    require(audit['manifest_sha256']==sha(session/'manifest.json') and audit['session_id']==m['session_id'] and audit['source']==m['source'],'Audit identity/manifest binding mismatch')
    require(audit['acquisition_complete'] and audit['coefficient_coverage']['exact_300_coverage'],'Full 300 audit required')
    for name in ('completed_microtiles.json','summary.json','protocol.json'):
        require(sha(session/name)==m['result_sha256'][name],'Changed bound '+name)
    require(sha(local(repo,ROUTING))==ROUTING_SHA,'Routing changed')
    wrappers=load(session/'completed_microtiles.json')
    for w in wrappers:
        origin=w['origin'];path=local(repo,origin['session'])/origin['path']
        require(path.resolve().is_relative_to(local(repo,origin['session'])),'Origin path escapes session')
        require(sha(path)==origin['sha256'] and load(path)==w['result'],'Changed contributing tile')
        if origin['reused']:require(not out.is_relative_to(local(repo,origin['session'])),'Output inside imported parent')
    report=analyze(wrappers,load(local(repo,ROUTING)));summary=load(session/'summary.json')
    for key,field in [('reset_probed','reset_probes'),('reset_eligible','quiet_confirmations'),('selected','assignments'),('set_attempted','SET_attempts')]:
        require(summary['address_coverage'][key]['count']==sum(a[field]>0 for a in report['addresses']),'Summary coverage differs: '+key)
    report.update(session=args.session,session_id=m['session_id'],source=m['source'],manifest_sha256=sha(session/'manifest.json'),
        independent_audit=args.audit,independent_audit_sha256=sha(audit_path),analysis_script_sha256=sha(Path(__file__)),hardware_io=False,
        scope='union of contributing new and imported microtiles; not final state or simultaneous retained capacity',new_counters=m['new_counters'],
        cumulative_impacts_including_all_parent_attempts=m['cumulative_impacts_including_all_parent_attempts'],
        historical_support_for_WL3_alternatives='Earlier singleton support is historical, not current quiet/programming qualification; no new measurements of these alternatives are invented.')
    out.mkdir(parents=True)
    (out/'capacity.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    csv_write(out/'addresses256.csv',[{k:v for k,v in a.items() if k!='evidence'} for a in report['addresses']]);csv_write(out/'microtiles.csv',report['microtiles'])
    (out/'coverage256.svg').write_text(svg(report),encoding='utf-8')
    lines=['# Охват ячеек и ограничения адаптивного разбиения','',f"Источник: {m['source']}; `{args.session}`. Microtiles: {len(wrappers)}, исторических: {report['imported_microtiles']}, разделённых исходных блоков: {report['split_parent_chunks']}.",'',
        'Цвета показывают только объединение наблюдений завершённых contributing new + imported microtiles. Failed/unimported attempts в эту карту не входят; их команды сохранены в отдельных cumulative totals. Карта не означает финальную проводимость, исправность ячейки или одновременно сохранённые веса. Для каждой ячейки CSV содержит флаги и причины; JSON — измеренные mean/SD/порог проверок, участвующих в этих вкладах, с происхождением.','',
        '| WL | В пуле | Проверено | Когда-либо quiet | Назначено | SET попытка |','|---|---:|---:|---:|---:|---:|']
    for r in report['rows']:lines.append('| '+ ' | '.join(str(r[k]) for k in ('wl','pool','reset_probed','ever_quiet','ever_assigned','SET_attempted'))+' |')
    lines+=['','| Пара WL | Макс. коэффициентов в одном microtile | min(quiet unions), НЕ одновременно | Ограниченных microtiles |','|---|---:|---:|---:|']
    for r in report['pairs']:lines.append(f"| {r['positive_WL']}/{r['negative_WL']} | {r['maximum_observed_assigned_coefficients_in_one_microtile']} | {r['min_ever_quiet_union_counts_not_simultaneous']} | {r['capacity_limited_microtiles']} |")
    lines+=['','WL3 BL12/13/31:']
    for a in report['WL3_alternatives']:lines.append(f"- BL{a['bl']}: в пуле={a['in_original_pool']}, проверок={a['reset_probes']}, quiet={a['quiet_confirmations']}, назначений={a['assignments']}.")
    affected=[r for r in report['microtiles'] if r['negative_WL']==3 and r['WL3_conditional_gain_if_three_unmeasured_alternatives_were_quiet']>0]
    lines+=['',f"В записанных решениях WL2/3 дополнительные три подтверждённые quiet-ячейки на WL3 могли бы увеличить k в {len(affected)} microtiles. Это условный расчёт по неизменным pending/P; такие проверки альтернатив не выполнены и выигрыш не установлен.",
        'Исторический singleton-отклик этих адресов не является их текущей квалификацией. Дополнительные адреса не меняют установленный предел двух microtiles и не доказывают физическую ёмкость всей платы.','',
        'Указанный max k наблюдался внутри одного microtile. Сумма максимумов разных пар не выдаётся за одновременную ёмкость: измерения последовательны, состояния перезаписываются. Quiet — только подтверждение данного RESET-критерия, не точности программирования/MVM.','',
        'Воспроизведение (новая выходная папка):','',f'```text\npython analyze_cifar_adaptive_capacity.py --session {args.session} --audit {args.audit} --output results/cifar_feedback_reviews/capacity_recheck\n```','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(dict(output=args.output,source=m['source'],category_counts=report['category_counts'],split_parent_chunks=report['split_parent_chunks'])))
    return 0


if __name__=='__main__':raise SystemExit(main())
