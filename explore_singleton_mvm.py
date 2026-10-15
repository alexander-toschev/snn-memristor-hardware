"""Exploratory offline singleton/recorded-dispersed analysis; no hardware imports.

This does not change or requalify the frozen coupled-MVM protocol. Selection and
test anchors remain independent acquisitions; no measured mixture is replaced.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean,stdev

ROOT=Path(__file__).resolve().parent


def local(name):
    path=(ROOT/name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(ROOT):raise ValueError('Use a repo-relative path')
    return path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return dict(n=len(values),mean=mean(values) if values else None,sd=stdev(values) if len(values)>1 else None)


def samples(vectors,phase,mask,wl):
    rows=sorted((v for v in vectors if v['phase']==phase and v['mask']==mask),key=lambda v:v['repeat'])
    if len(rows)!=5 or [v['repeat'] for v in rows]!=list(range(5)):return None
    return [v['adc'][wl] for v in rows]


def drift(reads,wl,bl):
    arrays=[[r['conductance_siemens'] for r in reads if r['phase']==phase and r['wl']==wl and r['bl']==bl] for phase in ('pre','post')]
    if any(len(v)!=3 for v in arrays):return dict(passes=False,reason='missing_readback')
    before,after=map(stats,arrays);delta=(after['mean']-before['mean'])/before['mean']
    tolerance=max(.05,3*math.sqrt(before['sd']**2/3+after['sd']**2/3)/before['mean'])
    return dict(passes=abs(delta)<=tolerance,fractional_delta=delta,tolerance=tolerance,pre=before,post=after)


def level(single,zero):
    s,z=stats(single),stats(zero);signal=s['mean']-z['mean'];noise=math.hypot(s['sd'],z['sd'])
    snr=signal/max(1.,noise);cv=noise/signal if signal>0 else None
    return dict(signal_adc=signal,combined_sd_adc=noise,snr=snr,cv=cv,
        quality=signal>0 and snr>=5 and cv<=.05,single=s,zero=z)


def anchor_test(single,zero,basis):
    measured=level(single,zero);expected=basis['signal_adc'];tolerance=max(2.,.05*abs(expected))
    # Identity with fixed calibration zero, exactly the original model coordinate.
    error=measured['single']['mean']-basis['zero']['mean']-expected
    zero_error=measured['zero']['mean']-basis['zero']['mean']
    measured.update(identity_error_adc=error,tolerance_adc=tolerance,
        independent_zero_error_adc=zero_error,independent_zero_pass=abs(zero_error)<=2 and measured['zero']['sd']<=2,
        own_zero_subtracted_error_adc=measured['signal_adc']-expected,
        identity_pass=abs(error)<=tolerance and measured['single']['sd']<=tolerance)
    measured['passes']=bool(measured['quality'] and measured['identity_pass'] and measured['independent_zero_pass'])
    return measured


def lif(currents):
    v=0.;pre=[];post=[];spikes=[]
    for current in currents:
        u=.5*v+current;s=int(u>=1.);v=0. if s else u
        pre.append(u);post.append(v);spikes.append(s)
    return dict(pre=pre,post=post,spikes=spikes)


def pair_partitions(rows):
    if not rows:yield [];return
    first=rows[0]
    for other in rows[1:]:
        for rest in pair_partitions([r for r in rows[1:] if r!=other]):yield [[first,other]]+rest


def signed_anchor_evidence(vectors,a,b,bl):
    evidence={}
    for phase in ('basis','selection_anchors','test_anchors'):
        arrays=[samples(vectors,phase,mask,wl) for mask in ([bl],[]) for wl in (a,b)]
        if any(v is None for v in arrays):return dict(supported=False,reason='missing_complete_pair_anchor')
        singles=[x-y for x,y in zip(arrays[0],arrays[1])];zeros=[x-y for x,y in zip(arrays[2],arrays[3])]
        s,z=stats(singles),stats(zeros);signal=s['mean']-z['mean'];noise=math.hypot(s['sd'],z['sd'])
        evidence[phase]=dict(signal_adc=signal,combined_sd_adc=noise,snr=abs(signal)/max(1.,noise),
            cv=noise/abs(signal) if signal else None,quality=abs(signal)>=5 and abs(signal)/max(1.,noise)>=5 and noise<=.05*abs(signal),
            single=s,zero=z)
    baseline=evidence['basis'];tolerance=max(2.,.05*abs(baseline['signal_adc']))
    for phase in ('selection_anchors','test_anchors'):
        row=evidence[phase];error=row['single']['mean']-baseline['single']['mean'];zero_error=row['zero']['mean']-baseline['zero']['mean']
        row.update(identity_error_adc=error,tolerance_adc=tolerance,
            identity_pass=abs(error)<=tolerance and row['single']['sd']<=tolerance,
            zero_pass=abs(zero_error)<=2 and row['zero']['sd']<=2)
    return dict(supported=baseline['quality'] and all(evidence[p]['quality'] and evidence[p]['identity_pass'] and evidence[p]['zero_pass'] for p in ('selection_anchors','test_anchors')),phases=evidence)


def evaluate(observed,plan):
    vectors,reads=observed['vectors'],observed['readbacks'];cells=[];lookup={};basis_values={};zero_values={}
    for wl in range(8):
        zero_values[wl]=samples(vectors,'basis',[],wl)
        for bl in range(32):
            cell=dict(wl=wl,bl=bl,status='unmeasured',exploratory_only=True)
            series={phase:(samples(vectors,phase,[bl],wl),samples(vectors,phase,[],wl)) for phase in ('basis','selection_anchors','test_anchors')}
            if any(a is None or b is None for a,b in series.values()):cell['reason']='missing_complete_anchor'
            else:
                basis=level(*series['basis']);selection=anchor_test(*series['selection_anchors'],basis);test=anchor_test(*series['test_anchors'],basis)
                g=drift(reads,wl,bl);supported=bool(basis['quality'] and selection['passes'] and test['passes'] and g['passes'])
                cell.update(status='exploratory_singleton_supported' if supported else 'not_supported',
                    basis=basis,selection=selection,test=test,conductance=g,
                    reasons=[name for name,ok in [('basis_quality',basis['quality']),('selection_anchor',selection['passes']),('test_anchor',test['passes']),('conductance_consistency',g['passes'])] if not ok])
                basis_values[wl,bl]=basis['signal_adc']
            cells.append(cell);lookup[wl,bl]=cell
    matching=[]
    signed_all=[]
    for a in range(8):
        for b in range(a+1,8):
            matching.append(dict(wl_pair=[a,b],common_supported_bl=[bl for bl in range(32) if all(lookup[w,bl]['status']=='exploratory_singleton_supported' for w in (a,b))]))
            for bl in range(32):
                evidence=signed_anchor_evidence(vectors,a,b,bl)
                evidence.update(wl_pair=[a,b],bl=bl,two_cells_supported=all(lookup[w,bl]['status']=='exploratory_singleton_supported' for w in (a,b)))
                evidence['exploratory_supported']=evidence['supported'] and evidence['two_cells_supported'];signed_all.append(evidence)
    fixed_pairs=[]
    for a,b in ((0,1),(2,3),(4,5),(6,7)):
        for bl in range(32):
            single_a=samples(vectors,'basis',[bl],a);single_b=samples(vectors,'basis',[bl],b)
            if any(v is None for v in (single_a,single_b,zero_values[a],zero_values[b])):continue
            ds=[x-y for x,y in zip(single_a,single_b)];dz=[x-y for x,y in zip(zero_values[a],zero_values[b])]
            signal=mean(ds)-mean(dz);noise=math.hypot(stdev(ds),stdev(dz))
            fixed_pairs.append(dict(wl_pair=[a,b],bl=bl,signal_adc=signal,combined_sd_adc=noise,
                signed_snr=abs(signal)/max(1.,noise),signed_cv=noise/abs(signal) if signal else None,
                signed_signal_supported=abs(signal)>=5 and abs(signal)/max(1.,noise)>=5 and noise<=.05*abs(signal),
                two_cells_supported=all(lookup[w,bl]['status']=='exploratory_singleton_supported' for w in (a,b))))
    capacities={tuple(p['wl_pair']):len(p['common_supported_bl']) for p in matching}
    partitions=[dict(pairs=p,total_differential_positions=sum(capacities[tuple(pair)] for pair in p)) for p in pair_partitions(list(range(8)))]
    maximum=max(p['total_differential_positions'] for p in partitions)
    signed_capacities={tuple(p['wl_pair']):sum(r['wl_pair']==p['wl_pair'] and r['exploratory_supported'] for r in signed_all) for p in matching}
    signed_partitions=[dict(pairs=p,total_signed_supported_positions=sum(signed_capacities[tuple(pair)] for pair in p)) for p in pair_partitions(list(range(8)))]
    signed_maximum=max(p['total_signed_supported_positions'] for p in signed_partitions)
    dispersed=[]
    recorded={(v['group_id'],v['repeat'],v['t']):v for v in vectors if v['phase']=='synthetic_lif' and v['pattern']=='dispersed'}
    for gid,columns in enumerate(plan['synthetic_groups']):
        for a,b in ((0,1),(2,3),(4,5),(6,7)):
            if any((w,bl) not in basis_values for w in (a,b) for bl in columns):continue
            d={bl:basis_values[a,bl]-basis_values[b,bl] for bl in columns}
            reference={name:[sum(d[columns[j]] for j in step) for step in pattern] for name,pattern in plan['synthetic_patterns'].items()}
            peak=0.
            for currents in reference.values():
                linear=0.
                for current in currents:linear=.5*linear+current;peak=max(peak,abs(linear))
            unsupported=[bl for bl in columns if any(lookup[w,bl]['status']!='exploratory_singleton_supported' for w in (a,b))]
            for polarity in (1,-1):
                result=dict(group_id=gid,columns=columns,pair=[a,b],polarity=polarity,status='inconclusive',
                    unsupported_bl=unsupported,mask_reselected=False,mixture_gate_omitted=True,
                    scale_policy='Unchanged v2 1.25/maxabs(no-reset reference over both fixed patterns); basis only')
                if peak<=0:result['reason']='zero_reference_peak';dispersed.append(result);continue
                scale=1.25/peak;expected=lif([polarity*scale*i for i in reference['dispersed']]);result.update(scale=scale,expected=expected)
                if not any(expected['spikes']) or all(expected['spikes']):result['reason']='expected_not_excited';dispersed.append(result);continue
                repeats=[]
                for repeat in range(5):
                    rows=[recorded.get((gid,repeat,t)) for t in range(12)]
                    if any(row is None for row in rows):repeats.append(dict(status='inconclusive',reason='missing_actual_timestep'));continue
                    currents=[polarity*scale*((row['adc'][a]-mean(zero_values[a]))-(row['adc'][b]-mean(zero_values[b]))) for row in rows]
                    actual=lif(currents);error=max(abs(x-y) for field in ('pre','post') for x,y in zip(actual[field],expected[field]))
                    repeats.append(dict(status='pass' if actual['spikes']==expected['spikes'] and error<=.05 else 'fail',
                        actual=actual,maximum_membrane_error=error,task_ids=[row['task_ids'] for row in rows]))
                result['repeats']=repeats
                result['numerical_agreement']=all(r['status']=='pass' for r in repeats)
                result['status']='exploratory_supported' if result['numerical_agreement'] and not unsupported else 'not_supported'
                dispersed.append(result)
    return dict(scope='Exploratory singleton identity without mixed-vector gate; not a new measurement or final qualification',
        historical_v2_reclassified=False,thresholds_changed=False,correction='identity with fixed basis zero',cells=cells,
        counts=dict(Counter(c['status'] for c in cells)),supported_by_wl=[sum(c['wl']==w and c['status']=='exploratory_singleton_supported' for c in cells) for w in range(8)],
        matching_pairs=matching,fixed_pair_basis=fixed_pairs,recorded_dispersed_lif=dispersed,
        signed_pair_anchors=signed_all,best_signed_capacity=signed_maximum,
        best_signed_exploratory_partitions=[p for p in signed_partitions if p['total_signed_supported_positions']==signed_maximum],
        partition_count=len(partitions),best_matching_capacity=maximum,
        best_exploratory_partitions=[p for p in partitions if p['total_differential_positions']==maximum],
        partition_selection_is_posthoc=True,new_measurement_needed_for_selected_partition=True,
        dispersed_counts=dict(Counter(r['status'] for r in dispersed)),
        simultaneous_mvm_qualified=False,sequential_new_protocol_qualified=False,cifar_qualified=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',required=True)
    parser.add_argument('--output',help='New repository-relative output directory; default session/exploratory_singletons')
    args=parser.parse_args(argv);session=local(args.session);output=local(args.output) if args.output else session/'exploratory_singletons'
    manifest=json.loads((session/'manifest.json').read_bytes())
    for name in ('observations.json','protocol.json'):
        if sha(session/name)!=manifest['result_sha256'][name]:raise ValueError('Input hash mismatch: '+name)
    observed=json.loads((session/'observations.json').read_bytes());plan=json.loads((session/'protocol.json').read_bytes())
    report=evaluate(observed,plan)
    report.update(source=manifest['source'],session=args.session,acquisition_status=manifest['status'],
        input_sha256={name:sha(session/name) for name in ('manifest.json','observations.json','protocol.json')},
        script_sha256=sha(Path(__file__)),raw_sha256_from_manifest=manifest['result_sha256'].get('raw.jsonl'),raw_reaudited_here=False)
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (output/'cells.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['wl','bl','status','basis_signal_adc','basis_snr','basis_cv','selection_pass','test_pass','g_consistent','reasons'],lineterminator='\n');writer.writeheader()
        for c in report['cells']:
            writer.writerow(dict(wl=c['wl'],bl=c['bl'],status=c['status'],basis_signal_adc=c.get('basis',{}).get('signal_adc'),
                basis_snr=c.get('basis',{}).get('snr'),basis_cv=c.get('basis',{}).get('cv'),selection_pass=c.get('selection',{}).get('passes'),
                test_pass=c.get('test',{}).get('passes'),g_consistent=c.get('conductance',{}).get('passes'),reasons=';'.join(c.get('reasons',[]))))
    text=['# Одиночные отклики: exploratory-анализ','',f"Источник: {report['source']}; сессия `{args.session}`.",'',
        'Это новый offline-разбор сохранённых данных. Историческая mixed-MVM квалификация не изменена; плата не вызывалась.',
        'Критерии одиночного отклика: SNR≥5, CV≤5%, независимые selection/test anchors, ошибка identity≤max(2 ADC,5% сигнала), стабильный G.',
        '',f"Результат: `{json.dumps(report['counts'],ensure_ascii=False)}`.",f"По WL0…7: `{report['supported_by_wl']}`.",
        f"Все 105 разбиений WL на 4 пары: максимум {report['best_matching_capacity']} дифференциальных позиций ({2*report['best_matching_capacity']} различных ячеек). Это post-hoc предложение карты, требующее новой проверки.",
        f"После отдельной проверки signed basis/selection/test SNR, CV и identity: максимум {report['best_signed_capacity']} разрешимых знаковых позиций. Это также exploratory, не новая qualification.",
        '',f"Существующий dispersed LIF с прежним масштабом: `{report['dispersed_counts']}`.",
        'Он использует фактически сохранённые одиночные ответы; неподдержанные адреса не заменены нулями и не исключены из траектории.',
        'Даже численное совпадение здесь не квалифицирует новый последовательный протокол: нужны отдельные повторные измерения.',
        '', '| WL-пара | Общие поддержанные BL |','|---|---|']
    text += [f"| {p['wl_pair']} | {', '.join(map(str,p['common_supported_bl'])) or 'нет'} |" for p in report['matching_pairs']]
    text += ['', 'Исходные observations/protocol сверены с manifest SHA256. Raw в этом скрипте заново не аудируется; его SHA сохранён в JSON.',
        'Нулевой ADC не является доказательством неисправного мемристора. Одновременное MVM, программируемость и CIFAR этим разбором не подтверждены.']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=output.relative_to(ROOT).as_posix(),counts=report['counts'],by_wl=report['supported_by_wl'],dispersed=report['dispersed_counts'],best_matching=report['best_matching_capacity'],best_signed=report['best_signed_capacity'])))
    return 0


if __name__=='__main__':raise SystemExit(main())
