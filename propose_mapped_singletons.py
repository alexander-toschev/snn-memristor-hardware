"""Offline post-hoc per-WL BL mapping proposal. Never runs a Connector."""
import argparse
import hashlib
import json
from pathlib import Path
from explore_singleton_mvm import pair_partitions

ROOT=Path(__file__).resolve().parent


def local(value):
    path=(ROOT/value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(ROOT):raise ValueError('Repository-relative path required')
    return path


def edge(a,b):
    """Different BL observations: sum-SD bound, no assumed covariance cancellation."""
    phases={}
    for key in ('basis','selection'):
        x,y=a[key],b[key]
        signal=x['signal_adc']-y['signal_adc']
        bound=x['single']['sd']+x['zero']['sd']+y['single']['sd']+y['zero']['sd']
        phases[key]=dict(signed_signal_adc=signal,sd_bound_adc=bound,snr_bound=abs(signal)/max(1.,bound),
            cv_bound=bound/abs(signal) if signal else None,
            contrast_supported=abs(signal)>=5 and abs(signal)/max(1.,bound)>=5 and bound<=.05*abs(signal))
    d=phases['basis']['signed_signal_adc'];tolerance=max(2.,.05*abs(d))
    error=(a['selection']['single']['mean']-b['selection']['single']['mean'])-(a['basis']['single']['mean']-b['basis']['single']['mean'])
    zero_error=(a['selection']['zero']['mean']-b['selection']['zero']['mean'])-(a['basis']['zero']['mean']-b['basis']['zero']['mean'])
    supported=all(p['contrast_supported'] for p in phases.values()) and abs(error)<=tolerance and abs(zero_error)<=2
    return dict(positive_cell=[a['wl'],a['bl']],negative_cell=[b['wl'],b['bl']],phases=phases,
        selection_identity_error_adc=error,selection_tolerance_adc=tolerance,selection_zero_error_adc=zero_error,
        contrast_supported=supported,physical_cell_defect_claim=False,programmability_qualified=False)


def maximum_matching(adj,left,right):
    assigned={};right=set(right)
    def place(a,seen):
        for b in sorted(adj.get(a,())):
            if b not in right or b in seen:continue
            seen.add(b)
            if b not in assigned or place(assigned[b],seen):assigned[b]=a;return True
        return False
    for a in sorted(left):place(a,set())
    return sorted((a,b) for b,a in assigned.items())


def lexical_matching(adj,left,right):
    """Maximum cardinality, then lexicographically smallest sorted physical pairs."""
    left=sorted(left);right=set(right);remaining=len(maximum_matching(adj,left,right));result=[]
    for index,a in enumerate(left):
        if not remaining:break
        later=left[index+1:]
        for b in sorted(set(adj.get(a,()))&right):
            if 1+len(maximum_matching(adj,later,right-{b}))>=remaining:
                result.append((a,b));right.remove(b);remaining-=1;break
    if remaining:raise AssertionError('Matching construction lost cardinality')
    return result


def propose(report):
    cells={(c['wl'],c['bl']):c for c in report['cells'] if c['status']=='exploratory_singleton_supported'}
    rows={wl:sorted(bl for w,bl in cells if w==wl) for wl in range(8)}
    pair_results={};edges=[]
    for a in range(8):
        for b in range(a+1,8):
            adj={bl:[] for bl in rows[a]};lookup={}
            for left in rows[a]:
                for right in rows[b]:
                    value=edge(cells[a,left],cells[b,right]);edges.append(value);lookup[left,right]=value
                    if value['contrast_supported']:adj[left].append(right)
            matched=lexical_matching(adj,rows[a],rows[b])
            pair_results[a,b]=dict(wl_pair=[a,b],strong_single_capacity=min(len(rows[a]),len(rows[b])),
                screened_contrast_capacity=len(matched),mapping=[lookup[p] for p in matched])
            signals=[lookup[p]['phases']['basis']['signed_signal_adc'] for p in matched]
            pair_results[a,b]['sign_counts']=dict(positive=sum(d>0 for d in signals),negative=sum(d<0 for d in signals),zero=sum(d==0 for d in signals))
            pair_results[a,b]['mixed_signs_present']=any(d>0 for d in signals) and any(d<0 for d in signals)
            pair_results[a,b]['cancellation_tested']=False
    partitions=[]
    for pairs in pair_partitions(list(range(8))):
        partitions.append(dict(pairs=pairs,strong_single_capacity=sum(pair_results[tuple(p)]['strong_single_capacity'] for p in pairs),
            screened_contrast_capacity=sum(pair_results[tuple(p)]['screened_contrast_capacity'] for p in pairs)))
    upper=max(p['strong_single_capacity'] for p in partitions);screened=max(p['screened_contrast_capacity'] for p in partitions)
    chosen=min((p for p in partitions if p['screened_contrast_capacity']==screened),key=lambda p:p['pairs'])
    mapping=[pair_results[tuple(p)] for p in chosen['pairs']]
    used={tuple(e[k]) for p in mapping for e in p['mapping'] for k in ('positive_cell','negative_cell')}
    return dict(scope='Post-hoc proposal from prior observations, not fresh qualification',
        source_singleton_count=len(cells),source_singleton_counts_by_wl=[len(rows[w]) for w in range(8)],
        unfiltered_pairing_upper_positions=upper,unfiltered_pairing_upper_cells=2*upper,
        current_resolvable_contrast_positions=screened,current_resolvable_contrast_cells=2*screened,
        chosen_partition=chosen,mapping=mapping,
        all_105_partitions=partitions,all_pair_capacities=[v for v in pair_results.values()],
        unused_strong_cells=[list(cell) for cell in sorted(cells) if cell not in used],
        unused_interpretation='Not selected by current contrast matching, not a defective or unprogrammable cell',
        all_edge_evidence=edges,
        selection_data='Prior basis+selection contrasts; prior102 mask also used old heldout test, so entire proposal is post-hoc',
        noise_rule='sum SD(singleA,zeroA,singleB,zeroB), covariance unknown; no independence-based noise discount',
        scale_rule='No scale fitted here; future scale must use new basis/selection before new heldout test',
        physical_operation='Separate public mode10 calls can route each WL to its own physical BL; every coefficient remains measured, never looked up for inference',
        actual_simultaneous_rows=False,new_confirmatory_measurement_required=True,
        programmed_weights_qualified=False,cifar_qualified=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);args=parser.parse_args(argv)
    source=local(args.input);output=local(args.output);report=propose(json.loads(source.read_bytes()))
    report['input_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();report['input']=args.input
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True)
    (output/'proposal.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    text=['# Предварительная раздельная адресация BL для WL-пар','',
        f"Исходные устойчивые одиночные отклики: {report['source_singleton_count']}.",
        f"Вместимость без проверки разности: {report['unfiltered_pairing_upper_positions']} позиций / {report['unfiltered_pairing_upper_cells']} ячеек.",
        f"Разрешимые текущие разности по basis+selection с консервативным SD: {report['current_resolvable_contrast_positions']} позиций / {report['current_resolvable_contrast_cells']} ячеек.",
        '', 'Это post-hoc предложение, не подтверждённая программируемость и не новая квалификация. Неиспользованная ячейка не объявляется неисправной.',
        '', '| WL+ / WL− | Логический индекс | BL+ | BL− | Signed basis ADC | SD bound ADC |','|---|---:|---:|---:|---:|---:|']
    for pair in report['mapping']:
        for index,e in enumerate(pair['mapping']):
            p=e['phases']['basis'];text.append(f"| {pair['wl_pair']} | {index} | {e['positive_cell'][1]} | {e['negative_cell'][1]} | {p['signed_signal_adc']:.6g} | {p['sd_bound_adc']:.6g} |")
    text += ['', '## Знаки текущих разностей', '', '| WL+ / WL− | D > 0 | D < 0 | D = 0 |', '|---|---:|---:|---:|']
    for pair in report['mapping']:
        counts=pair['sign_counts'];text.append(f"| {pair['wl_pair']} | {counts['positive']} | {counts['negative']} | {counts['zero']} |")
    text += ['', 'Наличие обоих знаков не подтверждает сложение или компенсацию вкладов. Цифровая смена полярности D/−D сама по себе не проверяет смешанную сумму положительных и отрицательных весов. Такие суммы требуют новых измерений каждого вклада; для предложенной адресации эта проверка ещё не выполнена.']
    text += ['', 'Каждый вклад будущего опыта требует нового вызова Connector с физическим BL этой WL; таблица используется для маршрутизации входных спайков, не вместо измерения.',
        'Маска и масштабы фиксируются до нового heldout. Actual CIFAR, SET/RESET и патч драйвера не входят в предложение.']
    (output/'PROPOSAL.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('unfiltered_pairing_upper_positions','current_resolvable_contrast_positions','chosen_partition')}));return 0


if __name__=='__main__':raise SystemExit(main())
