"""Offline fixed-amplitude hypothesis from one frozen, audited map; stdlib only."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev

SESSION = 'results/cell_maps/9923f9bc-1ffc-4977-b26f-fd68c6c84e90'
PINS = {
    'manifest.json': '0c2f2f9cc3be69251d49ccb99bb22b083bc5b5cb137af7ce7e619940b5899f6c',
    'raw.jsonl': 'a599fe5220a9360d90f9007d18d58bb3034e0e6936cbb755a09615a2f5cdd4c1',
    'criteria.json': 'b9527594380d2a92b5338d34cbf433d5ca98f32cc7e9ff322d4b168773252a34',
    'cells.json': '437487e4e346e178e6a4b336a5044e349b283f9194e15ab01d859df5707c6fd5',
    'summary.json': 'd1613608989710510f5c06ad8fd098f8aae627d0a347a8db660d6308a701a14a',
    'offline-audit/independent_cell_map_audit.json': '325d46ac1cac63e8692ddf2f52291ff1b8c44caa9c05fec5a739d7822030f9d4',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def stats(values):
    return dict(n=len(values), mean=mean(values), sd=stdev(values) if len(values)>1 else 0.)


def classify(level, zero, pre_g, post_g, rules):
    """SD is выборочное SD; combined SD describes observations, not the mean's SE."""
    counts = dict(level=len(level), zero=len(zero), pre_g=len(pre_g), post_g=len(post_g))
    if counts != dict(level=5, zero=5, pre_g=3, post_g=3):
        return dict(status='unmeasured', reasons=['invalid_counts'], counts=counts)
    if any(not math.isfinite(x) or x<0 for x in level+zero) or any(not math.isfinite(x) or x<=0 for x in pre_g+post_g):
        return dict(status='unmeasured', reasons=['invalid_samples'], counts=counts)
    a,z,p,f = map(stats, (level,zero,pre_g,post_g))
    signal=a['mean']-z['mean'];sd=math.hypot(a['sd'],z['sd'])
    snr=signal/max(rules['noise_floor_adc'],sd);cv=sd/signal if signal>0 else None
    drift=(f['mean']-p['mean'])/p['mean']
    drift_se=math.sqrt(p['sd']**2/3+f['sd']**2/3)/p['mean']
    tolerance=max(rules['drift_fraction'],rules['drift_se_multiplier']*drift_se)
    reasons=[]
    if signal<=0: reasons.append('nonpositive_signal')
    if snr<rules['minimum_snr']: reasons.append('snr_below_5')
    if cv is None: reasons.append('cv_undefined_nonpositive_signal')
    elif cv>rules['maximum_cv_high']: reasons.append('cv_above_0.05')
    if abs(drift)>tolerance: reasons.append('pre_post_g_inconsistent')
    return dict(status='provisional_candidate' if not reasons else 'reject',reasons=reasons,counts=counts,
        level_adc=a,zero_adc=z,signal_adc=signal,combined_sd_adc=sd,cv=cv,snr=snr,
        pre_g_siemens=p,post_g_siemens=f,g_fractional_delta=drift,g_fractional_se=drift_se,
        g_fractional_tolerance=tolerance,cross_amplitude_linearity_used_as_gate=False)


def pairings(rows=tuple(range(8))):
    if not rows:
        yield ()
        return
    a=rows[0]
    for b in rows[1:]:
        for rest in pairings(tuple(x for x in rows[1:] if x!=b)):
            yield ((a,b),)+rest


def coverage(candidates):
    common={(a,b):sorted(candidates[a]&candidates[b]) for a,b in itertools.combinations(range(8),2)}
    partitions=[]
    for pairs in pairings():
        counts=[len(common[p]) for p in pairs]
        partitions.append(dict(pairs=[dict(wl_a=a,wl_b=b,common_bl=common[a,b],count=len(common[a,b])) for a,b in pairs],
            capacity_sum=sum(counts),minimum_pair_capacity=min(counts),
            shared_bl_all_four_pairs=sorted(set.intersection(*(set(common[p]) for p in pairs)))))
    return dict(per_wl_counts=[len(candidates[w]) for w in range(8)],
        shared_bl_all_eight_wl=sorted(set.intersection(*candidates)),
        wl_pairs=[dict(wl_a=a,wl_b=b,common_bl=bs,count=len(bs)) for (a,b),bs in common.items()],
        partitions=partitions,partition_count=len(partitions),
        maximum_capacity_sum=max(p['capacity_sum'] for p in partitions),
        maximum_minimum_pair_capacity=max(p['minimum_pair_capacity'] for p in partitions),
        interpretation='Coordinate intersections at one amplitude only; no simultaneous pair, weight or filter capacity qualification.')


def local(root,name):
    if Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('Use repo-relative paths')
    path=(root/name).resolve()
    if not path.is_relative_to(root): raise ValueError('Path escapes repo')
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    parser.add_argument('--output-dir',default='results/binary_spike_review/20261015_fixed_amplitude')
    args=parser.parse_args(argv);root=Path(args.repo).resolve();source=local(root,SESSION);out=local(root,args.output_dir)
    if out.exists(): raise ValueError('Output must be a new directory')
    for name,digest in PINS.items():
        if sha(source/name)!=digest: raise ValueError('Input SHA mismatch: '+name)
    def load(name):return json.loads((source/name).read_text(encoding='utf-8'))
    manifest,audit,summary=load('manifest.json'),load('offline-audit/independent_cell_map_audit.json'),load('summary.json')
    if manifest['source']!='hardware' or summary['status']!='completed' or not summary['mask_usable']: raise ValueError('Incomplete source map')
    if audit['audit_status']!='passed' or audit['errors'] or audit['manifest_sha256']!=PINS['manifest.json'] or audit['raw_sha256']!=PINS['raw.jsonl']: raise ValueError('Audit linkage failed')
    for name in ('raw.jsonl','criteria.json','cells.json','summary.json'):
        if manifest['result_sha256'][name]!=PINS[name]: raise ValueError('Manifest/result linkage failed')
    criteria=load('criteria.json');rules=criteria['criteria'];old=load('cells.json')
    old={(c['wl'],c['bl']):c for c in old}
    keys={(w,b) for w in range(8) for b in range(32)}
    if set(old)!=keys: raise ValueError('Expected 256 distinct cells')
    data={k:dict(pre=[],post=[],low=[],high=[]) for k in keys};zero={w:[] for w in range(8)}
    sample_ids={k:dict(pre=[],post=[],low=[],high=[]) for k in keys};zero_ids={w:[] for w in range(8)}
    results=0
    with (source/'raw.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['event']!='result':continue
            results+=1;c=r['command_si'];adc=r['raw_adc'];w=c['wl']
            if r['status']!='ok' or r['source']!='hardware' or type(adc) is not int or not 0<=adc<=16382:raise ValueError('Invalid raw result')
            if c['mode_flag']==9:
                key=(w,c['bl']);kind=r['stage'];value=r['conductance_siemens']
                if kind not in ('pre','post') or adc<1:raise ValueError('Invalid readback')
            elif c['mode_flag']==10:
                active=[i for i,v in enumerate(c['vol']) if v]
                if not active:
                    zero[w].append(adc);zero_ids[w].append(r['task_id']);continue
                if len(active)!=1 or c['vol'][active[0]] not in (.05,.1):raise ValueError('Unsupported input')
                key=(w,active[0]);kind='low' if c['vol'][active[0]]==.05 else 'high';value=adc
            else:raise ValueError('Unexpected mode')
            data[key][kind].append(value);sample_ids[key][kind].append(r['task_id'])
    if results!=4136 or results!=audit['request_result_pairs']: raise ValueError('Result count mismatch')
    rows=[]
    for w,b in sorted(keys):
        d=data[w,b]
        rows.append(dict(wl=w,bl=b,original_status=old[w,b]['status'],original_reasons=old[w,b]['reasons'],
            samples=d,zero_samples=zero[w],task_ids=sample_ids[w,b],zero_task_ids=zero_ids[w],
            fixed_amplitudes={str(v):classify(d[k],zero[w],d['pre'],d['post'],rules) for v,k in ((.05,'low'),(.1,'high'))}))
    covers={};counts={}
    for level in ('0.05','0.1'):
        candidates=[{r['bl'] for r in rows if r['wl']==w and r['fixed_amplitudes'][level]['status']=='provisional_candidate'} for w in range(8)]
        covers[level]=coverage(candidates);counts[level]=sum(map(len,candidates))
    report=dict(hypothesis='fixed-amplitude-single-cell-screen-v1',offline_only=True,new_measurement=False,
        source_session=SESSION,source_commit=manifest['invocation']['git']['commit'],input_sha256={SESSION+'/'+k:v for k,v in PINS.items()},
        generator_sha256=sha(Path(__file__)),raw_protocol_reaudited=False,prior_raw_audit_checks=audit['checks'],
        integrity='Pinned bytes and audit/manifest/raw linkage checked; raw samples re-extracted, not a new protocol audit.',
        original_criteria=criteria,changed_hypothesis='Each amplitude screened independently; cross-amplitude linearity excluded; CV<=5% applied to each level.',
        additivity_tested=False,temporal_response_tested=False,physical_voltage_calibrated=False,
        qualification_flag_scope='Current fixed-amplitude candidates at this saved state; historical additivity tests exist and do not qualify them.',
        programming_qualified=False,cifar_mapping_qualified=False,current_hardware_state_confirmed=False,
        candidate_counts=counts,coverage=covers,cells=rows)
    out.mkdir(parents=True)
    (out/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    columns=['wl','bl','original_status','original_reasons']
    fields=['status','signal_adc','combined_sd_adc','cv','snr','g_fractional_delta','g_fractional_tolerance','reasons']
    columns += [level+'_'+f for level in ('0.05','0.1') for f in fields]
    with (out/'cells.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for r in rows:
            flat={k:r[k] for k in columns[:4]}
            for level in ('0.05','0.1'):
                flat.update({level+'_'+f:r['fixed_amplitudes'][level].get(f) for f in fields})
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in flat.items()})
    lines=['# Фиксированная амплитуда: отдельная offline-гипотеза','',
        'Новых измерений нет. Исходная карта с 7 кандидатами и её межамплитудный критерий не изменены. Ниже отдельная предварительная проверка одиночного отклика при каждой заданной амплитуде отдельно.', '',
        '| Заданный вход | Кандидатов /256 | По WL0…7 | Общие BL всех 8 WL | Максимум суммы четырёх пар | Максимум минимальной ёмкости пары |',
        '|---|---|---|---|---|---|']
    for level,c in covers.items():lines.append(f"| {level} V | {counts[level]} | {c['per_wl_counts']} | {c['shared_bl_all_eight_wl']} | {c['maximum_capacity_sum']} | {c['maximum_minimum_pair_capacity']} |")
    lines+=['','Для каждой амплитуды отдельно перебраны 28 пар WL и 105 разбиений всех 8 WL на 4 непересекающиеся пары. commonBL означает пересечение координат кандидатов при одном и том же V. Сумма — число таких координатных мест, не доказанная ёмкость весов/фильтров. Общий BL всех 8 одинаков для всех разбиений. Амплитуды не смешиваются.', '',
        'Критерии: signal=mean(level)−mean(zero)>0; SDcombined=√(SDlevel²+SDzero²), выборочное SD; SNR=signal/max(1,SDcombined)≥5; CV=SDcombined/signal≤5%. Нулевой/отрицательный signal делает CV неопределённым, а не «низким шумом». По 3 G в режиме 9 до/после: |ΔG/Gpre|≤max(5%,3√(SDpre²/3+SDpost²/3)/Gpre). Ровно 5 level/zero и 3 pre/post нужны для кандидата. Межамплитудная линейность .05→.1 здесь не является критерием отбора; это изменение гипотезы, не улучшение физических данных.', '',
        'Отклик на единичный активный BL ещё не проверяет суммирование одновременно активных бинарных входов. Для этих кандидатов в этом состоянии: `additivity_tested=False`, `temporal_response_tested=False`, `physical_voltage_calibrated=False`. Сохранённые команды не являются проверенной временной последовательностью LIF-спайков. Исторический отдельный тест другой пары/состояния уже показал нарушение аддитивности 44.063% (WL0, BL2/3); он не переносится количественно на текущую маску. Программируемость и CIFAR mapping не подтверждены; ADC0 не доказывает физический нулевой ток/offset. Pre/post G в режиме 9 проверяет только сохранённую сессию, не состояние платы сейчас.', '',
        '## Связь с Conv → BN → LIF','',
        'Проверенный LatencyEncoder формирует 0/1 на входе Conv1. Код модели выполняет Conv→BN[t]→LIF→AvgPool; LIF цифровой: v_new=tau*v+i, threshold, hard reset. Binary encoding не отменяет сумму Σw*s. При фиксированном V* возможный эффективный вес определяется как Fcell(V*)−Fcell(0), а не автоматически 330/R; это ещё требует проверки масштаба и совместного суммирования. BN/LIF не исправляют автоматически неизвестную нелинейность суммы. После AvgPool2×2 входы следующих слоёв могут быть 0,.25,.5,.75,1. Источники/хеши и ограниченное предложение следующего теста сохранены в BINARY_SPIKE_NOTES.md в корне репозитория. Модель здесь не исполнялась.', '',
        '## Воспроизведение','', '```sh',
        'python analyze_binary_spikes.py --repo . --output-dir results/binary_spike_review/new_review','```','',
        'Папка результата должна быть новой. Проверены фиксированные SHA шести входов, связь audit/manifest/raw; заново извлечены исходные отсчёты. Полный протокольный аудит исходного журнала повторно не выполнялся. analysis.json хранит все исходные task IDs/отсчёты, формулы,256 строк и оба набора 105 разбиений; cells.csv —256 строк с двумя амплитудами. Старые результаты не переписаны.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(candidate_counts=counts,coverage={k:{f:v[f] for f in ('maximum_capacity_sum','maximum_minimum_pair_capacity','shared_bl_all_eight_wl')} for k,v in covers.items()})))


if __name__=='__main__':main()
