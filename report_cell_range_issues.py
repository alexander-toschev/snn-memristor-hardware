"""Offline developer evidence report: latest full map plus separately versioned interval runs."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
from statistics import mean,stdev

HISTORY='results/cell_calibrations/3380d497-ded0-4881-9df2-85ef1c7e32d9/report/calibration.json'
HISTORY_SHA='fe60dc81922cea3c25f64b6336693b522caff220468bb6b7e0225ec6fcc2884a'
PROTOCOL_FLAGS={'three-cell-native-R-interval-v1':'--test-cell-ranges',
                'three-cell-native-R-interval-v2-refined':'--test-cell-ranges-refined'}
PALETTE={'candidate':('#39986b','Кандидат текущего screen'),
 'zero_both':('#b6bcc6','Оба наблюдённых сигнала нулевые'),
 'weak_both':('#9876bc','Оба сигнала слабые'),
 'measured_cv_fail':('#619bd1','Измеренный CV выше допуска'),
 'weak_low':('#edaa58','Слабый сигнал 0.05 V'),
 'weak_high':('#9876bc','Слабый сигнал 0.1 V'),
 'nonlinear':('#da6b67','Нелинейный отклик'),
 'g_drift':('#c09a70','G pre/post вне допуска'),
 'other':('#e4dddd','Не квалифицировано / другая причина')}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def primary(cell):
    m=cell['metrics'];r=cell['reasons']
    if cell['status']=='candidate':return 'candidate'
    if m['low_adc']['mean']==m['high_adc']['mean']==m['zero_adc']['mean']==0:return 'zero_both'
    if m.get('cv_high') is not None and 'unstable_high_signal' in r:return 'measured_cv_fail'
    low='weak_or_nonpositive_signal_low' in r;high='weak_or_nonpositive_signal_high' in r
    if low and high:return 'weak_both'
    if low:return 'weak_low'
    if high:return 'weak_high'
    if 'nonlinear_zero_adjusted_response' in r:return 'nonlinear'
    if any('conductance' in reason or 'drift' in reason for reason in r):return 'g_drift'
    return 'other'


def explanations(cell,criteria):
    m=cell['metrics'];out=[];rules=criteria['criteria']
    for side in ('low','high'):
        reason='weak_or_nonpositive_signal_'+side
        if reason in cell['reasons']:
            out.append(f"{side}: signal={m['signal_'+side+'_adc']:.9g} ADC; SNR={m['snr_'+side]:.9g}; требуется положительный signal и SNR≥{rules['minimum_snr']} (noise=max({rules['noise_floor_adc']},sqrt(SDlevel²+SDzero²))).")
    if 'unstable_high_signal' in cell['reasons']:
        if m['cv_high'] is None:out.append('CVhigh не определён: high signal≤0. Это не доказательство высокого шума.')
        else:out.append(f"CVhigh={m['cv_high']:.9g} >{rules['maximum_cv_high']}.")
    if 'nonlinear_zero_adjusted_response' in cell['reasons']:
        out.append(f"ε=high−2low+zero={m['linearity_epsilon_adc']:.9g} ADC; |ε|>{m['linearity_tolerance_adc']:.9g} ADC =max({100*rules['linearity_fraction']:g}%|high signal|,{rules['linearity_se_multiplier']}SE,{rules['linearity_floor_adc']} ADC).")
    if abs(m['g_fractional_delta'])>m['g_fractional_tolerance']:
        out.append(f"G pre/post: Δ/G={m['g_fractional_delta']:.9g}; |Δ/G|>{m['g_fractional_tolerance']:.9g}=max({rules['drift_fraction']},{rules['drift_se_multiplier']}SE/Gpre).")
    if not out:out=['Все предварительные критерии одиночного отклика текущей карты пройдены; программируемость и суммирование не подтверждены.']
    return out


def interval_explanations(attempts):
    lines=[]
    for t in attempts:
        prefix=f"{t['session_id']} cycle{t['cycle_id']} target q={t['target_q']}: {t['status']}"
        detail=f"; R inclusive={t['native_r_interval_ohm']} Ω"
        if 'stop_native_r_interval_ohm' in t:detail+=f"; stop R={t['stop_native_r_interval_ohm']} Ω, stop tolerance={t['stop_q_relative_tolerance']}"
        if 'evidence_task_id' in t:detail+=f"; evidence task={t['evidence_task_id']}, q={t.get('evidence_q')}"
        if 'hit_origin' in t:detail+='; origin='+t['hit_origin']
        for phase in ('pre','post','delayed'):
            if phase in t:
                data=t[phase];q=data['q']
                detail+=f"; {phase} all_inside={data['all_inside']}, q[min,max]=[{q['minimum']},{q['maximum']}]"
        if 'mvm_screen' in t:
            s=t['mvm_screen'];detail+=f"; MVM pass={s['passes']}, SNRlow/high={s['snr_low']:.7g}/{s['snr_high']:.7g}, CVhigh={s['cv_high']}, ε={s['epsilon_adc']:.7g}, tolerance={s['tolerance_adc']:.7g} ADC"
        if 'delay' in t:detail+=f"; delay={t['delay']['elapsed_s']:.6g}s, met={t['delay']['delay_met']}"
        if 'interval_and_mvm_pass' in t:detail+=f"; combined pass={t['interval_and_mvm_pass']}"
        if 'error' in t:detail+='; '+t['error']
        lines.append(prefix+detail)
    return lines


def mvm_transfer(targets):
    """Descriptive saved-data estimates only: no empirical gain correction or pair qualification."""
    groups={}
    for t in targets:
        if t['source']!='hardware' or len(t.get('mvm',[]))!=30 or 'pre' not in t:continue
        groups.setdefault((t['session_id'],t['wl'],t['bl'],t['target_q']),[]).append(t)
    states=[]
    for (sid,w,b,q),trials in sorted(groups.items()):
        levels=[]
        for pos,v in enumerate((0.,.025,.05,.075,.1,0.)):
            samples=[s for t in trials for s in t['mvm'] if s['position']==pos]
            if len(samples)!=5*len(trials) or any(s['nominal_v']!=v for s in samples):
                raise ValueError('Incomplete or relabelled MVM sequence')
            percycle=[mean(s['raw_adc'] for s in t['mvm'] if s['position']==pos) for t in trials]
            levels.append(dict(position=pos,requested_input_v=v,n=len(samples),
                adc_mean=mean(s['raw_adc'] for s in samples),output_v_mean=mean(s['output_voltage_v'] for s in samples),
                cycle_means_adc=percycle,cycle_mean_adc_sd=stdev(percycle) if len(percycle)>1 else None))
        preq=mean(t['pre']['q']['mean'] for t in trials)
        # Fit only nonzero requested levels; zero may be ADC-clipped.
        xs=[x['requested_input_v'] for x in levels[1:5]];ys=[x['output_v_mean'] for x in levels[1:5]]
        mx,my=mean(xs),mean(ys);slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
        intercept=my-slope*mx;rmse=math.sqrt(mean((y-(slope*x+intercept))**2 for x,y in zip(xs,ys)))
        states.append(dict(session_id=sid,wl=w,bl=b,target_q=q,cycles=len(trials),
            cycle_ids=[t['cycle_id'] for t in trials],all_combined_pass=all(t.get('interval_and_mvm_pass') is True for t in trials),
            pre_q_mean=preq,levels=levels,
            nominal_q_times_input_v=preq*.1,observed_output_over_q_times_input=levels[4]['output_v_mean']/(preq*.1),
            nonzero_requested_input_affine_fit=dict(slope_v_per_v=slope,intercept_v=intercept,rmse_v=rmse),
            zero_adc_proves_physical_zero_offset=False,physical_input_voltage_measured=False))
    pairs=[]
    for sid in sorted({s['session_id'] for s in states}):
        selected={(s['wl'],s['target_q']):s for s in states if s['session_id']==sid and s['bl']==8 and s['wl'] in (0,3) and s['target_q'] in (.045,.05)}
        if len(selected)!=4:continue
        levels=[]
        for pos in range(1,5):
            aL,aH,bL,bH=(selected[k]['levels'][pos]['adc_mean'] for k in ((0,.045),(0,.05),(3,.045),(3,.05)))
            hl,lh=aH-bL,aL-bH
            levels.append(dict(requested_input_v=selected[0,.045]['levels'][pos]['requested_input_v'],
                LL_adc=aL-bL,HH_adc=aH-bH,HL_adc=hl,LH_adc=lh,
                descriptive_center_adc=(hl+lh)/2,descriptive_half_swing_adc=(hl-lh)/2,
                delta_WL0_adc=aH-aL,delta_WL3_adc=bH-bL))
        eps=levels[3]['descriptive_half_swing_adc']-2*levels[1]['descriptive_half_swing_adc']
        pairs.append(dict(session_id=sid,logical_WLs=[0,3],BL=8,low_target_q=.045,high_target_q=.05,
            simultaneously_programmed_or_measured=False,pair_qualified=False,levels=levels,
            half_swing_high_minus_twice_low_adc=eps,
            half_swing_fractional_epsilon=eps/levels[3]['descriptive_half_swing_adc'] if levels[3]['descriptive_half_swing_adc'] else None))
    return dict(states=states,descriptive_non_simultaneous_pairs=pairs,
        interpretation='Output uses the native GUI a2v(gain=1). q uses mode7 native R and sum_gain=330. Requested input voltage is not independently measured. Ratios are observed unresolved discrepancies, not a physical gain calibration. Pair estimates subtract separate historical single-cell measurements; no pair was jointly programmed or measured.')


def transfer_markdown(transfer):
    lines=['## Масштаб MVM и пригодность для весов','',
        'Native GUI и эксперимент используют q=sum_gain/R и output=a2v(gain=1). Ниже отношение наблюдённого output к q×заданному Vin. Это не найденный коэффициент аппаратного усиления: фактическое напряжение и физический тракт независимо не откалиброваны. Нулевой ADC может означать отсечку и не доказывает физически нулевое смещение.','',
        '| Сессия | WL/BL | q target / pre mean | Циклы | Vout при заданных 0.1 V | Vout/(q×0.1) | slope / intercept fit |',
        '|---|---|---|---|---|---|---|']
    for s in transfer['states']:
        f=s['nonzero_requested_input_affine_fit']
        lines.append(f"| {s['session_id']} | {s['wl']}/{s['bl']} | {s['target_q']:.6g} / {s['pre_q_mean']:.9g} | {s['cycles']} | {s['levels'][4]['output_v_mean']:.9g} V | {s['observed_output_over_q_times_input']:.6g} | {f['slope_v_per_v']:.7g} V/V; {f['intercept_v']:.7g} V |")
    lines+=['','Fit использует четыре ненулевых заданных входа .025/.05/.075/.1 V; реальные DAC codes и native decoded V остаются в raw. Это описательная аппроксимация средних, без validation на отложенных данных. Все завершённые checkpoint MVM включены независимо от успешности interval screen.','',
        '### Расчётные разности двух ячеек — совместного измерения не было','',
        'D=WL0−WL3 на BL8, L=q .045, H=q .05. LL/HL/LH/HH ниже вычислены из отдельных исторических средних. Они не подтверждают совместное сохранение состояний, корреляцию шума или работу дифференциальной пары.','',
        ]
    for pair in transfer['descriptive_non_simultaneous_pairs']:
        lines+=['| Сессия | Заданный Vin | LL | HH | HL | LH | (HL−LH)/2 |','|---|---|---|---|---|---|---|']
        for v in pair['levels']:
            lines.append(f"| {pair['session_id']} | {v['requested_input_v']:.3g} | {v['LL_adc']:.7g} | {v['HH_adc']:.7g} | {v['HL_adc']:.7g} | {v['LH_adc']:.7g} | {v['descriptive_half_swing_adc']:.7g} |")
        ratio=pair['half_swing_fractional_epsilon']
        lines+=['',f"Сессия {pair['session_id']}: half-swing(.1)−2×half-swing(.05)={pair['half_swing_high_minus_twice_low_adc']:.9g} ADC; доля от high half-swing={ratio if ratio is None else format(ratio,'.9g')}. Все числа таблицы — ADC codes, не новые измерения.",'']
    lines+=['','Прохождение 5% screen отдельных больших выходов не гарантирует линейность меньшей разности между весовыми состояниями. Прямое q→MVM масштабирование и CIFAR mapping пока не квалифицированы.','',
        '### Следующий тест пары — предложение, здесь не выполнен','',
        'На логических WL0/WL3, BL8 последовательно установить LL, HL, LH, HH (L=.045, H=.05). После записи второго элемента перечитать оба, чтобы проверить совместное сохранение. Для каждого состояния чередовать чтения WL0/WL3 при zero/.025/.05/.075/.1 V, пять повторов; сохранять raw ADC, zero до/после, readback обоих состояний и токовые/ADC guards. Повторить независимо программируемые циклы с фиксированным порядком или заранее зафиксированной перестановкой состояний.','',
        'Оценивать совместный offset, знак и амплитуду HL/LH, равновесные LL/HH, дифференциальную линейность по входу и повторяемость. Параметры возможной эмпирической калибровки определять на части входов/циклов и проверять на остальных; не подгонять допуски по результату. Даже успешно проверенная одна пара — отдельный коэффициент, а не четыре CIFAR-фильтра или все256 ячеек.']
    return lines


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    parser.add_argument('--interval-session',action='append',required=True,help='Repeat for each separately versioned run')
    parser.add_argument('--map-session',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();root=Path(args.repo).resolve();inputs={}
    def local(name):
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Use repository-relative paths')
        p=(root/name).resolve()
        if not p.is_relative_to(root):raise ValueError('Path escapes repository')
        return p
    def read(name):
        p=local(name);inputs[name]=sha(p);return load(p)
    def verify_session(name,audit_name):
        manifest=read(name+'/manifest.json');audit=read(name+'/'+audit_name)
        if audit['audit_status']!='passed' or audit['errors']:raise ValueError('Independent audit did not pass: '+name)
        if audit['manifest_sha256']!=inputs[name+'/manifest.json']:raise ValueError('Audit/manifest hash disagreement')
        for file,digest in manifest['result_sha256'].items():
            p=(local(name)/file).resolve()
            if not p.is_relative_to(local(name)):raise ValueError('Manifest path escapes session')
            actual=sha(p)
            if actual!=digest:raise ValueError('Result SHA mismatch: '+str(p))
            inputs[name+'/'+file]=actual
        if audit['raw_sha256']!=inputs[name+'/raw.jsonl']:raise ValueError('Audit/raw hash disagreement')
        return manifest,audit
    out=local(args.output_dir)
    if out.exists():raise ValueError('Use a new output directory')
    history=read(HISTORY)
    if inputs[HISTORY]!=HISTORY_SHA:raise ValueError('Historical compact report fingerprint changed')
    map_manifest,map_audit=verify_session(args.map_session,'offline-audit/independent_cell_map_audit.json')
    map_summary=read(args.map_session+'/summary.json');criteria=read(args.map_session+'/criteria.json')
    if map_manifest['source']!='hardware' or map_summary['status']!='completed' or not map_summary['mask_usable']:
        raise ValueError('Current map must be a completed audited physical map')
    cells=read(args.map_session+'/cells.json')
    if len(cells)!=256 or {(c['wl'],c['bl']) for c in cells}!={(w,b) for w in range(8) for b in range(32)}:raise ValueError('Expected unique all256 map')
    runs=[];all_targets=[];raw_samples={(w,b):{'pre':[],'post':[],'low':[],'high':[]} for w in range(8) for b in range(32)}
    zero={w:[] for w in range(8)};map_modes=Counter()
    for line in local(args.map_session+'/raw.jsonl').open(encoding='utf-8'):
        r=json.loads(line)
        if r['event']!='result':continue
        command=r['command_si'];mode=command['mode_flag'];map_modes[mode]+=1
        sample={k:r.get(k) for k in ('task_id','raw_adc','native_r','resistance_ohm','conductance_siemens','output_voltage_v','repeat','status')}
        if mode==9:raw_samples[command['wl'],command['bl']][r['stage']].append(sample)
        elif mode==10:
            active=[i for i,v in enumerate(command['vol']) if v]
            if not active:zero[command['wl']].append(sample)
            elif len(active)==1:
                if max(command['vol']) not in (.05,.1):raise ValueError('Unsupported map input levels; do not silently relabel measurements')
                raw_samples[command['wl'],active[0]]['low' if max(command['vol'])==.05 else 'high'].append(sample)
    for name in args.interval_session:
        manifest,audit=verify_session(name,'offline-audit/independent_interval_audit.json')
        summary=read(name+'/summary.json');plan=read(name+'/protocol.json')
        if plan['protocol_id'] not in PROTOCOL_FLAGS:raise ValueError('Unknown reproduction protocol')
        targets=[json.loads(line) for line in local(name+'/targets.jsonl').read_text(encoding='utf-8').splitlines()]
        sid=manifest['session_id']
        run=dict(session=name,session_id=sid,source=manifest['source'],status=summary['status'],protocol=plan,
            source_commit=manifest['invocation']['git']['commit'],source_sha256=manifest['source_sha256'],
            manifest_sha256=inputs[name+'/manifest.json'],raw_sha256=inputs[name+'/raw.jsonl'],
            finished_utc=manifest['finished_utc'],audit_checks=audit['checks'],outcomes=audit['outcomes'],
            attempted_cells=sorted({(c['wl'],c['bl']) for c in summary['cycles'] if c['impact_attempts']>0}),
            reproduction_command='python experiment.py '+PROTOCOL_FLAGS[plan['protocol_id']]+' --run-'+manifest['source'],
            dry_run_command='python experiment.py '+PROTOCOL_FLAGS[plan['protocol_id']]+' --dry-run',
            thresholds_reinterpreted=False)
        runs.append(run)
        for target in targets:all_targets.append(dict(session_id=sid,source=manifest['source'],protocol_id=plan['protocol_id'],**target))
    latest_map_start=datetime.fromisoformat(map_manifest['started_utc'].replace('Z','+00:00'))
    current_after=all(latest_map_start>=datetime.fromisoformat(r['finished_utc'].replace('Z','+00:00')) for r in runs if r['source']=='hardware')
    if not current_after:raise ValueError('Map predates a supplied hardware interval run; obtain the final read-only map before calling it current')
    rows=[]
    for cell in sorted(cells,key=lambda c:(c['wl'],c['bl'])):
        w,b=cell['wl'],cell['bl'];samples={**raw_samples[w,b],'zero':zero[w]}
        observed=[s for s in history['states'] if (s['wl'],s['bl'])==(w,b)]
        historical_cell=next(c for c in history['cells'] if (c['wl'],c['bl'])==(w,b))
        old=[s for s in observed if s['passes']]
        attempts=[t for t in all_targets if (t['wl'],t['bl'])==(w,b)]
        rows.append(dict(wl=w,bl=b,primary=primary(cell),map_status=cell['status'],map_reasons=cell['reasons'],
            exact_map_findings=explanations(cell,criteria),map_metrics=cell['metrics'],map_raw_samples=samples,
            new_hardware_interval_tested=any(r['source']=='hardware' and (w,b) in r['attempted_cells'] for r in runs),interval_runs=attempts,
            exact_interval_findings=interval_explanations(attempts),
            historical_mode7_observed_states=observed,historical_mode7_passing_points=old,
            historical_calibration_cell=historical_cell,
            range_qualified=False,physical_defect_diagnosed=False))
    candidates={w:{c['bl'] for c in cells if c['wl']==w and c['status']=='candidate'} for w in range(8)}
    pair_coverage=[dict(wl_a=a,wl_b=b,common_candidate_bl=sorted(candidates[a]&candidates[b]),
        count=len(candidates[a]&candidates[b]),programmed_pair_or_sum_validated=False) for a,b in itertools.combinations(range(8),2)]
    zero_block=[]
    for row in rows:
        samples=row['map_raw_samples']
        if 16<=row['bl']<=30 and len(samples['low'])==len(samples['high'])==5 and all(s['raw_adc']==0 for s in samples['low']+samples['high']):
            zero_block.append([row['wl'],row['bl']])
    counts=dict(Counter(row['primary'] for row in rows))
    transfer=mvm_transfer(all_targets)
    report=dict(offline_only=True,new_measurement=False,generator_sha256=sha(Path(__file__)),input_sha256=inputs,
        historical_calibration_report=HISTORY,current_map=args.map_session,map_after_all_supplied_hardware_interval_runs=current_after,
        map_commit=map_manifest['invocation']['git']['commit'],map_source_sha256=map_manifest['source_sha256'],
        map_criteria=criteria,map_mode_counts=dict(map_modes),interval_runs=runs,primary_counts=counts,
        current_candidates=[[r['wl'],r['bl']] for r in rows if r['map_status']=='candidate'],
        new_hardware_interval_cell_count=sum(r['new_hardware_interval_tested'] for r in rows),
        zero_MVM_block_BL16_30=zero_block,pair_coverage=pair_coverage,cells=rows,
        measured_transfer_analysis=transfer,
        interpretation='Logical Connector addresses. Current single-cell screen, historical mode7 q and new interval repeats remain separate; no continuous programmable range, physical defect or CIFAR qualification is inferred.')
    out.mkdir(parents=True)
    (out/'developer_issues.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (out/'cells.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        fields=['wl','bl','primary','map_status','new_hardware_interval_tested','map_reasons','exact_map_findings','exact_interval_findings','interval_runs','historical_calibration_cell','historical_mode7_observed_states','historical_mode7_passing_points']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(row[k],ensure_ascii=False) if isinstance(row[k],(list,dict)) else row[k] for k in fields})
    lines=['# Проблемы ячеек и воспроизведение для разработчика','',
        f"Карта `{args.map_session}`: {len(report['current_candidates'])} предварительных кандидатов. Новые аппаратные интервальные прогоны относятся к {report['new_hardware_interval_cell_count']} адресам; остальные адреса не проходили этот новый interval test. Карта снята после всех приложенных аппаратных interval runs: **{current_after}**.",'',
        'Каждый интервал/порог сохранён со своим protocol.json и commit. Результаты исходного критерия не заменяются интерпретацией расширенного окна. Min/max наблюдённых q не доказывают непрерывный диапазон.', '',
        '| Сессия | Source / protocol | Acquisition | Targets | Interval+MVM observed pass |', '|---|---|---|---|---|']
    for run in runs:lines.append(f"| {run['session_id']} | {run['source']} / {run['protocol']['protocol_id']} | {run['status']} | {run['outcomes']['target_statuses']} | {run['outcomes']['target_observed_pass']} |")
    lines+=['',f"BL16–30: {len(zero_block)}/120 логических адресов имеют все 10 basis MVM ADC=0. Наличие mode9 отклика и систематический блок требуют проверки тракта/адресации разработчиком; причина не установлена. Нулевой ADC не доказывает нулевую физическую проводимость.",'',
        f"Ненулевые пересечения candidate BL для WL-пар: {[p for p in pair_coverage if p['count']]}. Это лишь пересечения маски одиночных откликов; физическое соответствие линий, совместная запись и дифференциальное суммирование ими не подтверждены.",'',
        '## Воспроизведение', '',
        'Команды выполняются оператором из корня репозитория на соответствующем записанном commit. dry-run проверяет подготовку; --run-hardware запускает указанное воздействие. Приватные драйверы предоставляются владельцем платы и проверяются по recorded pins; INI сохранён в каждой сессии.', '',
        '| Interval session | Source commit | Dry run | Run |','|---|---|---|---|']
    for run in runs:lines.append(f"| {run['session_id']} | `{run['source_commit']}` | `{run['dry_run_command']}` | `{run['reproduction_command']}` |")
    lines+=['',f"Карта: commit `{report['map_commit']}`.",'',
        '```sh','python experiment.py --map-cells --dry-run',
        'python experiment.py --map-cells --run-hardware','```','',
        'Interval-команда выполняет запись только трёх адресов, заданных протоколом выбранного commit. Map-команда читает все256 через mode9/mode10 без SET/RESET. CLI для произвольного одиночного адреса здесь не выдуман. Для каждого сравниваемого interval run нужен его source_commit и protocol; одного latest checkout недостаточно для точного повторения иной версии.', '',
        '| WL,BL | Текущий screen | Новые interval tests | Точные причины / данные |', '|---|---|---|---|']
    for row in rows:
        findings=' '.join(row['exact_map_findings']).replace('|','\\|')
        observed=row['historical_mode7_observed_states']
        findings+=' Исторические mode7 checkpoints (роль:q, MVM screen): '+', '.join(f"{s['role']}:{s['q_mean']} ({'pass' if s['passes'] else 'fail'})" for s in observed)+'. Это отдельные наблюдения, не непрерывный диапазон.'
        h=row['historical_calibration_cell']
        findings+=f" Исторический RESET crossing/confirmed={h.get('reset_reached')}/{h.get('reset_confirmed_before_mvm')}, RESET/SET pulses={h.get('reset_pulses')}/{h.get('set_pulses')}; terminal q={h.get('terminal_q')}, terminal MVM={h.get('terminal_transfer_pass')}, late G delta={h.get('final_g_fractional_delta')}, tolerance={h.get('final_g_tolerance')}. Изменение late G не приписывается одной причине."
        lines.append(f"| {row['wl']},{row['bl']} | {PALETTE[row['primary']][1]} | {'да' if row['new_hardware_interval_tested'] else 'нет; только исторические данные'} | {findings} |")
    lines+=['','## Новые интервальные наблюдения по адресам','']
    for row in rows:
        if not row['interval_runs']:continue
        lines += [f"### WL{row['wl']}/BL{row['bl']}",'']
        for run in runs:
            attempts=[t for t in row['interval_runs'] if t['session_id']==run['session_id']]
            for target in sorted({t['target_q'] for t in attempts}):
                trials=[t for t in attempts if t['target_q']==target]
                passed=sum(t.get('interval_and_mvm_pass') is True for t in trials)
                lines.append(f"- Сессия {run['session_id']}, q target={target}: {passed}/{len(trials)} наблюдённых combined pass; запланировано {run['protocol']['cycles_per_cell']} циклов. Малая серия, условный результат данного протокола.")
        lines+=['']+['- '+line for line in row['exact_interval_findings']]+['']
    lines+=['']+transfer_markdown(transfer)+['']
    lines+=['','Полные ADC/task IDs, наблюдения интервалов по циклам, INI/source commits/hashes находятся в developer_issues.json и исходных сессиях. Все доступные result SHA и связь audit/manifest/raw проверены; численные raw-аудиты отдельно выполнены специализированными аудиторами. Этот отчёт не запускает аппаратуру.','',
        'CIFAR: ни прохождение отдельных ячеек, ни пересечение масок не подтверждают размещение четырёх фильтров на всех256 ячейках. Нужны подтверждённые программируемые уровни, физическое соответствие линий и проверка суммирования; такой результат здесь не заявляется.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    esc=html.escape
    page=['<!doctype html><meta charset="utf-8"><title>Карта проблем ячеек</title><style>body{font:15px system-ui;margin:24px;max-width:1500px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}a{color:inherit}pre{white-space:pre-wrap;font-size:12px}details{border-bottom:1px solid #ddd;padding:8px}.matrix td{min-width:28px;text-align:center}.note{background:#fff3ca;padding:12px}</style>',
        '<h1>256 адресов: текущая карта и отдельные интервальные прогоны</h1>',
        '<p class="note">Цвет — предварительный screen текущего одиночного отклика. Он не означает физическую неисправность или подтверждённый программируемый диапазон. Нажмите адрес для точных чисел и ADC.</p>',
        f'<p>Карта после всех supplied hardware interval runs: <b>{current_after}</b>. Кандидатов: {len(report["current_candidates"])}. Новые interval tests: {report["new_hardware_interval_cell_count"]} адреса.</p>',
        '<p>'+ '; '.join(f'<span style="background:{PALETTE[k][0]};padding:3px">{esc(PALETTE[k][1])}: {n}</span>' for k,n in counts.items())+'</p>',
        '<table class="matrix"><tr><th>WL / BL</th>'+''.join(f'<th>{b}</th>' for b in range(32))+'</tr>']
    for w in range(8):
        page.append(f'<tr><th>{w}</th>')
        for row in [r for r in rows if r['wl']==w]:
            b=row['bl'];color=PALETTE[row['primary']][0];mark='*' if row['new_hardware_interval_tested'] else ''
            page.append(f'<td style="background:{color}"><a href="#cell-{w}-{b}">{w},{b}{mark}</a></td>')
        page.append('</tr>')
    page+=['</table><p>* — адрес имеет новый аппаратный interval test; результаты по отдельным целям и циклам ниже. Остальные адреса имеют текущую read-only карту и исторические данные.</p>',
        '<p><a href="REPORT.md">Отчёт и команды воспроизведения</a> · <a href="cells.csv">CSV</a> · <a href="developer_issues.json">Все данные JSON</a></p>']
    page+=['<details open><summary>Масштаб MVM и границы перехода к CIFAR</summary><pre>'+esc('\n'.join(transfer_markdown(transfer)))+'</pre></details>']
    for row in rows:
        page.append(f'<details id="cell-{row["wl"]}-{row["bl"]}"><summary>WL{row["wl"]}/BL{row["bl"]}: {esc(PALETTE[row["primary"]][1])}</summary><p>{esc(" ".join(row["exact_map_findings"]))}</p><pre>{esc(json.dumps(row,ensure_ascii=False,indent=2))}</pre></details>')
    (out/'index.html').write_text('\n'.join(page)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=args.output_dir,cells=len(rows),current_candidates=report['current_candidates'],counts=counts,new_interval_cells=report['new_hardware_interval_cell_count'],map_after_all_hardware_runs=current_after)))


if __name__=='__main__':main()
