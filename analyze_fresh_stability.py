"""Pure offline science report from independently audited fresh300 artifacts."""
import argparse
import csv
from collections import Counter,defaultdict
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics as st

AUDITORS={'bfab9b2044388c11a6362e94471b7872fab134451e8c09187eda59c9e8caa5bb':'fresh300-jit-capacity-fixed-gain-v1',
          '07d2dd0faecff90360df9c9c1771746b8d01045fb10119fa626263a3f201e9ca':'fresh300-jit-capacity-no-bl8-fixed-gain-v1'}
SIDES=('positive_cell','negative_cell')

def require(v,message):
    if not v:raise ValueError(message)

def load(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def local(root,name):
    p=(root/name).resolve();require(not Path(name).is_absolute() and p.is_relative_to(root),'Outside repository');return p

def rms(v):return math.sqrt(st.mean(x*x for x in v)) if v else None

def weight_metrics(rows):
    if not rows:return None
    w=[x['weight'] for x in rows];scale=rms(w)
    return dict(count=len(rows),trained_rms=scale,**{stage:dict(rmse=rms([x[stage]-x['weight'] for x in rows]),sse=sum((x[stage]-x['weight'])**2 for x in rows),
        nrmse=rms([x[stage]-x['weight'] for x in rows])/scale,
        sign_mismatches=sum(x[stage]*x['weight']<0 for x in rows),
        exactly_zero=sum(x[stage]==0 for x in rows)) for stage in ('post_own','post_all_raw','post_cifar_raw','effective')},
        later_write_change_rms=rms([x['post_all_raw']-x['post_own'] for x in rows]),
        during_cifar_change_rms=rms([x['post_cifar_raw']-x['post_all_raw'] for x in rows]))

def trace_metrics(trajectories,fixture):
    output={}
    for ref in ('trained','effective'):
        records=[r['comparison'][ref] for r in trajectories]
        output[ref]=dict(conv_rmse=rms([r['conv_rmse'] for r in records]),
            spike_differences=sum(r['spike_mismatches'] for r in records),total_timepoints=12*len(records),
            **{name+'_max_abs_error':max((r[name+'_max_abs_error'] for r in records),default=None)
               for name in ('bntt_current','lif_before_reset','lif_after_reset')})
    groups=defaultdict(list)
    for r in trajectories:
        for t,x in enumerate(r['actual']):groups[(r['filter_id'],r['case'],t)].append(x)
    require(all(len(v)==5 for v in groups.values()),'Missing repeats')
    unique={name:set() for name in ('trained','effective')}
    for row in trajectories:
        for t,x in enumerate(row['actual']):
            trained=fixture['cases'][row['case']]['reference_t12x64']['lif_spikes'][t][row['filter_id']]
            effective=row['effective_weight_reference'][t]['lif_spike']
            for name,reference in (('trained',trained),('effective',effective)):
                if x['lif_spike']!=reference:unique[name].add((row['filter_id'],row['case'],t))
    for name in unique:output[name]['unique_logical_points_with_any_spike_difference']=len(unique[name])
    output['repeat_variation']=dict(logical_points=len(groups),
        conv_sample_sd_rms=rms([st.stdev(x['conv_sum'] for x in v) for v in groups.values()]),
        lif_before_reset_sample_sd_rms=rms([st.stdev(x['lif_before_reset'] for x in v) for v in groups.values()]),
        points_with_any_spike_disagreement=sum(len({x['lif_spike'] for x in v})>1 for v in groups.values()))
    return output

def analyze(repo,session,audit):
    d=local(repo,session);ap=local(repo,audit);a=load(ap);m=load(d/'manifest.json')
    require(a['audit_status']=='passed' and not a['errors'] and a['auditor_sha256'] in AUDITORS,'Independent audit not PASS or wrong frozen auditor')
    require(m['protocol_id']==AUDITORS[a['auditor_sha256']],'Auditor/protocol mismatch')
    require(a['manifest_sha256']==sha(d/'manifest.json') and a['session_id']==m['session_id'] and a['source']==m['source'],'Audit/session binding mismatch')
    for name,digest in m['result_sha256'].items():require(sha(local(d,name))==digest,'Recorded artifact changed: '+name)
    fixture_path='weights/prepared_cifar100_conv1.json'
    require(sha(local(repo,fixture_path))==m['source_sha256'][fixture_path],'Input fixture hash mismatch')
    fixture=load(local(repo,fixture_path))
    route=load(d/'planned_route.json');protocol=load(d/'protocol.json')
    require(protocol['protocol_id']==m['protocol_id'],'Protocol/manifest mismatch')
    gain=protocol['target_gain_v_per_weight'][m['source']]
    require(gain==(.25 if m['source']=='hardware' else .03),'Unexpected fixed source gain')
    cells=defaultdict(lambda:dict(probes=[],selected=[],SET=[],failures=[]));weights=[];tiles=[];reset_stops=Counter();set_stops=Counter()
    for f in route['filters']:
        for t in f['tiles']:
            path=d/'tiles'/f"f{f['filter_id']}_t{t['tile_id']}"/'tile.json'
            if not path.exists():continue
            raw=load(path);fid=f['filter_id'];tid=t['tile_id']
            for reset in raw['reset_attempts']:
                reset_stops[reset['stop']]+=1
                for side in SIDES:
                    cell=cells[tuple(reset[side])];last=reset['joint_gates'][-1];j=SIDES.index(side)
                    values=[x['row_sums_v'][j] for x in last['samples']]
                    reason=dict(filter=fid,tile=tid,tap=reset['pair_tap'],attempt=reset['candidate_attempt'],
                        eligible_pair=reset['eligible'],own_final_joint_quiet=last['quiet_by_side'][side],
                        joint_mean_v=st.mean(values),joint_sd_v=st.stdev(values),quiet_limit_v=last['quiet_limit_v'],
                        reset_pulses=reset['legs'][side]['next_step'],joint_rounds=len(reset['joint_gates']),
                        observed_stop=reset['stop'])
                    cell['probes'].append(reason)
                    if not last['quiet_by_side'][side]:cell['failures'].append(reason)
            bytap={p['tap']:p for p in raw['pairs']}
            for p in raw['pairs']:
                set_stops[p['stop']]+=1
                evidence=dict(filter=fid,tile=tid,tap=p['tap'],stop=p['stop'],target_signed_v=p['target_signed_v'],
                    initial_signed_v=p['initial_signed_v'],post_mean_signed_v=p['post_mean_signed_v'],
                    post_sd_signed_v=p['post_sd_signed_v'],SET_pulses=p['programming_pulse_attempts'])
                for side in SIDES:cells[tuple(p[side])]['selected'].append({**evidence,'role':side,'active':side==p['active_set_side']})
                if p['programming_pulse_attempts']:cells[tuple(p[p['active_set_side']])]['SET'].append(evidence)
            if not raw['technical_completed']:continue
            anchors=load(path.parent/'calibration_anchors.json')
            for j,s in enumerate(raw['effective_route']):
                p=bytap[s['tap_index']]
                legs=leg_evidence(p,anchors,raw['final_anchors'],j)
                for side in SIDES:
                    cells[tuple(s[side])].setdefault('leg_observations',[]).append(dict(filter=fid,tile=tid,tap=s['tap_index'],role=side,**legs[side]))
                weights.append(dict(filter=fid,tile=tid,tap=s['tap_index'],weight=s['trained_weight'],positive_cell=s[SIDES[0]],negative_cell=s[SIDES[1]],
                    leg_evidence=legs,has_physical_bl8=any(s[side][1]==8 for side in SIDES),
                    post_own=p['post_mean_signed_v']/gain,post_all_raw=raw['physical_fidelity']['raw_measured_weight'][j],
                    post_cifar_raw=raw['final_physical_fidelity']['raw_measured_weight'][j],effective=raw['calibration']['effective_weights'][j],
                    observed_post_all_sd=raw['physical_fidelity']['observed_weight_sd'][j],stop=p['stop'],
                    programming_attempts=p['programming_pulse_attempts'],candidate_attempt=p['candidate_attempt']))
            tiles.append(dict(filter=fid,tile=tid,taps=len(raw['effective_route']),matrix=raw['matrix']['matrix'],matrix_model=raw['matrix']['selected_model'],
                quality=raw['physical_weight_quality_pass'],raw_nrmse=raw['physical_fidelity']['raw_target_nrmse'],
                final_raw_nrmse=raw['final_physical_fidelity']['raw_target_nrmse'],uncertainty_nrms=raw['physical_fidelity']['uncertainty_nrms']))
    sse_by_stop=defaultdict(float)
    for w in weights:sse_by_stop[w['stop']]+=(w['post_all_raw']-w['weight'])**2
    total_sse=sum(sse_by_stop.values())
    complete=a['acquisition_complete'];summary=load(d/'summary.json') if complete else None
    require(not complete or len(weights)==300,'Complete flag without300 independently audited weights')
    require(len({(x['filter'],x['tap']) for x in weights})==len(weights),'Duplicate coefficient')
    trajectories=summary['trajectories'] if summary else []
    return dict(session=session,source=m['source'],session_id=m['session_id'],source_commit=m['invocation']['git']['commit'],
        manifest_sha256=sha(d/'manifest.json'),fixture_sha256=sha(local(repo,fixture_path)),
        trained_weights_sha256=hashlib.sha256(json.dumps(sorted((x['filter'],x['tap'],x['weight']) for x in weights)).encode()).hexdigest(),
        excluded_physical_bl=protocol.get('excluded_physical_bl',[]),audit=audit,audit_sha256=sha(ap),auditor_sha256=a['auditor_sha256'],audit_checks=a['checks'],
        complete=complete,decoder_gain=gain,gains_are_not_physical_equivalence=True,all_coefficients_fresh=True,
        total_impacts=m['total_impacts'],total_writes=m['total_pulses'],counters=a['counters'],coverage=a['actual_coverage'],
        full_time_outputs=a['full_time_outputs'],weight_metrics=weight_metrics(weights),
        per_filter=[dict(filter=f,weight_metrics=weight_metrics([x for x in weights if x['filter']==f]),
            traces=trace_metrics([r for r in trajectories if r['filter_id']==f],fixture) if trajectories else None) for f in range(4)],
        trace_metrics=trace_metrics(trajectories,fixture) if trajectories else None,tiles=tiles,weights=weights,
        reset_stops=dict(reset_stops),program_stops=dict(set_stops),
        post_all_error_sse_by_stop={k:dict(sse=v,fraction=v/total_sse if total_sse else 0) for k,v in sse_by_stop.items()},quality_tiles=sum(x['quality'] for x in tiles),
        bl8_groups={name:weight_metrics([w for w in weights if w['has_physical_bl8']==flag])
            for name,flag in [('contains_bl8',True),('no_bl8',False)]},
        cells=all_cell_records(cells,protocol.get('excluded_physical_bl',[]))),trajectories

def compare_traces(first,second):
    def index(rows):return {(r['filter_id'],r['case'],r['repeat'],t):x for r in rows for t,x in enumerate(r['actual'])}
    a=index(first);b=index(second);require(a.keys()==b.keys() and len(a)==2400,'Unmatched full outputs')
    return dict(timepoints=len(a),**{name+'_rmse':rms([a[k][name]-b[k][name] for k in a])
        for name in ('conv_sum','bntt_current','lif_before_reset','lif_after_reset')},
        spike_differences=sum(a[k]['lif_spike']!=b[k]['lif_spike'] for k in a),
        interpretation='Independent acquisitions; different programmed states, routes and measurement times may affect comparison. No causal attribution.')

def svg_map(report):
    palette={'untested':'#f1f5f9','quiet_failed_only':'#fca5a5','probed_only':'#c4b5fd','selected_without_SET':'#fde68a','SET_attempted':'#67cbe1'}
    by={tuple(x['cell']):x for x in report['cells']};out=['<svg xmlns="http://www.w3.org/2000/svg" width="1460" height="630" viewBox="0 0 1460 630">','<rect width="1460" height="630" fill="white"/>']
    def text(x,y,s,size=16):out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="#172033">{escape(str(s))}</text>')
    text(35,32,'Fresh Conv1 — '+report['source']+' — '+report['session_id'],21)
    text(35,60,'Coverage and operations only. Color never means accurate weight, stable cell, or permanent suitability.',17)
    for bl in range(32):text(87+bl*41,94,bl,12)
    for wl in range(8):
        text(26,123+wl*43,'WL'+str(wl),14)
        for bl in range(32):
            row=by.get((wl,bl));kind='untested'
            if row:kind='SET_attempted' if row['SET'] else 'selected_without_SET' if row['selected'] else 'quiet_failed_only' if row['failures'] else 'probed_only'
            x=78+bl*41;y=104+wl*43;title=f'WL{wl}/BL{bl}: {kind}'
            if row:title+=f"; probes={len(row['probes'])}; selected={len(row['selected'])}; SET roles={len(row['SET'])}; failed joint gates={len(row['failures'])}"
            out.append(f'<rect x="{x}" y="{y}" width="38" height="38" rx="3" fill="{palette[kind]}" stroke="#64748b"><title>{escape(title)}</title></rect>')
            if row and row['failures']:text(x+14,y+25,'!',18)
    labels={'untested':'No fresh probe','quiet_failed_only':'Only probe; quiet failed','probed_only':'Probed, not selected','selected_without_SET':'Selected; no SET','SET_attempted':'At least one SET attempt'}
    for j,(k,label) in enumerate(labels.items()):
        x=35+(j%3)*460;y=470+(j//3)*38;out.append(f'<rect x="{x}" y="{y-17}" width="24" height="24" fill="{palette[k]}" stroke="#64748b"/>');text(x+34,y,label,16)
    text(35,562,'! = at least one failed joint quiet observation; later attempts may differ. Both legs contribute to a differential pair.',15)
    text(35,590,f"Audited complete={report['complete']}; coefficients={len(report['weights'])}; quality-passing tiles={report['quality_tiles']}/{len(report['tiles'])}.",16)
    text(35,615,'Serial single-input native measurements; four Conv1 filters with digital trained BN/LIF, not classifier accuracy.',15)
    out.append('</svg>');return '\n'.join(out)

def markdown_report(report):
    lines=['# Fresh Conv1: проверенные измерения и ограничения','',
        'Отчёт получен offline из файлов, связанных с независимым raw audit PASS. '+
        'Повторное исполнение raw-аудитора этим скриптом не выполняется. Все сохранённые result SHA проверяются.',
        '', 'Измерения последовательные, с одним активным входом в каждом native вызове. '+
        'Четыре Conv1-фильтра, обученный BN и цифровой LIF не являются проверкой точности полного классификатора.', '']
    for r in report['sessions']:
        lines += ['## '+r['source']+' — '+r['session_id'],'',
            f"Источник `{r['source_commit']}`; audit `{r['audit']}`: {r['audit_checks']:,} проверок, ошибок 0.",
            f"Свежих коэффициентов: {len(r['weights'])}; завершённость: {r['complete']}; временных выходов: {r['full_time_outputs']}.",
            f"Native calls: {r['total_impacts']}; SET/RESET: {r['total_writes']}. Frozen gain: {r['decoder_gain']} V/weight.",
            f"Quality-passing tiles: {r['quality_tiles']}/{len(r['tiles'])}; это отдельный результат от полноты сбора.", '',
            '| Стадия | RMSE весов | NRMSE | Изменённые знаки (без порога) |',
            '|---|---:|---:|---:|']
        wm=r['weight_metrics']
        if wm:
            for stage in ('post_own','post_all_raw','post_cifar_raw','effective'):
                m=wm[stage];lines.append(f"| {stage} | {m['rmse']:.8g} | {m['nrmse']:.4%} | {m['sign_mismatches']} |")
            lines += ['',f"RMS post-all минус post-own: {wm['later_write_change_rms']:.8g} weight; "+
                f"во время CIFAR: {wm['during_cifar_change_rms']:.8g} weight.",
                'Собственная проверка использует bracket zero, а anchors — общий zero: разность включает различие операторов, шум и возможное изменение состояния. Причинное объяснение не следует.', '']
        if r['trace_metrics']:
            lines += ['| Эталон | Conv RMSE | Различий спайков, включая 5 повторов | Уникальных логических точек с отличием |','|---|---:|---:|---:|']
            for name in ('trained','effective'):
                m=r['trace_metrics'][name];lines.append(f"| {name} | {m['conv_rmse']:.8g} | {m['spike_differences']}/{m['total_timepoints']} | {m['unique_logical_points_with_any_spike_difference']}/{m['total_timepoints']//5} |")
            rep=r['trace_metrics']['repeat_variation'];lines += ['',
                f"По пяти повторениям: RMS sample SD Conv={rep['conv_sample_sd_rms']:.8g}; "+
                f"точек с различающимся спайком {rep['points_with_any_spike_disagreement']}/{rep['logical_points']}.",
                'Согласие с собственным W_eff подтверждает арифметику измеренных весов, но не точность относительно обученных весов.', '']
        lines += ['### Наибольшие ошибки после всех записей тайла','',
            '| Filter/tap | P WL/BL | N WL/BL | Target w | Measured w | SET stop |',
            '|---|---|---|---:|---:|---|']
        for w in sorted(r['weights'],key=lambda x:abs(x['post_all_raw']-x['weight']),reverse=True)[:10]:
            lines.append(f"| {w['filter']}/{w['tap']} | {w['positive_cell']} | {w['negative_cell']} | {w['weight']:.7g} | {w['post_all_raw']:.7g} | {w['stop']} |")
        lines += ['', 'Stop сообщает факт конечного протокола: overshoot, исчерпание ramp или попадание в band. '+
            'Он не устанавливает физическую причину и не означает, что ячейка мертва. '+
            'Матрица C фиксирована до CIFAR и не превращает незаписанный вес в обученный вес.', '',
            'Карта показывает фактический охват и операции. Цвет не означает пригодность или точность; '+
            'знак ! отмечает хотя бы одно неподходящее joint-quiet наблюдение. История каждой ячейки сохранена в SCIENCE.json.', '']
    lines += ['## Сравнение источников','',
        'HW gain .25 и native SIM gain .03 заданы заранее из разных достижимых масштабов. '+
        'Это не физическая эквивалентность устройств и не подгонка к CIFAR-ответам.', '']
    if 'between_sessions' in report:lines += ['```json',json.dumps(report['between_sessions'],ensure_ascii=False,indent=2),'```','']
    return '\n'.join(lines)


def moments(values):
    require(len(values)==5 and all(math.isfinite(v) for v in values),'Exactly five finite observations required')
    return dict(mean_v=st.mean(values),sample_sd_v=st.stdev(values),values_v=values)


def leg_evidence(pair,anchors,final,index):
    """Individual measured row signals, not attribution of differential error."""
    require(len(pair['post_feedback'])==len(anchors)==len(final)==5,'Missing stage observations')
    result={}
    for j,side in enumerate(SIDES):
        own=[];own_ids=[];all_ids=[];final_ids=[]
        for block in pair['post_feedback']:
            rows=[r for r in block['measured_rows'] if r['side']==side]
            require(len(rows)==1 and len(rows[0]['signals'])==1,'Expected one physical signal per pair leg')
            row=rows[0]
            own.append(row['signals'][0]['voltage_v']-(row['zero_before']['voltage_v']+row['zero_after']['voltage_v'])/2)
            own_ids.append([row['zero_before']['task_id'],row['signals'][0]['task_id'],row['zero_after']['task_id']])
        def centered(records,ids):
            values=[]
            for record in records:
                row=record[j];signal=row['singles'][index];zero=row['zero']
                values.append(signal['voltage_v']-zero['voltage_v']);ids.append([zero['task_id'],signal['task_id']])
            return values
        a=centered(anchors,all_ids);f=centered(final,final_ids)
        result[side]=dict(own=moments(own),post_all=moments(a),final=moments(f),
            own_task_ids=own_ids,post_all_task_ids=all_ids,final_task_ids=final_ids,
            post_all_minus_own_v=st.mean(a)-st.mean(own),final_minus_post_all_v=st.mean(f)-st.mean(a),
            own_operator='signal-(zero_before+zero_after)/2',anchor_operator='signal-shared_zero',
            no_causal_attribution=True)
    return result


def all_cell_records(cells,excluded):
    result=[]
    for wl in range(8):
        for bl in range(32):
            row=dict(probes=[],selected=[],SET=[],failures=[],leg_observations=[])
            row.update(cells.get((wl,bl),{}));observed=bool(row['probes'] or row['selected'])
            require(not(bl in excluded and observed),'Excluded physical address was observed')
            role='excluded_by_protocol' if bl in excluded else 'not_observed' if not observed else \
                 'SET_attempted' if row['SET'] else 'selected_no_SET' if row['selected'] else 'quiet_probe_only'
            legs=row['leg_observations']
            delta=[v['post_all_minus_own_v'] for v in legs]
            finaldelta=[v['final_minus_post_all_v'] for v in legs]
            result.append(dict(cell=[wl,bl],observation_status=role,**row,
                selected_observations=len(legs),max_abs_post_all_minus_own_v=max(map(abs,delta),default=None),
                signed_mean_post_all_minus_own_v=st.mean(delta) if delta else None,
                max_abs_final_minus_post_all_v=max(map(abs,finaldelta),default=None),
                pair_errors_not_attributed_to_this_cell=True))
    return result


def evidence_map(report):
    """Fixed descriptive voltage bins; never a new pass/fail criterion."""
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1460" height="650" viewBox="0 0 1460 650">',
         '<rect width="1460" height="650" fill="white"/>']
    def text(x,y,value,size=16):out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="#172033">{escape(str(value))}</text>')
    text(30,30,'Наблюдения по физическим адресам: '+report['source']+' '+report['session_id'],20)
    text(30,58,'Цвет: max |отклик после всех записей − собственная проверка|, V; отдельная нога пары.',16)
    for bl in range(32):text(88+bl*41,92,bl,12)
    palette=[(0.00062,'#bde6e8'),(.01,'#ffe2a3'),(.1,'#ffac76'),(math.inf,'#e86d72')]
    for row in report['cells']:
        wl,bl=row['cell'];value=row['max_abs_post_all_minus_own_v'];status=row['observation_status']
        color='#d8c6ee' if status=='excluded_by_protocol' else '#edf0f3' if status=='not_observed' else '#a4b9dc' if value is None else next(c for limit,c in palette if value<=limit)
        x=78+bl*41;y=104+wl*43
        if bl==0:text(20,y+26,'WL'+str(wl),14)
        title=f"WL{wl}/BL{bl}; {status}; observations={row['selected_observations']}; max delta={value}; joint quiet failures={len(row['failures'])}"
        out.append(f'<rect x="{x}" y="{y}" width="38" height="38" rx="3" fill="{color}" stroke="#64748b"><title>{escape(title)}</title></rect>')
        if row['SET']:text(x+14,y+25,'S',14)
        if row['failures']:text(x+2,y+14,'!',12)
    labels=[('#d8c6ee','Исключён заранее'),('#edf0f3','Нет новых наблюдений'),('#a4b9dc','Только проверка, нет сравнения'),
            ('#bde6e8','Δ ≤ 0.00062 V'),('#ffe2a3','0.00062 < Δ ≤ 0.01 V'),('#ffac76','0.01 < Δ ≤ 0.1 V'),('#e86d72','Δ > 0.1 V')]
    for j,(color,label) in enumerate(labels):
        x=30+(j%3)*470;y=478+(j//3)*35
        out.append(f'<rect x="{x}" y="{y-17}" width="24" height="24" fill="{color}" stroke="#64748b"/>');text(x+33,y,label,15)
    text(30,590,'S: выполнялся SET; !: хотя бы один отказ joint quiet. Цвет не оценивает точность веса или дефект ячейки.',15)
    text(30,617,'Разные zero-операторы и парная запись ограничивают причинные выводы; диапазоны цвета описательные.',15)
    out.append('</svg>');return '\n'.join(out)


def export_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def extra_report(sessions):
    lines=['','## Отдельные ноги и ограничения интерпретации','',
        'В CSV/JSON сохранены все 256 адресов, включая исключённые заранее и не наблюдавшиеся. '+
        'Ошибка разности пары не приписывается двум ячейкам. Для каждой выбранной ноги приведены '+
        'собственные измерения, anchors после всех записей и final anchors, sample SD и ссылки task ID.',
        'У собственной проверки bracket-zero, у anchors общий zero. Разности между ними не являются '+
        'чистой оценкой воздействия соседней записи. Наблюдения одной ячейки в разных тайлах относятся к разным состояниям.',
        'W_eff в стандартном отчёте означает C×H/fixed_gain, полученный после записи всего тайла; '+
        'это не обученные веса и не эталон из собственной проверки каждой пары. Никакие C или веса здесь не подгоняются.',
        'Цветовые границы карты фиксированы только для наглядности; новые критерии пригодности не вводятся.','']
    for r in sessions:
        lines+=['### '+r['source']+' '+r['session_id'],'',
                f"Исключённые BL: {r['excluded_physical_bl']}; карта: [{r['artifact_prefix']}_MAP.svg]({r['artifact_prefix']}_MAP.svg).",'',
                '| Группа пар | Коэффициентов | Own NRMSE | Post-all NRMSE | Final NRMSE |','|---|---:|---:|---:|---:|']
        for group,m in r['bl8_groups'].items():
            if m:lines.append(f"| {group} | {m['count']} | {m['post_own']['nrmse']:.4%} | {m['post_all_raw']['nrmse']:.4%} | {m['post_cifar_raw']['nrmse']:.4%} |")
        if r['trace_metrics']:
            lines+=['','| Эталон | BN max abs | LIF до reset max abs | LIF после reset max abs |','|---|---:|---:|---:|']
            for name in ('trained','effective'):
                m=r['trace_metrics'][name]
                lines.append(f"| {name} | {m['bntt_current_max_abs_error']:.8g} | {m['lif_before_reset_max_abs_error']:.8g} | {m['lif_after_reset_max_abs_error']:.8g} |")
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='.');p.add_argument('--session',required=True);p.add_argument('--audit',required=True)
    p.add_argument('--other-session');p.add_argument('--other-audit');p.add_argument('--output',required=True);args=p.parse_args(argv)
    root=Path(args.repo).resolve();out=local(root,args.output);require(not out.exists(),'New output directory required')
    source_digest=sha(Path(__file__))
    r,t=analyze(root,args.session,args.audit)
    report=dict(schema='fresh300-stability-observations-v1',hardware_io=False,
        analyzer_sha256=source_digest,raw_reaudit_performed=False,sessions=[r])
    if args.other_session:
        require(args.other_audit,'Other audit required');s,u=analyze(root,args.other_session,args.other_audit)
        require(r['fixture_sha256']==s['fixture_sha256'],'Different CIFAR fixture')
        require(r['trained_weights_sha256']==s['trained_weights_sha256'],'Different trained coefficients/coverage')
        report['sessions'].append(s)
        report['between_sessions']=compare_traces(t,u) if r['complete'] and s['complete'] else dict(status='incomplete_no_full_comparison')
        report['between_sessions'].update(same_backend=r['source']==s['source'],different_fixed_gains=r['decoder_gain']!=s['decoder_gain'],
            routing_comparison='Old versus no-BL8 plan reassigns 160 primary pairs; observed fallback routes may differ further.',
            physical_equivalence_claimed=False)
    require(sha(Path(__file__))==source_digest,'Analyzer source changed during verification')
    out.mkdir(parents=True)
    for r in report['sessions']:
        prefix=r['source']+'_'+r['session_id'];r['artifact_prefix']=prefix
        (out/(prefix+'_MAP.svg')).write_text(evidence_map(r),encoding='utf8')
        export_csv(out/(prefix+'_WEIGHTS.csv'),[{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v)
            for k,v in row.items() if k!='leg_evidence'} for row in r['weights']])
        export_csv(out/(prefix+'_CELLS.csv'),[dict(wl=row['cell'][0],bl=row['cell'][1],status=row['observation_status'],
            probes=len(row['probes']),selected=len(row['selected']),SET_events=len(row['SET']),quiet_failures=len(row['failures']),
            selected_observations=row['selected_observations'],max_abs_post_all_minus_own_v=row['max_abs_post_all_minus_own_v'],
            signed_mean_post_all_minus_own_v=row['signed_mean_post_all_minus_own_v'],max_abs_final_minus_post_all_v=row['max_abs_final_minus_post_all_v'])
            for row in r['cells']])
    (out/'SCIENCE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    (out/'REPORT.md').write_text(markdown_report(report)+extra_report(report['sessions']),encoding='utf8')
    print(json.dumps([dict(session=x['session_id'],source=x['source'],complete=x['complete'],weights=len(x['weights']),
        cells=len(x['cells']),weight_metrics=x['weight_metrics'],trace_metrics=x['trace_metrics']) for x in report['sessions']],ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
