"""Pure offline science report from independently audited fresh300 artifacts."""
import argparse
from collections import Counter,defaultdict
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics as st

AUDITOR='bfab9b2044388c11a6362e94471b7872fab134451e8c09187eda59c9e8caa5bb'
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
    return dict(count=len(rows),trained_rms=scale,**{stage:dict(rmse=rms([x[stage]-x['weight'] for x in rows]),
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
    require(a['audit_status']=='passed' and not a['errors'] and a['auditor_sha256']==AUDITOR,'Independent audit not PASS or wrong frozen auditor')
    require(a['manifest_sha256']==sha(d/'manifest.json') and a['session_id']==m['session_id'] and a['source']==m['source'],'Audit/session binding mismatch')
    for name,digest in m['result_sha256'].items():require(sha(local(d,name))==digest,'Recorded artifact changed: '+name)
    fixture_path='weights/prepared_cifar100_conv1.json'
    require(sha(local(repo,fixture_path))==m['source_sha256'][fixture_path],'Input fixture hash mismatch')
    fixture=load(local(repo,fixture_path))
    route=load(d/'planned_route.json');gain=.25 if m['source']=='hardware' else .03
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
            for j,s in enumerate(raw['effective_route']):
                p=bytap[s['tap_index']]
                weights.append(dict(filter=fid,tile=tid,tap=s['tap_index'],weight=s['trained_weight'],positive_cell=s[SIDES[0]],negative_cell=s[SIDES[1]],
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
        manifest_sha256=sha(d/'manifest.json'),audit=audit,audit_sha256=sha(ap),auditor_sha256=AUDITOR,audit_checks=a['checks'],
        complete=complete,decoder_gain=gain,gains_are_not_physical_equivalence=True,all_coefficients_fresh=True,
        total_impacts=m['total_impacts'],total_writes=m['total_pulses'],counters=a['counters'],coverage=a['actual_coverage'],
        full_time_outputs=a['full_time_outputs'],weight_metrics=weight_metrics(weights),
        per_filter=[dict(filter=f,weight_metrics=weight_metrics([x for x in weights if x['filter']==f]),
            traces=trace_metrics([r for r in trajectories if r['filter_id']==f],fixture) if trajectories else None) for f in range(4)],
        trace_metrics=trace_metrics(trajectories,fixture) if trajectories else None,tiles=tiles,weights=weights,
        reset_stops=dict(reset_stops),program_stops=dict(set_stops),
        post_all_error_sse_by_stop={k:dict(sse=v,fraction=v/total_sse if total_sse else 0) for k,v in sse_by_stop.items()},quality_tiles=sum(x['quality'] for x in tiles),
        cells=[dict(cell=list(k),**v) for k,v in sorted(cells.items())]),trajectories

def compare_traces(first,second):
    def index(rows):return {(r['filter_id'],r['case'],r['repeat'],t):x for r in rows for t,x in enumerate(r['actual'])}
    a=index(first);b=index(second);require(a.keys()==b.keys() and len(a)==2400,'Unmatched full outputs')
    return dict(timepoints=len(a),**{name+'_rmse':rms([a[k][name]-b[k][name] for k in a])
        for name in ('conv_sum','bntt_current','lif_before_reset','lif_after_reset')},
        spike_differences=sum(a[k]['lif_spike']!=b[k]['lif_spike'] for k in a),
        interpretation='Separate physical/simulator acquisitions with different gains; not interchangeable device physics.')

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
            lines += ['',f"RMS изменения после других записей: {wm['later_write_change_rms']:.8g} weight; "+
                f"во время CIFAR: {wm['during_cifar_change_rms']:.8g} weight.",
                'Эти изменения включают измерительный шум и изменение состояния; причинное объяснение из одной разности не следует.', '']
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--session',required=True);p.add_argument('--audit',required=True);p.add_argument('--other-session');p.add_argument('--other-audit');p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.repo).resolve();out=local(root,args.output);require(not out.exists(),'New output directory required')
    r,t=analyze(root,args.session,args.audit);report=dict(schema='fresh300-science-v1',hardware_io=False,sessions=[r])
    if args.other_session:
        require(args.other_audit,'Other audit required');s,u=analyze(root,args.other_session,args.other_audit);report['sessions'].append(s)
        report['between_sessions']=compare_traces(t,u) if r['complete'] and s['complete'] else dict(status='incomplete_no_full_comparison')
    out.mkdir(parents=True)
    (out/'SCIENCE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for r in report['sessions']:(out/(r['source']+'_COVERAGE.svg')).write_text(svg_map(r),encoding='utf8')
    (out/'REPORT.md').write_text(markdown_report(report),encoding='utf8')
    print(json.dumps({x['source']:dict(complete=x['complete'],weight_metrics=x['weight_metrics'],trace_metrics=x['trace_metrics'],coverage=x['coverage']) for x in report['sessions']},ensure_ascii=False))
if __name__=='__main__':main()
