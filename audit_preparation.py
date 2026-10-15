"""Independent stdlib-only offline journal audit and preparation comparison.

No collector, calibration analyzer, MemriBoard or hardware imports. Matrix
selection is not independently refitted. Selected C/d are treated as supplied
analysis parameters; their held-out application is recomputed from audited ADC.
"""
import argparse
from collections import Counter, defaultdict
from configparser import ConfigParser
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path, PureWindowsPath
from statistics import mean, stdev
import uuid

ROOT = Path(__file__).resolve().parent
SCHEDULE_SHA = '7ff4c9de45ece848e09a89b29f7f14bcc1b8eee0161a33d217b05084e7252d88'
SOURCE_PINS = {
    'all256_mvm.py': 'c8391bb9125bc9f0d280f91d1ca56b4cf98293d702531805bac13e2836c168f0',
    'mvm_calibration.py': '5ee05de87d4c3ae10c85c202ca2f5b7e369fd338fa5369c4708400a1b30863ed',
    'configs/calibrated_mvm.json': 'cf410f1409ae72eb7851675881a97db880701002e5f35bf853e750af519f6562',
}


def local(value, root=ROOT):
    p, w = Path(value), PureWindowsPath(value)
    if p.is_absolute() or w.drive or w.root or '..' in p.parts or '..' in w.parts:
        raise ValueError('Repository-relative paths only')
    out = (root / p).resolve()
    if not out.is_relative_to(root.resolve()): raise ValueError('Path escapes repository')
    return out


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def load(path): return json.loads(path.read_text(encoding='utf-8'))
def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def require(condition, message):
    if not condition: raise ValueError(message)


def close(a, b):
    if isinstance(a,dict) and isinstance(b,dict): return a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list): return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if type(a) in (int,float) and type(b) in (int,float): return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
    return a==b


def stats(values): return dict(mean=mean(values),sd=stdev(values) if len(values)>1 else 0.,n=len(values))


def convert_read(adc):
    require(type(adc) is int and 20<=adc<=16382,'Invalid readback ADC')
    resistance=round(11*3000*.1*16384/(adc*5)-10-3000,2)
    require(resistance>1e-8,'Invalid mode9 resistance')
    return resistance,1/resistance


def recompute_guard(meta, g, basis):
    mask=meta['mask'];reasons=[];predictions=[]
    if len(mask)>4: reasons.append('predeclared_max_active_4')
    for wl in range(8):
        current=.1*sum(g[wl,b] for b in mask);model=330*current
        gui=sum(.1*round(330*g[wl,b],4) for b in mask)
        p=dict(wl=wl,mode9_model_current_a=current,mode9_model_output_v=model,gui_model_output_v=gui)
        if not all(math.isfinite(v) for v in (current,model,gui)) or current>.005 or model>4 or gui>=3.3:
            reasons.append(f'wl{wl}_mode9_model_guard')
        if meta['phase']!='basis':
            if () not in basis or any((b,) not in basis for b in mask): reasons.append(f'wl{wl}_basis_missing')
            else:
                zero=stats([v[wl] for v in basis[()]])
                upper=zero['mean']+3*zero['sd']
                for b in mask:
                    s=stats([v[wl] for v in basis[b,]])
                    upper+=2*max(0,s['mean']-zero['mean']+3*math.hypot(s['sd'],zero['sd']))
                p['basis_heuristic_upper_adc']=upper
                if upper>.8*16383: reasons.append(f'wl{wl}_fresh_basis_headroom')
        predictions.append(p)
    return dict(allowed=not reasons,reasons=reasons,predictions=predictions,
        upper_bound_certified=False,physical_current_compliance=False)


def audit_raw(session, expected_source):
    manifest=load(session/'manifest.json');plan=load(session/'protocol.json');schedule=load(session/'schedule.json')
    obs=load(session/'observations.json')
    require(manifest['source']==expected_source,'Wrong source')
    require(manifest['status']=='completed','Audit requires a completed session')
    require(plan['protocol_id']=='all256-preparation-mvm-lif-v2','Wrong protocol')
    require(plan['maximum_impacts']==18576 and plan['maximum_programming_pulses']==0,'Wrong budget')
    require(canonical(schedule)==SCHEDULE_SHA==manifest['schedule_sha256'],'Frozen schedule mismatch')
    for name,pin in SOURCE_PINS.items(): require(manifest['source_sha256'][name]==pin,'Acquisition source pin: '+name)
    require(sha(session/'protocol.json')==SOURCE_PINS['configs/calibrated_mvm.json'],'Protocol bytes differ')
    for name in ('raw.jsonl','observations.json','protocol.json','schedule.json','board.ini'):
        require(sha(session/name)==manifest['result_sha256'][name],'Result hash: '+name)
    cfg=ConfigParser();cfg.read_string((session/'board.ini').read_text(encoding='utf-8-sig'))
    for k,v in dict(dac_bit=12,vol_ref_dac=5,adc_bit=14,vol_ref_adc=5,gain=11,res_load=3000,res_switches=10,sum_gain=330).items():
        require(cfg.getfloat('board',k)==v,'Board parameter: '+k)
    require(len(schedule)==2130 and [v['vector_id'] for v in schedule]==list(range(1,2131)),'Schedule coverage')
    require(not any('cifar' in v['phase'] for v in schedule),'Forbidden CIFAR replay')
    pending=None;sequence=0;guard_index=0;current_guard=None;by_vector=defaultdict(list)
    reads=[];vectors=[];skips=[];basis=defaultdict(list);g={};mode_counts=Counter()
    with (session/'raw.jsonl').open(encoding='utf-8') as stream:
        for line_number,line in enumerate(stream,1):
            r=json.loads(line);event=r['event']
            require(r['session_id']==manifest['session_id'] and r['source']==expected_source,f'Identity line {line_number}')
            if event=='vector_guard':
                require(pending is None and guard_index<len(schedule),'Guard ordering')
                meta=schedule[guard_index];guard_index+=1
                require(all(r[k]==v for k,v in meta.items()),'Guard schedule metadata')
                if not g:
                    require(len(reads)==768,'Pre reads before guard')
                    g={(w,b):mean(x['conductance_siemens'] for x in reads if x['wl']==w and x['bl']==b) for w in range(8) for b in range(32)}
                valid_basis={k:v for k,v in basis.items() if len(v)==5}
                computed=recompute_guard(meta,g,valid_basis)
                require(close({k:r[k] for k in computed},computed),'Independent guard mismatch')
                current_guard=r
                if not r['allowed']: skips.append(dict(**meta,reasons=r['reasons']))
            elif event=='request':
                require(pending is None,'Nested request');sequence+=1
                require(r['task_id']==sequence and r['mode'] in (9,10),'Task ID/mode')
                command=r['command_si'];wl=r['wl'];require(type(wl) is int and 0<=wl<8,'WL address')
                require(command['wl']==wl and command['mode_flag']==r['mode'],'Command metadata')
                if r['mode']==10:
                    require(current_guard is not None and current_guard['allowed'],'Impact without allowed guard')
                    meta=schedule[current_guard['vector_id']-1]
                    require(all(r[k]==v for k,v in meta.items()),'MVM metadata')
                    mask=r['mask'];require(len(mask)<=4 and len(mask)==len(set(mask)),'Active bounds')
                    if r['phase']=='synthetic_lif':require(len(mask)<=2,'Synthetic active bounds')
                    require(command==dict(mode_flag=10,vol=[.1 if b in mask else 0. for b in range(32)],wl=wl,id=0),'MVM payload')
                    require(wl==len(by_vector[r['vector_id']]),'WL acquisition order')
                    expected_voltage=[dict(requested_v=v,dac_code=82 if v else 0,native_decoded_v=v) for v in command['vol']]
                else:
                    index=len(reads)%768;phase='pre' if len(reads)<768 else 'post'
                    require((r['phase'],wl,r['bl'],r['repeat'])==(phase,index//96,(index%96)//3,index%3),'Read schedule')
                    if phase=='post':require(guard_index==len(schedule),'Premature post read')
                    require(command==dict(mode_flag=9,vol=.1,pulse_width=0.,sign=0,wl=wl,bl=r['bl'],id=sequence),'Read payload')
                    expected_voltage=dict(requested_v=.1,dac_code=82,native_decoded_v=.1)
                require(close(r['native_voltage_conversion'],expected_voltage),'Voltage conversion metadata')
                pending=r
            elif event=='result':
                require(pending is not None and all(r[k]==v for k,v in pending.items() if k!='event'),'Request/result metadata mismatch')
                require(r['status']=='ok','Completed session contains failed result')
                adc=r['raw_adc'];require(type(adc) is int and 0<=adc<=16382,'Raw ADC range')
                require(type(r['returned_id']) is int and r['returned_id']==0,'Returned ID')
                require(r['native_result'][1:]==[0,adc],'Native result vs compact ADC')
                mode_counts[r['mode']]+=1
                if r['mode']==9:
                    resistance,conductance=convert_read(adc)
                    require(close([r['resistance_ohm'],r['conductance_siemens'],r['q'],r['read_voltage_parameter_v']],
                        [resistance,conductance,330/resistance,.1]),'Independent native-equivalent read conversion')
                    reads.append({k:v for k,v in r.items() if k not in ('event','latency_s','command_si','native_voltage_conversion','native_result')})
                else:
                    require(close(r['output_voltage_v'],round(adc*5/16383,5)) and r['output_voltage_v']<=4,'Output voltage limit/conversion')
                    group=by_vector[r['vector_id']];group.append(r)
                    if len(group)==8:
                        meta=schedule[r['vector_id']-1]
                        v=dict(**meta,adc=[x['raw_adc'] for x in group],task_ids=[x['task_id'] for x in group]);vectors.append(v)
                        if meta['phase']=='basis':basis[tuple(meta['mask'])].append(v['adc'])
                pending=None
            else: raise ValueError('Unknown raw event')
    require(pending is None and guard_index==2130 and len(reads)==1536,'Incomplete journal')
    require(all(len(v)==8 for v in by_vector.values()),'Partial vector')
    require(len(vectors)+len(skips)==2130,'Coverage ledger')
    require(sequence==manifest['impact_attempts']==1536+8*len(vectors)<=18576,'Impact counts')
    require(vectors==obs['vectors'] and reads==obs['readbacks'] and skips==obs['skips'],'Raw/compact mismatch')
    return dict(passed=True,session_id=manifest['session_id'],source=expected_source,paired_impacts=sequence,
        complete_vectors=len(vectors),skipped_vectors=len(skips),mode_counts=dict(mode_counts),
        schedule_sha256=SCHEDULE_SHA,guards_recomputed=True,conversion_recomputed=True,
        physical_guard_guarantee=False,hardware_io_by_auditor=False),obs


def grouped(vectors,phase):
    out=defaultdict(list)
    for v in vectors:
        if v['phase']==phase:out[tuple(v['mask'])].append(v['adc'])
    return out


def heldout_metrics(data,H,zero,C,d,signed=False):
    points=[]
    for mask,samples in data.items():
        if len(samples)!=5:continue
        corrected=[[sum(C[w][j]*(row[j]-zero[j]) for j in range(8))+d[w] for w in range(8)] for row in samples]
        expected=[sum(H[w][b] for b in mask) for w in range(8)]
        for channel in range(4 if signed else 8):
            a,b=(2*channel,2*channel+1) if signed else (channel,None)
            values=[r[a]-(r[b] if b is not None else 0) for r in corrected]
            target=expected[a]-(expected[b] if b is not None else 0);s=stats(values);tol=max(2.,.05*abs(target))
            points.append(dict(mask=list(mask),channel=channel,mean=s['mean'],expected=target,sd=s['sd'],error=s['mean']-target,tolerance=tol,
                mean_pass=abs(s['mean']-target)<=tol+1e-9,repeatability_pass=s['sd']<=tol+1e-9))
    return dict(points=len(points),mean_pass=sum(p['mean_pass'] for p in points),repeatability_pass=sum(p['repeatability_pass'] for p in points),
        both_pass=sum(p['mean_pass'] and p['repeatability_pass'] for p in points),details=points)


def scientific_review(obs,analysis):
    vectors,reads=obs['vectors'],obs['readbacks'];basis=grouped(vectors,'basis');anchors=grouped(vectors,'test_anchors')
    require(len(basis)==33 and all(len(v)==5 for v in basis.values()),'Incomplete basis: detailed matrix audit unavailable')
    zero=[mean(v[w] for v in basis[()]) for w in range(8)]
    H=[[mean(v[w] for v in basis[b,])-zero[w] for b in range(32)] for w in range(8)]
    require(close(H,analysis['basis_H']) and close(zero,analysis['zero']),'Analyzer basis differs from raw')
    C,d=analysis['C'],analysis['d'];require(len(C)==8 and all(len(r)==8 for r in C) and len(d)==8,'Matrix shape')
    require(all(math.isfinite(x) for row in C for x in row) and all(math.isfinite(x) for x in d),'Matrix finite')
    metrics={}
    for phase in ('test','test_anchors'):
        for signed in (False,True):metrics[phase+('_signed' if signed else '_rows')]=heldout_metrics(grouped(vectors,phase),H,zero,C,d,signed)
    cells=[]
    for w in range(8):
        z=stats([v[w] for v in basis[()]])
        for b in range(32):
            s=stats([v[w] for v in basis[b,]]);signal=s['mean']-z['mean'];noise=math.hypot(s['sd'],z['sd'])
            reasons=[];basis_ok=signal>0 and signal/max(1.,noise)>=5 and noise<=.05*signal
            if signal==0:reasons.append('Нулевой одиночный отклик: причина не установлена, это не диагноз мёртвой ячейки')
            elif not basis_ok:reasons.append('Одиночный базис не проходит SNR≥5 и CV≤5%')
            drift_values=[[r['conductance_siemens'] for r in reads if r['phase']==phase and r['wl']==w and r['bl']==b] for phase in ('pre','post')]
            pre,post=map(stats,drift_values);fraction=(post['mean']-pre['mean'])/pre['mean']
            limit=max(.05,3*math.hypot(pre['sd']/math.sqrt(3),post['sd']/math.sqrt(3))/pre['mean']);drift_ok=abs(fraction)<=limit
            if not drift_ok:reasons.append('Изменение G превышает допуск до/после')
            anchor_ok=False;anchor_error=None;anchor_noise=None
            if len(anchors.get((b,),[]))==5 and len(anchors.get((),[]))==5:
                av=stats([r[w] for r in anchors[b,]]);az=stats([r[w] for r in anchors[()]])
                anchor_error=av['mean']-az['mean']-signal;anchor_noise=math.hypot(av['sd'],az['sd']);tol=max(2.,.05*abs(signal))
                anchor_ok=abs(anchor_error)<=tol+1e-9 and anchor_noise<=tol+1e-9
                if not anchor_ok:reasons.append('Независимый одиночный ответ/шум не проходит допуск5% или2ADC')
            else:reasons.append('Нет полного независимого одиночного контрольного измерения')
            mixed=[p for p in metrics['test_rows']['details'] if p['channel']==w and b in p['mask']]
            mixed_ok=bool(mixed) and all(p['mean_pass'] and p['repeatability_pass'] for p in mixed)
            if not mixed:reasons.append('Нет совместных контрольных векторов для адреса')
            elif not mixed_ok:reasons.append('Совместный MVM после выбранной поправки не проходит все измеренные проверки')
            single_ok=bool(basis_ok and drift_ok and anchor_ok)
            cells.append(dict(wl=w,bl=b,single_supported=single_ok,basis_supported=basis_ok,drift_consistent=drift_ok,
                raw_signal_adc=signal,snr=signal/max(1.,noise),cv=noise/signal if signal>0 else None,
                conductance_fractional_delta=fraction,heldout_single_error_adc=anchor_error,heldout_single_noise_adc=anchor_noise,
                tested_mixed_points=len(mixed),mixed_observed_pass=mixed_ok,
                row_selection_claim=analysis.get('selection_by_wl',[{}]*8)[w].get('passed'),
                reasons=reasons or ['Одиночные и имеющиеся совместные проверки прошли; программируемость и CIFAR не доказаны'],
                arbitrary_mvm_qualified=False,programmability_qualified=False))
    return dict(matrix_parameters_supplied_by_analyzer=True,matrix_fit_selection_independently_audited=False,
        selected_model_claim=analysis.get('selected_model'),heldout_metrics=metrics,cells=cells,
        raw_single_supported=sum(c['single_supported'] for c in cells),basis_supported=sum(c['basis_supported'] for c in cells),
        single_and_observed_mixed_pass=sum(c['single_supported'] and c['mixed_observed_pass'] for c in cells),
        lif_status_claims=dict(Counter(r['status'] for r in analysis.get('lif',[]))),
        lif_claim_scope='Reported by frozen analyzer, not recomputed here; digital polarities reuse the same measurements',
        fully_qualified_cells_not_established=True,full_cifar_qualified=False,numerical_comparison_slack_adc=1e-9)


def inspect_session(path,source):
    audit,obs=audit_raw(path,source);analysis_path=path/'analysis'/'analysis.json';analysis=load(analysis_path)
    inputs=analysis.get('input_sha256',{})
    for name in ('observations.json','protocol.json'):
        matches=[value for key,value in inputs.items() if key.endswith('/'+name)]
        require(matches==[sha(path/name)],'Analyzer input provenance: '+name)
    evidence={name:sha(path/name) for name in ('raw.jsonl','manifest.json','observations.json','protocol.json','schedule.json','analysis/analysis.json')}
    return dict(raw_audit=audit,science=scientific_review(obs,analysis),input_sha256=evidence)


def render(report):
    h=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Подготовка MVM: аудит</title><style>body{font:15px system-ui;margin:24px;color:#18212a}table{border-collapse:collapse}th,td{padding:5px;border:1px solid #bbb;text-align:center}.cell{min-width:23px}.single{background:#a7d9f5}.weak{background:#ffdb92}.mixedok{box-shadow:inset 0 0 0 3px #237b3a}.mixedfail{box-shadow:inset 0 0 0 3px #a93232}details{margin:8px 0}.wrap{overflow:auto}pre{white-space:pre-wrap}.muted{color:#505b66}</style><h1>Подготовка MVM: независимый аудит</h1>',
       '<p>Голубой фон: отдельный одиночный отклик подтверждён базисом, независимыми чтениями и стабильностью G. Оранжевый: эти условия не подтверждены. Зелёная рамка: прошли все измеренные совместные проверки данного адреса; красная: есть отказ или нет покрытия. Ни один цвет не означает программируемость веса или готовый CIFAR.</p>']
    for source,result in report['sessions'].items():
        a,s=result['raw_audit'],result['science'];h.append(f'<h2>{html.escape(source)} · {html.escape(a["session_id"])}</h2>')
        h.append(f'<p>Raw audit PASS: {a["paired_impacts"]} impacts, {a["complete_vectors"]} полных векторов, {a["skipped_vectors"]} пропусков. Одноячейковые свидетельства: {s["raw_single_supported"]}/256; одновременно одиночные и имеющиеся mixed-проверки: {s["single_and_observed_mixed_pass"]}/256.</p>')
        h.append('<div class="wrap"><table><tr><th>WL / BL</th>'+''.join(f'<th>{b}</th>' for b in range(32))+'</tr>')
        for w in range(8):
            h.append(f'<tr><th>{w}</th>')
            for c in s['cells'][w*32:(w+1)*32]:
                title=f'WL{w}/BL{c["bl"]}; ΔADC={c["raw_signal_adc"]:.3g}; SNR={c["snr"]:.3g}; ΔG={c["conductance_fractional_delta"]:.2%}. '+ '; '.join(c['reasons'])
                classes=('single' if c['single_supported'] else 'weak')+' '+('mixedok' if c['mixed_observed_pass'] else 'mixedfail')
                h.append(f'<td class="cell {classes}" title="{html.escape(title,quote=True)}">{c["bl"]}</td>')
            h.append('</tr>')
        h.append('</table></div><h3>Независимо пересчитанный final-test после предоставленной C</h3><ul>')
        for name,m in s['heldout_metrics'].items():h.append(f'<li>{name}: mean {m["mean_pass"]}/{m["points"]}; repeatability {m["repeatability_pass"]}/{m["points"]}; оба {m["both_pass"]}/{m["points"]}</li>')
        h.append('</ul><p>Выбор/fit C не повторялся. LIF — отдельные заявления замороженного анализатора: '+html.escape(json.dumps(s['lif_status_claims']))+'.</p>')
        for c in s['cells']:
            h.append(f'<details><summary>WL{c["wl"]}/BL{c["bl"]}: одиночный {"подтверждён" if c["single_supported"] else "не подтверждён"}, mixed {"прошёл измеренные проверки" if c["mixed_observed_pass"] else "не подтверждён"}</summary><pre>'+html.escape(json.dumps(c,ensure_ascii=False,indent=2))+'</pre></details>')
    h.append('<p>Источники имеют собственные состояния и базисы. Сравниваются ошибки и покрытие, не равенство абсолютных ADC. Guard — проверенная программная эвристика, а не физически доказанное ограничение тока/насыщения. CIFAR не запускался этим аудитором.</p></html>')
    return ''.join(h)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware-session',required=True);parser.add_argument('--simulator-session',required=True)
    parser.add_argument('--output',default=None);args=parser.parse_args(argv)
    output=local(args.output or 'results/preparation_reviews/'+str(uuid.uuid4()))
    require(not output.exists(),'Output already exists; no overwrite')
    report=dict(schema='independent-preparation-review-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        audit_source_sha256=sha(Path(__file__)),hardware_io=False,sessions={})
    for source,value in [('hardware',args.hardware_session),('simulator',args.simulator_session)]:
        report['sessions'][source]=inspect_session(local(value),source)
    output.mkdir(parents=True)
    with (output/'summary.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    with (output/'index.html').open('x',encoding='utf-8') as f:f.write(render(report))
    print(json.dumps(dict(output=output.relative_to(ROOT).as_posix(),raw_audit_passed=True,
        sources={k:dict(single_supported=v['science']['raw_single_supported'],single_and_mixed=v['science']['single_and_observed_mixed_pass'],
            final_counts={n:{p:m[p] for p in ('points','mean_pass','repeatability_pass','both_pass')} for n,m in v['science']['heldout_metrics'].items()}) for k,v in report['sessions'].items()})))
    return 0


if __name__=='__main__':raise SystemExit(main())
