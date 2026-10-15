"""Offline summary of bounded conductance discovery. Never imports hardware modules."""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

CRITERIA=dict(minimum_snr=5,noise_floor_adc=1,maximum_cv_high=.05,linearity_fraction=.05,
              linearity_se_multiplier=3,linearity_floor_adc=2,drift_fraction=.05,drift_se_multiplier=3)


def affine_fit(points):
    xs,ys=zip(*points); xm=statistics.mean(xs); ym=statistics.mean(ys)
    slope=sum((x-xm)*(y-ym) for x,y in points)/sum((x-xm)**2 for x in xs)
    return slope,ym-slope*xm


def screen(state,source):
    result=dict(checkpoint_id=state['checkpoint_id'],wl=state['wl'],bl=state['bl'],role=state['role'],
                measured=state['status']=='completed',passes=False,reasons=[],per_state_g_stability='not measured after MVM')
    if not result['measured']: result['reasons']=['incomplete_checkpoint']; return result
    transfer=state['transfer']; zero,low,high=(transfer[i]['adc'] for i in (0,2,4))
    result.update(q_mean=state['q']['mean'],q_sd=state['q']['sd'],crossed_targets=state['crossed_targets'],
        crossed_targets_postread_reached=[target for target in state['crossed_targets'] if state['q']['mean']>=target])
    for name,values in (('low',low),('high',high)):
        signal=values['mean']-zero['mean']; sd=math.hypot(values['sd'],zero['sd'])
        snr=signal/max(1,sd); result.update({f'signal_{name}_adc':signal,f'signal_{name}_sd_adc':sd,f'snr_{name}':snr})
        if signal<=0 or snr<5: result['reasons'].append(f'weak_or_nonpositive_{name}_signal')
    result['cv_high']=result['signal_high_sd_adc']/result['signal_high_adc'] if result['signal_high_adc']>0 else None
    if result['cv_high'] is None or result['cv_high']>.05: result['reasons'].append('cv_high_undefined_or_above_limit')
    epsilon=high['mean']-2*low['mean']+zero['mean']
    se=math.sqrt(high['sd']**2/5+4*low['sd']**2/5+zero['sd']**2/5)
    tolerance=max(.05*abs(result['signal_high_adc']),3*se,2)
    result.update(linearity_epsilon_adc=epsilon,linearity_tolerance_adc=tolerance,linearity_se_adc=se,
        zero_before_mean_adc=zero['mean'],zero_after_mean_adc=transfer[5]['adc']['mean'],
        zero_after_minus_before_adc=transfer[5]['adc']['mean']-zero['mean'])
    if abs(epsilon)>tolerance: result['reasons'].append('nonlinear_two_level_response')
    result['passes']=not result['reasons']
    # Fit first four temporal repeats; keep the fifth separate. Never correct acquired outputs.
    xkey='native_decoded_v' if source=='simulator' else 'dac'
    points=[]; heldout=[]
    for position in range(5):
        values=[r for r in state['mvm'] if r['position']==position]
        x=transfer[position][xkey]
        points.append((x,statistics.mean(r['raw_adc'] for r in values if r['repeat']<4)))
        heldout.append((x,next(r['raw_adc'] for r in values if r['repeat']==4)))
    slope,intercept=affine_fit(points)
    result['descriptive_fit']=dict(input=xkey,output='raw_ADC',training_repeats=[0,1,2,3],heldout_repeat=4,
        slope=slope,intercept=intercept,in_sample_rmse_adc=math.sqrt(statistics.mean((y-slope*x-intercept)**2 for x,y in points)),
        heldout_rmse_adc=math.sqrt(statistics.mean((y-slope*x-intercept)**2 for x,y in heldout)),
        physical_voltage_calibrated=False,used_as_acceptance_gate=False)
    return result


def build(summary,checkpoints):
    screens={s['checkpoint_id']:screen(s,summary['source']) for s in checkpoints}
    cells={(c['wl'],c['bl']):c for c in summary['cells']}
    final={(c['wl'],c['bl']):c for c in summary['final_array']}
    output=[]
    for wl in range(8):
        for bl in range(32):
            cell=cells.get((wl,bl)); row=dict(wl=wl,bl=bl,status='unmeasured',session_completed=summary['status']=='completed')
            if not cell: output.append(row); continue
            states=[screens[s['checkpoint_id']] for s in cell['checkpoints']]
            passing=[s for s in states if s['passes']]
            best=max(passing,key=lambda s:min(s['snr_low'],s['snr_high'])) if passing else None
            terminal=screens.get(cell.get('terminal_checkpoint_id'))
            row.update(reset_reached=cell.get('reset_reached'),reset_confirmed_before_mvm=cell.get('reset_confirmed'),
                reset_pulses=cell['reset_pulses'],set_pulses=cell['set_pulses'],set_skipped_reason=cell.get('set_skipped_reason'),
                crossed_targets=cell['crossed_targets'],unreached_targets=cell.get('unreached_targets'),
                tested_window_upper_reached=cell.get('tested_window_upper_reached'),
                measured_state_count=sum(s['measured'] for s in states),passing_state_count=len(passing),
                best_observed_checkpoint_id=best['checkpoint_id'] if best else None,
                best_observed_q=best['q_mean'] if best else None,terminal_checkpoint_id=cell.get('terminal_checkpoint_id'),
                terminal_transfer_pass=terminal['passes'] if terminal else None,
                terminal_q=terminal.get('q_mean') if terminal else None,
                terminal_transfer_reasons=terminal['reasons'] if terminal else ['terminal_not_measured'],
                error=cell.get('error'),mvm_remeasured_after_other_cells=False)
            later=final.get((wl,bl))
            if later and cell.get('terminal_g'):
                before=cell['terminal_g']; after=later['conductance_siemens']
                delta=(after['mean']-before['mean'])/before['mean']
                se=math.sqrt(before['sd']**2/before['n']+after['sd']**2/after['n'])/before['mean']
                tolerance=max(.05,3*se)
                row.update(final_g_fractional_delta=delta,final_g_fractional_se=se,final_g_tolerance=tolerance,
                           final_g_difference_pass=abs(delta)<=tolerance,
                           final_g_fractional_difference_from_initial=later.get('fractional_difference_from_initial'))
            if not row['session_completed'] or terminal is None or later is None: row['status']='incomplete'
            elif terminal['passes'] and row['final_g_difference_pass']: row['status']='terminal_screen_and_g_pass'
            elif best: row['status']='historical_state_only'
            else: row['status']='no_observed_state_pass'
            output.append(row)
    return output,list(screens.values())


def save_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): value.update(block)
    return value.hexdigest()


def local(root,value):
    if Path(value).is_absolute() or '..' in Path(value).parts: raise ValueError('Use a repository-relative path')
    path=(root/value).resolve()
    if not path.is_relative_to(root): raise ValueError('Path escapes repository')
    return path


def render(root,session,output):
    manifest=json.loads((session/'manifest.json').read_text(encoding='utf-8-sig'))
    summary=json.loads((session/'summary.json').read_text(encoding='utf-8-sig'))
    hashes={}
    for name,expected in manifest['result_sha256'].items():
        hashes[name]=sha(local(session,name))
        if hashes[name]!=expected: raise ValueError(f'Session file hash mismatch: {name}')
    checkpoint_path=session/'checkpoints.jsonl'
    checkpoints=[json.loads(line) for line in checkpoint_path.read_text(encoding='utf-8').splitlines()] if checkpoint_path.exists() else []
    cells,states=build(summary,checkpoints)
    output.mkdir(parents=True,exist_ok=False)
    report=dict(session=session.relative_to(root).as_posix(),source=summary['source'],status=summary['status'],
        provisional_criteria=CRITERIA,verified_result_sha256=hashes,generator_sha256=sha(Path(__file__)),
        cells=cells,states=states,automatic_mask_exported=False,full_mvm_or_cifar_qualified=False)
    (output/'calibration.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save_csv(output/'cells.csv',cells); save_csv(output/'states.csv',states)
    colors={'terminal_screen_and_g_pass':('#8ac5a1','T','Терминальный отклик + поздний G прошли preliminary tests'),
            'historical_state_only':('#b4a0d5','H','Удачное историческое состояние; не готовая текущая ячейка'),
            'no_observed_state_pass':('#e7a09c','N','Ни одно наблюдённое состояние не прошло transfer tests'),
            'incomplete':('#f0cc83','!','Незавершённый сбор / нет терминального сравнения'),
            'unmeasured':('#ddd','?','Не измерено')}
    esc=lambda value:html.escape(str(value))
    lines=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Калибровка проводимостей</title>',
        '<style>body{font:14px system-ui;margin:28px;color:#172335}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #bbc4d0}a{color:inherit}.grid td{text-align:center;font-size:12px}.scroll{overflow:auto}pre{white-space:pre-wrap}.warn{padding:14px;background:#fff2d5}.full{font-size:12px}</style>',
        f'<h1>Discovery проводимостей: {esc(summary["source"])} · {esc(summary["status"])}</h1>',
        '<p class="warn">Один ограниченный цикл SET/RESET, не проверка повторяемости. Best observed state — историческое состояние, которое могло быть перезаписано. Терминальная MVM-кривая не измерялась заново после программирования остальных ячеек; поздний G не доказывает её сохранность. Это не калибровка абсолютного напряжения/усиления, не готовая маска, не квалификация 4 фильтров или CIFAR.</p>',
        '<p>Симулятор: 125 модельных шагов, один simulated ADC. Плата: длительность native-импульса неизвестна, mode7 возвращает среднее десяти ADC; пять таких ответов не являются 50 независимыми отсчётами. Mode7 G измерен при nominal .3 В, MVM при ≤.1 В: различие трактов не доказывает ошибку усиления.</p>',
        '<p><a href="cells.csv">Ячейки CSV</a> · <a href="states.csv">Все состояния CSV</a> · <a href="calibration.json">JSON / provenance</a> · <a href="../raw.jsonl">Исходный журнал</a> · <a href="../protocol.json">Протокол</a></p>']
    for key,(color,symbol,label) in colors.items():
        lines.append(f'<p><span style="background:{color};padding:4px">{symbol}</span> {esc(label)}: {sum(c["status"]==key for c in cells)}</p>')
    lines.append('<div class="scroll"><table class="grid"><tr><th>WL/BL</th>'+''.join(f'<th>{bl}</th>' for bl in range(32))+'</tr>')
    for wl in range(8):
        lines.append(f'<tr><th>{wl}</th>')
        for cell in cells[wl*32:(wl+1)*32]:
            color,symbol,label=colors[cell['status']]
            lines.append(f'<td style="background:{color}" title="{esc(label)}"><a href="#cell-{wl}-{cell["bl"]}">{symbol}<br>{wl},{cell["bl"]}</a></td>')
        lines.append('</tr>')
    lines.append('</table></div><h2>Все ячейки и состояния</h2>')
    for cell in cells:
        lines.append(f'<details id="cell-{cell["wl"]}-{cell["bl"]}"><summary>WL{cell["wl"]}/BL{cell["bl"]}: {esc(colors[cell["status"]][2])}; RESET={esc(cell.get("reset_confirmed_before_mvm"))}; q terminal={esc(cell.get("terminal_q"))}</summary><pre>{esc(json.dumps(cell,ensure_ascii=False,indent=2))}</pre>')
        for state in states:
            if state['wl']==cell['wl'] and state['bl']==cell['bl']:
                lines.append(f'<pre>{esc(json.dumps(state,ensure_ascii=False,indent=2))}</pre>')
        lines.append('</details>')
    lines.append('<p>Per-state screening: .05/.1 относительно пяти zero-before; zero-after отдельно. SNR≥5 при noise=max(1,combined SD), CVhigh≤5%; |ε|≤max(5%signalHigh,3SEε,2 ADC). G внутри checkpoint после MVM не измеряется. Поздний final G: |ΔG|/Gterminal≤max(5%,3SEfraction), SEfraction=√(SDterminal²/5+SDfinal²/3)/Gterminal. Эти пороги предварительные. .025/.075 — дополнительные точки, не отдельный gate. Фит hardware ADC по DAC-командам; simulator ADC по native decoded volts; первые 4 повтора — fit, пятый — временной holdout. Выходы не корректировались, fit не становится доказательством линейного физического умножения.</p></html>')
    (output/'index.html').write_text('\n'.join(lines),encoding='utf-8')
    return {'output':output.relative_to(root).as_posix(),'cell_count':len(cells),'state_count':len(states)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    parser.add_argument('--session',required=True)
    parser.add_argument('--output')
    args=parser.parse_args(); root=Path(args.repo).resolve()
    print(json.dumps(render(root,local(root,args.session),local(root,args.output or args.session+'/report'))))


if __name__=='__main__': main()
