"""Offline observed q points, never continuous qualified ranges. Standard library only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

CAL='results/cell_calibrations/3380d497-ded0-4881-9df2-85ef1c7e32d9'
MAP='results/cell_maps/0584b8fe-eb6c-4cb4-ad85-a37af1fbb873'
PINS={
    CAL+'/report/calibration.json':'fe60dc81922cea3c25f64b6336693b522caff220468bb6b7e0225ec6fcc2884a',
    CAL+'/manifest.json':'d5f9737d9a4e4f915202edfea1a0055031365cd78ee003bdccdad2b1fe0fdf12',
    CAL+'/protocol.json':'ffc2dc1f0c0560b7375ff2d3e1125b06a9dea445546a0cba24605060f266adf1',
    MAP+'/cells.json':'68ebc42331dc6d415381217878d7fc2251c8f480ff952eedac880b472beea56e',
    MAP+'/manifest.json':'89e49670c4897375262a220473aabc09772196a2784ae7b464c4037fcc963ee8',
    MAP+'/summary.json':'2567a21c99c063a8ba99d962c3a6f49b3c689dd43821f416551702a886bcd94a',
    MAP+'/criteria.json':'b9527594380d2a92b5338d34cbf433d5ca98f32cc7e9ff322d4b168773252a34'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(root,name):
    if Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('Use repository-relative paths')
    path=(root/name).resolve()
    if not path.is_relative_to(root): raise ValueError('Path escapes repository')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    parser.add_argument('--output',default='results/cell_ranges/v2_observed_q')
    args=parser.parse_args(); root=Path(args.repo).resolve(); out=local(root,args.output)
    inputs={}
    for name,pin in PINS.items():
        path=local(root,name)
        if sha(path)!=pin: raise ValueError('Frozen input SHA256 mismatch: '+name)
        inputs[name]=json.loads(path.read_text(encoding='utf-8-sig'))
    cal=inputs[CAL+'/report/calibration.json']; cmap=inputs[MAP+'/cells.json']
    cm=inputs[CAL+'/manifest.json']; mm=inputs[MAP+'/manifest.json']
    if cal['status']!='completed' or inputs[MAP+'/summary.json']['status']!='completed': raise ValueError('Expected completed sessions')
    if cal['verified_result_sha256']!=cm['result_sha256']: raise ValueError('Derived calibration source hashes disagree with manifest')
    for session,manifest in ((CAL,cm),(MAP,mm)):
        for name,digest in PINS.items():
            if name.startswith(session+'/') and name[len(session)+1:] in manifest['result_sha256']:
                if manifest['result_sha256'][name[len(session)+1:]]!=digest: raise ValueError('Input/manifest SHA disagreement: '+name)
    expected={(wl,bl) for wl in range(8) for bl in range(32)}
    cells={(c['wl'],c['bl']):c for c in cal['cells']}; fresh={(c['wl'],c['bl']):c for c in cmap}
    if set(cells)!=expected or set(fresh)!=expected or len(cal['cells'])!=256 or len(cmap)!=256: raise ValueError('Expected unique all256 inputs')
    states=cal['states']
    if len({s['checkpoint_id'] for s in states})!=len(states): raise ValueError('Repeated checkpoint ID')
    rows=[]
    fields=('checkpoint_id','role','q_mean','q_sd','crossed_targets','crossed_targets_postread_reached',
            'snr_low','snr_high','cv_high','linearity_epsilon_adc','linearity_tolerance_adc')
    for wl,bl in sorted(expected):
        saved=cells[wl,bl]; passing=[]
        for state in states:
            if (state['wl'],state['bl'])==(wl,bl) and state['measured'] and state['passes']:
                item={k:state[k] for k in fields}
                item['linearity_margin_adc']=state['linearity_tolerance_adc']-abs(state['linearity_epsilon_adc'])
                passing.append(item)
        programmed=[s for s in passing if s['role']=='set_crossing']
        category=('two_or_more_passing_SET_states' if len(programmed)>=2 else 'one_passing_SET_state' if programmed
                  else 'passing_non_SET_observations_only' if passing else 'no_passing_observation')
        rows.append(dict(wl=wl,bl=bl,category=category,passing_state_count=len(passing),passing_SET_count=len(programmed),
            observed_passing_q_min=min((s['q_mean'] for s in passing),default=None),
            observed_passing_q_max=max((s['q_mean'] for s in passing),default=None),passing_states=passing,
            fresh_map_status=fresh[wl,bl]['status'],fresh_map_reasons=fresh[wl,bl]['reasons'],
            discovery_status=saved['status'],terminal_checkpoint_id=saved.get('terminal_checkpoint_id'),
            late_G_pass=saved.get('final_g_difference_pass'),late_G_fractional_delta=saved.get('final_g_fractional_delta'),
            late_G_tolerance=saved.get('final_g_tolerance'),continuous_range_validated=False))
    metrics=dict(cells=256,cells_with_passing_states=sum(bool(r['passing_states']) for r in rows),
        cells_with_two_or_more_passing_SET_states=sum(r['passing_SET_count']>=2 for r in rows),
        fresh_candidates=sum(r['fresh_map_status']=='candidate' for r in rows))
    result=dict(offline_only=True,new_measurement=False,thresholds_changed=False,continuous_ranges_validated=False,
        input_sha256_verified=PINS,generator_sha256=sha(Path(__file__)),
        calibration_session=CAL,map_session=MAP,calibration_commit=cm['invocation']['git']['commit'],
        calibration_generator_sha256=cal['generator_sha256'],criteria=cal['provisional_criteria'],
        fresh_map_criteria=inputs[MAP+'/criteria.json'],raw_reaudited_here=False,
        limitation='Frozen derived inputs verified; original full calibration raw remains on Pi. q from mode7 nominal .3 only; no mode9 .1 q merged.',
        metrics=metrics,cells=rows)
    out.mkdir(parents=True,exist_ok=False)
    (out/'ranges.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (out/'cells.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
        for row in rows: writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    lines=['# Наблюдённые q-точки, прошедшие MVM-screen', '',
        '**Только offline-анализ сохранённых данных; новых измерений и ослабления критериев нет.**', '',
        f"Все 256 ячеек: {metrics['cells_with_passing_states']} имеют хотя бы одну passing-точку; только {metrics['cells_with_two_or_more_passing_SET_states']} — две или более passing SET-точки. Новая карта содержит {metrics['fresh_candidates']} кандидатов.", '',
        'Min/max описывают лишь крайние наблюдённые passing q, а не непрерывный рабочий диапазон. Baseline/postRESET не считаются SET-точками; несколько пересечённых целей одного checkpoint остаются одной точкой. Успешное состояние могло быть перезаписано.', '',
        'Screen не ограничивал q общим окном .01–.05: высокие baseline q также могли пройти. Мы не меняем пороги и никого не переводим в candidate подбором окна. q здесь только из mode7 при nominal .3 В; q свежей mode9 карты при .1 В к диапазонам не добавляется.', '',
        '## Все проходящие измеренные состояния', '',
        '| WL,BL | checkpoint / роль | q mean | q SD (5 mode7 ответов) | ε / допуск / запас, ADC | ≥2 SET-точек | новая карта | поздний G pass |',
        '|---|---|---:|---:|---|---|---|---|']
    for row in rows:
        for state in row['passing_states']:
            lines.append(f"| {row['wl']},{row['bl']} | {state['checkpoint_id']} / {state['role']} | {state['q_mean']:.17g} | {state['q_sd']:.17g} | {state['linearity_epsilon_adc']:.12g} / {state['linearity_tolerance_adc']:.12g} / {state['linearity_margin_adc']:.12g} | {row['passing_SET_count']>=2} | {row['fresh_map_status']} | {row['late_G_pass']} |")
    lines+=['', 'Три ячейки с двумя passing SET-точками: (0,8), (3,8), (5,8). Это приоритеты для отдельной проверки, не подтверждённые непрерывные диапазоны; все три отклонены свежей картой. У нижней SET-точки (5,8) ε=27.6 ADC при допуске27.9 — запас лишь0.3 ADC, пограничное наблюдение.', '',
        'Повторяемость записи, число различимых уровней/битность, совместная пригодность пар и точность CIFAR этим не установлены. Поздний G сравнивает terminal5 с final3, а не подтверждает сохранность каждой исторической MVM-кривой.', '',
        '## Все 256 адресов', '', '| WL,BL | Категория | Passing / SET | Наблюдённые q | новая карта | поздний G pass |', '|---|---|---|---|---|---|']
    for row in rows:
        points=', '.join(format(s['q_mean'],'.17g') for s in row['passing_states']) or '—'
        lines.append(f"| {row['wl']},{row['bl']} | {row['category']} | {row['passing_state_count']} / {row['passing_SET_count']} | {points} | {row['fresh_map_status']} | {row['late_G_pass']} |")
    lines+=['', 'SHA256 всех семи входов закреплены в скрипте и проверены до анализа; числовые поля/ID/роли сохранены в ranges.json и cells.csv. Полный raw здесь повторно не аудирован и этим компактным набором не заменён.', '',
        'Воспроизведение из корня репозитория (нужен новый выходной каталог):', '', '```sh',
        'python analyze_cell_ranges.py --repo . --output results/cell_ranges/v2_observed_q_recheck', '```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=out.relative_to(root).as_posix(),**metrics)))


if __name__=='__main__': main()
