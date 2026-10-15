"""Explain frozen sequential-MVM results offline, without changing qualification."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def local(value):
    path=(ROOT/value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(ROOT):raise ValueError('Repository-relative path required')
    return path


def reasons_for_cell(cell):
    reasons=[]
    for phase in ('basis','selection','final'):
        v=cell[phase];signal=v['signal_adc'];noise=v['noise_adc']
        if signal is None or noise is None:reasons.append(f'{phase}: отсутствуют пять полных zero/single отсчётов');continue
        if signal<5*max(1.,noise):reasons.append(f'{phase}: signal={signal:.9g} < 5×max(1,SD)={5*max(1.,noise):.9g} ADC')
        if noise>.05*signal:reasons.append(f'{phase}: SD={noise:.9g} > 0.05×signal={.05*signal:.9g} ADC')
    base=cell['basis']
    if base['signal_adc'] is not None:
        for phase in ('selection','final'):
            other=cell[phase]
            if other['signal_adc'] is None:continue
            error=other['single']['mean']-base['single']['mean'];limit=max(2.,.05*abs(base['signal_adc']))
            if abs(error)>limit:reasons.append(f'{phase}: raw single mean error={error:.9g}, |error|>{limit:.9g} ADC')
            ze=other['zero']['mean']-base['zero']['mean']
            if abs(ze)>2:reasons.append(f'{phase}: zero shift={ze:.9g}, |shift|>2 ADC')
            if other['zero']['sd']>2:reasons.append(f"{phase}: zero SD={other['zero']['sd']:.9g}>2 ADC")
    if not cell['g_consistent']:reasons.append(f"G: delta={cell['g_fraction_delta']}, limit={cell['g_fraction_limit']}; missing or inconsistent pre/post")
    return reasons


def explain(report):
    cells={(c['wl'],c['bl']):c for c in report['cells']};coefficients=[]
    for c in report['coefficients']:
        reasons=[]
        for side in ('positive_cell','negative_cell'):
            coord=tuple(c[side]);cell=cells[coord]
            if not cell['singleton_supported']:
                reasons.extend(f'{side} WL{coord[0]} BL{coord[1]} — {r}' for r in reasons_for_cell(cell))
        for phase in ('basis','selection'):
            value=c[phase];d=value['signal_adc'];sd=value['conservative_sd_adc']
            if d is None or sd is None:reasons.append(phase+': missing signed evidence');continue
            if abs(d)<5*max(1.,sd):reasons.append(f'{phase}: |D|={abs(d):.9g} < 5×max(1,SDbound)={5*max(1.,sd):.9g} ADC')
            if sd>.05*abs(d):reasons.append(f'{phase}: signed SDbound={sd:.9g} > .05×|D|={.05*abs(d):.9g} ADC')
        base=c['basis']['signal_adc'];selection=c['selection']['signal_adc'];final=c['final_signed_adc'];noise=c['final_sd_bound_adc']
        if base is not None:
            limit=max(2.,.05*abs(base))
            for phase,value in (('selection',selection),('final',final)):
                if value is None:reasons.append(phase+': missing signed response')
                elif abs(value-base)>limit:reasons.append(f'{phase}: signed error={value-base:.9g}, |error|>{limit:.9g} ADC')
        if final is not None and noise is not None:
            if abs(final)<5*max(1.,noise):reasons.append(f'final: |D|={abs(final):.9g} < 5×max(1,SDbound)={5*max(1.,noise):.9g} ADC')
            if noise>.05*abs(final):reasons.append(f'final: signed SDbound={noise:.9g} > .05×|D|={.05*abs(final):.9g} ADC')
        if not c['final_supported'] and not reasons:reasons.append('Unexplained saved rejection: review original analysis; not reclassified')
        coefficients.append(dict(pair_id=c['pair_id'],logical_index=c['logical_index'],positive_cell=c['positive_cell'],negative_cell=c['negative_cell'],
            basis_signed_adc=base,selection_signed_adc=selection,final_signed_adc=final,final_sd_bound_adc=noise,
            saved_supported=c['final_supported'],reasons=reasons))
    failures=[]
    for step in report['steps']:
        if step['status']!='fail':continue
        reasons=[]
        for name,value in [(f'WL{wl}',v) for wl,v in step['rows'].items()]+[('signed',step['signed'])]:
            if not value:continue
            if not value['mean_pass']:reasons.append(f"{name}: mean error={value['error_adc']:.9g}, tolerance={value['tolerance_adc']:.9g} ADC")
            if not value['repeatability_pass']:reasons.append(f"{name}: repeat SD={value['statistics']['sd']:.9g}, limit={value['tolerance_adc']:.9g} ADC")
        failures.append(dict(pair_id=step['pair_id'],pattern=step['pattern'],t=step['t'],active=step['active_logical_indices'],
            row_status=step['row_status'],signed_status=step['signed_status'],reasons=reasons,
            cancellation_ratio=step['expected_cancellation_ratio']))
    lif=[]
    for trace in report['lif']:
        reasons=[]
        if not trace['reference_excited_and_silent']:reasons.append('Эталон не содержит и спайк, и тихий такт')
        if not trace['all_coefficients_supported']:reasons.append('Не все заранее фиксированные коэффициенты подтверждены свежими anchors/G')
        if not trace['mvm_steps_pass']:reasons.append('Не все row/signed такты подтвердили среднее и повторяемость')
        if len(trace['actual'])!=5:reasons.append('Неполные пять последовательностей')
        if trace['numerical_lif_status']=='fail':reasons.append('Численная ошибка потенциала >0.05 или несовпадение спайков')
        lif.append(dict(pair_id=trace['pair_id'],pattern=trace['pattern'],polarity=trace['polarity'],
            saved_status=trace['status'],numerical_status=trace['numerical_lif_status'],reasons=reasons,
            maximum_membrane_error=max(trace['max_membrane_errors']) if trace['max_membrane_errors'] else None,
            exact_spikes_by_repeat=trace['exact_spikes_by_repeat']))
    proposals=[]
    for pair_id in range(4):
        selected=[c for c in coefficients if c['pair_id']==pair_id and c['saved_supported']]
        proposals.append(dict(pair_id=pair_id,previous_logical_indices=[c['logical_index'] for c in selected],
            positive_cells=[c['positive_cell'] for c in selected],negative_cells=[c['negative_cell'] for c in selected],
            positions=len(selected),physical_cells=2*len(selected),
            signs=dict(positive=sum(c['basis_signed_adc']>0 for c in selected),negative=sum(c['basis_signed_adc']<0 for c in selected)),
            status='posthoc_candidate_subset_requires_new_measurements',
            next_minimal_test='Freeze these routes first; new basis/selection, then newly measured dispersed/grouped/dense logical sums and LIF. No fitted correction from this final data.',
            limitation='Current differences only: not programmable trained weights or a remap qualification'))
    return dict(coefficients=coefficients,unsupported_coefficients=[c for c in coefficients if not c['saved_supported']],
        failed_steps=failures,lif=lif,future_candidate_subsets=proposals,
        counts=dict(singletons=report['singleton_supported'],contrasts=report['mapped_contrasts_supported'],
            step_statuses=report['step_statuses'],lif_statuses=report['lif_statuses'],
            numerical_lif_statuses=dict(Counter(t['numerical_status'] for t in lif))),
        thresholds_changed=False,new_measurement=False,new_remap_qualified=False,
        proposal_selection_is_posthoc=True,no_new_hardware_authorization=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',required=True);parser.add_argument('--output')
    args=parser.parse_args(argv);session=local(args.session);manifest=json.loads((session/'manifest.json').read_bytes())
    names=('analysis.json','summary.json','protocol.json','frozen_reference.json')
    hashes={}
    for name in names:
        hashes[name]=hashlib.sha256((session/name).read_bytes()).hexdigest()
        if manifest['result_sha256'].get(name)!=hashes[name]:raise ValueError('Session result fingerprint mismatch: '+name)
    hashes['manifest.json']=hashlib.sha256((session/'manifest.json').read_bytes()).hexdigest()
    protocol=json.loads((session/'protocol.json').read_bytes())
    if protocol['protocol_id']!='mapped-sequential-mvm-lif-v1':raise ValueError('Unsupported diagnostic protocol')
    result=explain(json.loads((session/'analysis.json').read_bytes()))
    result.update(session=args.session,source=manifest['source'],session_status=manifest['status'],input_sha256=hashes,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        recorded_raw_sha256=manifest['result_sha256'].get('raw.jsonl'),
        verification='Compact files checked against manifest. This script does not independently re-audit raw ADC or hardware acquisition.')
    output=local(args.output or args.session+'/diagnostics');output.mkdir(parents=True)
    (output/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (output/'coefficients.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result['coefficients'][0]),lineterminator='\n');writer.writeheader()
        for row in result['coefficients']:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in row.items()})
    text=['# Диагностика последовательного MVM', '',f"Источник: {result['source']}; сессия `{manifest['session_id']}`; acquisition `{manifest['status']}`.",
        f"Одиночные отклики: {result['counts']['singletons']}/256; разности: {result['counts']['contrasts']}/47.",
        f"Статусы тактов: {result['counts']['step_statuses']}; LIF с eligibility: {result['counts']['lif_statuses']}.",
        f"Отдельные численные статусы LIF: {result['counts']['numerical_lif_statuses']}.",
        '', 'Новые измерения и переоценка порогов не выполнялись. Compact-файлы сверены с manifest; это не повторный независимый аудит raw.',
        '', '## Неподтверждённые коэффициенты', '', '| WL-пара / индекс | Физические + / − | Причины |', '|---|---|---|']
    for c in result['unsupported_coefficients']:text.append(f"| {c['pair_id']} / {c['logical_index']} | {c['positive_cell']} / {c['negative_cell']} | {'; '.join(c['reasons']).replace('|','abs')} |")
    text+=['','## Провалившиеся такты', '', '| Пара / шаблон / t | Причины |','|---|---|']
    for s in result['failed_steps']:text.append(f"| {s['pair_id']} / {s['pattern']} / {s['t']} | {'; '.join(s['reasons'])} |")
    text+=['','## LIF: численный результат отдельно от допуска', '', '| Пара / шаблон / знак | Численно | Итог | Ограничения |','|---|---|---|---|']
    for t in result['lif']:text.append(f"| {t['pair_id']} / {t['pattern']} / {t['polarity']} | {t['numerical_status']} | {t['saved_status']} | {'; '.join(t['reasons'])} |")
    text+=['','## Предложение будущего минимального подтверждения', '',
        'Ниже — post-hoc подмножество текущих разностей, а не новая квалификация. Перед следующим опытом маршруты фиксируются, затем заново измеряются basis/selection и final. Не менять масштаб по неудачным final-отсчётам.',
        '', '| WL-пара | Старые индексы | Позиций / ячеек | Знаки + / − |','|---|---|---|---|']
    for p in result['future_candidate_subsets']:text.append(f"| {2*p['pair_id']}/{2*p['pair_id']+1} | {p['previous_logical_indices']} | {p['positions']} / {p['physical_cells']} | {p['signs']['positive']} / {p['signs']['negative']} |")
    text+=['','Новый опыт не запускается этим отчётом. Стабильные одиночные отсчёты не гарантируют плотные суммы: шум, общий baseline и компенсация знаков проверяются новыми измерениями каждого вклада. Никаких выводов о физической причине неисправности или готовности CIFAR.']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(result['counts']));return 0


if __name__=='__main__':raise SystemExit(main())
