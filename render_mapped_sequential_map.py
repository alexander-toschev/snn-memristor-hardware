"""Render frozen per-cell evidence; never changes acquisition or qualification."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path

from report_mapped_sequential_diagnostics import local, reasons_for_cell

COLORS={
    'mapped_contrast_supported':('#9ad4a4','C','Одиночный отклик и выбранная разность подтверждены'),
    'mapped_singleton_only':('#f1ca7d','S!','Одиночный отклик подтверждён; выбранная разность нет'),
    'unmapped_singleton_supported':('#9fc4e9','S','Одиночный отклик подтверждён; вне выбранных 94'),
    'mapped_singleton_unsupported':('#e9a5a5','?','Выбрана в 94, одиночный критерий не подтверждён'),
    'unmapped_singleton_unsupported':('#d6dbe1','·','Одиночный критерий не подтверждён; вне выбранных 94')}


def build(report):
    mapped={tuple(c[side]):c for c in report['coefficients'] for side in ('positive_cell','negative_cell')}
    cells=[]
    for cell in report['cells']:
        coordinate=(cell['wl'],cell['bl']);coefficient=mapped.get(coordinate)
        if coefficient:
            key='mapped_contrast_supported' if coefficient['final_supported'] else 'mapped_singleton_only' if cell['singleton_supported'] else 'mapped_singleton_unsupported'
        else:key='unmapped_singleton_supported' if cell['singleton_supported'] else 'unmapped_singleton_unsupported'
        cells.append(dict(**cell,map_category=key,mapped_coefficient=coefficient,
            singleton_reasons=reasons_for_cell(cell) if not cell['singleton_supported'] else []))
    numeric=[]
    for step in report['steps']:
        complete=bool(step['signed'] and step['signed']['statistics'] and step['signed']['statistics']['n']==5
            and all(v and v['statistics'] and v['statistics']['n']==5 for v in step['rows'].values()))
        failed_rows=[wl for wl,v in step['rows'].items() if v and not v['passes']]
        failed_signed=bool(step['signed'] and not step['signed']['passes'])
        if complete and (failed_rows or failed_signed):
            numeric.append(dict(pair_id=step['pair_id'],pattern=step['pattern'],t=step['t'],saved_status=step['status'],
                failed_rows=failed_rows,signed_failed=failed_signed,rows=step['rows'],signed=step['signed']))
    return dict(cells=cells,category_counts=dict(Counter(c['map_category'] for c in cells)),
        numerical_step_failures=numeric,numerical_step_failures_count=len(numeric),
        admitted_step_failures=sum(s['status']=='fail' for s in report['steps']),
        numerical_step_failures_despite_inconclusive=sum(s['saved_status']=='inconclusive' for s in numeric),
        lif_statuses=report['lif_statuses'],numerical_lif_statuses=dict(Counter(t['numerical_lif_status'] for t in report['lif'])),
        g_consistent=sum(c['g_consistent'] for c in cells),new_qualification=False,
        interpretation='Each category is saved singleton/contrast evidence only, not grouped MVM/LIF or CIFAR qualification.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',required=True);parser.add_argument('--output');args=parser.parse_args(argv)
    session=local(args.session);manifest=json.loads((session/'manifest.json').read_bytes())
    raw=(session/'analysis.json').read_bytes();pin=hashlib.sha256(raw).hexdigest()
    if manifest['result_sha256'].get('analysis.json')!=pin:raise ValueError('Analysis differs from acquisition manifest')
    report=build(json.loads(raw));report.update(session=args.session,source=manifest['source'],session_id=manifest['session_id'],
        input_sha256={'analysis.json':pin,'manifest.json':hashlib.sha256((session/'manifest.json').read_bytes()).hexdigest()},
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reason_helper_sha256=hashlib.sha256(Path(__file__).with_name('report_mapped_sequential_diagnostics.py').read_bytes()).hexdigest(),
        raw_reaudit=False)
    output=local(args.output or args.session+'/diagnostics/map');output.mkdir(parents=True)
    (output/'map.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    escape=html.escape;bycell={(c['wl'],c['bl']):c for c in report['cells']}
    table=['<table class="grid"><thead><tr><th>WL / BL</th>']+[f'<th>{bl}</th>' for bl in range(32)]+['</tr></thead><tbody>']
    for wl in range(8):
        table.append(f'<tr><th>WL{wl}</th>')
        for bl in range(32):
            cell=bycell[wl,bl];color,symbol,label=COLORS[cell['map_category']]
            table.append(f'<td><button style="background:{color}" data-cell="{wl},{bl}" data-kind="{cell["map_category"]}" title="WL{wl} BL{bl}: {escape(label)}">{symbol}<small>{wl},{bl}</small></button></td>')
        table.append('</tr>')
    table.append('</tbody></table>')
    counts=report['category_counts'];legend=''.join(f'<li><span style="background:{color}">{symbol}</span> {escape(label)}: <b>{counts.get(key,0)}</b></li>' for key,(color,symbol,label) in COLORS.items())
    options='<option value="all">Все категории</option>'+''.join(f'<option value="{key}">{escape(label)}</option>' for key,(_,_,label) in COLORS.items())
    details=[]
    for c in report['cells']:
        details.append(f'<tr><td>{c["wl"]}</td><td>{c["bl"]}</td><td>{escape(COLORS[c["map_category"]][2])}</td><td>{c["basis"]["signal_adc"]}</td><td>{c["selection"]["signal_adc"]}</td><td>{c["final"]["signal_adc"]}</td><td>{escape("; ".join(c["singleton_reasons"]))}</td></tr>')
    numeric=''.join(f'<tr><td>{s["pair_id"]}</td><td>{s["pattern"]}</td><td>{s["t"]}</td><td>{s["saved_status"]}</td><td>{escape(str(s["failed_rows"]))}</td><td>{s["signed_failed"]}</td></tr>' for s in report['numerical_step_failures'])
    data=json.dumps(report['cells'],ensure_ascii=False).replace('<','\\u003c')
    page=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Карта одиночных откликов и разностей</title>
<style>body{{font:15px system-ui,sans-serif;color:#203040;background:#f5f7fa;margin:24px;line-height:1.5}}h1{{font-size:26px}}article{{background:white;padding:20px;border-radius:12px;margin:16px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse}}th,td{{padding:6px;border:1px solid #dae0e7;text-align:left}}.grid th{{font-size:11px;text-align:center}}.grid td{{padding:2px}}button{{min-width:34px;height:48px;border:1px solid #536475;border-radius:4px;cursor:pointer;font-weight:700}}button small{{display:block;font-size:9px}}button:focus{{outline:3px solid #122c76}}li{{list-style:none;margin:6px 0}}li span{{display:inline-block;width:32px;text-align:center;border-radius:4px}}pre{{white-space:pre-wrap;word-break:break-word;max-height:500px;overflow:auto}}.muted{{opacity:.15}}a{{color:#164a93}}@media print{{button{{print-color-adjust:exact}}pre{{max-height:none}}}}</style>
<h1>256 адресов: одиночный отклик ≠ квалифицированный групповой MVM</h1>
<p>{escape(manifest['source'])}, сессия <code>{escape(manifest['session_id'])}</code>. Критерии не изменены; это отображение сохранённого анализа.</p>
<article><b>Одиночные критерии: {sum(c['singleton_supported'] for c in report['cells'])}/256.</b> Подтверждённые выбранные разности: {counts.get('mapped_contrast_supported',0)//2}/47, то есть {counts.get('mapped_contrast_supported',0)}/94 использованных ячеек. Согласованность G: {report['g_consistent']}/256.
<p>LIF с условиями допуска: {escape(str(report['lif_statuses']))}; численно отдельно: {escape(str(report['numerical_lif_statuses']))}. Зелёный цвет не означает готовность группового MVM, записи весов или CIFAR.</p><ul>{legend}</ul></article>
<article><label>Показать: <select id="filter">{options}</select></label><p>Нажмите ячейку или выберите её клавишей Tab + Enter. Координаты логические WL/BL, физическая причина отсутствия отклика не установлена.</p><div class="scroll">{''.join(table)}</div><h2 id="cell-title">Данные ячейки</h2><pre id="detail">Выберите ячейку: ниже появятся все mean/SD, причины и связанные физические адреса пары.</pre></article>
<article><h2>Численные ошибки нельзя скрывать статусом inconclusive</h2><p>Допущенных fail: {report['admitted_step_failures']}. Всего численно непрошедших полных тактов: {report['numerical_step_failures_count']}, из них {report['numerical_step_failures_despite_inconclusive']} имеют итог inconclusive из-за условий допуска. Итоговые статусы не пересчитаны.</p><table><tr><th>Пара</th><th>Шаблон</th><th>t</th><th>Сохранённый статус</th><th>Непрошедшие WL</th><th>Signed fail</th></tr>{numeric}</table></article>
<article><details><summary>Полная печатная таблица 256 адресов</summary><table><tr><th>WL</th><th>BL</th><th>Категория</th><th>Basis ADC</th><th>Selection ADC</th><th>Final ADC</th><th>Причины одиночного отказа</th></tr>{''.join(details)}</table></details></article>
<p><a href="../REPORT.md">Точные причины неподтверждённых разностей и LIF</a> · <a href="map.json">Все метрики JSON</a> · <a href="../../analysis.json">Исходный analysis</a> · <a href="../../manifest.json">Manifest</a>. SHA analysis: <code>{pin}</code>. Здесь проверены hashes compact-файлов; независимый raw-аудит выполняется отдельным инструментом. После программирования или изменения состояния эту карту следует измерить заново.</p>
<script>const cells={data};const by=new Map(cells.map(c=>[`${{c.wl}},${{c.bl}}`,c]));document.querySelectorAll('[data-cell]').forEach(b=>b.onclick=()=>{{const c=by.get(b.dataset.cell);document.getElementById('cell-title').textContent=`WL${{c.wl}} / BL${{c.bl}}`;document.getElementById('detail').textContent=JSON.stringify(c,null,2);}});document.getElementById('filter').onchange=e=>document.querySelectorAll('[data-cell]').forEach(b=>b.classList.toggle('muted',e.target.value!=='all'&&b.dataset.kind!==e.target.value));</script></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('category_counts','numerical_step_failures_count','numerical_step_failures_despite_inconclusive')}));return 0


if __name__=='__main__':raise SystemExit(main())
