"""Offline detailed cell map. Standard library only; no hardware imports or calls."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path

CATEGORIES = {
    'candidate': ('К', 'Предварительный кандидат', '#39986b'),
    'zero': ('0', 'Оба сигнала равны нулю', '#b6bcc6'),
    'weak_both': ('С', 'Оба SNR ниже порога; есть положительный отклик', '#9876bc'),
    'cv': ('V', 'Измеренный CV превышает порог', '#619bd1'),
    'low': ('L', 'Не прошёл SNR на 0,05 В', '#edaa58'),
    'high': ('H', 'Не прошёл SNR на 0,1 В', '#e7d86a'),
    'nonlinear': ('N', 'Оба SNR прошли; нелинейный отклик', '#da6b67'),
    'drift': ('G', 'Различие проводимости до/после', '#aa886b'),
    'other': ('?', 'Неполные данные / другая причина', '#eeeeee'),
}
DISCLAIMER = ('Карта текущего состояния при номинальных 0,05–0,1 В. Пороги предварительные. '
 'Отказ критерия не устанавливает физическую неисправность. Нулевой ADC может быть отсечкой. '
 'Программируемость, совместный MVM выбранных пар и CIFAR accuracy не проверены. После записи весов карта требует обновления. '
 'WL/BL — логические адреса Connector; физическая адресация независимо не калибровалась.')


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()


def relative(root, value):
    if Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('Paths must be repository-relative without ..')
    path=(root/value).resolve()
    if not path.is_relative_to(root): raise ValueError('Path escapes repository')
    return path


def number(value):
    return 'не определено' if value is None else format(value,'.12g')


def details(cell, criteria):
    m=cell['metrics']; reasons=cell['reasons']
    low='weak_or_nonpositive_signal_low' in reasons
    high='weak_or_nonpositive_signal_high' in reasons
    if cell['status']=='candidate': category='candidate'
    elif cell['status']=='unmeasured' or 'incomplete_measurements' in reasons: category='other'
    elif m.get('signal_low_adc')==0 and m.get('signal_high_adc')==0: category='zero'
    elif low and high: category='weak_both'
    elif m.get('cv_high') is not None and m['cv_high']>criteria['maximum_cv_high']: category='cv'
    elif low: category='low'
    elif high: category='high'
    elif 'nonlinear_zero_adjusted_response' in reasons: category='nonlinear'
    elif 'pre_post_conductance_difference' in reasons: category='drift'
    else: category='other'
    checks=[]
    for key,label in [('low','0,05 В'),('high','0,1 В')]:
        signal=m.get(f'signal_{key}_adc'); snr=m.get(f'snr_{key}')
        ok=snr is not None and signal>0 and snr>=criteria['minimum_snr']
        checks.append({'name':f'SNR {label}','pass':ok,'expression':
            f'сигнал {number(signal)} ADC; SNR {number(snr)} '+('≥' if ok else '<')+
            f' {number(criteria["minimum_snr"])}; требуется положительный сигнал; порог сигнала {number(m.get(f"signal_{key}_threshold_adc"))} ADC'})
    cv=m.get('cv_high'); limit=criteria['maximum_cv_high']
    checks.append({'name':'CV 0,1 В','pass':cv is not None and cv<=limit,'expression':
        ('CV не определён: сигнал 0,1 В неположительный; это НЕ измерение повышенного шума.' if cv is None else
         f'CV {number(cv)} '+('≤' if cv<=limit else '>')+f' {number(limit)}')})
    for label,value,tolerance in [('Линейность ε, ADC',m.get('linearity_epsilon_adc'),m.get('linearity_tolerance_adc')),
                                 ('Изменение G, доля',m.get('g_fractional_delta'),m.get('g_fractional_tolerance'))]:
        ok=value is not None and tolerance is not None and abs(value)<=tolerance
        expression='Недостаточно данных' if value is None or tolerance is None else (
            f'значение {number(value)}; |значение| {number(abs(value))} '+('≤' if ok else '>')+
            f' допуск {number(tolerance)}; |значение|/допуск = {number(abs(value)/tolerance)}')
        checks.append({'name':label,'pass':ok,'expression':expression})
    return {**cell,'category':category,'checks':checks,'drift_marker':'pre_post_conductance_difference' in reasons}


def generate(root, session, output):
    names=('cells.json','criteria.json','summary.json','manifest.json')
    loaded={name:json.loads((session/name).read_text(encoding='utf-8-sig')) for name in names}
    cells=loaded['cells.json']; criteria=loaded['criteria.json']['criteria']; manifest=loaded['manifest.json']
    if len(cells)!=256 or {(c['wl'],c['bl']) for c in cells}!={(w,b) for w in range(8) for b in range(32)}:
        raise ValueError('Expected exactly one record per logical WL0..7 / BL0..31')
    verified={}
    for name,expected in manifest['result_sha256'].items():
        path=relative(session,name)
        actual=digest(path)
        if actual!=expected: raise ValueError(f'Frozen result hash mismatch: {name}')
        verified[name]=actual
    if loaded['criteria.json']!=manifest['criteria']: raise ValueError('Criteria snapshot differs from manifest')
    enriched=[details(c,criteria) for c in sorted(cells,key=lambda c:(c['wl'],c['bl']))]
    counts=Counter(c['category'] for c in enriched)
    output.mkdir(parents=True,exist_ok=False)
    provenance={'session':session.relative_to(root).as_posix(),'inputs_sha256':{n:digest(session/n) for n in names},
                'verified_result_sha256':verified,'measurement_source_sha256_recorded':manifest['source_sha256'],
                'current_runtime_source_comparison':'not performed; recorded hashes describe measurement version',
                'generator_sha256':digest(Path(__file__)),'category_counts':dict(counts),'criteria':criteria}
    (output/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    csv_rows=[]
    for c in enriched:
        row={'wl':c['wl'],'bl':c['bl'],'status':c['status'],'category_ru':CATEGORIES[c['category']][1],
             'reason_codes':'; '.join(c['reasons']),'checks_ru':' | '.join(x['name']+': '+x['expression'] for x in c['checks'])}
        for key,value in c['metrics'].items():
            if isinstance(value,dict): row.update({key+'_'+k:v for k,v in value.items()})
            else: row[key]=value
        csv_rows.append(row)
    fields=list(csv_rows[0])
    for row in csv_rows:
        fields.extend(k for k in row if k not in fields)
    with (output/'cells_detailed.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(csv_rows)
    legend=''.join(f'<span class="legend" style="--color:{color}"><b>{symbol}</b> {html.escape(label)} <strong>{counts[key]}</strong></span>'
                   for key,(symbol,label,color) in CATEGORIES.items() if counts[key])
    grid='<table class="matrix"><thead><tr><th>WL / BL</th>'+''.join(f'<th>{b}</th>' for b in range(32))+'</tr></thead><tbody>'
    for wl in range(8):
        grid+=f'<tr><th>{wl}</th>'
        for c in enriched[wl*32:(wl+1)*32]:
            symbol,label,color=CATEGORIES[c['category']]; marker=' Δ' if c['drift_marker'] else ''
            grid+=f'<td><button data-index="{wl*32+c["bl"]}" style="background:{color}" aria-label="WL {wl}, BL {c["bl"]}: {html.escape(label)}{marker}"><b>{symbol}{marker}</b><small>{wl},{c["bl"]}</small></button></td>'
        grid+='</tr>'
    grid+='</tbody></table>'
    rows=''
    for c in enriched:
        m=c['metrics']; adc='; '.join(f'{label}: {number(m[name]["mean"])} ± {number(m[name]["sd"])} (n={m[name]["n"]})' for name,label in [('zero_adc','0'),('low_adc','0,05'),('high_adc','0,1')])
        reasons='<br>'.join(html.escape(x['name']+': '+x['expression']) for x in c['checks'])
        rows+=f'<tr><td>{c["wl"]}</td><td>{c["bl"]}</td><td>{html.escape(CATEGORIES[c["category"]][1])}</td><td>{html.escape(adc)}</td><td>{reasons}</td><td>{html.escape("; ".join(c["reasons"]) or "все критерии выполнены")}</td></tr>'
    data=json.dumps({'cells':enriched,'criteria':criteria,'categories':CATEGORIES},ensure_ascii=False).replace('<','\\u003c')
    template='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Подробная карта 256 ячеек</title><style>
*{box-sizing:border-box}body{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f4f6f8;color:#182436}main{max-width:1680px;margin:auto;padding:28px}h1{font-size:30px;margin:0 0 8px}h2{font-size:22px}.intro,.panel{background:white;padding:20px;border:1px solid #d8dfe8;border-radius:12px;margin:16px 0}.muted{color:#516078}.legend{display:inline-flex;gap:8px;align-items:center;margin:6px 14px 6px 0}.legend b{background:var(--color);border:1px solid #677485;padding:2px 8px;border-radius:4px}.scroll{overflow:auto}.matrix{border-collapse:separate;border-spacing:3px;width:100%;min-width:1350px}.matrix th{font-size:12px}.matrix td{padding:0}.matrix button{border:1px solid #7b8794;border-radius:5px;min-width:36px;width:100%;padding:7px 1px;color:#142030;cursor:pointer}.matrix button:focus-visible,.matrix button.selected{outline:3px solid #172d57;outline-offset:1px}.matrix small{display:block;font-size:10px}.dim{opacity:.18}.toolbar{display:flex;gap:16px;flex-wrap:wrap;align-items:center}select{padding:8px}table.detail,table.full{border-collapse:collapse;width:100%}td,th{text-align:left;vertical-align:top;padding:7px;border-bottom:1px solid #e0e5ec}.pass{color:#17653b}.fail{color:#9b252b}.full{font-size:11px}.full td{min-width:24px}a{color:#1e5294}.badge{font-size:13px;color:#43536a}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:850px){.metrics{grid-template-columns:1fr}}@media print{body{background:white}main{padding:0}.toolbar,.panel a{display:none}.scroll{overflow:visible}.matrix{min-width:0}.matrix button{min-width:0;padding:2px;font-size:9px;print-color-adjust:exact}.matrix small{font-size:7px}.full{font-size:8px}thead{display:table-header-group}tr{break-inside:avoid}.panel{break-inside:avoid}details>summary{display:none}details:not([open])>table{display:table}.intro{border:0}}
</style><main><header><div class="badge">OFFLINE · ФИКСИРОВАННЫЕ ДАННЫЕ · 8 WL × 32 BL</div><h1>Почему ячейка прошла или не прошла отбор</h1><p>@@DISCLAIMER@@</p></header>
<section class="intro"><strong>Сессия @@SESSION@@</strong><p>@@STATUS@@ · 4 136 пар request/result · без mode7 и записи весов. Все сохранённые result SHA256 сверены с manifest; исходники измерения указаны в provenance, текущий runtime не пересравнивался.</p><p><a href="../REPORT.md">Независимый аудит</a> · <a href="../candidate_cells.csv">Маска кандидатов wl,bl</a> · <a href="cells_detailed.csv">Подробный CSV</a> · <a href="overview.svg">Статичная SVG-карта</a> · <a href="provenance.json">Hashes / provenance</a></p><p>Цвет — взаимоисключающий приоритет отображения, а не физический диагноз. Все пересекающиеся причины сохранены ниже. Δ — превышен допуск изменения G. Неположительный сигнал делает CV неопределённым; это не доказательство шума.</p>@@LEGEND@@</section>
<div class="toolbar"><label>Категория <select id="category"><option value="">Все</option></select></label><label>Любая причина <select id="reason"><option value="">Все</option></select></label><label>WL <select id="wl"><option value="">Все</option></select></label><span id="matches"></span></div><div class="scroll">@@GRID@@</div><p class="muted">Выберите ячейку мышью или Tab + Enter; стрелки перемещают фокус по физически фиксированной сетке.</p><section class="panel" id="detail" aria-live="polite"></section>
<section class="intro"><h2>Как читать проверки</h2><p>SNR = (μV−μ0) / max(1 ADC, √(SDV²+SD0²)); SD выборочное. CV = √(SDH²+SD0²)/(μH−μ0), только при положительном знаменателе. ε = μH−2μL+μ0; SEε = √(SDH²/5+4SDL²/5+SD0²/5). Допуск ε = max(5%|μH−μ0|, 3SEε, 2 ADC). ΔG = Gpost−Gpre; SEΔ = √(SDpre²/3+SDpost²/3), допуск доли = max(5%,3SEΔ/Gpre). Фактические значения критериев этой сессии сохранены в criteria.json и provenance.json.</p><p>Отрицательный ε означает суб-линейный, положительный — супер-линейный двухуровневый отклик; это описание измерения, не установленная причина. Общий zero для WL учитывается в шуме, но не делает оценки разных ячеек независимыми. Показанные числа округлены для чтения до 12 значащих цифр; CSV сохраняет числовую точность JSON.</p></section>
<details open><summary><h2>Все 256 ячеек: таблица для печати</h2></summary><table class="full"><thead><tr><th>WL</th><th>BL</th><th>Категория</th><th>ADC μ ± SD</th><th>Точные проверки</th><th>Все коды причин</th></tr></thead><tbody>@@ROWS@@</tbody></table></details>
<script type="application/json" id="data">@@DATA@@</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),buttons=[...document.querySelectorAll('[data-index]')];
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const f=x=>x===null||x===undefined?'не определено':typeof x==='number'?Number(x.toPrecision(12)).toString():String(x);
Object.entries(D.categories).filter(([k])=>D.cells.some(c=>c.category===k)).forEach(([k,v])=>$('category').add(new Option(v[1],k)));
[...new Set(D.cells.flatMap(c=>c.reasons))].sort().forEach(r=>$('reason').add(new Option(r,r)));for(let w=0;w<8;w++)$('wl').add(new Option(w,w));
function filter(){let n=0;buttons.forEach((b,i)=>{const c=D.cells[i],ok=(!$('category').value||c.category===$('category').value)&&(!$('reason').value||c.reasons.includes($('reason').value))&&(!$('wl').value||c.wl===Number($('wl').value));b.classList.toggle('dim',!ok);n+=ok});$('matches').textContent=`Подходят фильтру: ${n}/256`}
['category','reason','wl'].forEach(k=>$(k).addEventListener('change',filter));
function select(i){const c=D.cells[i],m=c.metrics;buttons.forEach((b,j)=>b.classList.toggle('selected',j===i));let groups=['zero_adc','low_adc','high_adc','pre_g','post_g'],labels=['ADC zero','ADC 0,05 В','ADC 0,1 В','G до, Сименс','G после, Сименс'];let body=groups.map((k,j)=>`<tr><th>${labels[j]}</th><td>${f(m[k].mean)}</td><td>${f(m[k].sd)}</td><td>${m[k].n}</td></tr>`).join('');let all=Object.entries(m).filter(([k,v])=>typeof v!=='object'||v===null).map(([k,v])=>`<tr><th>${esc(k)}</th><td>${esc(f(v))}</td></tr>`).join('');$('detail').innerHTML=`<h2>WL ${c.wl} · BL ${c.bl} — ${esc(D.categories[c.category][1])}</h2><p><strong>${esc(c.status)}</strong> · ${esc(c.reasons.join('; ')||'Все критерии выполнены')}</p><div class="metrics"><div><table class="detail"><tr><th>Измерение</th><th>Среднее</th><th>SD</th><th>n</th></tr>${body}</table>${c.checks.map(x=>`<p class="${x.pass?'pass':'fail'}"><b>${x.pass?'✓':'×'} ${esc(x.name)}</b><br>${esc(x.expression)}</p>`).join('')}</div><div><table class="detail">${all}</table></div></div><p>Установлен результат теста, физическая причина не установлена. CV «не определено» не означает измеренный шум.</p>`}
buttons.forEach((b,i)=>{b.onclick=()=>select(i);b.onkeydown=e=>{let j=i;if(e.key==='ArrowRight')j=i+1;if(e.key==='ArrowLeft')j=i-1;if(e.key==='ArrowDown')j=i+32;if(e.key==='ArrowUp')j=i-32;if(j!==i&&j>=0&&j<256){e.preventDefault();buttons[j].focus();select(j)}}});filter();select(0);
</script></main></html>'''
    tokens={'DISCLAIMER':html.escape(DISCLAIMER),'SESSION':html.escape(session.name),'STATUS':html.escape(loaded['summary.json']['status']),
            'LEGEND':legend,'GRID':grid,'ROWS':rows,'DATA':data}
    for key,value in tokens.items(): template=template.replace('@@'+key+'@@',value)
    (output/'index.html').write_text(template,encoding='utf-8')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1680" height="760" viewBox="0 0 1680 760"><rect width="1680" height="760" fill="white"/>',
         '<g font-family="sans-serif" fill="#182436"><text x="30" y="35" font-size="24">Карта 256 ячеек: текущие проводимости, 0,05–0,1 В</text>',
         '<text x="30" y="62" font-size="14">WL/BL — логические адреса; цвет = приоритет причины. Δ = изменение G. Пороги предварительные.</text>']
    for b in range(32): svg.append(f'<text x="{87+b*49}" y="95" text-anchor="middle" font-size="13">{b}</text>')
    for w in range(8): svg.append(f'<text x="20" y="{138+w*48}" font-size="14">WL {w}</text>')
    for c in enriched:
        symbol,label,color=CATEGORIES[c['category']]; x=64+c['bl']*49;y=107+c['wl']*48
        svg.append(f'<rect x="{x}" y="{y}" width="46" height="44" rx="3" fill="{color}" stroke="#657385"/><text x="{x+23}" y="{y+21}" text-anchor="middle" font-size="15">{symbol}{" Δ" if c["drift_marker"] else ""}</text><text x="{x+23}" y="{y+36}" text-anchor="middle" font-size="10">{c["wl"]},{c["bl"]}</text>')
    for index,(key,count) in enumerate((k,counts[k]) for k in CATEGORIES if counts[k]):
        symbol,label,color=CATEGORIES[key]; y=530+index*25
        svg.append(f'<rect x="30" y="{y-16}" width="30" height="21" fill="{color}"/><text x="45" y="{y}" text-anchor="middle" font-size="13">{symbol}</text><text x="72" y="{y}" font-size="14">{html.escape(label)} — {count}</text>')
    svg.append('<text x="30" y="735" font-size="13">Не подтверждает программируемость, совместный MVM или физическую неисправность. Данные и все причины: index.html / cells_detailed.csv.</text></g></svg>')
    (output/'overview.svg').write_text('\n'.join(svg),encoding='utf-8')
    return dict(output=output.relative_to(root).as_posix(),categories=dict(counts),verified_files=len(verified))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.',help='Repository root relative to current directory')
    parser.add_argument('--session',required=True,help='Frozen session relative to repository')
    parser.add_argument('--output',help='New output directory relative to repository; default session/detailed')
    args=parser.parse_args(); root=Path(args.repo).resolve()
    print(json.dumps(generate(root,relative(root,args.session),relative(root,args.output or args.session+'/detailed')),ensure_ascii=False))


if __name__=='__main__': main()
