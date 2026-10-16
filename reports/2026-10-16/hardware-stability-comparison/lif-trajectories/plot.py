"""Offline SVG scientific plot; validates published traces, no backend imports.

Run from repository root: python -B reports/2026-10-16/hardware-stability-comparison/lif-trajectories/plot.py --repo .
Then: node reports/2026-10-16/hardware-stability-comparison/lif-trajectories/render.cjs
Node requires the standard sharp package. No network installation is performed.
"""
import argparse,hashlib,importlib.util,json,math,statistics
from html import escape
from pathlib import Path

COMPARATOR_SHA='1538619642c41510f0f519b8df91c4fcfcf4a1c301955700c1bd14e89b68c2f6'
BASE='reports/2026-10-16/hardware-stability-comparison'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def require(v,s):
    if not v:raise ValueError(s)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',default='.');args=parser.parse_args()
    root=Path(args.repo).resolve();out=root/BASE/'lif-trajectories'
    helper=root/'compare_fresh_traces.py';require(sha(helper)==COMPARATOR_SHA,'Comparator pin changed')
    spec=importlib.util.spec_from_file_location('verified_trace_reader',helper);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    hw=mod.read_trace(root,BASE+'/actual-trace-v2')
    sim=mod.read_trace(root,'reports/2026-10-16/no-bl8-simulator/actual-trace-v2')
    mod.compare(hw,sim) # validates identical fixture, trained weights, canonical route and keys
    fixture_path=root/'weights/prepared_cifar100_conv1.json';fixture=read(fixture_path)
    require(sha(fixture_path)==hw['metadata']['fixture_sha256']==sim['metadata']['fixture_sha256'],'Fixture hash mismatch')
    diagnostic_path=root/BASE/'DIAGNOSTICS.json';diagnostic=read(diagnostic_path)
    require(diagnostic['provenance']['manifest_sha256']==hw['metadata']['manifest_sha256'],'Diagnostic session mismatch')
    selectors=sorted({(r['filter'],r['case'],r['t']) for r in diagnostic['mismatch_rows']})
    require(len(selectors)==5,'Expected five diagnostic logical points')
    def idx(rows):return {(r['filter'],r['case'],r['repeat'],r['t']):r for r in rows}
    h,s=idx(hw['rows']),idx(sim['rows']);panels=[]
    for f,c,focus in selectors:
        means=[];lower=[];upper=[];simvalues=[];trained=[]
        for t in range(12):
            hv=[h[f,c,r,t]['v_before'] for r in range(5)];sv=[s[f,c,r,t]['v_before'] for r in range(5)]
            require(max(sv)==min(sv),'SIM repeats differ: do not silently hide their variation')
            means.append(statistics.mean(hv));lower.append(min(hv));upper.append(max(hv));simvalues.append(sv[0])
            trained.append(fixture['cases'][c]['reference_t12x64']['lif_before_reset'][t][f])
        require(all(math.isfinite(v) for seq in [means,lower,upper,simvalues,trained] for v in seq),'Nonfinite plot value')
        panels.append(dict(filter=f,case=c,original_test_index=fixture['cases'][c]['original_test_index'],focus_t=focus,
            hw_mean=means,hw_min=lower,hw_max=upper,sim=simvalues,trained=trained))
    W,H=1440,1740;svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.small{font-size:15px}</style>']
    def text(x,y,t,size=17,anchor='start',color=None):
        style=f' style="fill:{color}"' if color else ''
        svg.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}"{style}>{escape(str(t))}</text>')
    def line(x1,y1,x2,y2,color,width=1,dash=None):
        svg.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
    text(45,36,'LIF: пять логических точек с расхождениями',28)
    text(45,65,'Все 12 шагов • четыре Conv1-фильтра • десять patches • потенциал непосредственно перед сбросом',17)
    legends=[(45,'#1276b8','HW: среднее и диапазон 5 повторов'),(535,'#dc5b27','native SIM: все повторы совпали'),(1020,'#703cb1','trained: CPU-эталон')]
    for x,color,label in legends:line(x,92,x+30,92,color,3);text(x+40,98,label,16)
    line(45,120,75,120,'#333333',2,'7 5');text(85,126,'Порог = 1; spike при V ≥ 1',16)
    text(535,126,'Светло-оранжевая полоса: выбранный диагностический шаг',16)
    left,right=105,1395
    for n,p in enumerate(panels):
        top=169+n*285;bottom=top+190;title=top-15;focus=p['focus_t']
        text(45,title,f"F{p['filter']} · case {p['case']} (test index {p['original_test_index']}) · проблемный шаг t={focus}",19)
        values=p['hw_min']+p['hw_max']+p['sim']+p['trained']+[0,1]
        lo,hi=min(values),max(values);pad=max(.12,(hi-lo)*.10);lo-=pad;hi+=pad
        xx=lambda t:left+(right-left)*t/11
        yy=lambda v:bottom-(v-lo)/(hi-lo)*(bottom-top)
        svg.append(f'<rect x="{xx(focus)-16:.2f}" y="{top}" width="32" height="190" fill="#fff0db"/>')
        for j in range(5):
            v=lo+(hi-lo)*j/4;y=yy(v);line(left,y,right,y,'#e5e9ee');text(left-12,y+5,f'{v:.2f}',14,'end')
        line(left,yy(1),right,yy(1),'#333333',1.7,'7 5')
        line(left,bottom,right,bottom,'#66717c');line(left,top,left,bottom,'#66717c')
        polygon=[(xx(t),yy(v)) for t,v in enumerate(p['hw_min'])]+[(xx(t),yy(p['hw_max'][t])) for t in reversed(range(12))]
        svg.append('<polygon points="'+' '.join(f'{x:.3f},{y:.3f}' for x,y in polygon)+'" fill="#1276b8" fill-opacity="0.20"/>')
        for series,color,dash in [('sim','#dc5b27','7 4'),('trained','#703cb1','3 4'),('hw_mean','#1276b8',None)]:
            coords=' '.join(f'{xx(t):.3f},{yy(v):.3f}' for t,v in enumerate(p[series]))
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.6"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
            for t,v in enumerate(p[series]):svg.append(f'<circle cx="{xx(t):.3f}" cy="{yy(v):.3f}" r="3.1" fill="{color}"/>')
        for t in range(12):text(xx(t),bottom+21,t,14,'middle')
        text(right,bottom+42,'логический шаг t',14,'end');text(45,top+95,'V',17)
        text(left,bottom+43,f"t={focus}: HW [{p['hw_min'][focus]:.6f}, {p['hw_max'][focus]:.6f}] · SIM {p['sim'][focus]:.6f} · trained {p['trained'][focus]:.6f}",15)
    text(45,1624,'Оси Y индивидуальны. Полоса HW — min–max наблюдений, не доверительный интервал.',16)
    text(45,1651,'Отрицательные V сохранены. После spike V_after = 0: соединения точек не изображают непрерывную физическую динамику.',16)
    text(45,1678,'HW и SIM — независимые состояния с разными gain и C. График не является оценкой точности полного классификатора.',16)
    text(45,1705,'V — потенциал цифровой модели LIF в её условных единицах, не напряжение платы в вольтах.',17)
    svg.append('</svg>');svgpath=out/'LIF_TRAJECTORIES.svg';svgpath.write_text('\n'.join(svg)+'\n',encoding='utf8')
    caption='''# Как читать траектории LIF

[PNG](LIF_TRAJECTORIES.png) · [SVG](LIF_TRAJECTORIES.svg)

Показаны все12 логических шагов пяти точек, обнаруженных в фактической аппаратной диагностике. Синий — среднее пяти HW-повторов, полоса — их min–max; оранжевый — native SIM, фиолетовый — обученный CPU-эталон. Все пять SIM-повторов в этих точках проверены на совпадение. ОсиY имеют индивидуальные пределы. Числа под панелями относятся к выделенному шагу.

В каждом шаге сначала суммируются75 измеренных вкладов, затем применяется обученный BN и цифровой LIF: V_before=0,5·V_previous_after+I_BN. Порог1, spike при V_before≥1; после spike V_after=0. Отрицательные потенциалы сохраняются. Соединительные линии помогают читать последовательность; они не являются измеренной непрерывной динамикой. Полоса повторов не доверительный интервал и не повторная программировка массива.

**V — потенциал цифровой модели LIF в её условных единицах, не напряжение платы в вольтах.** Порог1 относится к этой цифровой модели; он не задаёт1 V на мемристоре.

Панели F0/case8/t1 и F2/case8/t6 показывают пересечение порога разными HW-повторами. В F1/case1/t7 и F1/case3/t5 расхождение с trained систематическое, а не только колебание около порога. По одним этим траекториям физическую причину ошибки конкретной ячейки определить нельзя.

W_eff здесь не изображён: его полная траектория в данном компактном пакете отсутствует. Отдельный DIAGNOSTIC_REVIEW.md объясняет различие trained и W_eff=C×H_post-all/fixed_gain. HW/SIM используют разные заранее заданные gain0,25/0,03 и могут иметь разные C и записанные веса; дополнительная нормировка на графике не выполнена. Это четыре Conv1-фильтра на десяти patches, не полный классификатор.

Входы проверены закреплённым read_trace: SHA/размеры CSV, exporter/auditor-контракт,2400 ключей, единицы, порог и сброс. Fixture SHA совпал с обоими экспортами. Выбор панелей связан с DIAGNOSTICS. Raw на плате здесь повторно не исполнялся. Источники и выходы связаны в PLOT_PROVENANCE.json.

Воспроизведение из корня репозитория: `python -B reports/2026-10-16/hardware-stability-comparison/lif-trajectories/plot.py --repo .`, затем `node reports/2026-10-16/hardware-stability-comparison/lif-trajectories/render.cjs`. Для PNG требуется установленный пакет sharp; SVG создаётся стандартной библиотекой Python, без backend/сети.
'''
    captionpath=out/'CAPTION.md';captionpath.write_text(caption,encoding='utf8')
    inputs={str(p.relative_to(root)):sha(p) for p in [helper,fixture_path,diagnostic_path]}
    for t in (hw,sim):
        for name in ('TRACE_PROVENANCE.json','ACTUAL_TRACE.csv'):
            p=root/t['path']/name;inputs[str(p.relative_to(root))]=sha(p)
    provenance=dict(schema='lif-trajectory-figure-v1',inputs=inputs,script_sha256=sha(Path(__file__)),
        selectors=selectors,panels=panels,raw_replay_performed=False,hardware_io=False,
        additional_fitting=False,sim_repeats_exactly_equal=True,outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [svgpath,captionpath]})
    (out/'PLOT_PROVENANCE.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(panels=len(panels),svg=str(svgpath))))
if __name__=='__main__':main()
