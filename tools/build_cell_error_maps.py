"""Descriptive physical-address maps from hash-bound compact CSV; no backend imports."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path('reports/2026-10-16/hardware-stability-comparison')
OUTPUT = Path('docs/developer-reports/2026-10-17/cell-errors')
STOPS = {'feedback_hit': 'hit', 'feedback_overshoot': 'overshoot',
         'set_ramp_exhausted': 'exhausted', 'initial_overshoot_no_SET': 'initial_overshoot_no_SET'}
THRESHOLDS_V = [.00062, .005, .05, .1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def address(value):
    c = json.loads(value)
    if len(c) != 2 or any(type(n) is not int for n in c) or not (0 <= c[0] < 8 and 0 <= c[1] < 32):
        raise ValueError('Invalid physical WL/BL')
    return tuple(c)


def aggregate(cells, weights, excluded_bl8=False):
    if len(cells) != 256 or len(weights) != 300:
        raise ValueError('Expected 256 address rows and 300 coefficient records')
    indexed = {(int(c['wl']), int(c['bl'])): c for c in cells}
    if set(indexed) != {(w,b) for w in range(8) for b in range(32)}:
        raise ValueError('Missing or duplicate physical address')
    keys = [(int(w['filter']), int(w['tap'])) for w in weights]
    if len(set(keys)) != 300 or set(keys) != {(f,t) for f in range(4) for t in range(75)}:
        raise ValueError('Missing/duplicate filter tap; logical tap is not physical BL')
    result = {}
    for key, c in indexed.items():
        delta = c['max_abs_post_all_minus_own_v']
        value = float(delta) if delta else None
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError('Invalid leg delta')
        result[key] = dict(wl=key[0], bl=key[1], source_status=c['status'],
            excluded_by_protocol=excluded_bl8 and key[1] == 8,
            probes=int(c['probes']), selected_observations=int(c['selected_observations']),
            active_assignments=0, actual_SET_events=0, paired_role_events=0,
            hit=0, overshoot=0, exhausted=0, unknown_stop=0, initial_overshoot_no_SET=0,
            no_SET_events=0, max_abs_post_all_minus_own_v=value, references=[])
    for w in weights:
        p, n = address(w['positive_cell']), address(w['negative_cell'])
        weight = float(w['weight'])
        if not math.isfinite(weight): raise ValueError('Nonfinite weight')
        active = p if weight >= 0 else n  # fresh_conv1.py program_pair: planned sign, no reversal.
        pulses = int(w['programming_attempts'])
        if pulses < 0: raise ValueError('Negative pulse count')
        for c in (p, n):
            r = result[c]
            if r['excluded_by_protocol']: raise ValueError('Excluded BL8 was assigned')
            is_active = c == active
            r['references'].append(dict(filter=int(w['filter']), block=int(w['tile']), tap=int(w['tap']),
                role='planned_SET_leg' if is_active else 'paired_leg',
                stop=w['stop'], SET_pulses=pulses if is_active else 0))
            if is_active:
                r['active_assignments'] += 1
                r['actual_SET_events'] += int(pulses > 0)
                r['no_SET_events'] += int(pulses == 0)
                r[STOPS.get(w['stop'], 'unknown_stop')] += 1
            else:
                r['paired_role_events'] += 1
    for key, r in result.items():
        c = indexed[key]
        if len(r['references']) != r['selected_observations'] or r['actual_SET_events'] != int(c['SET_events']):
            raise ValueError('Compact CELLS/WEIGHTS role counts disagree')
        r['observed'] = bool(r['probes'] or r['references'])
        if r['excluded_by_protocol']: category = 'excluded_BL8'
        elif not r['observed']: category = 'not_observed'
        elif not r['actual_SET_events']: category = 'read_or_pair_only_no_SET'
        elif r['unknown_stop']: category = 'unknown_stop'
        elif r['exhausted']: category = 'exhausted_present'
        elif r['overshoot'] or r['initial_overshoot_no_SET']: category = 'overshoot_present'
        else: category = 'hit_only'
        r['stop_map_category'] = category
    return [result[(w,b)] for w in range(8) for b in range(32)]


def load(root):
    folder = root / INPUT
    provenance = json.loads((folder/'PROVENANCE.json').read_text(encoding='utf-8'))
    out = []
    for number in (1, 2):
        parsed = []
        for kind in ('CELLS', 'WEIGHTS'):
            name = f'SESSION{number}_{kind}.csv'; path = folder/name
            if sha(path) != provenance['compact_files'][name]['sha256']:
                raise ValueError('Compact CSV hash mismatch: '+name)
            with path.open(encoding='utf-8', newline='') as f: parsed.append(list(csv.DictReader(f)))
        out.append(aggregate(*parsed, excluded_bl8=number == 2))
    expected = provenance['sessions'][0]['acquisition_source_sha256']['fresh_conv1.py']
    if sha(root/'fresh_conv1.py') != expected:
        raise ValueError('Programming-side source changed; review sign convention first')
    return out, provenance


def render(rows, path, mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none', 'svg.hashsalt':'cell-errors-v1'})
    fig, ax = plt.subplots(figsize=(20, 7.5))
    gray = '#e5e7eb'; blue = '#bbd6ef'; teal = '#87cbbb'; orange = '#ffce80'; red = '#ed9696'
    if mode == 'shift':
        palette = [teal, '#cbe7ad', '#ffe197', '#f6b784', '#e98686']
        legends = [('не наблюдалась',gray), ('наблюдалась, нет leg-delta',blue)] + list(zip(
            ['≤0,62 мВ','(0,62;5] мВ','(5;50] мВ','(50;100] мВ','>100 мВ'],palette))
        title = 'До исключения BL8: изменение отклика после записи других пар'
        subtitle = 'Все 256 физических адресов; числа в клетках — мВ. Это изменение наблюдения, не диагноз и не ошибка веса.'
    else:
        colors = {'not_observed':gray,'excluded_BL8':'#ddd2ec','read_or_pair_only_no_SET':blue,
                  'hit_only':teal,'overshoot_present':orange,'exhausted_present':red,'unknown_stop':'#b8a2d5'}
        legends = list(zip(['нет наблюдений','BL8 исключён','без SET / парная роль','только H (попадание)','есть O (перескок)','есть E (лимит)','неизвестная остановка'],colors.values()))
        title = 'Без BL8: результаты подбора веса по ячейкам'
        subtitle = 'H — попадание в интервал при остановке; O — перескок; E — исчерпан лимит SET.\nВсе события сохранены; цвет показывает приоритет E > O > H, а не исправность.'
    for r in rows:
        w,b=r['wl'],r['bl']; value=r['max_abs_post_all_minus_own_v']
        if mode == 'shift':
            color = gray if not r['observed'] else blue if value is None else palette[sum(value > t for t in THRESHOLDS_V)]
            label = '—' if not r['observed'] else 'R' if value is None else f'{value*1000:.2f}'
        else:
            color=colors[r['stop_map_category']]
            label='\n'.join(f'{s}{r[k]}' for s,k in [('H','hit'),('O','overshoot'),('E','exhausted'),('?','unknown_stop')] if r[k])
            if not r['actual_SET_events']:
                label=('X8' if r['excluded_by_protocol'] else '—' if not r['observed'] else 'R') + ('\n'+label if label else '')
        ax.add_patch(Rectangle((b-.5,w-.5),1,1,facecolor=color,edgecolor='white',linewidth=1))
        ax.text(b,w,label,ha='center',va='center',fontsize=8.5,color='#17212e')
    if mode != 'shift':
        ax.add_patch(Rectangle((14.5,2.5),1,1,fill=False,edgecolor='#701b1b',linewidth=2.5))
    ax.set(xlim=(-.5,31.5),ylim=(7.5,-.5),xticks=range(32),yticks=range(8),xlabel='BL (физический столбец)',ylabel='WL (физическая строка)')
    ax.xaxis.tick_top(); ax.xaxis.set_label_position('top'); ax.tick_params(length=0,labelsize=10)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle(title,fontsize=17,fontweight='bold',y=.98)
    fig.text(.5,.922,subtitle,ha='center',va='top',fontsize=11)
    ax.legend(handles=[Patch(facecolor=c,label=l) for l,c in legends],loc='upper center',bbox_to_anchor=(.5,-.04),ncol=4,frameon=False,fontsize=11)
    footer = ('Максимум |post-all − post-own|: own с bracket-zero, post-all с общим zero. Операторы различаются. Пороги — только отображение.' if mode=='shift' else
              'WL3/BL15: E8 + H1. H — решение в момент остановки; последующие чтения могут не подтвердить допуск. R не означает отсутствие RESET.')
    fig.text(.5,.025,footer,ha='center',fontsize=11)
    fig.subplots_adjust(left=.055,right=.99,top=.82,bottom=.2)
    fig.savefig(path.with_suffix('.svg'),metadata={'Date':None})
    fig.savefig(path.with_suffix('.png'),dpi=160)
    plt.close(fig)


def build(root, output):
    rows, source = load(root)
    if output.exists(): raise ValueError('Output already exists; never overwrite an existing report')
    output.mkdir(parents=True)
    summaries=[]
    for number, data in enumerate(rows, 1):
        name='BASELINE_SHIFT' if number==1 else 'NO_BL8_STOPS'
        with (output/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader()
            for r in data: writer.writerow({**r,'references':json.dumps(r['references'],separators=(',',':'))})
        render(data,output/name,'shift' if number==1 else 'stops')
        summaries.append(dict(session=source['sessions'][number-1]['session_id'],cells=256,
            assigned_cells=sum(bool(r['references']) for r in data),actual_SET_cells=sum(bool(r['actual_SET_events']) for r in data),
            counts={k:sum(r[k] for r in data) for k in ['active_assignments','actual_SET_events','no_SET_events','hit','overshoot','exhausted','unknown_stop']},
            categories=dict(collections.Counter(r['stop_map_category'] for r in data))))
    with (output/'CELL_ERRORS.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['session_number',*rows[0][0]])
        writer.writeheader()
        for number, data in enumerate(rows,1):
            for r in data:writer.writerow({'session_number':number,**r,'references':json.dumps(r['references'],separators=(',',':'))})
    proof=dict(schema='descriptive-cell-errors-v1',generator_sha256=sha(Path(__file__)),
        inputs={str(INPUT/p.name):sha(p) for p in (root/INPUT).glob('SESSION*_*.csv')},
        source_provenance_sha256=sha(root/INPUT/'PROVENANCE.json'),programming_source_sha256=sha(root/'fresh_conv1.py'),
        thresholds_v=THRESHOLDS_V,sessions=summaries,
        wl3_bl15_no_bl8=rows[1][3*32+15], raw_reaudit_performed=False,
        outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'MAP_README.md').write_text('''# Две карты наблюдений по физическим адресам

[Baseline: сдвиг отклика](BASELINE_SHIFT.png) · [SVG](BASELINE_SHIFT.svg) · [256 строк CSV](BASELINE_SHIFT.csv)

[noBL8: остановки SET](NO_BL8_STOPS.png) · [SVG](NO_BL8_STOPS.svg) · [256 строк CSV](NO_BL8_STOPS.csv)

[Общий CSV: 2 сессии × 256 ячеек](CELL_ERRORS.csv), поле session_number=1 baseline, 2 noBL8.

Первая карта: максимум абсолютного post-all минус post-own отклика **одной ноги** по её назначениям, в вольтах (подписи — мВ). Границы цветов: 0,00062; 0,005; 0,05; 0,1 V, верхняя граница включена. Это границы отображения, выбранные после измерений, не тест значимости и не критерий исправности. Own использует bracket zero, post-all — общий zero: разность не изолирует физическую причину изменения.

Вторая карта: связанный с активной ногой исход feedback-loop. Положительный trained weight выбирает positive_cell, отрицательный — negative_cell; frozen fresh_conv1.py program_pair не меняет эту ногу по остаточной ошибке. H=feedback_hit, O=feedback_overshoot, E=set_ramp_exhausted. H означает решение при остановке, не последующее соответствие веса и не исправность. При нескольких исходах цвет имеет приоритет E>O>H, **все счётчики остаются на карте и в CSV**. WL3/BL15: E8/H1. Число SET-событий учитывает только назначения с programming_attempts>0; отдельно сохранены no_SET и парные роли.

Серый — не наблюдалась; синий R — наблюдалась без SET (парная роль/чтения; RESET возможен); фиолетовый X8 — исключена маршрутом noBL8, не deadcell и не электрическая изоляция. Фиолетовый неизвестный stop предусмотрен отдельно. Отсутствие события не заменяется нулевой ошибкой.

В CSV для каждой из 256 ячеек: наблюдаемость, число назначений/SET/no_SET, H/O/E/unknown, максимум leg-delta и полный список filter/block/tap/role/stop. **Ошибка дифференциальной пары не присваивается обеим ячейкам как физический дефект.** 300 — коэффициенты четырёх фильтров, переиспользующие адреса; блок (тайл) — порция до 7 пар / 14 выбранных ячеек, а не одна ячейка. RESET-кандидатов может быть больше.

Источник — неизменённые компактные SESSION1/2_CELLS/WEIGHTS.csv. Их SHA проверены по исходному PROVENANCE.json, SHA programming source — по manifest-привязке компактного пакета. Новый raw-аудит не выполнялся. Полные исходные поля, sessions и SHA выходов — [PROVENANCE.json](PROVENANCE.json). Карта ничего не подгоняет и не квалифицирует.

Воспроизведение из корня: `python tools/build_cell_error_maps.py --output <новая-папка>`. Требуется matplotlib; CLI не импортирует аппаратный backend. Уже существующий выход не перезаписывается.
''',encoding='utf-8')
    proof['outputs']['MAP_README.md']=sha(output/'MAP_README.md')
    (output/'PROVENANCE.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(OUTPUT))
    args=parser.parse_args();path=Path(args.output)
    if path.is_absolute() or not (ROOT/path).resolve().is_relative_to(ROOT): parser.error('Output must be repository-relative')
    print(json.dumps(build(ROOT,ROOT/path),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
