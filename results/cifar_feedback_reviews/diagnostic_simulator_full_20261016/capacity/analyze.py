"""Offline historical address-capacity calculation; no runtime/driver imports."""
import argparse
import hashlib
import json
from pathlib import Path

INPUTS={
 'weights/candidate_cifar75_routing.json':'bbe71222138fd28ef928dcca22b319abc1223ec87b6680edd56b03c63dac0245',
 'results/mapped_sequential_mvm/07c6d866-de36-4d02-a9ef-93347d744409/analysis.json':'f931e833c71f41a17c48b51a83411e83824515fc571d06e451596db4d7237c61'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    parser.add_argument('--output',default='.tmp/coverage-capacity/REPORT.md')
    args=parser.parse_args();repo=Path(args.repo).resolve();objects=[]
    for name,expected in INPUTS.items():
        payload=(repo/name).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Input hash changed: '+name)
        objects.append(json.loads(payload))
    routing,analysis=objects
    addresses={(r['wl'],r['bl']) for r in analysis['cells']}
    if addresses!={(w,b) for w in range(8) for b in range(32)} or len(analysis['cells'])!=256:
        raise ValueError('Expected exactly one historical record per address')
    good={(r['wl'],r['bl']) for r in analysis['cells'] if r['singleton_supported']}
    original=set();candidate=set();pair_sets=[]
    for filt in routing['filters']:
        pairs=set()
        for tile in filt['tiles']:
            active=[s for s in tile['slots'] if not s['padding']]
            pairs.update((tuple(s['positive_cell']),tuple(s['negative_cell'])) for s in active)
            for side in ('positive_cell','negative_cell'):
                cells=[tuple(s[side]) for s in active];original.update(cells);candidate.update(cells)
                wl=cells[0][0];used={b for _,b in cells}
                if any(w!=wl for w,_ in cells):raise ValueError('Nonuniform routing leg')
                spare=[b for b in [*range(16),31] if b not in used][:3]
                candidate.update((wl,b) for b in spare)
        pair_sets.append(pairs)
    if [f['wl_pair'] for f in routing['filters']]!=[[0,1],[2,3],[4,5],[6,7]]:
        raise ValueError('Unexpected WL pairs')
    def counts(cells):return [sum(w==i for w,b in cells) for i in range(8)]
    def capacity(cells):
        ns=counts(cells);return [min(ns[i],ns[i+1]) for i in range(0,8,2)]
    subset=good&candidate;outside=good-candidate
    capacities=[len(p) for p in pair_sets]
    lines=['# Историческая геометрическая ёмкость фиксированных пар WL','',
        'Offline-пересчёт H15 и текущего маршрута. Плата/симулятор не запускались; сценарии и исходные данные не изменены. '
        'Исторический singleton_supported — результат состояния 15 октября, не текущая квалификация после записи.','',
        'Если разрешены разные BL двух ног и каждая ячейка используется ровно в одном коэффициенте одновременно, '
        'геометрический максимум для пары WL равен min(Nplus,Nminus). Он достижим как адресное паросочетание, '
        'но не доказывает измерительную пригодность полученных разностей.','',
        '| WL | H15 supported из32 | Candidate pool | Supported ∩ candidate | Supported вне candidate |',
        '|---:|---:|---:|---:|---:|']
    for w in range(8):lines.append(f'| {w} | {counts(good)[w]} | {counts(candidate)[w]} | {counts(subset)[w]} | {counts(outside)[w]} |')
    lines+=['', '| Фиксированная пара WL | Текущих позиций маршрута | Max по всем101 supported | Max по96 supported∩118 candidates | Геометрия всех118 без quality-фильтра |',
        '|---|---:|---:|---:|---:|']
    for i,(a,b) in enumerate(((0,1),(2,3),(4,5),(6,7))):
        lines.append(f'| {a}/{b} | {capacities[i]} | {capacity(good)[i]} | {capacity(subset)[i]} | {capacity(candidate)[i]} |')
    lines.append(f'| **Итого** | **{sum(capacities)}** | **{sum(capacity(good))}** | **{sum(capacity(subset))}** | **{sum(capacity(candidate))}** |')
    lines+=['','## Что добавляют пять адресов','',
        'Supported вне candidate: '+', '.join(f'WL{w}/BL{b}' for w,b in sorted(outside))+'.','',
        '**Они не увеличивают исторический геометрический потолок выше 47 коэффициентов текущего маршрута.** '
        'Однако они увеличивают максимум, построенный только из исторически поддержанных адресов внутри пула 118, '
        '**с 46 до 47**. Причина — пара WL2/WL3: внутри пула9/8 supported, во всём H15 —9/11. '
        'Достаточно одного из WL3/BL12, WL3/BL13, WL3/BL31, чтобы снять нехватку одной отрицательной ноги; '
        'два других остаются запасом. Выбор конкретной пары по singleton-карте не заменяет её измерение.','',
        'WL5/BL15 добавляет запас на уже более многочисленной стороне пары WL4/WL5 (11/11 внутри пула →11/12). '
        'WL7/BL15 также добавляет запас (13/15 →13/16). Они не увеличивают одновременную ёмкость при фиксированных парах WL.','',
        f'Исходный маршрут содержит {len(original)} уникальных ячейки; из них {len(original&good)} исторически supported. '
        'Единственный original-адрес, не прошедший H15 singleton-критерий, — '+', '.join(f'WL{w}/BL{b}' for w,b in sorted(original-good))+'. '
        'Поэтому наличие 47 слотов в файле не означает 47 исторически подтверждённых разностей. '
        f'Обе ноги singleton-supported у {sum(a in good and b in good for pairs in pair_sets for a,b in pairs)}/47 фиксированных пар; '
        f'фактический более строгий H15 contrast-критерий прошли лишь {sum(bool(r["final_supported"]) for r in analysis["coefficients"])}/47.','',
        '## Граница результата','',
        '59 — только комбинаторная ёмкость118 candidates без отбора по исторической пригодности. '
        '101 supported дают 47 пар/94 участвующих ячейки и 7 непарных остатков из-за дисбаланса WL. '
        '96 supported внутри candidate дают 46 пар/92 ячейки и 4 остатка. '
        'Ни одно из этих чисел не является новым числом программируемых коэффициентов, '
        'одновременно доступных обученных фильтров либо абсолютным максимумом массива 256.','',
        'Для максимума охвата полезно документировать пропущенную supported-ногу WL3 как конечную возможность замены/ротации. '
        'Увеличить потолок 47 по этой исторической карте при тех же WL-парах нельзя простым добавлением пяти адресов: '
        'потребуются новые данные о других состояниях/адресах либо иной reviewed layout. Новая квалификация здесь не выполнялась.','',
        '## Воспроизведение','',
        '```text','python results/cifar_feedback_reviews/diagnostic_simulator_full_20261016/capacity/analyze.py --repo . --output .tmp/coverage-capacity/REPORT.md','```','',
        'Скрипт использует только stdlib, проверяет SHA обоих входов и вычисляет candidate pools из порядка слотов плюс 3 первых разрешённых запасных BL.']
    for name,digest in INPUTS.items():lines+=['',f'- `{name}`: `{digest}`.']
    output=(repo/args.output).resolve()
    if Path(args.output).is_absolute() or not output.is_relative_to(repo/'.tmp/coverage-capacity'):
        raise ValueError('Output must remain within .tmp/coverage-capacity')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(supported=counts(good),candidate=counts(candidate),intersection=counts(subset),
        capacity_all_supported=capacity(good),capacity_intersection=capacity(subset),capacity_all_candidates=capacity(candidate),
        omitted_supported=sorted(outside),report=output.relative_to(repo).as_posix())))


if __name__=='__main__':main()
