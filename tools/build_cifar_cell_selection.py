"""Describe the actually used compact-CIFAR layout, without claiming quality qualification."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from annotate_cell_problem_map import annotate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/2026-10-26/cell-problems'
SESSION = 'results/compact_cifar_physical_20261020/69c5bfda-17bf-41ad-8805-704dc9b22ac4'


def prepare() -> None:
    folder = ROOT / SESSION
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    measured = json.loads((folder / 'measured_tiles.json').read_text(encoding='utf-8'))
    map_path = ROOT / protocol['map_session'] / 'map.json'
    map_bytes = map_path.read_bytes()
    if hashlib.sha256(map_bytes).hexdigest() != protocol['map_sha256']:
        raise ValueError('Parent selection map checksum mismatch')
    source_map = json.loads(map_bytes)
    if manifest['source'] != 'hardware' or manifest['status'] != 'completed':
        raise ValueError('Expected a completed hardware acquisition')
    if measured['completed_stages'] != 48 or measured['status'] != 'completed':
        raise ValueError('Incomplete acquisition')
    actual = {s['stage']: s for s in measured['stages']}
    if set(actual) != set(range(48)):
        raise ValueError('Missing/duplicate stage coverage')
    used: set[tuple[int, int]] = set()
    for stage in protocol['stages']:
        completed = actual[stage['stage']]
        if completed['inputs'] != stage['inputs'] or completed['neurons'] != stage['neurons']:
            raise ValueError('Measured stages disagree with frozen layout')
        used.update((wl, bl) for pair in stage['pairs'] for wl in pair for bl in stage['bls'])
    if len(used) != 16 or source_map['selection']['quality_qualified']:
        raise ValueError('Expected the recorded 16-cell diagnostic fallback')
    rows: list[dict[str, str]] = []
    for cell in sorted(source_map['cells'], key=lambda c: (c['wl'], c['bl'])):
        wl, bl = cell['wl'], cell['bl']
        block, partner = '', ''
        if (wl, bl) in used:
            category = 'used'
            block, partner = 'WL0/WL7', f'WL{7 if wl == 0 else 0}/BL{bl}'
            reason = ('Использовали в завершённом аппаратном сборе CIFAR compact (48 этапов). '
                      'Вошла в диагностическую раскладку WL0/WL7 с 8 общими BL: '
                      'максимум коэффициентов, затем минимальная худшая нормированная ошибка. '
                      'Полная проверка качества карты не пройдена; зелёный не означает точную запись веса.')
        elif not cell['hard_eligible']:
            category = 'range'
            reason = ('Не прошла базовый критерий воспроизводимого диапазона. '
                      f'Диапазон: {cell["span_adc"]:.6g} ADC. '
                      f'Зафиксированный критерий: {", ".join(cell["hard_reasons"])}. '
                      'Это результат данной калибровки, не доказательство постоянной неисправности.')
        else:
            category = 'layout'
            reason = ('Базовый критерий диапазона пройден, но ячейка не вошла в выбранную общую '
                      'раскладку WL0/WL7 × BL2,3,6,10,11,12,14,15. '
                      'Выбор ограничен одной парой WL и 8 общими BL; это не индивидуальный отказ. '
                      f'Худшая нормированная ошибка: {cell["worst_normalized_error"]:.6g}.')
        rows.append(dict(wl=str(wl), bl=str(bl), category=category, block=block,
                         partner=partner, reason=reason, writes='', misses='', max_abs_error_us=''))
    with (OUT / 'CIFAR_SELECTION.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths = [folder / n for n in ['protocol.json', 'manifest.json', 'measured_tiles.json']] + [map_path]
    evidence = dict(schema='cifar-compact-cell-selection-v1', acquisition=SESSION,
                    physical_cells=16, completed_stages=48, map_quality_qualified=False,
                    classification_performed_by_acquisition=measured['classification_performed'],
                    source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    selection=source_map['selection']['selected'], cells=rows)
    (OUT / 'CIFAR_SELECTION.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    annotate('mnist')
    (OUT / 'CIFAR.html').write_bytes((OUT / 'MAP.html').read_bytes())
    annotate('cifar')


if __name__ == '__main__':
    prepare()
