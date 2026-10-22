"""Build a compact map from saved block measurements; no board I/O."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiment_common.selective_metrology import object_dict


def objects(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError('Expected array')
    return [object_dict(item) for item in value]


def integers(value: object) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError('Expected integer addresses')
    return [int(item) for item in value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(args.summary).resolve()
    out = Path(args.output).resolve()
    source.relative_to(ROOT)
    out.relative_to(ROOT)
    data = object_dict(json.loads(source.read_text(encoding='utf-8')))
    if data['source'] != 'hardware' or data['status'] != 'completed' or data['error'] is not None:
        raise ValueError('A completed hardware session is required')
    if data['impacts'] != 74518 or data['complete_pairs'] != 2352 or data['pulses'] != 0:
        raise ValueError('Incomplete planned schedule')
    primary = objects(data['primary'])
    selected = objects(data['selected_blocks'])
    if len(primary) != 45 or not all(row['complete'] is True for row in primary):
        raise ValueError('Missing block profiles')
    chosen = {str(row['block']): row for row in selected}
    by_block: dict[str, list[dict[str, object]]] = {}
    cells: dict[tuple[int, int], str] = {}
    for row in primary:
        name = str(row['block'])
        by_block.setdefault(name, []).append(row)
    for name, profiles in by_block.items():
        first = profiles[0]
        for wl in integers(first['wl_pair']):
            for bl in integers(first['bls']):
                if (wl, bl) in cells:
                    raise ValueError('Overlapping blocks')
                cells[wl, bl] = name
        if name in chosen:
            matching = [r for r in profiles if r['profile'] == chosen[name]['profile']]
            if len(matching) != 1 or matching[0]['status'] != 'stable_current_state_readout':
                raise ValueError('Selected block did not pass')
    if len(cells) != 66 or len(by_block) != 15:
        raise ValueError('Incorrect candidate map')
    slots = sum(len(integers(row['bls'])) for row in selected)
    if 2 * slots != data['selected_cells']:
        raise ValueError('Selected count mismatch')
    export: dict[str, object] = dict(
        schema='memriboard-stable-block-map-v1', session_id=data['session_id'],
        source_summary=source.relative_to(ROOT).as_posix(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        qualification='current_conductance_state_and_tested_binary_profile_only',
        selected_blocks=selected, coefficients=slots, physical_cells=2 * slots,
        readout='each leg: fresh current guard; baseline mean4, signal mean4, baseline mean4; signed difference',
        lif_rule='sum all measured partial contributions before one logical LIF update',
        all_nonempty_binary_patterns_validated=True, validation_repeats=5,
        after_programming='repeat readout validation at the new conductance state',
        mnist_scheduler_implemented=False, mnist_run_performed=False,
        cpu_ideal_matmul_is_not_a_replacement=True,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out/'STABLE_BLOCKS.json').write_text(json.dumps(export, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    h = html.escape
    matrix = '<tr><th>WL / BL</th>'+''.join(f'<th>{bl}</th>' for bl in range(32))+'</tr>'
    for wl in range(8):
        matrix += f'<tr><th>{wl}</th>'
        for bl in range(32):
            name = cells.get((wl, bl), '')
            color = 'stable' if name in chosen else ('candidate' if name else 'unused')
            matrix += f'<td class="{color}" title="WL{wl} BL{bl}">{h(name) or "·"}</td>'
        matrix += '</tr>'
    rows = ''
    for name, profiles in sorted(by_block.items()):
        first = profiles[0]
        statuses = []
        for row in profiles:
            reasons = row['reasons']
            count = len(reasons) if isinstance(reasons, list) else 0
            passed = row['status'] == 'stable_current_state_readout'
            details = h(json.dumps(reasons, ensure_ascii=False, indent=2))
            statuses.append('<td>'+('PASS' if passed else f'Не прошёл ({count})')+
                            f'<details><summary>Причины</summary><pre>{details}</pre></details></td>')
        selected_name = str(chosen[name]['profile']) if name in chosen else 'Не выбран'
        rows += f'<tr><th>{h(name)}</th><td>{h(str(first["wl_pair"]))}</td><td>{h(str(first["bls"]))}</td>'+''.join(statuses)+f'<td>{h(selected_name)}</td></tr>'
    report = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Стабильные блоки MemriBoard</title>
<style>body{{font:16px/1.4 system-ui;max-width:1250px;margin:24px auto;padding:0 16px}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #bbb;padding:5px}}.scroll{{overflow:auto}}.stable{{background:#a8e6b1}}.candidate{{background:#ffd79a}}.unused{{background:#eee}}pre{{white-space:pre-wrap;max-width:320px}}details{{font-size:12px}}</style>
<h1>Проверены все 15 блоков через MemriBoard</h1>
<p><b>Выбрано {len(selected)} блоков: {slots} пар весов, {2*slots} физических ячеек.</b></p>
<p>74 518 вызовов, 0 SET/RESET. Все 137 непустых бинарных масок: 3 профиля, 5 проверочных проходов с чередованием блоков и плеч. Каждый уровень — среднее четырёх реальных чтений.</p>
<p>Зелёный — блок прошёл хотя бы один заранее заданный профиль. Оранжевый — блок не прошёл критерии целиком; это не диагноз поломки каждой его ячейки. Серый — вне этой проверки.</p>
<div class="scroll"><table>{matrix}</table></div><h2>Результаты по профилям</h2>
<div class="scroll"><table><tr><th>Блок</th><th>WL-пара</th><th>BL</th><th>0 → 0,05 В</th><th>0,025 → 0,05 В</th><th>0,025 → 0,075 В</th><th>Выбранный профиль</th></tr>{rows}</table></div>
<p>Пороги до измерения: дрейф ≤5 ADC, signed SD ≤5 ADC, отклик каждой ноги больше max(5 ADC, 5 SD), ошибка суммы ≤max(5 ADC, 10% масштаба). Компенсация двух плеч сама по себе допустима. Нелинейность между разными амплитудами — отдельная диагностика: входы SNN бинарные.</p>
<p>Это проверка чтения при нынешних проводимостях. После записи весов проверка должна повторяться. MNIST этим прогоном не запускался; аппаратный планировщик чередования ещё не реализован. Частичные измеренные суммы складываются до одного обновления LIF.</p>
<p><a href="STABLE_BLOCKS.json">Карта для подготовки планировщика</a>. Сессия: {h(str(data['session_id']))}.</p></html>'''
    (out/'RESULTS.html').write_text(report, encoding='utf-8')
    print(json.dumps(dict(selected_blocks=list(chosen), coefficients=slots, cells=2*slots)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
