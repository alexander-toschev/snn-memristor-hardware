"""Локальный typed отчёт выборочного чтения. Connector не создаётся.

Повторно вычисляет статистику из сохранённых ADC brackets, проверяет hashes
наблюдений/выборки. Это не полный replay raw request/result и не калибровка G.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import html
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiment_common.durable import atomic_json, sha
from experiment_common.runtime import repo_path
from experiment_common.selective_metrology import Cell, load_sample, number, object_dict

ROLES = ('initial', 'delay_0', 'delay_10', 'delay_60')


@dataclass(frozen=True)
class CellReport:
    wl: int
    bl: int
    selected: bool
    sample_reason: str
    historical_hard_eligible: bool
    historical_quality_qualified: bool
    historical_span_adc: float
    checkpoint_count: int
    passed: bool
    reasons: list[str]
    initial_mean_adc: float | None
    final_mean_adc: float | None
    maximum_sd_adc: float | None
    noise_limit_adc: float
    maximum_drift_adc: float | None
    drift_limit_adc: float
    maximum_zero_shift_ratio: float | None
    initial_g_s: float | None
    final_g_s: float | None
    elapsed_initial_to_final_s: float | None


@dataclass(frozen=True)
class Checkpoint:
    mean: float
    sd: float
    zero_mean: float
    zero_limit: float
    mean_g: float
    time: float


def read_json(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding='utf-8'))
    return object_dict(value)


def address(row: dict[str, object]) -> tuple[int, int]:
    wl, bl = row['wl'], row['bl']
    if type(wl) is not int or type(bl) is not int or not isinstance(wl, int) or not isinstance(bl, int):
        raise ValueError('Integer cell address required')
    if not 0 <= wl < 8 or not 0 <= bl < 32:
        raise ValueError('Cell outside32x8')
    return wl, bl


def recompute(record: dict[str, object], cell: Cell, initial: float | None) -> Checkpoint:
    brackets = record['brackets']
    if not isinstance(brackets, list) or len(brackets) != 10:
        raise ValueError('Checkpoint requires10brackets')
    values: list[float] = []
    zeros: list[float] = []
    gs: list[float] = []
    for value in brackets:
        row = object_dict(value)
        readings = {name: object_dict(row[name]) for name in ('before', 'signal', 'after', 'guard')}
        codes: dict[str, int] = {}
        for name, reading in readings.items():
            adc = reading['adc']
            if type(adc) is not int or not isinstance(adc, int) or not 0 <= adc <= 16382:
                raise ValueError('Invalid ADC')
            codes[name] = adc
        resistance = number(readings['guard']['resistance_ohm'])
        if resistance <= 1e-8 or codes['guard'] < 20:
            raise ValueError('Invalid native guard resistance/ADC')
        centered = codes['signal'] - (codes['before'] + codes['after']) / 2
        if centered != number(row['centered_adc']):
            raise ValueError('Saved centeredADC disagrees with readings')
        values.append(centered)
        zeros.append(float(codes['after'] - codes['before']))
        gs.append(1 / resistance)
    point = Checkpoint(statistics.mean(values), statistics.stdev(values), statistics.mean(zeros),
                       max(2., 3 * statistics.stdev(zeros) / math.sqrt(10)), statistics.mean(gs),
                       number(record['monotonic_s']))
    noise_limit, drift_limit = max(2., .05 * cell.span_adc), max(3., .05 * cell.span_adc)
    drift = 0. if initial is None else abs(point.mean - initial)
    expected: dict[str, float | bool] = dict(
        mean_adc=point.mean, sd_adc=point.sd, mean_g_s=point.mean_g,
        zero_difference_adc=point.zero_mean, zero_shift_limit_adc=point.zero_limit,
        noise_limit_adc=noise_limit, drift_limit_adc=drift_limit, drift_from_initial_adc=drift,
        noise_pass=point.sd <= noise_limit, zero_shift_pass=abs(point.zero_mean) <= point.zero_limit,
        drift_pass=drift <= drift_limit)
    saved = object_dict(record['summary'])
    for key, reference in expected.items():
        actual = saved[key]
        if isinstance(reference, bool):
            if type(actual) is not bool or actual != reference:
                raise ValueError('Saved summary flag mismatch: ' + key)
        elif not math.isclose(number(actual), reference, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Saved summary number mismatch: ' + key)
    return point


def analyze_rows(cells: list[Cell], records: object) -> list[CellReport]:
    if not isinstance(records, list):
        raise ValueError('Expected checkpoint list')
    by_cell = {(cell.wl, cell.bl): cell for cell in cells}
    indexed: dict[tuple[int, int, str], dict[str, object]] = {}
    for value in records:
        record = object_dict(value)
        cell_data = object_dict(record['cell'])
        wl, bl = address(cell_data)
        role = record['role']
        if (wl, bl) not in by_cell or not isinstance(role, str) or role not in ROLES:
            raise ValueError('Unplanned cell/role')
        if cell_data != asdict(by_cell[wl, bl]):
            raise ValueError('Checkpoint cell metadata differs from frozen sample')
        key = wl, bl, role
        if key in indexed:
            raise ValueError('Duplicate checkpoint')
        indexed[key] = record
    output: list[CellReport] = []
    for cell in cells:
        reasons: list[str] = []
        points: dict[str, Checkpoint] = {}
        for role in ROLES:
            found = indexed.get((cell.wl, cell.bl, role))
            if found is None:
                reasons.append('missing:' + role)
                continue
            initial = points.get('initial')
            point = recompute(found, cell, initial.mean if initial else None)
            points[role] = point
            if point.sd > max(2., .05 * cell.span_adc):
                reasons.append('noise:' + role)
            if abs(point.zero_mean) > point.zero_limit:
                reasons.append('zero_shift:' + role)
            if initial and abs(point.mean - initial.mean) > max(3., .05 * cell.span_adc):
                reasons.append('drift:' + role)
        first, last = points.get('initial'), points.get('delay_60')
        values = list(points.values())
        output.append(CellReport(
            cell.wl, cell.bl, cell.sample_reason == 'original_selected_rectangle', cell.sample_reason,
            cell.historical_hard_eligible, cell.historical_quality_qualified, cell.span_adc,
            len(points), not reasons, reasons, first.mean if first else None, last.mean if last else None,
            max(p.sd for p in values) if values else None, max(2., .05 * cell.span_adc),
            max(abs(p.mean - first.mean) for p in values) if first else None,
            max(3., .05 * cell.span_adc), max(abs(p.zero_mean) / p.zero_limit for p in values) if values else None,
            first.mean_g if first else None, last.mean_g if last else None,
            last.time - first.time if first and last else None))
    return output


def count_group(rows: list[CellReport]) -> dict[str, int]:
    return dict(cells=len(rows), passed=sum(r.passed for r in rows), failed=sum(not r.passed for r in rows),
                incomplete=sum(r.checkpoint_count != 4 for r in rows))


def format_number(value: float | None) -> str:
    return '—' if value is None else f'{value:.4g}'


def table(rows: list[CellReport]) -> str:
    header = '<tr><th>WL/BL</th><th>Причина выборки</th><th>Текущий итог</th><th>ADC initial→final</th><th>max SD / допуск</th><th>max drift / допуск</th><th>G initial→final, µS</th><th>Δt,s</th></tr>'
    body = []
    for row in rows:
        reason = ', '.join(row.reasons) if row.reasons else 'PASS чтения'
        fields = [f'{row.wl}/{row.bl}', row.sample_reason, reason,
                  f'{format_number(row.initial_mean_adc)} → {format_number(row.final_mean_adc)}',
                  f'{format_number(row.maximum_sd_adc)} / {row.noise_limit_adc:.4g}',
                  f'{format_number(row.maximum_drift_adc)} / {row.drift_limit_adc:.4g}',
                  f'{format_number(row.initial_g_s * 1e6 if row.initial_g_s is not None else None)} → {format_number(row.final_g_s * 1e6 if row.final_g_s is not None else None)}',
                  format_number(row.elapsed_initial_to_final_s)]
        body.append('<tr>' + ''.join('<td>' + html.escape(field) + '</td>' for field in fields) + '</tr>')
    return '<table>' + header + ''.join(body) + '</table>'


def render_html(rows: list[CellReport], source: str, session: str) -> str:
    by_cell = {(row.wl, row.bl): row for row in rows}
    grid = ['<tr><th>WL/BL</th>' + ''.join(f'<th>{b}</th>' for b in range(32)) + '</tr>']
    for wl in range(8):
        cells = []
        for bl in range(32):
            row = by_cell.get((wl, bl))
            color = 'gray' if row is None or row.checkpoint_count == 0 else 'green' if row.passed else 'red'
            tooltip = 'Не проверено' if row is None else '; '.join(row.reasons) or 'Все fresh-критерии чтения выполнены'
            marker = '•' if row and row.selected else ''
            cells.append(f'<td class="{color}" title="{html.escape(tooltip, quote=True)}">{marker}</td>')
        grid.append(f'<tr><th>{wl}</th>' + ''.join(cells) + '</tr>')
    return ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Выборочная метрология</title>'
            '<style>body{font:15px system-ui;margin:24px;color:#17202a}table{border-collapse:collapse;margin:16px 0;font-size:12px}td,th{border:1px solid #bbb;padding:6px}.map td{width:22px;text-align:center}.gray{background:#ddd}.green{background:#86d39b}.red{background:#ef9898}p{max-width:1000px}</style>'
            f'<h1>Выборочная проверка40/256 ячеек — {html.escape(source)}</h1><p>Сессия {html.escape(session)}</p>'
            '<p>Зелёный: все текущие критерии чтения. Красный: отклонение/неполные данные. Серый: не проверено. • — исходные16selectedcells.</p>'
            '<p><strong>Это не полная калибровка и не проверка программируемого диапазона.</strong> Старая карта остаётся diagnostic_best_available, qualified=false. Изменения G после прежних записей не трактуются как ошибка retention.</p>'
            '<table class="map">' + ''.join(grid) + '</table><h2>16 исходно выбранных ячеек</h2>'
            + table([row for row in rows if row.selected]) + '<h2>Все40 выбранных ячеек</h2>' + table(rows) + '</html>')


def report(session: Path, output: Path) -> dict[str, object]:
    manifest = read_json(session / 'manifest.json')
    if manifest.get('status') not in ('completed', 'failed'):
        raise ValueError('Terminal manifest required')
    source = manifest.get('source')
    sid = manifest.get('session_id')
    if source not in ('hardware', 'simulator') or not isinstance(source, str) or not isinstance(sid, str):
        raise ValueError('Invalid session/source')
    if number(manifest['pulses']) != 0:
        raise ValueError('Not a read-only session')
    inventory = object_dict(manifest['result_sha256'])
    for name in ('sample.json', 'observations.json', 'protocol.json'):
        if inventory.get(name) != sha(session / name):
            raise ValueError('Result hash mismatch: ' + name)
    if manifest['sample_sha256'] != sha(session / 'sample.json'):
        raise ValueError('Sample identity mismatch')
    sample, cells = load_sample(session / 'sample.json')
    if sample['parent_map_sha256'] != manifest['parent_map_sha256'] or sample['parent_session'] != manifest['parent_session']:
        raise ValueError('Parent map identity mismatch')
    observations = read_json(session / 'observations.json')
    rows = analyze_rows(cells, observations['checkpoints'])
    if manifest.get('status') == 'completed':
        if number(manifest['impacts']) != 6400 or any(row.checkpoint_count != 4 for row in rows):
            raise ValueError('Completed coverage/count inconsistent')
        if manifest.get('sampled_metrology_pass') != all(row.passed for row in rows):
            raise ValueError('Overall metrology verdict inconsistent')
    summary: dict[str, object] = dict(
        schema='selective-metrology-offline-report-v1', session_id=sid, source=source,
        status=manifest['status'], impacts=manifest.get('impacts'), pulses=0,
        elapsed_seconds=manifest.get('elapsed_seconds'), all_sample=count_group(rows),
        original_selected=count_group([r for r in rows if r.selected]),
        not_sampled_cells=256-len(rows), parent_session=sample['parent_session'],
        parent_map_sha256=sample['parent_map_sha256'], original_quality_qualified=sample['parent_qualified'],
        programmable_range_validated=False, full_array_validated=False,
        raw_request_replay_performed=False, adc_checkpoint_statistics_recomputed=True,
        observation_sha256=sha(session/'observations.json'), manifest_sha256=sha(session/'manifest.json'),
        sample_sha256=sha(session/'sample.json'), report_helper_sha256=sha(Path(__file__)),
        cells=[asdict(row) for row in rows])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'SUMMARY.json', summary)
    (output / 'MAP.html').write_text(render_html(rows, source, sid), encoding='utf-8', newline='\n')
    selected = count_group([row for row in rows if row.selected])
    lines = [f'# Выборочная метрология — {source}', '', f'Сессия `{sid}`: {manifest["status"]}.', '',
             f'Свежие критерии чтения: {sum(row.passed for row in rows)}/40; исходные selected: {selected["passed"]}/16.',
             'Остальные216ячеек не проверялись. Полная таблица и цветовая карта: [MAP.html](MAP.html).', '',
             'Статистика повторно рассчитана по сохранённым ADC brackets; hashes проверены. Полный raw request/result replay не выполнялся.',
             'Исходная карта остаётся диагностической. Программируемый диапазон и signed orientations этим read-only опытом не проверены.', '',
             '| WL/BL | selected | Результат | max SD / limit ADC | max drift / limit ADC |',
             '|---|---|---|---|---|']
    for row in rows:
        lines.append(f'| {row.wl}/{row.bl} | {row.selected} | {", ".join(row.reasons) or "PASS"} | {format_number(row.maximum_sd_adc)} / {row.noise_limit_adc:.4g} | {format_number(row.maximum_drift_adc)} / {row.drift_limit_adc:.4g} |')
    (output / 'REPORT_RU.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    result = report(repo_path(args.session), repo_path(args.output_dir))
    print(json.dumps({key: result[key] for key in ('session_id', 'source', 'status', 'all_sample', 'original_selected')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
