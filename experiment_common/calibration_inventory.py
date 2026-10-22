"""Инвентаризация разновременных измерений и кандидаты раскладки200cells.

Ни количество хороших ячеек само по себе, ни эта таблица не квалифицируют
аппаратный forward: signed-пары/MVM и выбранное переназначение ещё нужны.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
import html
import json

from experiment_common.programming_validation import CELLS, Point
from experiment_common.selective_metrology import number, object_dict

REMAINING = tuple((w, b) for w in range(8) for b in range(32) if (w, b) not in CELLS)
SMOKE = ((0, 0), (2, 1), (4, 16), (7, 31))


def render_map(rows: list[dict[str, object]], summary: dict[str, object]) -> str:
    """Статическая карта: диапазон, качество и разновременность видны отдельно."""
    cells = row_map(rows)
    body = '<tr><th>WL / BL</th>' + ''.join(f'<th>{b}</th>' for b in range(32)) + '</tr>'
    for w in range(8):
        body += f'<tr><th>{w}</th>'
        for b in range(32):
            row = cells.get((w, b))
            color, text = '#ddd', '—'
            if row is not None:
                fresh = row.get('fresh_readonly_recheck')
                current = fresh is None or object_dict(fresh)['passed'] is True
                color = '#b8e4b8' if row.get('eligible') is True and current else '#ffd58a' if row.get('hard_eligible') is True and current else '#f2b3b3'
                text = ('стар.' if row.get('reprogrammed_this_session') is False else '') + str(round(number(row.get('span_adc', 0.)), 1))
                if 'missing_required_state_checkpoint' in str(row.get('hard_reasons', [])):
                    color, text = '#ddd', 'неполные данные'
            title = html.escape(json.dumps(row, ensure_ascii=False), quote=True)
            body += f'<td style="background:{color}" title="{title}">{text}</td>'
        body += '</tr>'
    return ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Карта программируемости</title>'
            '<style>body{font:14px sans-serif}table{border-collapse:collapse}td,th{border:1px solid #999;padding:5px}pre{white-space:pre-wrap}</style>'
            '<h1>Измеренные диапазоны R/M/H, ADC</h1><p>Зелёный: качество пройдено; оранжевый: только различимый диапазон; красный: критерии не пройдены; серый: не измерено. '
            '«стар.» — прежняя программируемость и свежая проверка чтением. Это не одновременная калибровка 256 ячеек и не квалификация MNIST/MVM.</p><table>'
            + body + '</table><pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre></html>')


def row_map(value: object) -> dict[tuple[int, int], dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError('Expected cell records')
    out: dict[tuple[int, int], dict[str, object]] = {}
    for item in value:
        row = object_dict(item)
        wl, bl = row['wl'], row['bl']
        if type(wl) is not int or type(bl) is not int or not isinstance(wl, int) or not isinstance(bl, int) or not 0 <= wl < 8 or not 0 <= bl < 32:
            raise ValueError('Invalid address')
        if (wl, bl) in out:
            raise ValueError('Duplicate address')
        out[wl, bl] = row
    return out


def pairings(wls: tuple[int, ...] = tuple(range(8))) -> Iterator[tuple[tuple[int, int], ...]]:
    if not wls:
        yield ()
        return
    first, *rest = wls
    for index, second in enumerate(rest):
        for tail in pairings(tuple(rest[:index] + rest[index+1:])):
            yield ((first, second),) + tail


def mapping_candidates(allowed: set[tuple[int, int]]) -> dict[str, object]:
    common = [b for b in range(32) if all((w, b) in allowed for w in range(8))]
    fixed = all((w, b) in allowed for w in range(8) for b in range(25))
    remap: list[dict[str, object]] | None = None
    for pairs in pairings():
        parts: list[dict[str, object]] = []
        for a, b in pairs:
            bls = [k for k in range(32) if (a, k) in allowed and (b, k) in allowed]
            if len(bls) < 25:
                break
            parts.append(dict(pair=[a, b], bls=bls[:25]))
        if len(parts) == 4:
            remap = parts
            break
    return dict(eligible_cells=len(allowed), original_bl0_24_all8wl=fixed,
                common_bls_all8wl=common, common25_candidate=common[:25] if len(common) >= 25 else None,
                four_disjoint_pairs_with_own25bl=remap,
                signed_mvm_qualified=False, remapped_forward_implemented=False,
                total_count_alone_is_not_layout=True)


def inventory(current_rows: list[dict[str, object]], prior: dict[str, object],
              prior_manifest: dict[str, object], rechecks: list[Point],
              current_session: str, current_finished: str) -> dict[str, object]:
    old = row_map(prior['cells'])
    new = row_map(current_rows)
    if set(old) != set(CELLS) or set(new) != set(REMAINING):
        raise ValueError('Expected exact disjoint16+240 inventory')
    points = prior['checkpoints']
    if not isinstance(points, list):
        raise ValueError('Missing prior checkpoint evidence')
    old_final: dict[tuple[int, int], dict[str, object]] = {}
    for value in points:
        old_point = object_dict(value)
        if old_point['role'] == 'final':
            old_final[int(number(old_point['wl'])), int(number(old_point['bl']))] = old_point
    fresh = {(point.wl, point.bl): point for point in rechecks if point.role == 'final_recheck'}
    rows: list[dict[str, object]] = []
    hard: set[tuple[int, int]] = set()
    quality: set[tuple[int, int]] = set()
    for cell in ((w, b) for w in range(8) for b in range(32)):
        previous = cell in old
        row = dict(old[cell] if previous else new[cell])
        row.update(programmability_session=prior_manifest['session_id'] if previous else current_session,
                   programmability_finished_utc=prior_manifest['finished_utc'] if previous else current_finished,
                   reprogrammed_this_session=not previous)
        freshness = True
        if previous:
            point, baseline = fresh.get(cell), old_final.get(cell)
            reasons: list[str] = []
            if point is None or baseline is None:
                reasons.append('old16_fresh_recheck_missing')
            else:
                span = number(row.get('span_adc', 0.))
                if point.sd > max(2., .05 * span):
                    reasons.append('repeat_noise')
                if abs(point.mean - number(baseline['mean'])) > max(3., .05 * span):
                    reasons.append('change_from_previous_final_H')
                if abs(point.zero_mean) > point.zero_limit:
                    reasons.append('zero_shift')
            freshness = not reasons
            row['fresh_readonly_recheck'] = dict(passed=freshness, reasons=reasons,
                                                point=asdict(point) if point else None,
                                                baseline=baseline,
                                                current_programmable_range_revalidated=False)
        if row.get('hard_eligible') is True and freshness:
            hard.add(cell)
        if row.get('eligible') is True and freshness:
            quality.add(cell)
        rows.append(row)
    return dict(schema='mixed-time-256-programmability-inventory-v1', cells=rows,
                acquired_now=240, previous_programmability=16, prior16_readonly_rechecks=len(fresh),
                simultaneous_256_calibration=False, different_programming_times=True,
                historical_range_not_revalidated_by_readonly=True,
                hard_diagnostic_candidates=mapping_candidates(hard),
                strict_quality_candidates=mapping_candidates(quality),
                hardware_mnist_200_qualified=False,
                note='Layout candidates only; prior16 range is historical, even if final read passes. No B/combined MVM in this acquisition.')
