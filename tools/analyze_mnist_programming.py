"""Offline diagnosis of immutable native programming records; no board access."""
from __future__ import annotations

from collections import Counter
import gzip
import html
import json
from pathlib import Path
import statistics
import sys
import tarfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_common.selective_metrology import object_dict, number
from experiment_common.durable import atomic_json, sha

BASE = ROOT / 'results/mnist_tiled_transfer_20261022'
SOURCES = {'hardware': 'cb53be86-ae2b-485f-bf40-3b7d994e7a50',
           'simulator': '0c7d20fc-045f-4f27-a031-cdcf1889500c'}


def read(path: Path) -> dict[str, object]:
    return object_dict(json.loads(path.read_text(encoding='utf-8')))


def analyze(sid: str) -> dict[str, object]:
    folder = BASE / sid
    manifest = read(folder / 'manifest.json')
    if manifest['status'] != 'completed':
        raise ValueError('Completed acquisition required')
    hashes = object_dict(manifest['result_sha256'])
    tolerance = number(read(folder / 'summary.json')['target_tolerance_s'])
    records: dict[tuple[int, int, int], dict[str, object]] = {}
    for path in sorted(folder.glob('load_*_programming.json')):
        if sha(path) != hashes[path.name]:
            raise ValueError('Programming checksum mismatch')
        raw: object = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(raw, list):
            raise ValueError('List required')
        load = int(path.name.split('_')[1])
        for item in raw:
            row = object_dict(item)
            records[load, int(number(row['wl'])), int(number(row['bl']))] = row
    traces: dict[tuple[int, int, int], list[float]] = {k: [number(v['initial_g_s'])] for k, v in records.items()}
    decisions: Counter[str] = Counter()
    directions: dict[tuple[int, int, int], list[int]] = {k: [] for k in records}
    noise: dict[tuple[int, int], list[float]] = {}
    for path in sorted((folder / 'raw').glob('*.jsonl.gz')):
        relative = path.relative_to(folder).as_posix()
        if sha(path) != hashes[relative]:
            raise ValueError('Raw checksum mismatch')
        with gzip.open(path, 'rb') as stream:
            for line in stream:
                if b'"event":"pulse_decision"' not in line and b'"event":"conductance_read"' not in line:
                    continue
                row = object_dict(json.loads(line))
                meta = object_dict(row['metadata'])
                if 'load' not in meta:
                    continue
                wl, bl = int(number(row['wl'])), int(number(row['bl']))
                key = int(number(meta['load'])), wl, bl
                if key not in records:
                    continue
                if row['event'] == 'pulse_decision':
                    sign = int(number(row['sign']))
                    directions[key].append(sign)
                    label = 'SET' if sign else 'RESET'
                    decisions[label + ('_admitted' if row['admitted'] else '_blocked')] += 1
                elif meta.get('role') == 'post_program':
                    traces[key].append(number(row['mean_g_s']))
                    rs = row['resistance_ohm']
                    if not isinstance(rs, list):
                        raise ValueError('Resistance samples required')
                    noise.setdefault((wl, bl), []).append(statistics.stdev([1 / number(r) for r in rs]))
    cells: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for wl, bl in sorted({(k[1], k[2]) for k in records}):
        selected = [(k, v) for k, v in records.items() if k[1:] == (wl, bl)]
        local: Counter[str] = Counter()
        final_values: list[float] = []
        goals: list[float] = []
        shifts: list[float] = []
        swings: list[float] = []
        for key, row in selected:
            target, last, final = (number(row[n]) for n in ('target_g_s', 'last_g_s', 'final_g_s'))
            before_ok, after_ok = abs(last-target) <= tolerance, abs(final-target) <= tolerance
            local['writes'] += 1
            local['final_hits' if after_ok else 'final_misses'] += 1
            local['terminal_hits' if before_ok else 'terminal_misses'] += 1
            if before_ok and not after_ok:
                local['lost_after_other_writes'] += 1
            if not before_ok and after_ok:
                local['gained_after_other_writes'] += 1
            if number(row['attempts']) == 96:
                local['limit_96'] += 1
            trace, signs = traces[key], directions[key]
            if len(trace) != 1 + int(number(row['attempts'])) or len(signs) != len(trace)-1:
                raise ValueError('Incomplete trace')
            swing = max(trace)-min(trace)
            if not after_ok and swing <= tolerance:
                local['miss_and_trace_range_within_tolerance'] += 1
            changes = sum(a != b for a, b in zip(signs, signs[1:]))
            if not after_ok and changes >= 4:
                local['miss_with_at_least_four_reversals'] += 1
            if not after_ok and (target < min(trace)-tolerance or target > max(trace)+tolerance):
                local['miss_target_outside_observed_trace'] += 1
            if min(abs(g-target) for g in trace) <= tolerance:
                local['ever_hit_target'] += 1
            if signs:
                local['direction_changes'] += changes
                local['SET'] += sum(signs)
                local['RESET'] += len(signs)-sum(signs)
            final_values.append(final*1e6)
            goals.append(target*1e6)
            shifts.append(abs(final-last)*1e6)
            swings.append(swing*1e6)
        counts.update(local)
        cells.append(dict(wl=wl, bl=bl, **dict(local), target_range_us=[min(goals), max(goals)],
            observed_final_range_us=[min(final_values), max(final_values)],
            median_trace_range_us=statistics.median(swings), max_trace_range_us=max(swings),
            median_read_triplet_sd_us=statistics.median(noise.get((wl, bl), [0.]))*1e6,
            median_post_write_shift_us=statistics.median(shifts), max_post_write_shift_us=max(shifts)))
    summary = read(folder/'summary.json')
    if (counts['writes'] != summary['programmed_cell_loads']
        or counts['final_misses'] != summary['target_misses']
        or decisions['SET_admitted'] + decisions['RESET_admitted'] != summary['pulses']
        or counts['terminal_hits'] - counts['lost_after_other_writes'] + counts['gained_after_other_writes'] != counts['final_hits']):
        raise ValueError('Independent summary/trace reconciliation failed')
    return dict(session=sid, tolerance_us=tolerance*1e6, counts=dict(counts), pulse_decisions=dict(decisions), cells=cells)


def survey() -> dict[str, object]:
    """Reuse historical 256-cell evidence; no declaration of current eligibility."""
    archive = ROOT / 'results/remaining_programming_validation_20261022/archives/3ddb2d16-0a97-4092-a681-8b7cabb477a4.tar.gz'
    receipt = read(archive.with_suffix('').with_suffix('.json'))
    if sha(archive) != receipt['archive_sha256']:
        raise ValueError('Survey archive checksum mismatch')
    files = object_dict(receipt['files'])
    with tarfile.open(archive) as tar:
        def member(name: str) -> dict[str, object]:
            names = [n for n in tar.getnames() if n.endswith('/'+name)]
            if len(names) != 1:
                raise ValueError('Ambiguous archive member')
            stream = tar.extractfile(names[0])
            if stream is None:
                raise ValueError('Missing archive member')
            raw = stream.read()
            if hashlib.sha256(raw).hexdigest() != object_dict(files[name])['sha256']:
                raise ValueError('Survey member checksum mismatch')
            return object_dict(json.loads(raw))
        inventory = member('inventory256.json')
        observations = [member('observations.json'), member('prior16_observations.json')]
    raw_cells = inventory['cells']
    if not isinstance(raw_cells, list):
        raise ValueError('Inventory list required')
    checkpoints: list[dict[str, object]] = []
    for obs in observations:
        values = obs['checkpoints']
        if not isinstance(values, list):
            raise ValueError('Checkpoint list required')
        checkpoints.extend(object_dict(v) for v in values)
    counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    stable = read(ROOT/'reports/2026-10-22/block-readout-validation/STABLE_BLOCKS.json')['selected_blocks']
    if not isinstance(stable, list):
        raise ValueError('Block list required')
    used: set[tuple[int, int]] = set()
    for block_value in stable:
        block = object_dict(block_value)
        wls, bls = block['wl_pair'], block['bls']
        if not isinstance(wls, list) or not isinstance(bls, list):
            raise ValueError('Coordinates required')
        used.update((int(number(w)), int(number(b))) for w in wls for b in bls)
    for value in raw_cells:
        row = object_dict(value)
        wl, bl = int(number(row['wl'])), int(number(row['bl']))
        points = [p for p in checkpoints if p['wl'] == wl and p['bl'] == bl]
        spans: list[float] = []
        for cycle in range(2):
            endpoints = {str(p['state']): number(p['mean_g_s']) for p in points if p['cycle'] == cycle and p['role'] == 'immediate'}
            if 'H' in endpoints and 'R' in endpoints:
                spans.append((endpoints['H']-endpoints['R'])*1e6)
        responsive = len(spans) == 2 and min(spans) > 2*2.5603554203687646
        # This screen only prioritizes remeasurement; it cannot qualify fine
        # programming, delay stability, arbitrary masks, or a new MNIST layout.
        label = ('used36' if (wl, bl) in used else
                 'candidate_both_paths' if responsive and row['hard_eligible'] else
                 'candidate_readout_diagnosis' if responsive else 'needs_response_diagnosis')
        fresh = row.get('fresh_readonly_recheck')
        if label == 'candidate_both_paths' and isinstance(fresh, dict) and fresh.get('passed') is False:
            label = 'candidate_recheck_failed'
        counts[label] += 1
        if row['hard_eligible']:
            counts['historical_hard_eligible'] += 1
        if row['eligible']:
            counts['historical_strict_eligible'] += 1
        if responsive:
            counts['historical_G_response_both_cycles'] += 1
        rows.append(dict(wl=wl, bl=bl, priority=label, used_in_mnist=(wl, bl) in used,
            historical_g_span_us=spans, historical_mvm_hard_eligible=row['hard_eligible'],
            historical_strict_eligible=row['eligible'], reasons=row['reasons'],
            time=row['programmability_finished_utc']))
    return dict(counts=dict(counts), cells=rows, source_sha256=receipt['archive_sha256'],
        screen='Historical H-R mode7 G span >5.12071uS in both cycles; candidate only, no current qualification. Mixed acquisition times; subsequent writes invalidate current state.')


def render(result: dict[str, object], out: Path) -> None:
    survey_data = object_dict(result['survey256'])
    cells = survey_data['cells']
    hardware = object_dict(result['hardware'])['cells']
    if not isinstance(cells, list) or not isinstance(hardware, list):
        raise ValueError('Cell lists required')
    indexed = {(int(number(r['wl'])), int(number(r['bl']))): r for value in cells for r in [object_dict(value)]}
    labels = {'used36': 'Использована в MNIST; подробные ошибки записи ниже',
              'candidate_both_paths': 'Дополнительный кандидат: исторически G и MVM откликаются',
              'candidate_recheck_failed': 'Исторический кандидат, последняя перепроверка не прошла',
              'candidate_readout_diagnosis': 'G менялась, требуется диагностика MVM',
              'needs_response_diagnosis': 'Недостаточно свидетельств рабочего диапазона по этому screening'}
    grid = '<tr><th>WL / BL</th>'+''.join(f'<th>{b}</th>' for b in range(32))+'</tr>'
    for wl in range(8):
        grid += f'<tr><th>{wl}</th>'
        for bl in range(32):
            r = indexed[wl, bl]
            label = str(r['priority'])
            detail = f'WL{wl}/BL{bl}: {labels[label]}; H−R, мкСм: {r["historical_g_span_us"]}; причины старой проверки: {r["reasons"]}'
            grid += f'<td class="{label}" title="{html.escape(detail, quote=True)}"><a href="#c{wl}_{bl}">{bl}</a></td>'
        grid += '</tr>'
    detail_rows = ''
    for value in cells:
        r = object_dict(value)
        detail_rows += f'<tr id="c{r["wl"]}_{r["bl"]}"><td>WL{r["wl"]}/BL{r["bl"]}</td><td>{labels[str(r["priority"])]}</td><td>{html.escape(str(r["historical_g_span_us"]))}</td><td>{html.escape(str(r["reasons"]))}</td></tr>'
    write_rows = ''
    for value in hardware:
        r = object_dict(value)
        write_rows += f'<tr><td>WL{r["wl"]}/BL{r["bl"]}</td><td>{r.get("final_hits",0)}/{r["writes"]}</td><td>{r.get("miss_and_trace_range_within_tolerance",0)}</td><td>{r.get("miss_with_at_least_four_reversals",0)}</td><td>{r.get("lost_after_other_writes",0)}</td><td>{number(r["median_trace_range_us"]):.3f}</td></tr>'
    prose = '''<h1>Почему запись MNIST промахнулась и как расширить карту</h1>
<p>22 октября 2026. Только анализ сохранённых измерений, новых воздействий на плату нет. Плата: 55,07%; сопоставимый SIM: 55,34%; прежняя раскладка обученной сети: 66,36%. Гибридные замороженные измеренные MVM + CPU LIF/head.</p>
<h2>Подтверждённые причины на уровне алгоритма</h2>
<ul><li>Для всех ячеек использовались одинаковые SET 1,7 В и RESET 1,4 В, максимум 96 попыток. Индивидуальные диапазоны не учитывались. Все 210354 решения об импульсах допущены токовым guard: пропуски по току не объясняют результат.</li>
<li>2107 из 3200 записей уже промахнулись при выходе из подбора. После общего readback: 2295 промахов; 189 прежних попаданий потеряны, 1 появился. Это временной сдвиг/взаимное влияние/шум, их физические причины данным опытом не разделены.</li>
<li>Среди финальных промахов 747 имеют полный разброс траектории не больше допуска ±2,560 мкСм; 1148 сопровождались минимум четырьмя сменами SET↔RESET. Категории могут пересекаться. Для 1027 целей траектория вообще не подошла к целевому диапазону.</li>
<li>WL0/BL8: 8544 RESET, все 89 целей мимо; итоговые G 369,8–399,1 мкСм при целях 10,0–116,6. WL3/BL8 и WL1/BL15 также по 8544 RESET и 89/89 промахов. Это отсутствие управляемости при данном режиме, не доказательство поломки ячейки.</li>
<li>WL2/BL15: 6090 смен направления, 61/89 промахов; типичный размах 24,6 мкСм. Сильный фиксированный импульс перескакивает узкий допуск. WL4/BL3: 82/89 попаданий, но только для проверенных целей 10,0–28,3 мкСм; переносить этот процент на другие цели нельзя.</li>
<li>Проблема есть и в SIM: 873/3200 промахов. Разница между 66,36% и 55,34% уже возникает без платы при новой схеме переноса; устранение аппаратных промахов само по себе не гарантирует возвращения 66%.</li></ul>
<h2>Что менять в следующем протоколе</h2>
<ol><li>Сначала определить для каждой пары достижимые уровни, полярность и минимальный полезный шаг. Амплитуду адаптировать внутри проверенных ограничений: при отсутствии реакции менять режим; при перескоке уменьшать шаг/амплитуду. Не повторять 96 одинаковых импульсов без прогресса. Длительность физического импульса остаётся неизвестной: менять pulse_width как подтверждённый аппаратный регулятор нельзя.</li>
<li>Подбирать пару совместно по G+−G−, допускается общий сдвиг обеих проводимостей внутри рабочих диапазонов. Если необходим масштаб, фиксировать обратное преобразование до LIF. Общий сдвиг и масштаб проверить реальным MVM: идеальная компенсация на нелинейной плате не гарантирована.</li>
<li>Останавливать подбор внутри допуска с запасом на измеренный шум/дрейф, затем подтверждать после записи всего блока. При колебаниях или недостижимой цели явно отмечать неуспех и переназначать коэффициент. Проверять не только G через mode7, но и фактический дифференциальный MVM при рабочем входном напряжении.</li>
<li>Калибровку масштаба/смещения/связей строить по отдельным измерительным шаблонам без test-меток, проверять на других шаблонах и после перепрограммирования. Матрица не восстанавливает информацию при нулевом отклике или вырожденном канале.</li>
<li>После стабильного измеренного forward заново обучить только scaler/head на train-признаках этой схемы, локально. Выбирать параметры на train/validation, затем один итоговый test. Исходный frozen-head результат сохранить как baseline. Сравнить без коррекции, с калибровкой и с адаптированным head; рост точности пока не измерен.</li></ol>
<h2>Расширение: все 256 адресов</h2>
<p>В историческом двухцикловом обследовании 202 адреса имели H−R по mode7 больше 5,121 мкСм в обоих циклах. Это screening изменения G, а не доказанная точная запись или пригодность для MNIST. В исходном inventory 78 адресов имели historical hard_eligible; старая сводка 75 дополнительно исключала три ранее измеренных адреса после неуспешной свежей перепроверки. Строгую старую проверку прошли 7.</p>
<p>Помимо использованных 36 адресов выделено 40 исторических кандидатов с изменением G и различимым MVM. Три из них (WL0/BL3,10,14) не прошли последнюю сохранённую перепроверку: приоритетных кандидатов остаётся 37. Ещё 127 адресов — с изменением G, но требующих диагностики MVM (104 из них BL16–31). Совокупность 36+37 адресов допускает не более 32 непересекающихся пар на общих BL, если заново перепаковать карту; это условная топологическая оценка, не 32 квалифицированных рабочих слота. В случае успешной проверки всех этих пар 1600 коэффициентов потребуют минимум 50 загрузок вместо 89. Текущий production mapping ограничивает BL&lt;25: BL31 нельзя включить без изменения и проверки раскладки.</p>
<p>Ближайший короткий аппаратный пилот: представители плато WL0/BL8, WL3/BL8, WL1/BL15; колебаний WL2/BL15; контроль WL4/BL3. Затем проверить дополнительные пары из списка ниже и двоичные групповые маски, включая контекст положительного фона. Новые импульсные параметры сначала проверить через MemriBoard Simulator; на плате запускать отдельным foreground experiment.py с конечным бюджетом и сохранением результатов. Никаких прямых вызовов драйвера.</p>
<p>Карта историческая: измерения сделаны в разные моменты, а последующие записи изменили состояние. Цвет означает приоритет диагностики, не разрешение использовать ячейку без проверки. Нажмите адрес для деталей.</p>'''
    legend = ''.join(f'<span class="{k}" style="padding:.4em;display:inline-block;margin:.2em">{v}</span>' for k,v in labels.items())
    page = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Запись MNIST и карта 256 ячеек</title>'
            '<style>body{font:16px system-ui;max-width:1400px;margin:2em auto;padding:1em}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:.3em;vertical-align:top}td{overflow-wrap:anywhere}a{color:inherit}.used36{background:#badcff}.candidate_both_paths{background:#c8e9bb}.candidate_recheck_failed{background:#f2b9bb}.candidate_readout_diagnosis{background:#ffe2a3}.needs_response_diagnosis{background:#ddd}pre{white-space:pre-wrap}.map td{text-align:center}.detail{font-size:12px}</style>'+prose+legend+'<table class="map">'+grid+'</table>'
            '<h2>Запись по 36 использованным ячейкам</h2><table><tr><th>Адрес</th><th>Попадания</th><th>Промахи с малым размахом</th><th>Промахи с ≥4 сменами</th><th>Потеряны после записи блока</th><th>Медиана размаха, мкСм</th></tr>'+write_rows+'</table>'
            '<h2>Подробности всех адресов</h2><table class="detail"><tr><th>Адрес</th><th>Приоритет</th><th>H−R по двум циклам, мкСм</th><th>Исходные причины</th></tr>'+detail_rows+'</table></html>')
    (out/'RESULTS.html').write_text(page, encoding='utf-8')


def main() -> int:
    out = ROOT / 'reports/2026-10-22/mnist-programming-diagnosis'
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, object]
    if '--render-only' in sys.argv:
        render(read(out/'RESULTS.json'), out)
        return 0
    result = {name: analyze(sid) for name, sid in SOURCES.items()}
    result['survey256'] = survey()
    atomic_json(out / 'RESULTS.json', result)
    for name, data in result.items():
        print(name, json.dumps({k: v for k, v in object_dict(data).items() if k != 'cells'}))
    render(result, out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
