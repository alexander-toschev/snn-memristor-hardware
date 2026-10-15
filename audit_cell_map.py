"""Offline cell-map audit. No Connector creation or hardware calls; native numerical converters only."""
import argparse
from collections import Counter
from configparser import ConfigParser
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys

CONVERTER_SHA = '28c6495c56afa225501dd6b0dea999215f08b73f30e016ff7ea34d6c5412701c'
PRIVATE_PINS = {
 'vendor/MemriBoard/MemriCORE/rp5_fpga_python/rpi_FPGAed.py': 'bc717659fa5d4d36d2037eef978bf1cd8dcf0fe92f9bbfad51f1eb1d0d27df9c',
 'vendor/MemriBoard/MemriCORE/rp5_fpga_python/FPGA_SIM.py': '5bcb02c04bc185e332b2ec96acabc2d4303d21724c9c052de540ef92d603b634'}
CELL_MAP_SHA = '48edea5ba8644342851e295aa21157815f832ac09912ab3f050d2b224bc22525'
BOARD_SHA = '01106c85f52b625b15d6fd08b68df07a133d5810c609a4f9cb9ab24abe01877c'
CELLS = [(wl, bl) for wl in range(8) for bl in range(32)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def child(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes root')
    return path


def verify_sources(repo, commit, sources, check):
    """Private operator files are pinned bytes, never Git blobs."""
    verified, missing = [], {}
    for name, digest in sources.items():
        path = child(repo, name)
        if name in PRIVATE_PINS:
            check(digest == PRIVATE_PINS[name], 'private_source/pin/' + name)
            if path.is_file():
                check(sha(path) == digest, 'local_source/' + name)
            else:
                missing[name] = digest
            continue
        if path.is_file():
            check(sha(path) == digest, 'local_source/' + name)
            blob = subprocess.run(['git', '-C', str(repo), 'show', commit + ':' + name], capture_output=True, check=True).stdout
            check(hashlib.sha256(blob).hexdigest() == digest, 'git_source/' + name)
            verified.append(name)
        else:
            check(False, 'local_source/missing/' + name)
    return verified, missing


def describe(values):
    return dict(n=len(values), mean=mean(values) if values else None,
                sd=stdev(values) if len(values) > 1 else None)


def independent_classification(key, samples, zero, reasons, limits):
    """Recompute the documented screening equations without importing acquisition helpers."""
    groups = {name: describe(samples[name]) for name in ('pre_g', 'post_g', 'low_adc', 'high_adc')}
    groups['zero_adc'] = describe(zero)
    result = dict(wl=key[0], bl=key[1], status='reject' if reasons else 'unmeasured',
                  reasons=list(reasons), metrics=groups)
    if [groups[name]['n'] for name in ('pre_g', 'post_g', 'low_adc', 'high_adc', 'zero_adc')] != [3, 3, 5, 5, 5]:
        result['reasons'].append('incomplete_measurements')
        return result
    base = groups['zero_adc']
    for name in ('low', 'high'):
        signal = groups[name + '_adc']['mean'] - base['mean']
        sd = math.hypot(groups[name + '_adc']['sd'], base['sd'])
        denominator = max(limits['noise_floor_adc'], sd)
        threshold = limits['minimum_snr'] * denominator
        groups.update({f'signal_{name}_adc': signal, f'signal_{name}_sd_adc': sd,
                       f'signal_{name}_threshold_adc': threshold, f'snr_{name}': signal / denominator})
        if signal <= 0 or signal < threshold:
            result['reasons'].append('weak_or_nonpositive_signal_' + name)
    high_signal = groups['signal_high_adc']
    groups['cv_high'] = groups['signal_high_sd_adc'] / high_signal if high_signal > 0 else None
    if groups['cv_high'] is None or groups['cv_high'] > limits['maximum_cv_high']:
        result['reasons'].append('unstable_high_signal')
    epsilon = groups['high_adc']['mean'] - 2 * groups['low_adc']['mean'] + base['mean']
    se = math.sqrt((groups['high_adc']['sd'] ** 2 + 4 * groups['low_adc']['sd'] ** 2 + base['sd'] ** 2) / 5)
    tolerance = max(limits['linearity_fraction'] * abs(high_signal), limits['linearity_se_multiplier'] * se,
                    limits['linearity_floor_adc'])
    pre, post = groups['pre_g'], groups['post_g']
    delta = post['mean'] - pre['mean']
    delta_se = math.sqrt((pre['sd'] ** 2 + post['sd'] ** 2) / 3)
    fraction = delta / pre['mean']
    fraction_tolerance = max(limits['drift_fraction'], limits['drift_se_multiplier'] * delta_se / pre['mean'])
    groups.update(linearity_epsilon_adc=epsilon, linearity_se_adc=se, linearity_tolerance_adc=tolerance,
                  g_delta_siemens=delta, g_delta_se_siemens=delta_se,
                  g_fractional_delta=fraction, g_fractional_tolerance=fraction_tolerance)
    if abs(epsilon) > tolerance:
        result['reasons'].append('nonlinear_zero_adjusted_response')
    if abs(fraction) > fraction_tolerance:
        result['reasons'].append('pre_post_conductance_difference')
    result['status'] = 'reject' if result['reasons'] else 'candidate'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-repo', 'session', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    repo, run, out = (Path(p).resolve() for p in (args.source_repo, args.session, args.output_dir))
    manifest, summary, document = (load(run / name) for name in ('manifest.json', 'summary.json', 'criteria.json'))
    records = [json.loads(line) for line in (run / 'raw.jsonl').read_text(encoding='utf-8').splitlines()]
    errors, checks, max_difference = [], 0, 0.0

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(label)

    def compare(saved, expected, label):
        nonlocal max_difference
        if isinstance(expected, dict):
            check(isinstance(saved, dict) and set(saved) == set(expected), label + '/keys')
            if isinstance(saved, dict):
                for name in expected:
                    if name in saved:
                        compare(saved[name], expected[name], label + '/' + name)
        elif isinstance(expected, list):
            check(isinstance(saved, list) and len(saved) == len(expected), label + '/length')
            if isinstance(saved, list):
                for i, (a, b) in enumerate(zip(saved, expected)):
                    compare(a, b, label + '/' + str(i))
        elif type(expected) in (int, float) and type(saved) in (int, float):
            max_difference = max(max_difference, abs(saved - expected))
            check(math.isclose(saved, expected, rel_tol=1e-12, abs_tol=1e-12), label)
        else:
            check(saved == expected, label)

    cfg = ConfigParser()
    cfg.read_string((run / 'board.ini').read_text(encoding='utf-8-sig'))
    board = {k: cfg.getfloat('board', k) for k in ('gain', 'res_load', 'res_switches', 'sum_gain', 'vol_read', 'vol_ref_adc', 'vol_ref_dac')}
    board.update(adc_bit=cfg.getint('board', 'adc_bit'), dac_bit=cfg.getint('board', 'dac_bit'))
    check(sha(run / 'board.ini') == BOARD_SHA, 'board_snapshot/pin')
    check(manifest['source_sha256']['cell_map.py'] == CELL_MAP_SHA, 'reviewed_cell_map/pin')
    if sha(repo / 'vendor/MemriBoard/manager/service/converters.py') != CONVERTER_SHA:
        raise ValueError('Unexpected converter source; import refused')
    sys.dont_write_bytecode = True
    for name in ('.deps', 'vendor/MemriBoard'):
        if (repo / name).is_dir():
            sys.path.insert(0, str(repo / name))
    from manager.service import converters as c

    def resistance(adc, voltage):
        return c.convert_adc_to_res(board['gain'], board['res_load'], voltage, board['adc_bit'],
                                    board['vol_ref_adc'], board['res_switches'], adc)

    git = manifest['invocation']['git']
    commit = git['commit']
    verified, missing = verify_sources(repo, commit, manifest['source_sha256'], check)
    for name, digest in manifest['result_sha256'].items():
        check(sha(child(run, name)) == digest, 'result_sha/' + name)
    check(git['dirty_tracked'] is False, 'git/clean_tracked_claim')
    check(all(line.startswith('?? ') for line in git['status_porcelain'].splitlines()), 'git/porcelain')
    check(git['untracked'] == [line[3:] for line in git['status_porcelain'].splitlines()], 'git/untracked_list')
    check(manifest['invocation']['process_argv'] == ['experiment.py', '--run-hardware', '--map-cells'], 'invocation')
    compare(manifest['criteria'], document, 'criteria/snapshot')
    compare(document, load(repo / 'configs/cell_map.json'), 'criteria/source')
    limits = document['criteria']
    check(all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in limits.values()), 'criteria/finite_positive')
    check(manifest['source'] == 'hardware' and summary['weights_written'] is False, 'scope/no_writes')
    check(manifest['model_is_not_calibration'] is True and manifest['programmability_tested'] is False
          and manifest['pair_additivity_tested'] is False and summary['screening_validated'] is False, 'scope/limitations')
    check((manifest['maximum_impacts'], manifest['mode9_maximum'], manifest['mode10_maximum'], manifest['cells']) == (4136, 1536, 2600, 256), 'planned_counts')
    check(manifest['status'] == summary['status'], 'status/manifest_summary')
    state = {key: {name: [] for name in ('pre_g', 'post_g', 'low_adc', 'high_adc')} for key in CELLS}
    reasons, zeros = {key: [] for key in CELLS}, {wl: [] for wl in range(8)}
    schedule = [(9, 'pre', wl, bl, .1, repeat) for wl, bl in CELLS for repeat in range(3)]
    schedule += [(10, 'mvm', wl, bl, level, repeat) for repeat in range(5)
                 for bl, level in [(None, 0.0)] + [(bl, level) for bl in range(32) for level in (.05, .1)] for wl in range(8)]
    schedule += [(9, 'post', wl, bl, .1, repeat) for wl, bl in CELLS for repeat in range(3)]
    position, slot, attempted, responded, skipped = 0, 0, 0, 0, 0
    mode_counts, status_counts = Counter(), Counter()
    previous_monotonic = -math.inf
    previous_time = None
    errored = False
    while position < len(records):
        if slot >= len(schedule):
            check(False, 'raw/excess_events')
            break
        if slot == 768:
            for key in CELLS:
                if len(state[key]['pre_g']) == 3:
                    g = mean(state[key]['pre_g'])
                    current = .1 * g
                    if current > .005 or current * board['sum_gain'] > .8 * board['vol_ref_adc'] or .1 * round(board['sum_gain'] * g, 4) >= 3.3:
                        reasons[key].append('single_cell_model_bound')
        mode, stage, wl, bl, level, repeat = schedule[slot]
        key = (wl, bl)
        event = records[position]
        label = 'slot/' + str(slot + 1)
        stamp = datetime.fromisoformat(event['timestamp_utc'].replace('Z', '+00:00'))
        check(stamp.tzinfo is not None, label + '/timestamp_timezone')
        check(previous_time is None or stamp >= previous_time, label + '/timestamp_order')
        previous_time = stamp
        check(event['source'] == 'hardware' and event['session_id'] == manifest['session_id'], label + '/identity')
        must_skip = mode == 10 and bl is not None and bool(reasons[key])
        if event['event'] == 'skipped':
            check(must_skip, label + '/skip_eligibility')
            compare({k: event[k] for k in ('wl', 'bl', 'nominal_v', 'repeat', 'reasons')},
                    dict(wl=wl, bl=bl, nominal_v=level, repeat=repeat, reasons=reasons[key]), label + '/skip')
            check('raw_adc' not in event and 'command_si' not in event and 'task_id' not in event, label + '/skip_no_fake_response')
            skipped += 1
            position += 1
            slot += 1
            continue
        check(not must_skip, label + '/eligible_before_command')
        check(event['event'] == 'request', label + '/request')
        if position + 1 >= len(records):
            check(False, label + '/missing_result')
            break
        result = records[position + 1]
        check(result['event'] == 'result', label + '/result')
        for name, value in event.items():
            if name != 'event':
                compare(result.get(name), value, label + '/pair/' + name)
        attempted += 1
        mode_counts[mode] += 1
        check(event['task_id'] == attempted, label + '/unique_task_id')
        values = [level if i == bl else 0.0 for i in range(32)] if mode == 10 else level
        expected_task = dict(mode_flag=10, vol=values, id=0, wl=wl) if mode == 10 else dict(mode_flag=9, vol=level, id=attempted, wl=wl, bl=bl, pulse_width=0.0, sign=0)
        compare(event['command_si'], expected_task, label + '/command')
        check((event['stage'], event['repeat']) == (stage, repeat), label + '/stage_repeat')
        if mode == 10:
            check((event['bl_basis'], event['nominal_v']) == (bl, level), label + '/basis')
        dac = lambda v: c.convert_volt_to_dac(board['dac_bit'], board['vol_ref_dac'], v)
        compare(event['vol_dac'], [dac(v) for v in values] if mode == 10 else dac(values), label + '/dac')
        check(math.isfinite(event['monotonic_time_s']) and event['monotonic_time_s'] >= previous_monotonic, label + '/monotonic')
        previous_monotonic = event['monotonic_time_s']
        check(math.isfinite(result['latency_s']) and result['latency_s'] >= 0, label + '/latency_not_pulse_width')
        native = result.get('native_result')
        protocol_ok = isinstance(native, list) and len(native) == 3 and type(native[1]) is int and native[1] == 0 and type(native[2]) is int and 0 <= native[2] < 2 ** board['adc_bit'] - 1
        expected_status = 'error'
        if protocol_ok:
            responded += 1
            adc = native[2]
            compare([result['native_r'], result['returned_id'], result['raw_adc']], native, label + '/native_tuple')
            compare(native[0], resistance(adc, board['vol_read']), label + '/native_R')
            if mode == 9:
                r = resistance(adc, .1)
                valid = adc >= 20 and math.isfinite(r) and r > 1e-8
                expected_status = 'ok' if valid else 'invalid_readback_model'
                if valid:
                    for name, value in dict(resistance_ohm=r, conductance_siemens=1/r, q=board['sum_gain']/r, read_voltage_parameter_v=.1).items():
                        compare(result[name], value, label + '/' + name)
                    state[key][stage + '_g'].append(1/r)
                else:
                    check(all(name not in result for name in ('resistance_ohm', 'conductance_siemens', 'q')), label + '/no_invalid_model_numbers')
                    reason = 'invalid_' + stage + '_readback_model'
                    if reason not in reasons[key]:
                        reasons[key].append(reason)
            else:
                vout = c.convert_adc_to_volt(1, board['adc_bit'], board['vol_ref_adc'], adc)
                compare(result['output_voltage_v'], vout, label + '/output_voltage')
                expected_status = 'ok' if vout <= .8 * board['vol_ref_adc'] else 'error'
                if expected_status == 'ok':
                    (zeros[wl] if bl is None else state[key]['low_adc' if level == .05 else 'high_adc']).append(adc)
        check(result['status'] == expected_status, label + '/status')
        status_counts[result['status']] += 1
        position += 2
        slot += 1
        if result['status'] == 'error':
            errored = True
            check(bool(result.get('error')), label + '/error_preserved')
            check(position == len(records), 'raw/stopped_after_error')
            break
    check(position == len(records), 'raw/all_events_consumed')
    check(summary['attempted'] == manifest['attempted'] == attempted, 'counts/attempted')
    check(summary['responded'] == responded and summary['skipped_mvm'] == skipped, 'counts/responded_skipped')
    completed = summary['status'] == 'completed'
    check(summary['mask_usable'] is completed, 'mask/usable_status')
    if completed:
        check(slot == 4136 and attempted + skipped == 4136 and mode_counts[9] == 1536 and mode_counts[10] + skipped == 2600 and not errored, 'completed/full_coverage')
    else:
        check(bool(summary.get('error') or summary.get('close_error')), 'failed/error_reason')
    cells = [independent_classification(key, state[key], zeros[key[0]], reasons[key], limits) for key in CELLS]
    compare(load(run / 'cells.json'), cells, 'cells/recomputed')
    counts = {name: sum(cell['status'] == name for cell in cells) for name in ('candidate', 'reject', 'unmeasured')}
    compare(summary['counts'], counts, 'classification/counts')
    with (run / 'candidate_cells.csv').open(encoding='utf-8', newline='') as stream:
        mask = list(csv.reader(stream))
    expected_mask = [['wl', 'bl']] + ([[str(cell['wl']), str(cell['bl'])] for cell in cells if cell['status'] == 'candidate'] if completed else [])
    compare(mask, expected_mask, 'candidate_csv/complete_only')
    flattened = []
    for cell in cells:
        flat = {k: cell[k] for k in ('wl', 'bl', 'status')}
        flat['reasons'] = '; '.join(cell['reasons'])
        for name, value in cell['metrics'].items():
            if isinstance(value, dict):
                flat.update({name + '_' + k: v for k, v in value.items()})
            else:
                flat[name] = value
        flattened.append(flat)
    with (run / 'cell_map.csv').open(encoding='utf-8', newline='') as stream:
        table = list(csv.DictReader(stream))
    check(len(table) == 256, 'csv/256_cells')
    for i, (row, expected) in enumerate(zip(table, flattened)):
        for name, value in expected.items():
            saved = row.get(name)
            if value is None:
                check(saved == '', f'csv/{i}/{name}')
            elif type(value) in (float, int):
                compare(float(saved), value, f'csv/{i}/{name}')
            else:
                compare(saved, value, f'csv/{i}/{name}')
    page = (run / 'map.html').read_text(encoding='utf-8')
    check(f'Run status: {summary["status"]}. Mask usable: {completed}.' in page, 'html/status_banner')
    reason_counts = dict(Counter(reason for cell in cells for reason in cell['reasons']))
    high_cv_unavailable = sum('cv_high' in cell['metrics'] and cell['metrics']['cv_high'] is None for cell in cells)
    high_cv_above_limit = sum(cell['metrics'].get('cv_high') is not None and cell['metrics']['cv_high'] > limits['maximum_cv_high'] for cell in cells)
    both_snr_pass = [cell for cell in cells if 'snr_high' in cell['metrics'] and not any(reason.startswith('weak_or_nonpositive_signal_') for reason in cell['reasons'])]
    only_linearity_reject = sum(cell['reasons'] == ['nonlinear_zero_adjusted_response'] for cell in both_snr_pass)
    report = dict(audit_status='failed' if errors else 'passed', checks=checks, errors=errors,
                  session_id=manifest['session_id'], acquisition_status=summary['status'], mask_usable=completed,
                  auditor_sha256=sha(Path(__file__)), raw_sha256=sha(run/'raw.jsonl'), manifest_sha256=sha(run/'manifest.json'),
                  git_commit=commit, verified_local_and_git_source_count=len(verified), unavailable_private_sources_pins_only=missing,
                  criteria=document, maximum_numeric_difference=max_difference, request_result_pairs=attempted,
                  mode9_count=mode_counts[9], mode10_count=mode_counts[10], mode7_count=mode_counts[7],
                  skipped_mvm=skipped, responded=responded, result_status_counts=dict(status_counts),
                  classification_counts=counts, reason_counts=reason_counts,
                  high_cv_unavailable_count=high_cv_unavailable, high_cv_above_limit_count=high_cv_above_limit,
                  both_snr_pass_count=len(both_snr_pass), both_snr_pass_reject_only_linearity_count=only_linearity_reject,
                  row_counts={str(wl): dict(Counter(cell['status'] for cell in cells if cell['wl'] == wl)) for wl in range(8)},
                  candidate_cells=[[cell['wl'],cell['bl']] for cell in cells if cell['status']=='candidate'] if completed else [],
                  candidate_metrics=[cell for cell in cells if cell['status']=='candidate'],
                  previous_basis_cells=[cell for cell in cells if cell['bl'] in (2,3)],
                  recomputed_cells=cells,
                  limitations=['Candidate applies only to present state and sampled 0.05/0.1 V basis responses.',
                               'Engineering thresholds are provisional, not validated scientific qualification.',
                               'No programming reachability, pair additivity, selected-mask summation or CIFAR accuracy established.',
                               'Zero ADC may indicate clipping; reject does not prove a physically defective cell.',
                               'Native readback/model conversions are not physical voltage/current calibration.',
                               'Sequential samples need not be independent; standard errors are descriptive.',
                               'Private driver sources checked against reviewed pins and local bytes when available; never Git blobs.',
                               'Hardware cleanup cannot be independently proved from software close_port.'])
    out.mkdir(parents=True, exist_ok=True)
    (out/'independent_cell_map_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    text = ['# Независимый аудит карты ячеек', '',
            f"Сессия `{manifest['session_id']}`: сбор `{summary['status']}`, аудит **{report['audit_status']}** — {checks} проверок, {len(errors)} расхождений.", '',
            f"Проверены {attempted} пар request/result: mode9 {mode_counts[9]}, mode10 {mode_counts[10]}, mode7 {mode_counts[7]}; явных пропусков MVM {skipped}. Пересчитаны native conversions, охват/порядок, DAC, ID, временные отметки, модельные guards, классификация всех 256 ячеек и CSV маска. Проверены result SHA и {len(verified)} исходников по локальным байтам и Git `{commit}`; приватные файлы — согласованные SHA и локальные байты при наличии, без Git.", '',
            f"Классификация: candidate {counts['candidate']}, reject {counts['reject']}, unmeasured {counts['unmeasured']}. Маска пригодна к рассмотрению: **{completed}**. Частичный failed-прогон не экспортирует рабочую маску.", '',
            '| Причина | Число ячеек |', '|---|---:|']
    text += [f'| {reason} | {number} |' for reason,number in sorted(reason_counts.items())]
    text += ['', 'Причины пересекаются: одна ячейка может не пройти несколько критериев.', '',
             f"Название unstable_high_signal не означает, что все эти ячейки шумные: у {high_cv_unavailable} CV не определён из-за неположительного high-сигнала, только у {high_cv_above_limit} измеренный CV превышает порог. Оба SNR-порога прошли {len(both_snr_pass)} ячейки; из них {only_linearity_reject} исключены только по двухуровневой линейности.", '',
             '## Предварительные кандидаты', '',
             '| WL, BL | Сигнал .05 / .1 В, ADC | SNR low / high | CV high | ε / допуск, ADC | ΔG / допуск, % |',
             '|---|---:|---:|---:|---:|---:|']
    for cell in cells:
        if cell['status'] == 'candidate':
            m = cell['metrics']
            text.append(f"| {cell['wl']}, {cell['bl']} | {m['signal_low_adc']:.3f} / {m['signal_high_adc']:.3f} | {m['snr_low']:.2f} / {m['snr_high']:.2f} | {m['cv_high']:.4f} | {m['linearity_epsilon_adc']:+.3f} / {m['linearity_tolerance_adc']:.3f} | {100*m['g_fractional_delta']:+.3f} / {100*m['g_fractional_tolerance']:.3f} |")
    text += ['', '## Ранее использованные BL2/BL3', '',
             'Здесь показаны новые одиночные измерения карты, а не повтор прежнего AB-теста. В таблице отсутствует проверка суммирования.', '',
             '| WL, BL | Сигнал .05 / .1 В, ADC | ε / допуск, ADC | Причины |', '|---|---:|---:|---|']
    for cell in cells:
        if cell['bl'] in (2,3):
            m = cell['metrics']
            if 'signal_low_adc' in m:
                text.append(f"| {cell['wl']}, {cell['bl']} | {m['signal_low_adc']:.3f} / {m['signal_high_adc']:.3f} | {m['linearity_epsilon_adc']:+.3f} / {m['linearity_tolerance_adc']:.3f} | {'; '.join(cell['reasons']) or 'candidate'} |")
            else:
                text.append(f"| {cell['wl']}, {cell['bl']} | — | — | {'; '.join(cell['reasons'])} |")
    text += ['', 'Координаты WL/BL — логические адреса Connector; соответствие физическим ячейкам независимо не калибровалось. Candidate означает предварительный отклик текущего состояния при 0.05/0.1 В по критериям criteria.json. Reject не доказывает физическую неисправность. Нулевой ADC может быть отсечкой. Программируемость, выбранные пары, суммирование комбинаций и CIFAR accuracy не проверены; после записи весов карта требует обновления. Пороги инженерные, не валидированные научные нормы.', '',
             'Аудитор не создаёт Connector и не вызывает аппаратуру. Исходные файлы сессии не изменяются. Воспроизведение после сохранения скрипта в корне репозитория:', '', '```sh',
             f"python -B audit_cell_map.py --source-repo . --session results/cell_maps/{manifest['session_id']} --output-dir results/cell_maps/{manifest['session_id']}", '```']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('audit_status','checks','errors','request_result_pairs','mode9_count','mode10_count','skipped_mvm','classification_counts','reason_counts','maximum_numeric_difference')}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
