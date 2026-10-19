"""Bounded all-cell survey through public MemriBoard Connector; explicit launch only."""
import argparse
import csv
import html
import importlib.metadata
import json
import logging
import math
import platform
from pathlib import Path
import random
import statistics
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cell_calibration import Discovery, CELLS, WRITE_GUARD, SIM_INI, LOCK, check_sources, stats
from read_all import EXPECTED_BOARD, local_path, native_connector, now, sha256, save_json
from experiment_hardware import launch_provenance

DIRECTORY = 'experiments/survey256_20261019'
CONFIG = DIRECTORY + '/protocol.json'
PINS = {
    'cell_calibration.py': 'eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8',
    'read_all.py': 'b30ae2f4dd3cd166dbc5cefb07b558127de2bbdba1f6674c40b1b45f4dd74d5f',
    'native_conversions.py': '2ad2e9c057bb14979b083a6e846e8c417c080a6c5273d848e31f893b7970253e',
    'experiment_hardware.py': '059dc4f827943a9df631631f52e776b57182afc9d36db0c1ad5246427f73548c',
}
SPEC = dict(
    protocol_id='survey256-independent-polarities-v1-20261019',
    cells=[[wl, bl] for wl, bl in CELLS],
    programming_guard=WRITE_GUARD,
    ramp_v=[round(.3+.05*i, 2) for i in range(55)],
    reset_endpoint_q=.005, set_endpoint_q=.05,
    endpoint_policy='check after at least one admitted pulse; SET independent of RESET endpoint',
    checkpoints=['initial_array', 'local_pre', 'post_reset', 'post_set', 'final_array'],
    checkpoint_reads=3, mvm_levels_v=[0., .05, .1, 0.], mvm_repeats=3,
    maximum_programming_decisions=28160, maximum_programming_pulses=28160,
    maximum_impacts=47360, simulator_seed=20261019, simulator_pulse_width_s=.000125,
    hardware_submitted_pulse_width_s=0., hardware_pulse_width_s=None,
    current_model_limit_a=.005, adc_headroom_fraction=.8,
    model_guard_policy='skip checkpoint MVM with reason; retain valid mode7 reads',
    invalid_measurement_policy='global stop; retain raw and mark missing cells',
    baseline_scope='full array before any programming, plus immediate pre-own-programming',
    qualification='one-cycle sampled response; not endurance, continuous range or CIFAR qualification',
)


def validate(plan):
    if plan != SPEC:
        raise ValueError('Protocol differs from frozen survey256 specification')


def execution_environment():
    packages = {}
    for name in ('numpy', 'SQLAlchemy', 'greenlet'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=packages)


def metric_checkpoint(state):
    """Descriptive diagnostics, never an eligibility mask or weight correction."""
    if state is None or state.get('status') != 'completed':
        return {'status': 'missing_or_incomplete', 'error': state.get('error') if state else None}
    result = dict(status='measured', q=state['q'], conductance_siemens=state['conductance_siemens'],
                  mvm_status=state['mvm_status'])
    if state['mvm_status'] != 'measured':
        result['reason'] = state.get('mvm_skip_reason')
        return result
    samples = [[r['raw_adc'] for r in state['mvm'] if r['position'] == pos] for pos in range(4)]
    means = [statistics.mean(v) for v in samples]
    deviations = [statistics.stdev(v) for v in samples]
    low, high = means[1]-means[0], means[2]-means[0]
    noise = math.hypot(deviations[0], deviations[2])
    epsilon = means[2]-2*means[1]+means[0]
    se = math.sqrt((deviations[2]**2+4*deviations[1]**2+deviations[0]**2)/3)
    tolerance = max(2., .05*abs(high), 3*se)
    zero_tolerance = max(2., 3*math.sqrt((deviations[0]**2+deviations[3]**2)/3))
    result.update(zero_before_adc=means[0], zero_after_adc=means[3],
                  zero_shift_adc=means[3]-means[0], signal_low_adc=low, signal_high_adc=high,
                  zero_shift_tolerance_adc=zero_tolerance,
                  zero_shift_flag=abs(means[3]-means[0])>zero_tolerance,
                  noise_high_adc=noise, snr_high=high/max(1., noise),
                  cv_high=noise/abs(high) if high else None,
                  two_level_epsilon_adc=epsilon, two_level_tolerance_adc=tolerance,
                  two_level_flag=abs(epsilon)>tolerance)
    return result


def build_map(summary):
    existing = {(c['wl'], c['bl']): c for c in summary['cells']}
    initial = {(s['wl'], s['bl']): s for s in summary['initial_array']}
    final = {(s['wl'], s['bl']): s for s in summary['final_array']}
    for state in summary.get('failed_checkpoints', []):
        if state['role'] == 'initial_array':
            initial[state['wl'], state['bl']] = state
        if state['role'] == 'final_array':
            final[state['wl'], state['bl']] = state
    rows = []
    for wl, bl in CELLS:
        cell = existing.get((wl, bl), {})
        states = cell.get('states', {})
        pre, terminal, post = initial.get((wl, bl)), states.get('post_set'), final.get((wl, bl))
        row = dict(wl=wl, bl=bl, status='not_measured', reasons=[],
                   initial=metric_checkpoint(pre), terminal=metric_checkpoint(terminal),
                   local_pre=metric_checkpoint(states.get('local_pre')),
                   post_reset=metric_checkpoint(states.get('post_reset')),
                   final=metric_checkpoint(post), programming=cell.get('programming', {}),
                   automatically_excluded=False, programming_repeatability_qualified=False)
        observed = [s['q']['mean'] for s in [pre, *states.values(), post]
                    if s and s.get('status') == 'completed']
        row['observed_q_min'] = min(observed) if observed else None
        row['observed_q_max'] = max(observed) if observed else None
        row['observed_q_scope'] = 'minimum and maximum checkpoint means; not full pulse trajectory or continuous range'
        if pre or cell or post:
            row['status'] = 'incomplete'
        if not cell:
            row['reasons'].append('programming_not_reached_before_global_stop')
        if cell.get('error'):
            row['reasons'].append(cell['error'])
        if not post or post.get('status') != 'completed':
            row['reasons'].append('final_array_checkpoint_missing_or_incomplete')
        for phase, details in cell.get('programming', {}).items():
            if details['admitted_pulses'] == 0:
                row['reasons'].append(phase+'_no_admitted_excitation')
            if details['current_skips']:
                row['reasons'].append(phase+'_current_limited')
            if not details['endpoint_reached']:
                row['reasons'].append(phase+'_endpoint_not_reached_within_budget')
        if terminal and post and terminal.get('status') == post.get('status') == 'completed':
            row['status'] = 'measured'
            before, after = terminal['conductance_siemens'], post['conductance_siemens']
            delta = (after['mean']-before['mean'])/before['mean']
            tolerance = max(.05, 3*math.sqrt(before['sd']**2/3+after['sd']**2/3)/before['mean'])
            row.update(final_g_fractional_delta=delta, final_g_tolerance=tolerance,
                       final_g_drift_flag=abs(delta)>tolerance)
            if row['final_g_drift_flag']:
                row['reasons'].append('conductance_changed_after_later_operations')
            a, b = row['terminal'], row['final']
            if a.get('mvm_status') == b.get('mvm_status') == 'measured':
                row['final_signal_delta_adc'] = b['signal_high_adc']-a['signal_high_adc']
                bound = max(2., .05*abs(a['signal_high_adc']),
                            3*math.hypot(a['noise_high_adc'], b['noise_high_adc'])/math.sqrt(3))
                row['final_signal_drift_tolerance_adc'] = bound
                if abs(row['final_signal_delta_adc']) > bound:
                    row['reasons'].append('mvm_response_changed_after_later_operations')
        for name in ('initial', 'local_pre', 'post_reset', 'terminal', 'final'):
            info = row[name]
            if info.get('error'):
                row['reasons'].append(name+'_error: '+info['error'])
            if info.get('mvm_status') == 'skipped_model_guard':
                row['reasons'].append(name+'_mvm_model_guard')
            if info.get('mvm_status') == 'measured':
                if info['signal_high_adc'] <= 0 or info['snr_high'] < 5:
                    row['reasons'].append(name+'_weak_or_nonpositive_signal')
                if info['two_level_flag']:
                    row['reasons'].append(name+'_two_level_nonlinearity')
                if info['zero_shift_flag']:
                    row['reasons'].append(name+'_zero_baseline_shift')
                if info['cv_high'] is None or info['cv_high'] > .05:
                    row['reasons'].append(name+'_repeat_variation')
        if row['status'] == 'measured' and row['reasons']:
            row['status'] = 'measured_with_flags'
        rows.append(row)
    return rows


def save_map(output, summary):
    rows = build_map(summary)
    save_json(output/'cell_map.json', dict(cells=rows, classification='descriptive, not acceptance mask',
                                          source=summary['source'], session_status=summary['status']))
    with (output/'cell_map.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                          for k, v in row.items()} for row in rows)
    colors = {'not_measured':'#aaa', 'incomplete':'#e88989', 'measured':'#86cf9a', 'measured_with_flags':'#ffd37a'}
    content = ['<!doctype html><meta charset="utf-8"><title>All256 survey</title>',
               '<h1>All256 survey: '+html.escape(summary['source'])+'</h1>',
               '<p>Green: measured without diagnostic flags. Amber: measured with flags. Red: incomplete. Gray: not measured. Flags do not automatically exclude cells. One cycle does not qualify programmability. No causal claim about disturbance.</p>',
               '<table><tr><th>WL / BL</th>'+''.join('<th>'+str(bl)+'</th>' for bl in range(32))+'</tr>']
    for wl in range(8):
        content.append('<tr><th>'+str(wl)+'</th>')
        for row in rows[wl*32:(wl+1)*32]:
            label = str(len(row['reasons'])) if row['status'].startswith('measured') else '?'
            content.append('<td style="background:'+colors[row['status']]+';padding:5px"><a href="#c'+str(wl)+'-'+str(row['bl'])+'">'+label+'</a></td>')
        content.append('</tr>')
    content.append('</table>')
    for row in rows:
        content.append('<details id="c'+str(row['wl'])+'-'+str(row['bl'])+'"><summary>WL'+str(row['wl'])+'/BL'+str(row['bl'])+': '+row['status']+'</summary><pre>'+html.escape(json.dumps(row, indent=2, ensure_ascii=False))+'</pre></details>')
    (output/'cell_map.html').write_text('\n'.join(content), encoding='utf-8')


class Survey(Discovery):
    def __init__(self, connector, converters, output, session_id, plan, source):
        validate(plan)
        super().__init__(connector, converters, output, session_id, plan, source)
        self.summary.update(protocol_id=plan['protocol_id'], physical_pulse_width_s=None,
                            checkpoint_count=0, qualification=plan['qualification'])

    def checkpoint(self, wl, bl, role):
        self.summary['checkpoint_count'] += 1
        state = dict(wl=wl, bl=bl, role=role, checkpoint_id=self.summary['checkpoint_count'],
                     status='failed', reads=[], mvm=[], mvm_status='not_attempted')
        try:
            for repeat in range(3):
                row = self.impact(wl, bl, 7, 0., role+'_read', repeat=repeat, checkpoint_id=state['checkpoint_id'])
                state['reads'].append({k:row[k] for k in ('task_id', 'raw_adc', 'q', 'conductance_siemens', 'native_r')})
            state['q'] = stats([r['q'] for r in state['reads']])
            state['conductance_siemens'] = stats([r['conductance_siemens'] for r in state['reads']])
            # Upper observed G is a conservative model check, not a physical guarantee.
            g = max(r['conductance_siemens'] for r in state['reads'])
            state['model_prediction'] = dict(current_a=.1*g, output_v=.1*g*EXPECTED_BOARD['sum_gain'], calibrated=False)
            if .1*g > self.plan['current_model_limit_a'] or .1*g*EXPECTED_BOARD['sum_gain'] > .8*EXPECTED_BOARD['vol_ref_adc']:
                state.update(mvm_status='skipped_model_guard', mvm_skip_reason='predicted_single_cell_current_or_headroom')
                self.emit(dict(event='checkpoint_mvm_skipped', **state))
            else:
                for repeat in range(3):
                    for position, level in enumerate(self.plan['mvm_levels_v']):
                        vector = [level if column == bl else 0. for column in range(32)]
                        row = self.impact(wl, bl, 10, vector, role+'_mvm', repeat=repeat,
                                          position=position, nominal_v=level, checkpoint_id=state['checkpoint_id'])
                        state['mvm'].append({k:row[k] for k in ('task_id', 'raw_adc', 'output_voltage_v', 'repeat', 'position', 'nominal_v')})
                state['mvm_status'] = 'measured'
            state['status'] = 'completed'
        except (Exception, KeyboardInterrupt) as exc:
            state['error'] = f'{type(exc).__name__}: {exc}'
            self.summary.setdefault('failed_checkpoints', []).append(state)
            raise
        finally:
            self.append('checkpoints.jsonl', state)
        return state

    def program(self, cell, phase, sign):
        details = dict(admitted_pulses=0, current_skips=0, decisions=0, endpoint_reached=False,
                       stop_reason='ramp_exhausted', terminal_q=None)
        cell['programming'][phase] = details
        for step, value in enumerate(self.plan['ramp_v']):
            before = self.pulses, self.decisions, self.skipped
            try:
                row = self.impact(cell['wl'], cell['bl'], 7, value, phase, sign=sign, step=step)
            finally:
                details['admitted_pulses'] += self.pulses-before[0]
                details['decisions'] += self.decisions-before[1]
                details['current_skips'] += self.skipped-before[2]
            if row is None:
                continue
            details['terminal_q'] = row['q']
            reached = row['q'] <= self.plan['reset_endpoint_q'] if sign == 0 else row['q'] >= self.plan['set_endpoint_q']
            if reached:
                details.update(endpoint_reached=True, stop_reason='single_response_endpoint_after_admitted_pulse')
                break
        if details['admitted_pulses'] == 0:
            details['stop_reason'] = 'all_decisions_current_blocked'

    def run(self):
        try:
            if not self.connector.open_port():
                raise ConnectionError('Connector.open_port returned false')
            for wl, bl in CELLS:
                self.summary['initial_array'].append(self.checkpoint(wl, bl, 'initial_array'))
            for wl, bl in CELLS:
                cell = dict(wl=wl, bl=bl, states={}, programming={}, status='failed')
                self.summary['cells'].append(cell)
                try:
                    cell['states']['local_pre'] = self.checkpoint(wl, bl, 'local_pre')
                    self.program(cell, 'reset', 0)
                    cell['states']['post_reset'] = self.checkpoint(wl, bl, 'post_reset')
                    self.program(cell, 'set', 1)
                    cell['states']['post_set'] = self.checkpoint(wl, bl, 'post_set')
                    cell['status'] = 'survey_completed'
                except (Exception, KeyboardInterrupt) as exc:
                    cell['error'] = f'{type(exc).__name__}: {exc}'
                    raise
                finally:
                    self.append('cell_results.jsonl', cell)
                print(f'survey256 WL{wl}/BL{bl}: bounded polarities complete', flush=True)
            for wl, bl in CELLS:
                self.summary['final_array'].append(self.checkpoint(wl, bl, 'final_array'))
            self.summary['status'] = 'completed'
        except (Exception, KeyboardInterrupt) as exc:
            self.summary['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            try:
                if not self.connector.close_port():
                    raise ConnectionError('Connector.close_port returned false')
            except Exception as exc:
                self.summary.update(status='failed', close_error=f'{type(exc).__name__}: {exc}')
            self.log.close()
            self.summary.update(impact_attempts=self.sequence, programming_pulse_attempts=self.pulses,
                                programming_decisions=self.decisions, programming_skipped_current=self.skipped,
                                finished_utc=now())
            save_json(self.output/'summary.json', self.summary)
            save_map(self.output, self.summary)
        return self.summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for mode in ('run-simulator', 'run-hardware', 'dry-run'):
        modes.add_argument('--'+mode, action='store_true')
    args = parser.parse_args(argv)
    if not (args.run_simulator or args.run_hardware or args.dry_run):
        parser.print_help()
        return 0
    source = 'simulator' if args.run_simulator else 'hardware'
    plan_bytes = local_path(CONFIG).read_bytes()
    plan = json.loads(plan_bytes.decode('utf-8-sig'))
    validate(plan)
    for name, expected in PINS.items():
        if sha256(local_path(name)) != expected:
            raise ValueError('Unreviewed helper: '+name)
    config, snapshot, hashes = check_sources(source)
    for name in [*PINS, CONFIG, DIRECTORY+'/experiment.py']:
        hashes[name] = sha256(local_path(name))
    description = dict(source=source, protocol=plan, source_sha256=hashes,
                       hardware_access=False if args.dry_run else source == 'hardware')
    if args.dry_run:
        print(json.dumps(description))
        return 0
    lock = local_path(LOCK)
    locked = False
    if source == 'hardware':
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open('x', encoding='utf-8') as stream:
            stream.write('survey256 '+now()+'\n')
        locked = True
    try:
        sid = str(uuid.uuid4())
        output = local_path('results/survey256_20261019')/sid
        output.mkdir(parents=True, exist_ok=False)
        (output/'protocol.json').write_bytes(plan_bytes)
        (output/'board.ini').write_bytes(snapshot)
        manifest = dict(session_id=sid, status='failed', started_utc=now(),
                        invocation=launch_provenance(argv), **description)
        save_json(output/'start_manifest.json', manifest)
        try:
            if source == 'hardware':
                connector = native_connector(config)
            else:
                from experiments.unn_rev3_20261014.paths import use_local_dependencies
                use_local_dependencies()
                import numpy as np
                from manager.board import Connector
                from simulator.src import create_crossbar_array
                random.seed(plan['simulator_seed'])
                np.random.seed(plan['simulator_seed'])
                serial = output/'array'
                (output/'simulator_settings.ini').write_bytes(local_path(SIM_INI).read_bytes())
                create_crossbar_array(str(serial), 32, 8)
                manifest['array_initial_sha256'] = sha256(serial.with_suffix('.cb'))
                connector = Connector(silent=True, logger=logging.getLogger(__name__), cb_type='simulator',
                                      board_type=config.get('board', 'board_type'), task_format='SI',
                                      config=config, crossbar_serial=str(serial))
            from manager.service import converters
            manifest['execution_environment'] = execution_environment()
            result = Survey(connector, converters, output, sid, plan, source).run()
            for key in ('status', 'impact_attempts', 'programming_pulse_attempts', 'programming_decisions',
                        'programming_skipped_current', 'error'):
                if key in result:
                    manifest[key] = result[key]
        except (Exception, KeyboardInterrupt) as exc:
            manifest['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            if not (output/'summary.json').exists():
                unavailable = dict(source=source, status='failed', cells=[], initial_array=[], final_array=[],
                                   error=manifest.get('error', 'Acquisition did not produce a summary'))
                save_json(output/'summary.json', unavailable)
                save_map(output, unavailable)
            manifest['finished_utc'] = now()
            manifest['result_sha256'] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
            save_json(output/'manifest.json', manifest)
        print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(), status=manifest['status'])))
        return 0 if manifest['status'] == 'completed' else 2
    finally:
        if locked:
            lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
