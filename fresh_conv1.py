"""Fresh 300-weight acquisition with bounded JIT quiet verification."""
import copy
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import statistics as st
import time
import uuid

import cifar_feedback_adaptive as adaptive
from cifar_feedback_adaptive import AdaptiveRun, measurement_tail
import cifar_feedback_diagnostic as diagnostic
from cifar_feedback_diagnostic import fixed_calibration, decode_measured
from cifar_feedback import fidelity_report
from cifar_memristor import EXPECTED_BOARD, save_json, sha256, slots, tiles, ROOT, local_path, now
from cifar_memristor_analysis import apply_matrix, aggregate
from native_conversions import mvm_output
from cell_calibration import check_sources, LOCK
from read_all import native_connector
from experiment_hardware import launch_provenance
from fresh_conv1_plan import build_route

SIDES = ('positive_cell', 'negative_cell')
RESET_LEG_BOUND = 4 + 55*4 + 56*16
QUIET_ATTEMPT_BOUND = 2*RESET_LEG_BOUND + 3*32
SET_BOUND = 811
HERE = Path(__file__).resolve().parent
CONFIG = HERE/'configs/fresh_conv1.json'
ROUTE = HERE/'weights/fresh_conv1_route.json'
EVIDENCE = HERE/'weights/fresh_conv1_capacity.json'
PINS = {**adaptive.PINS,
    'cifar_feedback_adaptive.py':'a46eb7b94891d1adf48d10f126227e10ecfa696aba61a98f293199a8dcb9f2ba',
    'experiments/unn_rev3_20261014/paths.py':'2c841b4a446a767595c68aad243abd966cee94ffd91f74c3c2823998fdc9ec1c'}
SPEC = {k:copy.deepcopy(diagnostic.SPEC[k]) for k in ('programming_guard','simulator_pulse_width_s',
    'reset_ramp_v','set_ramp_v','feedback_relative_tolerance','feedback_absolute_adc_tolerance',
    'adc_headroom_fraction','raw_part_maximum_bytes','simulator_seed')}
SPEC['programming_guard']['skip_policy'] = 'global_stop_without_impact_or_state_update'
SPEC.update(protocol_id='fresh300-jit-capacity-fixed-gain-v1', review_pending=False,
    target_gain_v_per_weight={'hardware':.25,'simulator':.03}, cutoff_utc='2026-10-16T18:50:00+00:00',
    max_joint_repair_rounds=2, max_candidate_pairs_per_tap=2, max_microtile_width=7,
    maximum_impacts=1702644, maximum_programming_pulses=91770, maximum_programming_decisions=91770,
    capacity_evidence_sha256='f1c109655557e8186d7356250c80194facd7f2e5ab209fdacfe5ab0cb38c2d72',
    reset_policy='single crossing requires five triples; failed confirmation continues remaining ramp; never restart',
    baseline_scope='per_pair_JIT_after_previous_neighbor_writes; joint_gate_after_both_RESET_legs',
    joint_policy='five paired brackets after both legs; at most two repairs; consume a new reset level on each repaired leg',
    set_policy='planned sign(w) leg only; opposite residual outside tolerance is initial_overshoot_no_SET; no reversal',
    quality_failure_policy='collect all actual coefficients; quality is never completion; quiet route exhaustion and technical faults abort',
    decoder='signed_C_actual_rows / fixed_source_gain; no fitted alpha decoding',
    collection_scope='all300_fresh; no ancestor measurements',
    physical_weight_quality=dict(mean_nrmse_max=.20, uncertainty_nrms_max=.20,
        uncertainty='sqrt(sampleSD(D)^2+ADC_LSB^2)/fixed_gain; no sqrt(repeats)',
        significant_target_adc=2, measured_sign_resolution='abs(meanD)>2ADC+2sampleSD(D)'),
    matrix='inherited C2 linear d=0; independent anchors/train/selection; frozen before CIFAR',
    source_gain_is_not_physical_equivalence=True)


def budget(selected, fixture):
    n = sum(len(slots(t)) for _,t in selected)
    attempts = sum(len(s['candidate_pairs']) for _,t in selected for s in slots(t))
    tail = sum(measurement_tail(slots(t), fixture) for _,t in selected)
    return dict(tiles=len(selected), trained_coefficients=n, candidate_pair_attempts=attempts,
        maximum_impacts=attempts*QUIET_ATTEMPT_BOUND+n*SET_BOUND+tail,
        maximum_programming_pulses=attempts*110+n*109, measurement_tail_impacts=tail)


def load_inputs():
    for name, pin in PINS.items():
        if sha256(local_path(name)) != pin: raise ValueError('Pinned source changed: '+name)
    plan = json.loads(CONFIG.read_text(encoding='utf-8'))
    if plan != SPEC: raise ValueError('Protocol differs from reviewed specification')
    evidence_path = EVIDENCE
    if sha256(evidence_path) != SPEC['capacity_evidence_sha256']: raise ValueError('Capacity evidence changed')
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    for name, pin in evidence['historical_leaf_sha256'].items():
        if sha256(local_path(name)) != pin: raise ValueError('Historical capacity source changed: '+name)
    for name, pin in evidence['source_manifest_sha256'].items():
        if sha256(local_path(name)/'manifest.json') != pin: raise ValueError('Capacity manifest changed')
    original = json.loads(local_path('weights/candidate_cifar75_routing.json').read_text(encoding='utf-8'))
    route = json.loads(ROUTE.read_text(encoding='utf-8'))
    if route != build_route(original, evidence): raise ValueError('Frozen route differs from deterministic plan')
    for _,tile in tiles(route): programming_order(slots(tile))
    fixture = json.loads(local_path('weights/prepared_cifar100_conv1.json').read_text(encoding='utf-8'))
    bound = budget(tiles(route), fixture)
    if any(bound[k] != SPEC[k] for k in ('maximum_impacts','maximum_programming_pulses')):
        raise ValueError('Frozen global budget mismatch')
    return plan, route, fixture


class QuietRouteExhausted(RuntimeError): pass
class DeadlineReached(RuntimeError): pass
class CurrentGuardStop(RuntimeError): pass


def utc_now(): return datetime.now(timezone.utc)


def quiet(values, limit):
    return len(values) == 5 and abs(st.mean(values)) <= limit and st.stdev(values) <= limit


def programming_order(active):
    """Only the observed WL3/BL3 -> WL3/BL8 dependency; no general BL8 claim."""
    pending = sorted(active, key=lambda s: s['tap_index']); ordered = []
    before = {s['tap_index'] for s in active if any(
        c[SIDES[0] if s['trained_weight'] >= 0 else SIDES[1]] == [3, 3] for c in s['candidate_pairs'])}
    after = {s['tap_index'] for s in active if any(
        c[side] == [3, 8] for c in s['candidate_pairs'] for side in SIDES)}
    edges = {(a, b) for a in before for b in after if a != b}
    while pending:
        ready = [s for s in pending if not any(b == s['tap_index'] and
                 any(p['tap_index'] == a for p in pending) for a, b in edges)]
        if not ready: raise ValueError('Frozen programming dependency cycle')
        chosen = ready[0]; pending.remove(chosen); ordered.append(chosen)
    return ordered


class FreshRun(AdaptiveRun):
    def __init__(self, *args, clock=None, monotonic=None, global_budget=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.clock = clock or utc_now
        self.monotonic = monotonic or time.monotonic
        self.cutoff = datetime.fromisoformat(self.plan['cutoff_utc'])
        self.monotonic_cutoff = self.monotonic()+max(0., (self.cutoff-self.clock()).total_seconds())
        self.attempt_start = None
        self.global_budget = global_budget

    def impact(self, wl, bl, mode, vol, phase, sign=0, **meta):
        if self.clock() >= self.cutoff or self.monotonic() >= self.monotonic_cutoff:
            self.emit(dict(event='deadline_stop', session_id=self.session_id, source=self.source, next_task_id=self.sequence+1,
                           timestamp_utc=self.clock().isoformat(), impact_issued=False))
            raise DeadlineReached('Fixed UTC/monotonic cutoff')
        if self.attempt_start is not None:
            calls, writes = self.attempt_start
            if self.sequence-calls >= QUIET_ATTEMPT_BOUND:
                raise ValueError('Quiet-attempt impact bound')
            if mode == 7 and vol and self.pulses-writes >= 110:
                raise ValueError('Quiet-attempt write bound')
        if self.global_budget is not None:
            if self.global_budget['impacts'] >= self.global_budget['maximum_impacts']:
                raise ValueError('Global impact budget exhausted')
            if mode == 7 and vol and self.global_budget['pulses'] >= self.global_budget['maximum_programming_pulses']:
                raise ValueError('Global programming budget exhausted')
        start = self.sequence, self.pulses
        try:
            row = super().impact(wl, bl, mode, vol, phase, sign, **meta)
        finally:
            if self.global_budget is not None:
                self.global_budget['impacts'] += self.sequence-start[0]
                self.global_budget['pulses'] += self.pulses-start[1]
        if row is None:
            raise CurrentGuardStop('Programming current guard refused command; stop all')
        return row

    def start_leg(self, cell, meta):
        guard = self.read(cell, 1, 'jit_reset_initial_mode7', **meta)[0]
        obs = self.leg_triple(cell, 'jit_reset_initial', **meta)
        return dict(cell=cell, next_step=0, confirmations=0, initial_guard_task_id=guard['task_id'],
                    last=obs, eligible=False)

    def advance_leg(self, state, meta, force_next=False):
        """A failed confirmation always consumes a new level; never restart the ramp."""
        limit = 2*mvm_output(1, EXPECTED_BOARD, self.converters)
        while True:
            if not force_next and abs(state['last']['voltage_v']) <= limit:
                confirmation = state['confirmations']; state['confirmations'] += 1
                guard = self.read(state['cell'], 1, 'jit_reset_confirm_mode7',
                                  confirmation=confirmation, **meta)[0]
                observations = [self.leg_triple(state['cell'], 'jit_reset_confirm',
                                 confirmation=confirmation, confirmation_repeat=i, **meta) for i in range(5)]
                values = [r['voltage_v'] for r in observations]
                state['eligible'] = quiet(values, limit)
                state['last'] = observations[-1]
                self.append('reset_confirmations.jsonl', dict(**meta, cell=state['cell'],
                    confirmation=confirmation, next_step=state['next_step'], guard_task_id=guard['task_id'],
                    observations=observations, mean_v=st.mean(values), sd_v=st.stdev(values),
                    quiet_limit_v=limit, eligible=state['eligible']))
                if state['eligible']:
                    return True
            # Includes failure of a quiet-looking endpoint's five observations.
            state['eligible'] = False
            if state['next_step'] == len(self.plan['reset_ramp_v']):
                return False
            step = state['next_step']; state['next_step'] += 1
            row = self.impact(*state['cell'], 7, self.plan['reset_ramp_v'][step],
                              'jit_reset', sign=0, reset_step=step, **meta)
            state['last'] = self.leg_triple(state['cell'], 'jit_reset_feedback', reset_step=step, **meta)
            self.append('reset_feedback.jsonl', dict(**meta, cell=state['cell'], step=step,
                        write_task_id=row['task_id'], measurement=state['last']))
            force_next = False

    def paired_snapshot(self, slot, phase, meta):
        guards = [self.read(slot[side], 1, phase+'_mode7', guard_side=side, **meta)[0]['task_id']
                  for side in SIDES]
        samples = []
        for k in range(5):
            values, rows = self.block([slot], [slot], phase, dict(**meta, bracket_repeat=k))
            samples.append(dict(row_sums_v=values, measured_rows=rows, signed_voltage_v=values[0]-values[1]))
        limit = 2*mvm_output(1, EXPECTED_BOARD, self.converters)
        flags = [quiet([r['row_sums_v'][j] for r in samples], limit) for j in range(2)]
        record = dict(**meta, phase=phase, guard_task_ids=guards, samples=samples,
                      quiet_by_side=dict(zip(SIDES, flags)), both_legs_quiet=all(flags), quiet_limit_v=limit)
        self.append('joint_gates.jsonl', record)
        return record

    def reset_pair(self, slot, meta):
        start = self.sequence, self.pulses
        self.attempt_start = start
        record = dict(**meta, positive_cell=slot[SIDES[0]], negative_cell=slot[SIDES[1]],
                      status='failed', eligible=False, joint_gates=[])
        try:
            states = {}
            for side in SIDES:
                state = self.start_leg(slot[side], dict(**meta, side=side)); states[side] = state
                self.advance_leg(state, dict(**meta, side=side))
            for round_index in range(3):
                gate = self.paired_snapshot(slot, 'jit_joint_gate', dict(**meta, joint_round=round_index))
                record['joint_gates'].append(gate)
                if gate['both_legs_quiet']:
                    record['eligible'] = True; break
                if round_index == 2:
                    break
                progressed = False
                for side in SIDES:
                    if not gate['quiet_by_side'][side] and states[side]['next_step'] < 55:
                        # Joint failure must not re-confirm the same crossing indefinitely.
                        self.advance_leg(states[side], dict(**meta, side=side), force_next=True)
                        progressed = True
                if not progressed:
                    break
            record.update(status='completed', legs=states,
                          stop='fresh_joint_quiet' if record['eligible'] else 'bounded_quiet_failure')
            return record
        finally:
            record.update(impact_attempts=self.sequence-start[0], programming_pulse_attempts=self.pulses-start[1])
            self.attempt_start = None
            self.append('reset_pairs.jsonl', record)

    def program_pair(self, slot, meta):
        target = self.plan['target_gain_v_per_weight'][self.source]*slot['trained_weight']
        tolerance = max(.5*mvm_output(1, EXPECTED_BOARD, self.converters), .05*abs(target))
        direction = 1 if slot['trained_weight'] >= 0 else -1
        side = SIDES[0] if direction == 1 else SIDES[1]
        result = dict(**meta, tap=slot['tap_index'], trained_weight=slot['trained_weight'],
            positive_cell=slot[SIDES[0]], negative_cell=slot[SIDES[1]], target_signed_v=target,
            active_set_side=side, feedback_stop_tolerance_v=tolerance, status='failed')
        start = self.sequence, self.pulses
        try:
            result['guard_refresh'] = [self.read(slot[s], 1, 'pair_guard_refresh', side=s, **meta)[0] for s in SIDES]
            values, rows = self.block([slot], [slot], 'pair_initial_feedback', meta)
            current = values[0]-values[1]
            result.update(initial_mvm=rows, initial_signed_v=current)
            stop = 'set_ramp_exhausted'
            for step, voltage in enumerate(self.plan['set_ramp_v']):
                if abs(current-target) <= tolerance:
                    stop = 'feedback_hit'; break
                if direction*(current-target) > tolerance:
                    stop = 'initial_overshoot_no_SET' if step == 0 else 'feedback_overshoot'; break
                row = self.impact(*slot[side], 7, voltage, 'pair_set', sign=1, step=step, **meta)
                values, rows = self.block([slot], [slot], 'pair_set_feedback', dict(**meta, step=step))
                current = values[0]-values[1]
                self.append('set_feedback.jsonl', dict(**meta, step=step, write_task_id=row['task_id'],
                                                     measured_rows=rows, signed_voltage_v=current))
            if abs(current-target) <= tolerance:
                stop = 'feedback_hit'
            elif direction*(current-target) > tolerance and stop != 'initial_overshoot_no_SET':
                stop = 'feedback_overshoot'
            result['post_mode7'] = [self.read(slot[s], 5, 'pair_post_mode7', side=s, **meta) for s in SIDES]
            samples = []
            for k in range(5):
                values, rows = self.block([slot], [slot], 'pair_post_feedback', dict(**meta, repeat=k))
                samples.append(dict(measured_rows=rows, signed_voltage_v=values[0]-values[1]))
            result.update(status='completed', stop=stop, endpoint_signed_v=current, post_feedback=samples,
                          post_mean_signed_v=st.mean(r['signed_voltage_v'] for r in samples),
                          post_sd_signed_v=st.stdev(r['signed_voltage_v'] for r in samples))
            return result
        finally:
            result.update(impact_attempts=self.sequence-start[0], programming_pulse_attempts=self.pulses-start[1])
            self.append('pair_programming.jsonl', result)

    def collect_tile(self, f, tile, fixture):
        active = copy.deepcopy(slots(tile)); meta = dict(filter_id=f['filter_id'], tile_id=tile['tile_id'])
        result = dict(**meta, status='failed', technical_completed=False, pairs=[], reset_attempts=[],
                      programming=[], partials=[], no_imported_measurements=True)
        try:
            used = set(); actual = {}
            for planned in programming_order(active):
                selected = None
                for attempt, candidate in enumerate(planned['candidate_pairs']):
                    if any(tuple(candidate[side]) in used for side in SIDES):
                        raise ValueError('Frozen candidate pair collides with an assigned address')
                    slot = {**planned, **candidate}
                    details = dict(**meta, pair_tap=slot['tap_index'], candidate_attempt=attempt)
                    reset = self.reset_pair(slot, details); result['reset_attempts'].append(reset)
                    if reset['eligible']:
                        selected = slot; break
                if selected is None:
                    raise QuietRouteExhausted(f'No fresh quiet pair for tap {planned["tap_index"]}')
                pair = self.program_pair(selected, details); result['pairs'].append(pair)
                actual[selected['tap_index']] = selected
                used.update(tuple(selected[s]) for s in SIDES)
                for side in SIDES:
                    result['programming'].append(dict(cell=selected[side], tap=selected['tap_index'], side=side,
                        participated_in_SET=side == pair['active_set_side'] and pair['programming_pulse_attempts'] > 0))
            active = [actual[s['tap_index']] for s in active]
            result['effective_route'] = active
            save_json(self.output/'frozen_route.json', dict(slots=active, planned_slots=slots(tile),
                programming_tap_order=[p['tap'] for p in result['pairs']], frozen_before_task_id=self.sequence+1))
            self.emit(dict(event='route_frozen', session_id=self.session_id, **meta, sha256=sha256(self.output/'frozen_route.json'),
                           before_task_id=self.sequence+1))
            self.collect_measurements(f, active, fixture, meta, result)
            result.update(status='completed', technical_completed=True, decode_usable=True)
            return result
        except (Exception, KeyboardInterrupt) as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            result.update(impact_attempts=self.sequence, programming_pulse_attempts=self.pulses,
                          programming_decisions=self.decisions, programming_skipped_current=self.skipped)
            self.log.close(); save_json(self.output/'tile.json', result)

    def collect_measurements(self, f, active, fixture, meta, result):
        """All target-dependent programming finishes before calibration or fresh CIFAR."""
        weights = [s['trained_weight'] for s in active]
        gain = self.plan['target_gain_v_per_weight'][self.source]
        adc = mvm_output(1, EXPECTED_BOARD, self.converters)
        result['pre_readback'] = [self.read(s[side], 3, 'tile_pre_readback', tap=s['tap_index'], side=side, **meta)
                                 for s in active for side in SIDES]
        anchors = self.anchors(active, 'calibration', meta)
        save_json(self.output/'calibration_anchors.json', anchors)
        matrix = self.matrix_calibration(active, anchors, meta); result['matrix'] = matrix
        raw = []; corrected = []
        for row in anchors:
            pairs = [[row[j]['singles'][i]['voltage_v']-row[j]['zero']['voltage_v'] for j in range(2)]
                     for i in range(len(active))]
            raw.append([p[0]-p[1] for p in pairs])
            changed = [apply_matrix(matrix['matrix'], p) for p in pairs]
            corrected.append([p[0]-p[1] for p in changed])
        cal = fixed_calibration(weights, corrected, gain, mvm_output(5, EXPECTED_BOARD, self.converters))
        cal.update(frozen_before_task_id=self.sequence+1, trained_weights=weights,
                   wmax_per_filter=max(abs(s['trained_weight']) for t in f['tiles'] for s in slots(t)))
        save_json(self.output/'frozen_calibration.json', cal); result['calibration'] = cal
        result['physical_fidelity'] = fidelity_report(weights, raw, gain, adc)
        self.emit(dict(event='calibration_frozen', session_id=self.session_id, **meta, before_task_id=self.sequence+1,
            sha256=sha256(self.output/'frozen_calibration.json'), matrix_sha256=sha256(self.output/'frozen_matrix.json')))
        for ci, case in enumerate(fixture['cases']):
            for repeat in range(5):
                for timestep, x in enumerate(case['spikes_t12x75']):
                    chosen = [s for s in active if x[s['tap_index']]]
                    event = dict(**meta, case=ci, original_test_index=case['original_test_index'], repeat=repeat, t=timestep)
                    values, ids = self.block(active, chosen, 'cifar', event)
                    corrected_values = apply_matrix(matrix['matrix'], values)
                    part = dict(**event, active_taps=[s['tap_index'] for s in chosen], measured_rows=ids,
                        raw_row_sums_v=values, corrected_row_sums_v=corrected_values,
                        raw_signed_voltage_v=values[0]-values[1],
                        signed_voltage_v=corrected_values[0]-corrected_values[1],
                        trained_unit_contribution=decode_measured(matrix['matrix'], values, gain))
                    result['partials'].append(part); self.append('partials.jsonl', part)
        final = self.anchors(active, 'final_anchors', meta); result['final_anchors'] = final
        raw_final = [[row[0]['singles'][i]['voltage_v']-row[0]['zero']['voltage_v']-
                      row[1]['singles'][i]['voltage_v']+row[1]['zero']['voltage_v'] for i in range(len(active))]
                     for row in final]
        result['final_physical_fidelity'] = fidelity_report(weights, raw_final, gain, adc)
        result['physical_weight_quality_pass'] = (result['physical_fidelity']['physical_fidelity_pass'] and
                                                   result['final_physical_fidelity']['physical_fidelity_pass'])
        result['post_readback'] = [self.read(s[side], 3, 'tile_post_readback', tap=s['tap_index'], side=side, **meta)
                                  for s in active for side in SIDES]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    for flag in ('run-simulator','run-hardware','dry-run'): mode.add_argument('--'+flag, action='store_true')
    args = parser.parse_args(argv)
    if not (args.run_simulator or args.run_hardware or args.dry_run): parser.print_help(); return 0
    plan, route, fixture = load_inputs()
    source = 'simulator' if args.run_simulator else 'hardware'
    if not args.dry_run and plan['review_pending']:
        raise ValueError('Draft protocol: native execution disabled until source review and commit')
    config, snapshot, hashes = check_sources(source)
    for name in PINS: hashes[name] = sha256(local_path(name))
    for path in (Path(__file__), HERE/'fresh_conv1_plan.py', CONFIG, ROUTE, EVIDENCE):
        hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
    if (ROOT/'experiment.py').is_file(): hashes['experiment.py'] = sha256(ROOT/'experiment.py')
    description = dict(protocol_id=plan['protocol_id'], source=source, source_sha256=hashes,
        hardware_access=source=='hardware' and not args.dry_run, **budget(tiles(route), fixture),
        fixed_decoder_gain_v_per_weight=plan['target_gain_v_per_weight'][source],
        no_imported_measurements=True, cutoff_utc=plan['cutoff_utc'], physical_pulse_width_s=None)
    if args.dry_run: print(json.dumps(description)); return 0
    if utc_now() >= datetime.fromisoformat(plan['cutoff_utc']):
        raise DeadlineReached('Acquisition cutoff expired before open_port')
    lock = local_path(LOCK); locked = False; connector = None; manifest = None; output = None
    results = []; current = None
    if source == 'hardware':
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open('x', encoding='utf-8') as stream: stream.write(f'Fresh Conv1 {now()}\n')
        locked = True
    try:
        sid = str(uuid.uuid4()); output = local_path('results/cifar_fresh_conv1')/sid
        output.mkdir(parents=True)
        save_json(output/'protocol.json', plan); save_json(output/'planned_route.json', route)
        (output/'board.ini').write_bytes(snapshot)
        manifest = dict(session_id=sid, status='failed', started_utc=now(), invocation=launch_provenance(argv), **description)
        save_json(output/'start_manifest.json', manifest)
        if utc_now() >= datetime.fromisoformat(plan['cutoff_utc']):
            raise DeadlineReached('Cutoff reached before native backend initialization')
        if source == 'hardware':
            connector = native_connector(config)
        else:
            from experiments.unn_rev3_20261014.paths import use_local_dependencies
            use_local_dependencies()
            import numpy as np
            from manager.board import Connector
            from simulator.src import create_crossbar_array
            random.seed(plan['simulator_seed']); np.random.seed(plan['simulator_seed'])
            serial = output/'array'; create_crossbar_array(str(serial), 32, 8)
            manifest['array_initial_sha256'] = sha256(serial.with_suffix('.cb'))
            connector = Connector(silent=True, logger=logging.getLogger(__name__), cb_type='simulator',
                board_type='rp5_fpga_python', task_format='SI', config=config, crossbar_serial=str(serial))
        from manager.service import converters
        if utc_now() >= datetime.fromisoformat(plan['cutoff_utc']):
            raise DeadlineReached('Cutoff reached before open_port')
        if not connector.open_port(): raise ConnectionError('Connector.open_port returned false')
        global_budget = dict(impacts=0, pulses=0, maximum_impacts=plan['maximum_impacts'],
                             maximum_programming_pulses=plan['maximum_programming_pulses'])
        for f, tile in tiles(route):
            folder = output/'tiles'/f"f{f['filter_id']}_t{tile['tile_id']}"; folder.mkdir(parents=True)
            per = budget([(f,tile)], fixture)
            tile_plan = {**plan, **per, 'maximum_programming_decisions':per['maximum_programming_pulses']}
            current = FreshRun(connector, converters, folder, sid, tile_plan, source, global_budget=global_budget)
            result = current.collect_tile(f, tile, fixture); results.append(result); current = None
            if sum(r['impact_attempts'] for r in results) > plan['maximum_impacts'] or sum(r['programming_pulse_attempts'] for r in results) > plan['maximum_programming_pulses']:
                raise ValueError('Global acquisition budget exceeded')
            print(f"filter {f['filter_id']} tile {tile['tile_id']}: acquired {len(result['effective_route'])} weights; quality={result['physical_weight_quality_pass']}", flush=True)
        summary = aggregate(results, route, fixture)
        represented = sorted({tuple(s[side]) for r in results for s in r['effective_route'] for side in SIDES})
        summary.update(status='completed', technical_completed=True, acquisition_complete=True,
            decoded_conv_bn_lif_complete=len(summary['trajectories'])==200,
            fresh_trained_coefficients=sum(len(r['effective_route']) for r in results),
            completed_microtiles=len(results), no_imported_measurements=True,
            physical_weight_quality_pass=all(r['physical_weight_quality_pass'] for r in results),
            represented_addresses=[list(c) for c in represented], represented_address_count=len(represented),
            address_coverage_is_not_weight_qualification=True,
            total_impacts=sum(r['impact_attempts'] for r in results),
            total_pulses=sum(r['programming_pulse_attempts'] for r in results))
        if summary['fresh_trained_coefficients'] != 300 or not summary['decoded_conv_bn_lif_complete']:
            raise ValueError('Incomplete 300-weight/2400-point aggregation')
        save_json(output/'summary.json', summary); manifest.update(status='completed', acquisition_complete=True)
    except (Exception, KeyboardInterrupt) as exc:
        if manifest is None: raise
        manifest['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if connector is not None:
            try:
                if not connector.close_port(): raise ConnectionError('Connector.close_port returned false')
            except Exception as exc:
                if manifest is not None: manifest.update(status='failed', close_error=f'{type(exc).__name__}: {exc}')
        if manifest is not None:
            if current is not None: current.log.close()
            manifest.update(completed_microtiles=len(results),
                total_impacts=sum(r['impact_attempts'] for r in results)+(current.sequence if current else 0),
                total_pulses=sum(r['programming_pulse_attempts'] for r in results)+(current.pulses if current else 0),
                finished_utc=now(), result_sha256={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file()})
            save_json(output/'manifest.json', manifest)
        if locked: lock.unlink()
    print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(), status=manifest['status'])))
    return 0 if manifest['status']=='completed' else 2


if __name__ == '__main__': raise SystemExit(main())
