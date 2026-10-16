"""Fresh 300-weight no-BL8 experiment; immutable JIT acquisition helpers."""
import argparse
import copy
from datetime import datetime
import json
import logging
from pathlib import Path
import random
import uuid

import fresh_conv1 as inherited
from fresh_conv1 import (FreshRun, DeadlineReached, SIDES, budget, utc_now,
    check_sources, LOCK, native_connector, launch_provenance, programming_order)
from fresh_conv1_plan import build_route
from cifar_memristor import (ROOT, local_path, now, save_json, sha256, slots, tiles)
from cifar_memristor_analysis import aggregate

HERE = Path(__file__).resolve().parent
CONFIG = HERE/'configs/fresh_conv1_no_bl8.json'
ROUTE = HERE/'weights/fresh_conv1_no_bl8_route.json'
EVIDENCE = HERE/'weights/fresh_conv1_capacity.json'
PINS = {**inherited.PINS,
    'fresh_conv1.py':'031150dbb5cfe285e5c3f8edc18757978f145ecb0b8528126ca4957371d60cfb',
    'fresh_conv1_plan.py':'465d9520acea8aecbb175e29aa18719b1787d17b6e4a54ba8c5577648a877aab',
    'configs/fresh_conv1.json':'5b47aff5d26245f8ab8aa742473c5b097d7722fe929d7c8eeaa558c935260922',
    'weights/fresh_conv1_route.json':'f3bc1279ec53ef76f364c2cb02ecbc0b32b2f1329bba869cf308e0edf91377f9'}
SPEC = copy.deepcopy(inherited.SPEC)
SPEC.update(protocol_id='fresh300-jit-capacity-no-bl8-fixed-gain-v1', review_pending=False,
    excluded_physical_bl=[8], maximum_impacts=1686292,
    maximum_programming_pulses=91000, maximum_programming_decisions=91000,
    exclusion_scope='all primary and fallback legs; physical BL8, not logical tap8',
    exclusion_is_not_dead_cell_diagnosis=True)


def planned_route(original, evidence):
    filtered = copy.deepcopy(evidence)
    filtered['cells'] = [cell for cell in evidence['cells'] if cell['cell'][1] != 8]
    route = build_route(original, filtered)
    route.update(excluded_physical_bl=[8], exclusion_is_not_dead_cell_diagnosis=True)
    return route


def load_inputs():
    for name, pin in PINS.items():
        if sha256(local_path(name)) != pin: raise ValueError('Pinned source changed: '+name)
    # Preserve every historical leaf/manifest validation in the frozen loader.
    # It returns only plans and inputs, never acquired ancestor contributions.
    _, _, fixture = inherited.load_inputs()
    plan = json.loads(CONFIG.read_text(encoding='utf-8'))
    if plan != SPEC: raise ValueError('Protocol differs from reviewed specification')
    evidence = json.loads(EVIDENCE.read_text(encoding='utf-8'))
    original = json.loads(local_path('weights/candidate_cifar75_routing.json').read_text(encoding='utf-8'))
    route = json.loads(ROUTE.read_text(encoding='utf-8'))
    if route != planned_route(original, evidence): raise ValueError('Frozen no-BL8 route differs from deterministic plan')
    for _, tile in tiles(route):
        programming_order(slots(tile))
        for slot in slots(tile):
            for pair in slot['candidate_pairs']:
                if any(pair[side][1] == 8 for side in SIDES): raise ValueError('Excluded physical BL8')
    bound = budget(tiles(route), fixture)
    if (bound['tiles'], bound['trained_coefficients'], bound['candidate_pair_attempts'],
        len(route['primary_distinct_addresses'])) != (44, 300, 530, 108):
        raise ValueError('No-BL8 coverage contract mismatch')
    if any(bound[k] != SPEC[k] for k in ('maximum_impacts','maximum_programming_pulses')):
        raise ValueError('Frozen global budget mismatch')
    return plan, route, fixture


class NoBL8Run(FreshRun):
    def impact(self, wl, bl, mode, vol, phase, sign=0, **meta):
        if mode == 7 and bl == 8:
            raise ValueError('Excluded physical BL8 mode7 command')
        if mode == 10 and (len(vol) != 32 or vol[8] != 0):
            raise ValueError('Excluded physical BL8 mode10 vector')
        return super().impact(wl, bl, mode, vol, phase, sign, **meta)


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
        no_imported_measurements=True, excluded_physical_bl=[8], cutoff_utc=plan['cutoff_utc'], physical_pulse_width_s=None)
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
        sid = str(uuid.uuid4()); output = local_path('results/cifar_fresh_conv1_no_bl8')/sid
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
            current = NoBL8Run(connector, converters, folder, sid, tile_plan, source, global_budget=global_budget, clock=utc_now)
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
