"""Post-v2 Conv1-all-64 BoardSimulator runner. Frozen final; parent stage authorization required.

No simulator import in verify/dry tests. Execution requires --execute + an exact-hash
approval JSON, and refuses interrupted filter replay, stale inputs or failed gates.
Loads Conv1 weights from a numpy file (conv1_weights.npy) only -- no torch.load, so it
runs on remote hosts with torch 1.13 (no weights_only kwarg) or no torch at all.
"""
import argparse
import configparser
import importlib.util
import json
import os
import pathlib
import pickle
import random
import signal
import sys
import time
import types
import numpy as np
from extension_core import TILES, assemble, metric, predrive, require, sha, summarize, validation_gate

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get('SNN_ROOT', '/mnt/d/projects/snn-mnist'))
UP = pathlib.Path(os.environ.get('MEMRIBOARD_UP', '/mnt/d/projects/memristor-aware-twd-snn/memristor_hw/MemriBoard'))
V2 = pathlib.Path(os.environ.get('V2_DIR', str(ROOT / '.tmp/openclaw-spikes/memriboard-conv1-v2-20261007')))
WEIGHTS = pathlib.Path(os.environ.get('CONV1_WEIGHTS_NPY', str(HERE / 'conv1_weights.npy')))
P = json.loads((HERE / 'PROTOCOL.json').read_text())
STAGES = {'pilot': list(range(1)), 'small': list(range(1, 4)),
          'medium': list(range(4, 16)), 'remaining': list(range(16, 64))}
SOURCE = ('runner.py', 'extension_core.py', 'PROTOCOL.json')


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def emit(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def check_inputs():
    require(sha(V2 / 'PROTOCOL.json') == P['ancestor_v2_protocol_sha256'], 'v2 protocol changed')
    require(sha(V2 / 'pilot.py') == P['ancestor_v2_runner_sha256'], 'v2 runner changed')
    paths = {'cifar_train': ROOT / 'data/cifar-100-python/train',
             'core': UP / 'MemriNeurons/cores.py', 'simulator': UP / 'simulator/src.py',
             'converter': UP / 'manager/service/converters.py',
             'simulator_config': UP / 'simulator/simulator_settings.ini'}
    for k, path in paths.items():
        require(sha(path) == P['inputs_sha256'][k], 'input changed: ' + k)
    require(sha(WEIGHTS) == P['conv1_weights_npy_sha256'], 'conv1 weights npy changed')
    return WEIGHTS


def source_hashes():
    return {name: sha(HERE / name) for name in SOURCE}


def approval(stage, path):
    """Parent-owned exact hash authorization, not a runner-created freeze/signature."""
    data = json.loads(path.read_text())
    require(data.get('decision') == 'APPROVED_BY_PARENT' and data.get('stage') == stage,
            'stage not explicitly approved')
    require(data.get('source_sha256') == source_hashes(), 'approval source hashes mismatch')
    require(data.get('checkpoint_sha256') == P['checkpoint_sha256'], 'approval checkpoint mismatch')
    require(data.get('filters') == STAGES[stage], 'approval filter set mismatch')
    require(data.get('no_concurrent_training_confirmed') is True,
            'parent has not confirmed a training-quiescent execution window')
    return sha(path)


def patch_rows():
    """Exact frozen v2 deterministic CIFAR-100 *training-file* split, no test access."""
    with (ROOT / 'data/cifar-100-python/train').open('rb') as f:
        raw = pickle.load(f, encoding='bytes')
    labels = np.asarray(raw[b'fine_labels'])
    selected = np.flatnonzero(np.isin(labels, [0, 1, 2, 3, 5, 8, 13, 14, 17, 19]))
    filtered = labels[selected]; rng = np.random.default_rng(20261006); val = []
    for cls in np.unique(filtered):
        pool = np.flatnonzero(filtered == cls)
        val.extend(rng.permutation(pool)[:round(len(pool) * .1)].tolist())
    val = sorted(val); mask = np.ones(len(selected), bool); mask[val] = False
    train = np.flatnonzero(mask).tolist()
    import hashlib
    digest = hashlib.sha256(json.dumps({'train_indices': train, 'val_indices': val},
                  sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(digest == P['canonical_split_digest'] and len(train) == 4500 and len(val) == 500,
            'canonical split mismatch')
    rows = []
    for split, indices, positions in (('train', train, P['train_split_positions']),
                                      ('val', val, P['val_split_positions'])):
        for pos in positions:
            fi = indices[pos]; oi = int(selected[fi]); img = raw[b'data'][oi].reshape(3, 32, 32).astype(np.float32) / 255
            for top, left in P['spatial_top_left']:
                px = img[:, top:top + 5, left:left + 5].reshape(75)
                fire = np.floor((1 - px) * (P['T'] - 1)).astype(int).clip(0, P['T'] - 1)
                for t in P['spike_times']:
                    x = ((px >= P['x_min']) & (fire == t)).astype(float)
                    rows.append((dict(split=split, split_position=pos, filtered_index=fi,
                              original_train_index=oi, top=top, left=left, t=t), x))
    require(len(rows) == 256, 'fixed patch count mismatch')
    return rows, digest


def simulator_api():
    """Import only local BoardSimulator via the frozen v2 headless adapter."""
    spec = importlib.util.spec_from_file_location('extension_converters', UP / 'manager/service/converters.py')
    cv = importlib.util.module_from_spec(spec); spec.loader.exec_module(cv)
    manager = types.ModuleType('manager'); manager.__path__ = []
    app = types.ModuleType('manager.app'); service = types.ModuleType('manager.service')
    for a, b in [('d2v', 'convert_dac_to_volt'), ('v2d', 'convert_volt_to_dac'),
                 ('a2r', 'convert_adc_to_res'), ('r2a', 'convert_res_to_adc'),
                 ('r2w', 'convert_res_to_weight'), ('w2r', 'convert_weight_to_res'),
                 ('a2v', 'convert_adc_to_volt'), ('a2c', 'convert_adc_to_current')]:
        setattr(service, a, getattr(cv, b))
    cfg = configparser.ConfigParser()
    require(bool(cfg.read(UP / 'simulator/simulator_settings.ini')), 'simulator config missing')
    class FakeApplication:
        def __init__(self):
            for k in ('dac_bit', 'adc_bit', 'res_load', 'sum_gain'): setattr(self, k, int(cfg['board'][k]))
            for k in ('vol_ref_dac', 'vol_ref_adc', 'vol_read', 'res_switches', 'gain', 'soft_cc'):
                setattr(self, k, float(cfg['board'][k]))
    app.Application = FakeApplication
    sys.modules.update({'manager': manager, 'manager.app': app, 'manager.service': service})
    sys.path.insert(0, str(UP))
    from simulator.src import BoardSimulator, create_crossbar_array
    from MemriNeurons.cores import HardCore
    return cv, cfg, BoardSimulator, create_crossbar_array, HardCore


def create_guarded(base, cv, budget, clock, log, counter):
    class GuardedBoard(base):
        def __init__(self):
            super().__init__(); self.core = None; self.meta = {}; self.calls = 0
        def mode_mvm(self, vols, tms, tus, rtms, rums, wl, task_id):
            require(type(self) is GuardedBoard and wl in (0, 1), 'connector or column invalid')
            require(len(vols) == 32 and all(isinstance(v, (int, np.integer)) and 0 <= v <= 4095 for v in vols),
                    'invalid DAC codes')
            decoded = [cv.convert_dac_to_volt(self.core.dac_bit, self.core.vol_ref_dac, int(v)) for v in vols]
            observed = self.core.current_resistances[:, wl]
            actual = [self.crossbar[i][wl].last_resistance for i in range(32)]
            info = predrive(np.asarray(decoded), decoded, observed, actual)
            emit(log, dict(self.meta, column=wl, guard='mode_mvm', call=self.calls, **info))
            require(info['permitted'], 'ABORT before mode_mvm: >=15mA')
            require(counter[0] < budget['max_mvm_calls'] and time.monotonic() - clock < budget['max_seconds'],
                    'filter MVM/time budget exhausted')
            self.calls += 1; counter[0] += 1
            result = super().mode_mvm(vols, tms, tus, rtms, rums, wl, task_id)
            require((result[0] > 0 or not any(vols)) and 0 <= result[0] < 2 ** self.core.adc_bit - 1,
                    'ADC saturation or nonzero drive with zero ADC')
            return result
    return GuardedBoard


def drive(board, core, cv, x, col, meta, log):
    require(np.asarray(x).shape == (32,) and np.all(np.isfinite(x)) and np.all(np.asarray(x) >= 0),
            'invalid spike vector')
    requested = .3 * np.asarray(x, dtype=float)
    # Check requested voltages BEFORE conversion (converter/library may clip).
    require(np.all(requested <= .3), 'requested >0.3V')
    codes = [cv.convert_volt_to_dac(core.dac_bit, core.vol_ref_dac, float(v)) for v in requested]
    require(all(isinstance(n, (int, np.integer)) and 0 <= n <= core.max_dac_value for n in codes),
            'DAC code out of permitted range')
    decoded = [cv.convert_dac_to_volt(core.dac_bit, core.vol_ref_dac, n) for n in codes]
    info = predrive(requested, decoded, core.current_resistances[:, col],
                    [board.crossbar[i][col].last_resistance for i in range(32)])
    emit(log, dict(meta, column=col, guard='predrive', **info))
    require(info['permitted'], 'ABORT before dot_product: >=15mA')
    board.meta = meta
    out, adc = core.dot_product(requested.tolist(), col)  # NOT the unscaled 0/1 x!
    return float(out), np.asarray(decoded), int(adc)


def expired(*_):
    raise TimeoutError('20s cell write alarm')


def run_tile(fid, rep, tile, w, scale, rows, folder, api, clock, budget, counter):
    cv, cfg, base, create, HardCore = api
    lo, hi = TILES[tile]; seed = 50000 + 100000 * fid + 1000 * rep + 101 * tile
    random.seed(seed); np.random.seed(seed)
    serial = str(folder / f'array_r{rep}_tile{tile}')
    require(not pathlib.Path(serial + '.cb').exists(), 'existing array; never reuse')
    create(serial, 32, 8)
    Guard = create_guarded(base, cv, budget, clock, folder / 'safety.jsonl', counter)
    board = Guard(); board.config = cfg
    require(board.connect(serial) == 1 and type(board) is Guard, 'simulator connection failed')
    core = HardCore(board); board.core = core; core.set_mvm()
    require((core.ROW_NUM, core.COL_NUM) == (32, 8), 'orientation wrong')
    arr = dict(filter=fid, replicate=rep, tile=tile, seed=seed, taps=[lo, hi],
               array_id=serial, initial_sha256=sha(serial + '.cb'))
    emit(folder / 'arrays.jsonl', arr)
    target = np.stack((.03 + .02 * w / scale, .03 - .02 * w / scale), axis=1)
    start = time.monotonic()
    for row, tap in enumerate(range(lo, hi)):
        for col in (0, 1):
            signal.alarm(budget['cell_alarm_seconds'])
            try:
                before = time.monotonic(); final, hist, pulses = core.write_weight(row, col, float(target[tap, col]))
                elapsed = time.monotonic() - before
            finally:
                signal.alarm(0)
            emit(folder / 'programming.jsonl', dict(arr, row=row, column=col, tap=tap,
                 target=float(target[tap, col]), final=float(final), write_seconds=elapsed,
                 pulses_v=pulses, weight_history=hist, array_sha256=sha(serial + '.cb')))
            if fid == 0 and rep == 0 and tile == 0 and row == 0 and col == 0:
                require(64 * elapsed <= budget['first_write_projection_seconds_64'],
                        'first-write cost projection exceeds pilot budget')
            require(time.monotonic() - start <= budget['stage32_seconds_per_array'] * (hi - lo) / 32,
                    'programming time budget exceeded')
    measured = []; res = []; repeats = []
    for row in range(hi - lo):
        m = []; rs = []; repeat = []
        for col in (0, 1):
            r = [core.read_one_weight(row, col) for _ in range(3)]
            m.append(float(np.mean([z[0] for z in r]))); rs.append(float(np.mean([z[1] for z in r])))
            repeat.append([float(z[0]) for z in r])
        measured.append(m); res.append(rs); repeats.append(repeat)
    measured = np.asarray(measured)
    require(np.all(np.isfinite(measured)) and np.all(np.isfinite(res)) and np.all(np.asarray(res) > 0),
            'invalid measured cells')
    error = (measured[:, 0] - measured[:, 1] - (target[lo:hi, 0] - target[lo:hi, 1])) * scale / .04
    median = float(np.median(np.abs(error)) / scale)
    significant = np.abs(w[lo:hi]) > .05 * scale
    sign = float(np.mean(np.sign(measured[significant, 0] - measured[significant, 1]) ==
                         np.sign(w[lo:hi][significant]))) if any(significant) else 1.
    emit(folder / 'readback.jsonl', dict(arr, final_sha256=sha(serial + '.cb'), target=target[lo:hi].tolist(),
         measured=measured.tolist(), resistance_ohm=res, repeat_read_weights=repeats,
         sign_fraction=sign, median_abs_cell_error_over_maxabs=median, cell_error=error.tolist()))
    require(median <= P['gates']['median_cell_error_over_maxabs_max'], 'cell readback gate failed')
    grouped = {'train': [[], [], []], 'val': [[], [], []]}; zero_max = 0.; calls_before = board.calls
    for noise in (0, 1):
        for line in board.crossbar:
            for mem in line: mem.noise_flag = noise
        # Exact v2 deterministic control probes (single-active tap0, tap15, all-active); no zero probe.
        controls = [(dict(split='control', split_position=-1, filtered_index=-1,
                  original_train_index=-1, top=-1, left=-1, t=-1, probe=j),
                  np.pad(z, (lo, 75 - hi))) for j, z in enumerate(
                  [np.eye(hi - lo)[0], np.eye(hi - lo)[min(hi - lo - 1, 15)], np.ones(hi - lo)])]
        sequence = rows[:8] + controls if noise == 0 else rows
        for meta, x in sequence:
            z = x[lo:hi]; pad = np.pad(z, (0, 32 - len(z)))
            ideal = float(z @ w[lo:hi]); outs = []
            for read in range(1 if noise == 0 else 3):
                columns = []; digit = []; adcs = []; dec = None
                for col in (0, 1):
                    mv, decoded, adc = drive(board, core, cv, pad, col,
                             dict(arr, **meta, noise_flag=noise, read=read), folder / 'safety.jsonl')
                    columns.append(mv); digit.append(float(decoded[:hi - lo] @ measured[:, col])); adcs.append(adc); dec = decoded
                denom = .04 * .3
                q_ideal = float(dec[:hi - lo] @ (target[lo:hi, 0] - target[lo:hi, 1]) * scale / denom)
                q_prog = (digit[0] - digit[1]) * scale / denom
                sim = (columns[0] - columns[1]) * scale / denom
                prog = float(z @ (measured[:, 0] - measured[:, 1]) * scale / .04)
                emit(folder / 'patches.jsonl', dict(arr, **meta, noise_flag=noise, read=read,
                     ideal=ideal, quantized_ideal=q_ideal, quantized_programmed=q_prog,
                     programmed_readback=prog, sim_mvm=sim, raw_columns=columns, adc_columns=adcs))
                if noise == 0:
                    zero_max = max(zero_max, abs(sim - q_prog) / (scale * max(1, int(sum(z)))))
                elif meta['split'] in grouped:
                    seq = grouped[meta['split']]; seq[0].append(ideal); seq[1].append(prog); seq[2].append(sim)
    stats = dict(arr, sign_fraction=sign, median_abs_cell_error_over_maxabs=median,
                 zero_noise_max_error_per_active_tap_over_maxabs=zero_max,
                 seconds=time.monotonic() - start, mvm_calls=board.calls - calls_before,
                 train=metric(*grouped['train']), val=metric(*grouped['val']))
    emit(folder / 'metrics.jsonl', stats)
    require(zero_max <= P['gates']['zero_noise_error_per_active_tap_over_maxabs_max'],
            'zero-noise MVM/readback gate failed')
    return stats


def filter_run(fid, rows, stage, approved_sha, results):
    weights = np.load(WEIGHTS)
    require(tuple(weights.shape) == (64, 3, 5, 5), 'Conv1 must have 64 filters x 75 taps')
    w = weights[fid].ravel().astype(float)
    require(w.shape == (75,) and np.all(np.isfinite(w)), 'filter geometry/finite mismatch')
    scale = float(np.max(np.abs(w))); require(scale > 0, 'zero filter')
    target = np.stack((.03 + .02 * w / scale, .03 - .02 * w / scale), axis=1)
    require(np.all(target >= .01 - 1e-12) and np.all(target <= .05 + 1e-12) and
            np.allclose((target[:, 0] - target[:, 1]) * scale / .04, w), 'mapping reconstruction invalid')
    folder = results / f'filter_{fid:02d}'
    folder.mkdir()  # no replay of interrupted filter
    atomic_json(folder / 'preflight.json', dict(filter=fid, stage=stage, approval_sha256=approved_sha,
        checkpoint_sha256=P['checkpoint_sha256'], conv1_weights_npy_sha256=P['conv1_weights_npy_sha256'],
        source_sha256=source_hashes(), filter_weights=w.tolist(), maxabs=scale,
        split_digest=P['canonical_split_digest']))
    api = simulator_api(); start = time.monotonic(); budget = P['budgets_per_filter']
    signal.signal(signal.SIGALRM, expired)
    stage32 = []; counter = [0]
    for rep in range(3):
        stage32.append(run_tile(fid, rep, 0, w, scale, rows, folder, api, start, budget, counter))
    gates = P['gates']
    gate32 = all(s['val']['mvm_readback_rmse'] <= gates['stage32_mvm_readback_rmse_over_maxabs32_max'] * scale * 32
                 and validation_gate(s['val'], gates) for s in stage32)
    emit(folder / 'gates.jsonl', dict(stage='32', passed=gate32, metrics=stage32))
    require(gate32, 'stage32 gate failed, no further tiles')
    for rep in range(3):
        for tile in (1, 2):
            run_tile(fid, rep, tile, w, scale, rows, folder, api, start, budget, counter)
    records = [json.loads(line) for line in (folder / 'patches.jsonl').read_text().splitlines()]
    require(len(records) == 9 * (8 + 3 + 768) and counter[0] == 2 * len(records),
            'per-filter raw-record/MVM count mismatch')
    assembled = assemble(records, fid)
    summary = summarize(assembled, gates)
    require(summary['all_passed'], 'full75 validation gate failed; stop stage')
    atomic_json(folder / 'full75_summary.json', summary)
    # Completion marker hashes all upstream output artifacts; verify on restart.
    files = {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != 'COMPLETE.json'}
    atomic_json(folder / 'COMPLETE.json', dict(filter=fid, stage=stage, approval_sha256=approved_sha,
                                              source_sha256=source_hashes(), files=files))
    return summary


def completed(results, fid):
    folder = results / f'filter_{fid:02d}'; marker = folder / 'COMPLETE.json'
    if not folder.exists(): return False
    require(marker.exists(), 'interrupted filter: no array reuse/retry; parent review required: ' + str(folder))
    record = json.loads(marker.read_text())
    require(record['filter'] == fid and record['source_sha256'] == source_hashes(),
            'completed filter source mismatch')
    for name, digest in record['files'].items():
        require(sha(folder / name) == digest, 'completed artifact changed: ' + name)
    require({p.name for p in folder.iterdir() if p.is_file()} == set(record['files']) | {'COMPLETE.json'},
            'completed filter has unexpected files')
    require(json.loads((folder / 'full75_summary.json').read_text())['all_passed'],
            'completed filter did not pass')
    return True


def execute(stage, authorization):
    check_inputs(); approved_sha = approval(stage, authorization)
    rows, digest = patch_rows(); require(digest == P['canonical_split_digest'], 'patch digest mismatch')
    weights = np.load(WEIGHTS)
    require(tuple(weights.shape) == (64, 3, 5, 5), '64-filter tensor mismatch')
    results = HERE / 'results'; results.mkdir(exist_ok=True)
    lock = results / 'LOCK'
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, (str(os.getpid()) + '\n').encode()); os.close(fd)
        earlier = []
        for name in list(STAGES)[:list(STAGES).index(stage)]: earlier += STAGES[name]
        require(all(completed(results, i) for i in earlier), 'previous stage not fully passed')
        for preceding in list(STAGES)[:list(STAGES).index(stage)]:
            manifest = results / (preceding + '_MANIFEST.json')
            require(manifest.exists(), 'previous stage manifest absent: ' + preceding)
            data = json.loads(manifest.read_text())
            require(data['stage'] == preceding and data['source_sha256'] == source_hashes() and
                    data['checkpoint_sha256'] == P['checkpoint_sha256'] and
                    data['completion_sha256'] == {str(i): sha(results / f'filter_{i:02d}' / 'COMPLETE.json')
                        for i in STAGES[preceding]}, 'previous stage manifest mismatch: ' + preceding)
        for fid in STAGES[stage]:
            if completed(results, fid): continue
            filter_run(fid, rows, stage, approved_sha, results)
        # Stage manifest only after ALL filters have passed; hash completion markers.
        stage_path = results / (stage + '_MANIFEST.json')
        if stage_path.exists():
            m = json.loads(stage_path.read_text())
            require(m['completion_sha256'] == {str(i): sha(results / f'filter_{i:02d}' / 'COMPLETE.json')
                      for i in STAGES[stage]}, 'existing stage manifest changed')
        else:
            atomic_json(stage_path, dict(stage=stage, approval_sha256=approved_sha,
                source_sha256=source_hashes(), checkpoint_sha256=P['checkpoint_sha256'],
                completion_sha256={str(i): sha(results / f'filter_{i:02d}' / 'COMPLETE.json') for i in STAGES[stage]}))
    finally:
        try: os.close(fd)
        except OSError: pass
        lock.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='read-only source/input hash verification')
    parser.add_argument('--execute', action='store_true', help='parent-authorized simulator measurements')
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--approval', type=pathlib.Path)
    args = parser.parse_args()
    require(args.verify != args.execute, 'choose exactly one of --verify or --execute')
    if args.verify:
        check_inputs(); print(json.dumps({'status': 'VERIFIED; NO MEASUREMENTS',
             'source_sha256': source_hashes(), 'filters': 64, 'stage_filters': STAGES}, indent=2))
    else:
        require(args.stage is not None and args.approval is not None, 'stage and parent approval required')
        execute(args.stage, args.approval)


if __name__ == '__main__':
    main()
