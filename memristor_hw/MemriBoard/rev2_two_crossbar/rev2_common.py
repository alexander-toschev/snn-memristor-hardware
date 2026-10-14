"""Shared headless BoardSimulator adapter + frozen protocol constants for the
REV 2 two-crossbar general rehearsal (2026-10-13).

Identical frozen source hashes / split digest / mapping as the proven
seq_reprogram_dryrun_20261013 erase-fix run (protocol 2eae1ff8 source lineage).
"""
import configparser
import hashlib
import importlib.util
import json
import math
import pathlib
import pickle
import sys
import types

import numpy as np

ROOT = pathlib.Path('/mnt/d/projects/snn-mnist')
UP = pathlib.Path(__file__).resolve().parent.parent  # MemriBoard repo root

FILTERS = [63, 41, 32, 30, 38, 43, 60, 47, 2, 9, 46, 12, 0, 1]
RANK = {fid: r for r, fid in enumerate(FILTERS)}
TILES = ((0, 32), (32, 64), (64, 75))

SPLIT_DIGEST = 'f3a5acaf7d61d1221b28a6ba66e468f905a44463630a8bdc5cf38cbb956f0492'

GATES = dict(pearson_min=0.8, rmse_over_ideal_sd_max=0.6, rmse_scale_floor=0.05,
             median_cell_error_over_maxabs_max=0.35,
             zero_noise_error_per_active_tap_over_maxabs_max=0.05)

TRAIN_POSITIONS = list(range(8))
VAL_POSITIONS = list(range(8))
SPATIAL_TOP_LEFT = ((0, 0), (7, 7), (14, 14), (21, 21))
SPIKE_TIMES = (0, 3, 6, 9)
T_STEPS = 12
X_MIN = 0.1

MASTER_SEED = 20261013
SEED_BASE = 90000  # per-filter programming seed = SEED_BASE + 1000*rank


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    tmp.replace(path)


def simulator_api():
    spec = importlib.util.spec_from_file_location('extension_converters',
                                                  UP / 'manager/service/converters.py')
    cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv)
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
            for k in ('dac_bit', 'adc_bit', 'res_load', 'sum_gain'):
                setattr(self, k, int(cfg['board'][k]))
            for k in ('vol_ref_dac', 'vol_ref_adc', 'vol_read', 'res_switches', 'gain', 'soft_cc'):
                setattr(self, k, float(cfg['board'][k]))

    app.Application = FakeApplication
    sys.modules.update({'manager': manager, 'manager.app': app, 'manager.service': service})
    sys.path.insert(0, str(UP))
    from simulator.src import BoardSimulator, create_crossbar_array
    from MemriNeurons.cores import HardCore
    return cv, cfg, BoardSimulator, create_crossbar_array, HardCore


def predrive(requested, decoded, observed, actual):
    v = np.asarray(requested, dtype=float); d = np.asarray(decoded, dtype=float)
    require(v.shape == d.shape == (32,), 'voltage vector must have 32 rows')
    require(np.all(np.isfinite(v)) and np.all(v >= 0) and np.all(v <= .3), 'voltage invalid/>0.3V')
    require(np.all(np.isfinite(d)) and np.all(d >= 0) and np.all(d <= .3), 'decoded voltage invalid/>0.3V')
    currents = []
    for name, raw in (('observed', observed), ('actual', actual)):
        resistance = np.asarray(raw, dtype=float)
        require(resistance.shape == (32,) and np.all(np.isfinite(resistance)) and np.all(resistance > 0),
                name + ' resistance invalid')
        current = float(np.sum(d / resistance))
        require(math.isfinite(current), name + ' predicted current nonfinite')
        currents.append(current)
    peak = max(currents)
    return dict(requested_max_v=float(v.max()), decoded_max_v=float(d.max()),
                observed_current_a=currents[0], actual_current_a=currents[1],
                current_a=peak, margin_a=.015 - peak, permitted=peak < .015)


def drive(board, core, cv, x, col, safety_log):
    """One differential MVM column drive with the <=0.3V / <15mA predrive guard."""
    require(np.asarray(x).shape == (32,) and np.all(np.isfinite(x)) and np.all(np.asarray(x) >= 0),
            'invalid spike vector')
    requested = .3 * np.asarray(x, dtype=float)
    require(np.all(requested <= .3), 'requested >0.3V')
    codes = [cv.convert_volt_to_dac(core.dac_bit, core.vol_ref_dac, float(v)) for v in requested]
    require(all(isinstance(n, (int, np.integer)) and 0 <= n <= core.max_dac_value for n in codes),
            'DAC code out of range')
    decoded = [cv.convert_dac_to_volt(core.dac_bit, core.vol_ref_dac, n) for n in codes]
    info = predrive(requested, decoded, core.current_resistances[:, col],
                    [board.crossbar[i][col].last_resistance for i in range(32)])
    require(info['permitted'], 'ABORT before dot_product: >=15mA')
    safety_log.append(dict(column=col, **info))
    out, adc = core.dot_product(requested.tolist(), col)  # scaled voltage vector
    return float(out), int(adc)


def patch_rows():
    """Exact frozen v2 deterministic CIFAR-100 *training-file* split (no test access)."""
    with (ROOT / 'data/cifar-100-python/train').open('rb') as f:
        raw = pickle.load(f, encoding='bytes')
    labels = np.asarray(raw[b'fine_labels'])
    selected = np.flatnonzero(np.isin(labels, [0, 1, 2, 3, 5, 8, 13, 14, 17, 19]))
    filtered = labels[selected]
    rng = np.random.default_rng(20261006)
    val = []
    for cls in np.unique(filtered):
        pool = np.flatnonzero(filtered == cls)
        val.extend(rng.permutation(pool)[:round(len(pool) * .1)].tolist())
    val = sorted(val)
    mask = np.ones(len(selected), bool); mask[val] = False
    train = np.flatnonzero(mask).tolist()
    digest = hashlib.sha256(json.dumps({'train_indices': train, 'val_indices': val},
                                       sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(digest == SPLIT_DIGEST and len(train) == 4500 and len(val) == 500, 'canonical split mismatch')
    rows = []
    for split, indices, positions in (('train', train, TRAIN_POSITIONS), ('val', val, VAL_POSITIONS)):
        for pos in positions:
            fi = indices[pos]; oi = int(selected[fi])
            img = raw[b'data'][oi].reshape(3, 32, 32).astype(np.float32) / 255
            for top, left in SPATIAL_TOP_LEFT:
                px = img[:, top:top + 5, left:left + 5].reshape(75)
                fire = np.floor((1 - px) * (T_STEPS - 1)).astype(int).clip(0, T_STEPS - 1)
                for t in SPIKE_TIMES:
                    x = ((px >= X_MIN) & (fire == t)).astype(float)
                    rows.append((dict(split=split, split_position=pos, filtered_index=fi,
                                      original_train_index=oi, top=top, left=left, t=t), x))
    require(len(rows) == 256, 'fixed patch count mismatch')
    return rows, digest


def weight_metric(w_hat, w):
    a = np.asarray(w_hat, dtype=float); b = np.asarray(w, dtype=float)
    require(a.shape == b.shape == (75,), 'weight geometry mismatch')
    sd = float(np.std(b))
    return dict(count=75, ideal_sd=sd,
                rmse=float(np.sqrt(np.mean((a - b) ** 2))),
                corr=float(np.corrcoef(a, b)[0, 1]) if sd > 0 and np.std(a) > 0 else None)


def erase_cell(core, row, col, pulses=2):
    """RESET used cells to ~min conductance (fixed low write start state)."""
    for _ in range(pulses):
        core.apply_voltage(3.0, 0, row, col)  # RESET (rev=0, positive 3 V)


def read_avg(core, row, col, n=3):
    return float(np.mean([core.read_one_weight(row, col)[0] for _ in range(n)]))
