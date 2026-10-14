"""REV 2 two-crossbar sequential-reprogram dry-run (BoardSimulator, software-only).

VALIDATES: the simulator creates EXACTLY 2 virtual 32x8 arrays ONCE and REUSES them
across all 14 Conv1 filters via sequential erase+reprogram (write-verify), instead of a
fresh independent array per filter. Runs the same differential mapping + full-75
val-patch metric as the frozen stratified protocol 2eae1ff8, on the same frozen source
hashes, so results are directly comparable to prior fresh-array numbers.

Column layout (documented): one filter uses 6 of 8 columns of ONE array --
  tile0 -> cols (0,1), tile1 -> cols (2,3), tile2 -> cols (4,5); cols 6,7 spare.
Filters are processed strictly in the frozen order, ping-ponging across the 2 arrays:
  array A holds even-index filters (rank 0,2,4,...), array B holds odd (rank 1,3,5,...).
Each array is therefore erase+reprogrammed 7 times (14 filters / 2 arrays).

Metric: full-75 val-patch RMSE/Pearson(ideal, sim_mvm) on the frozen validation patches
(digest f3a5acaf...f0492), 3 MVM reads per patch, identical to prior. Also reports
weight-space RMSE/Pearson (reconstructed 75-tap weight vs ideal) and the median
cell-error gate. Bound = max(0.6*SD(ideal_full_75_val), 0.05*maxabs_w).

Read-only: does not touch prior MemriBoard/champion artifacts or the test set.
"""
import configparser
import hashlib
import importlib.util
import json
import math
import pathlib
import pickle
import random
import sys
import time
import types
import numpy as np

# ----------------------------------------------------------------------------
# Frozen constants (from stratified protocol 2eae1ff8 / PROTOCOL.json)
# ----------------------------------------------------------------------------
ROOT = pathlib.Path('/mnt/d/projects/snn-mnist')
UP = pathlib.Path('/mnt/d/projects/memristor-aware-twd-snn/memristor_hw/MemriBoard')
WEIGHTS_NPY = pathlib.Path(
    '/mnt/d/projects/snn-mnist/.tmp/openclaw-spikes/memriboard-conv1-stratified-20261008/extension/conv1_weights.npy')
OUT = pathlib.Path(__file__).resolve().parent
ARRAYS_DIR = OUT / 'arrays'
RESULTS = OUT / 'results'

FILTERS = [63, 41, 32, 30, 38, 43, 60, 47, 2, 9, 46, 12, 0, 1]
RANK = {fid: r for r, fid in enumerate(FILTERS)}
TILES = ((0, 32), (32, 64), (64, 75))

WEIGHTS_SHA = '8be312b9eafb9de23f8340b17f848e7e0e2e7884293c826a6d9f6cbb5125d440'
SPLIT_DIGEST = 'f3a5acaf7d61d1221b28a6ba66e468f905a44463630a8bdc5cf38cbb956f0492'

GATES = dict(pearson_min=0.8, rmse_over_ideal_sd_max=0.6, rmse_scale_floor=0.05,
             median_cell_error_over_maxabs_max=0.35,
             zero_noise_error_per_active_tap_over_maxabs_max=0.05)

SERIAL_A = 'array_A'   # .cb path = ARRAYS_DIR / 'array_A.cb'
SERIAL_B = 'array_B'
SERIALS = (SERIAL_A, SERIAL_B)

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


# ----------------------------------------------------------------------------
# Headless BoardSimulator adapter (identical to frozen runner simulator_api())
# ----------------------------------------------------------------------------
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


def main():
    OUT.mkdir(exist_ok=True)
    ARRAYS_DIR.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    t0 = time.monotonic()

    # ---- frozen input integrity ----
    require(sha(WEIGHTS_NPY) == WEIGHTS_SHA, 'conv1 weights npy changed')
    require(sha(UP / 'MemriNeurons/cores.py') ==
            '22973863b5688090d2ab87ea28c611c4275d44cdc17115e8ce9655987c42b716', 'cores.py changed')
    require(sha(UP / 'simulator/src.py') ==
            '5b751d5925b7af05db8ed751dde73ac3f9a31262a0514c5a6c9948e96c8a96d4', 'src.py changed')
    require(sha(UP / 'manager/service/converters.py') ==
            '28c6495c56afa225501dd6b0dea999215f08b73f30e016ff7ea34d6c5412701c', 'converters.py changed')
    require(sha(UP / 'simulator/simulator_settings.ini') ==
            '9ef3229595cbbba0ccc7d7450981d7a63266092792f4573d405beb2f364d3ddb', 'config changed')

    weights = np.load(WEIGHTS_NPY)
    require(tuple(weights.shape) == (64, 3, 5, 5), 'Conv1 must be 64 x 3 x 5 x 5')

    rows, digest = patch_rows()
    require(digest == SPLIT_DIGEST, 'patch digest mismatch')
    val_patches = [(m, x) for m, x in rows if m['split'] == 'val']
    require(len(val_patches) == 128, 'val patch count mismatch')

    cv, cfg, BoardSimulator, create_crossbar_array, HardCore = simulator_api()

    # ---- CREATE EXACTLY 2 ARRAYS ONCE (instrumented) ----
    create_log = []

    def counted_create(serial, r, c):
        create_log.append(dict(serial=serial, rows=r, cols=c, wall=time.monotonic() - t0))
        create_crossbar_array(serial, r, c)

    random.seed(MASTER_SEED); np.random.seed(MASTER_SEED)
    counted_create(str(ARRAYS_DIR / SERIAL_A), 32, 8)
    counted_create(str(ARRAYS_DIR / SERIAL_B), 32, 8)
    require(len(create_log) == 2, 'must create exactly 2 arrays')

    # Two persistent board+core handles (reused across filters; object identity preserved)
    boards = {}
    cores = {}
    for serial in SERIALS:
        path = str(ARRAYS_DIR / serial)
        b = BoardSimulator(); b.config = cfg
        require(b.connect(path) == 1, 'connect failed: ' + serial)
        c = HardCore(b); c.set_mvm()
        require((c.ROW_NUM, c.COL_NUM) == (32, 8), 'orientation wrong')
        boards[serial] = b; cores[serial] = c

    # ---- process 14 filters, ping-pong across the 2 arrays ----
    summary_rows = []
    for i, fid in enumerate(FILTERS):
        rank = RANK[fid]
        serial = SERIALS[i % 2]
        board = boards[serial]; core = cores[serial]
        w = weights[fid].ravel().astype(float)
        scale = float(np.max(np.abs(w)))
        require(scale > 0, 'zero filter')
        target = np.stack((.03 + .02 * w / scale, .03 - .02 * w / scale), axis=1)  # (75,2)
        require(np.all(target >= .01 - 1e-12) and np.all(target <= .05 + 1e-12), 'mapping range invalid')

        random.seed(SEED_BASE + 1000 * rank); np.random.seed(SEED_BASE + 1000 * rank)
        # ensure write noise ON (default) before reprogramming
        for line in board.crossbar:
            for mem in line:
                mem.noise_flag = 1

        ft0 = time.monotonic()
        # (a.0) EXPLICIT ERASE before programming: RESET all 6 used columns to near-min
        # conductance so write_weight's write-verify always starts from a FIXED low state,
        # not the prior filter's residual conductance (fixes sequential-reprogram
        # cross-contamination observed without this pass).
        for col in range(6):
            for row in range(32):
                core.apply_voltage(3.0, 0, row, col)  # RESET (rev=0, positive 3V)
                core.apply_voltage(3.0, 0, row, col)  # 2 pulses -> converged ~0.0033
        # (a) program 3 tiles (6 differential columns) into this array, in place
        program_log = []
        for tile, (lo, hi) in enumerate(TILES):
            col0, col1 = 2 * tile, 2 * tile + 1
            for row, tap in enumerate(range(lo, hi)):
                for col, sign in ((col0, +1), (col1, -1)):
                    final, hist, pulses = core.write_weight(row, col, float(target[tap, 0 if sign > 0 else 1]))
                    program_log.append(dict(tile=tile, row=row, col=col, tap=tap,
                                            target=float(target[tap, 0 if sign > 0 else 1]),
                                            final=float(final), pulses=len(pulses)))
        program_s = time.monotonic() - ft0

        # (b) read back all cells (3 reads averaged), reconstruct 75-tap differential weight
        measured = np.zeros((75, 2))
        cell_res = np.zeros((75, 2))
        for tile, (lo, hi) in enumerate(TILES):
            col0, col1 = 2 * tile, 2 * tile + 1
            for row, tap in enumerate(range(lo, hi)):
                for j, col in enumerate((col0, col1)):
                    r = [core.read_one_weight(row, col) for _ in range(3)]
                    measured[tap, j] = float(np.mean([z[0] for z in r]))
                    cell_res[tap, j] = float(np.mean([z[1] for z in r]))
        w_hat = (measured[:, 0] - measured[:, 1]) * scale / .04
        cell_error = (measured[:, 0] - measured[:, 1] -
                      (target[:, 0] - target[:, 1])) * scale / .04
        median_cell_err = float(np.median(np.abs(cell_error)) / scale)
        significant = np.abs(w) > .05 * scale
        sign_frac = float(np.mean(np.sign(measured[significant, 0] - measured[significant, 1]) ==
                                  np.sign(w[significant]))) if any(significant) else 1.0
        wm = weight_metric(w_hat, w)

        # (c) full-75 val-patch metric (ideal vs sim_mvm), 3 MVM reads, matching prior
        val_ideal = [float(x @ w) for _, x in val_patches]
        val_sim = []
        safety_log = []
        for meta, x in val_patches:
            for read in range(3):
                sim_sum = 0.0
                for tile, (lo, hi) in enumerate(TILES):
                    z = x[lo:hi]
                    pad = np.pad(z, (0, 32 - len(z)))
                    cols = []
                    for col in (2 * tile, 2 * tile + 1):
                        mv, adc = drive(board, core, cv, pad, col, safety_log)
                        cols.append(mv)
                    denom = .04 * .3
                    sim_sum += (cols[0] - cols[1]) * scale / denom
                val_sim.append(sim_sum)
        # 128 patches x 3 MVM reads = 384 records; repeat ideal per read to align
        val_ideal_arr = np.repeat(np.asarray(val_ideal), 3)
        val_sim_arr = np.asarray(val_sim)
        require(val_ideal_arr.shape == val_sim_arr.shape == (384,), 'val record alignment')
        ideal_sd = float(np.std(val_ideal_arr))
        bound = max(GATES['rmse_over_ideal_sd_max'] * ideal_sd, GATES['rmse_scale_floor'] * scale)
        val_rmse = float(np.sqrt(np.mean((val_ideal_arr - val_sim_arr) ** 2)))
        val_corr = (float(np.corrcoef(val_ideal_arr, val_sim_arr)[0, 1])
                    if ideal_sd > 0 and np.std(val_sim_arr) > 0 else None)
        passed = bool(val_corr is not None and val_corr >= GATES['pearson_min'] and
                      ideal_sd > 0 and val_rmse <= bound)

        # (d) save per-filter result
        fdir = RESULTS / f'filter_{fid:02d}'
        fdir.mkdir(exist_ok=True)
        atomic_json(fdir / 'result.json', dict(
            filter=fid, rank=rank, array=serial, maxabs_w=scale,
            weight_space=wm, median_abs_cell_error_over_maxabs=median_cell_err,
            sign_fraction=sign_frac,
            val=dict(ideal_sd=ideal_sd, rmse=val_rmse, corr=val_corr, bound=bound, passed=passed,
                     n_val_records=int(val_sim_arr.size)),
            wall_seconds=dict(total=time.monotonic() - t0, filter=time.monotonic() - ft0,
                              program=program_s),
            n_program_writes=len(program_log)))
        atomic_json(fdir / 'readback.json', dict(
            filter=fid, w_hat=w_hat.tolist(), w_ideal=w.tolist(),
            measured=measured.tolist(), cell_resistance=cell_res.tolist(),
            cell_error=cell_error.tolist(), program=program_log))
        summary_rows.append(dict(filter=fid, rank=rank, array=serial, maxabs_w=scale,
                                 val_rmse=val_rmse, val_corr=val_corr, bound=bound,
                                 val_passed=passed, w_rmse=wm['rmse'], w_corr=wm['corr'],
                                 median_cell_err=median_cell_err, sign_fraction=sign_frac,
                                 wall_filter_s=round(time.monotonic() - ft0, 1)))
        print(f"[{i+1:2d}/14] fid={fid:2d} array={serial} maxabs={scale:.4f} "
              f"val_rmse={val_rmse:.4f} val_corr={val_corr:.4f} bound={bound:.4f} "
              f"pass={passed} | w_rmse={wm['rmse']:.4f} w_corr={wm['corr']:.4f} "
              f"cell_err={median_cell_err:.4f} | {time.monotonic()-ft0:.1f}s",
              flush=True)

    # ---- reuse evidence + final summary ----
    reuse_evidence = dict(
        create_crossbar_array_calls=len(create_log),
        create_log=create_log,
        distinct_serials=sorted(set(SERIALS)),
        board_object_ids={s: id(boards[s]) for s in SERIALS},
        core_object_ids={s: id(cores[s]) for s in SERIALS},
        arrays_reused_across_all_filters=True,
        note='create_crossbar_array called exactly 2x total; each filter uses board/core object '
             'already connected to one of the 2 persistent arrays (no re-create, no re-connect). '
             'Each array reprogrammed 7x via write_weight write-verify (erase+program in place).')
    require(len(create_log) == 2, 'BUG: more than 2 arrays created')
    atomic_json(OUT / 'reuse_evidence.json', reuse_evidence)
    atomic_json(OUT / 'summary.json', dict(
        filters=FILTERS, split_digest=SPLIT_DIGEST, weights_sha=WEIGHTS_SHA,
        gates=GATES, column_layout='tile0->cols(0,1), tile1->cols(2,3), tile2->cols(4,5); '
                                   'cols 6,7 spare; 1 filter/array; ping-pong A/B',
        seed=dict(master=MASTER_SEED, per_filter_base=SEED_BASE, formula='90000+1000*rank'),
        wall_seconds=round(time.monotonic() - t0, 1),
        results=summary_rows))
    print('DONE wall_s=', round(time.monotonic() - t0, 1))
    print('create_crossbar_array calls =', len(create_log), '(expect exactly 2)')


if __name__ == '__main__':
    main()
