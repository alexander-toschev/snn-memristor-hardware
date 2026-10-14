"""Phase 2 — sequential differential mapping of 14 Conv1 filters through exactly
2 reused 32x8 arrays (erase -> program -> readback -> reconstruct -> save).

CRITICAL: create_crossbar_array called EXACTLY 2x total; the 2 board/core handles
are reused in place across all 14 filters (ping-pong even-rank->A, odd-rank->B),
each array erase+reprogrammed 7x. Erase-before-write RESET (2x apply_voltage 3V) is
applied before every filter so writes start from a fixed low state (the fix that
removes sequential-reprogram cross-contamination).

Per filter: full-75 val-patch RMSE/Pearson vs ideal (frozen val split), weight-space
RMSE/Pearson, median cell error, sign fraction, gate pass/fail. Saves each filter's
reconstructed 75-tap readback weight to results/filter_NN/readback_weight.npy
(weight space, (75,)) for Phase 3.

Output: summary.json, reuse_evidence.json, results/filter_NN/{result.json, readback.json,
readback_weight.npy}.
"""
import pathlib
import random
import time

import numpy as np

import rev2_common as C

OUT = pathlib.Path(__file__).resolve().parent
ARRAYS_DIR = OUT / 'arrays'
RESULTS = OUT / 'results'

WEIGHTS_NPY = pathlib.Path(
    '/mnt/d/projects/snn-mnist/.tmp/openclaw-spikes/memriboard-conv1-stratified-20261008/extension/conv1_weights.npy')
WEIGHTS_SHA = '8be312b9eafb9de23f8340b17f848e7e0e2e7884293c826a6d9f6cbb5125d440'
CHECKPOINT_SHA = 'ee4fc4f58a3d010218116e9ad4c7ba808787959f9ed7687cd58cbb32d90f62fb'

SERIAL_A = 'array_A'
SERIAL_B = 'array_B'
SERIALS = (SERIAL_A, SERIAL_B)


def main():
    OUT.mkdir(exist_ok=True)
    ARRAYS_DIR.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    t0 = time.monotonic()

    C.require(C.sha(WEIGHTS_NPY) == WEIGHTS_SHA, 'conv1 weights npy changed')
    C.require(C.sha(C.UP / 'MemriNeurons/cores.py') ==
              '22973863b5688090d2ab87ea28c611c4275d44cdc17115e8ce9655987c42b716', 'cores.py changed')
    C.require(C.sha(C.UP / 'simulator/src.py') ==
              '5b751d5925b7af05db8ed751dde73ac3f9a31262a0514c5a6c9948e96c8a96d4', 'src.py changed')
    C.require(C.sha(C.UP / 'manager/service/converters.py') ==
              '28c6495c56afa225501dd6b0dea999215f08b73f30e016ff7ea34d6c5412701c', 'converters.py changed')
    C.require(C.sha(C.UP / 'simulator/simulator_settings.ini') ==
              '9ef3229595cbbba0ccc7d7450981d7a63266092792f4573d405beb2f364d3ddb', 'config changed')

    weights = np.load(WEIGHTS_NPY)
    C.require(tuple(weights.shape) == (64, 3, 5, 5), 'Conv1 must be 64 x 3 x 5 x 5')

    rows, digest = C.patch_rows()
    C.require(digest == C.SPLIT_DIGEST, 'patch digest mismatch')
    val_patches = [(m, x) for m, x in rows if m['split'] == 'val']
    C.require(len(val_patches) == 128, 'val patch count mismatch')

    cv, cfg, BoardSimulator, create_crossbar_array, HardCore = C.simulator_api()

    create_log = []

    def counted_create(serial, r, c):
        create_log.append(dict(serial=serial, rows=r, cols=c, wall=time.monotonic() - t0))
        create_crossbar_array(serial, r, c)

    random.seed(C.MASTER_SEED); np.random.seed(C.MASTER_SEED)
    counted_create(str(ARRAYS_DIR / SERIAL_A), 32, 8)
    counted_create(str(ARRAYS_DIR / SERIAL_B), 32, 8)
    C.require(len(create_log) == 2, 'must create exactly 2 arrays')

    boards, cores = {}, {}
    for serial in SERIALS:
        path = str(ARRAYS_DIR / serial)
        b = BoardSimulator(); b.config = cfg
        C.require(b.connect(path) == 1, 'connect failed: ' + serial)
        c = HardCore(b); c.set_mvm()
        C.require((c.ROW_NUM, c.COL_NUM) == (32, 8), 'orientation wrong')
        boards[serial] = b; cores[serial] = c

    summary_rows = []
    for i, fid in enumerate(C.FILTERS):
        rank = C.RANK[fid]
        serial = SERIALS[i % 2]
        board = boards[serial]; core = cores[serial]
        w = weights[fid].ravel().astype(float)
        scale = float(np.max(np.abs(w)))
        C.require(scale > 0, 'zero filter')
        target = np.stack((.03 + .02 * w / scale, .03 - .02 * w / scale), axis=1)
        C.require(np.all(target >= .01 - 1e-12) and np.all(target <= .05 + 1e-12), 'mapping range invalid')

        random.seed(C.SEED_BASE + 1000 * rank); np.random.seed(C.SEED_BASE + 1000 * rank)
        for line in board.crossbar:
            for mem in line:
                mem.noise_flag = 1

        ft0 = time.monotonic()
        # explicit erase before programming
        for col in range(6):
            for row in range(32):
                C.erase_cell(core, row, col, pulses=2)
        program_log = []
        for tile, (lo, hi) in enumerate(C.TILES):
            col0, col1 = 2 * tile, 2 * tile + 1
            for row, tap in enumerate(range(lo, hi)):
                for col, sign in ((col0, +1), (col1, -1)):
                    final, hist, pulses = core.write_weight(row, col, float(target[tap, 0 if sign > 0 else 1]))
                    program_log.append(dict(tile=tile, row=row, col=col, tap=tap,
                                            target=float(target[tap, 0 if sign > 0 else 1]),
                                            final=float(final), pulses=len(pulses)))
        program_s = time.monotonic() - ft0

        measured = np.zeros((75, 2))
        cell_res = np.zeros((75, 2))
        for tile, (lo, hi) in enumerate(C.TILES):
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
        wm = C.weight_metric(w_hat, w)

        val_ideal = [float(x @ w) for _, x in val_patches]
        val_sim = []
        safety_log = []
        for meta, x in val_patches:
            for read in range(3):
                sim_sum = 0.0
                for tile, (lo, hi) in enumerate(C.TILES):
                    z = x[lo:hi]
                    pad = np.pad(z, (0, 32 - len(z)))
                    cols = []
                    for col in (2 * tile, 2 * tile + 1):
                        mv, adc = C.drive(board, core, cv, pad, col, safety_log)
                        cols.append(mv)
                    denom = .04 * .3
                    sim_sum += (cols[0] - cols[1]) * scale / denom
                val_sim.append(sim_sum)
        val_ideal_arr = np.repeat(np.asarray(val_ideal), 3)
        val_sim_arr = np.asarray(val_sim)
        C.require(val_ideal_arr.shape == val_sim_arr.shape == (384,), 'val record alignment')
        ideal_sd = float(np.std(val_ideal_arr))
        bound = max(C.GATES['rmse_over_ideal_sd_max'] * ideal_sd, C.GATES['rmse_scale_floor'] * scale)
        val_rmse = float(np.sqrt(np.mean((val_ideal_arr - val_sim_arr) ** 2)))
        val_corr = (float(np.corrcoef(val_ideal_arr, val_sim_arr)[0, 1])
                    if ideal_sd > 0 and np.std(val_sim_arr) > 0 else None)
        passed = bool(val_corr is not None and val_corr >= C.GATES['pearson_min'] and
                      ideal_sd > 0 and val_rmse <= bound)

        fdir = RESULTS / f'filter_{fid:02d}'
        fdir.mkdir(exist_ok=True)
        C.atomic_json(fdir / 'result.json', dict(
            filter=fid, rank=rank, array=serial, maxabs_w=scale,
            weight_space=wm, median_abs_cell_error_over_maxabs=median_cell_err,
            sign_fraction=sign_frac,
            val=dict(ideal_sd=ideal_sd, rmse=val_rmse, corr=val_corr, bound=bound, passed=passed,
                     n_val_records=int(val_sim_arr.size)),
            wall_seconds=dict(total=time.monotonic() - t0, filter=time.monotonic() - ft0,
                              program=program_s),
            n_program_writes=len(program_log)))
        C.atomic_json(fdir / 'readback.json', dict(
            filter=fid, w_hat=w_hat.tolist(), w_ideal=w.tolist(),
            measured=measured.tolist(), cell_resistance=cell_res.tolist(),
            cell_error=cell_error.tolist(), program=program_log))
        np.save(fdir / 'readback_weight.npy', w_hat.astype(np.float64))  # (75,) weight space

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
    C.require(len(create_log) == 2, 'BUG: more than 2 arrays created')
    C.atomic_json(OUT / 'reuse_evidence.json', reuse_evidence)
    C.atomic_json(OUT / 'summary.json', dict(
        filters=C.FILTERS, split_digest=C.SPLIT_DIGEST, weights_sha=WEIGHTS_SHA,
        checkpoint_sha=CHECKPOINT_SHA, gates=C.GATES,
        column_layout='tile0->cols(0,1), tile1->cols(2,3), tile2->cols(4,5); '
                      'cols 6,7 spare; 1 filter/array; ping-pong A/B',
        seed=dict(master=C.MASTER_SEED, per_filter_base=C.SEED_BASE, formula='90000+1000*rank'),
        wall_seconds=round(time.monotonic() - t0, 1),
        results=summary_rows))
    print('DONE wall_s=', round(time.monotonic() - t0, 1))
    print('create_crossbar_array calls =', len(create_log), '(expect exactly 2)')


if __name__ == '__main__':
    main()
