"""Fresh-array seed-matched CONTROL for the sequential-reprogram dry-run.

Purpose: distinguish cross-contamination (reuse bug) from seed-scheme variance.

The reuse dry-run creates array_A then array_B with MASTER_SEED=20261013 and reprograms
each filter with write-seed = 90000 + 1000*rank (rank = 0-based position in FILTERS).
Even-rank filters use array_A (1st create), odd-rank use array_B (2nd create).

This control reproduces the EXACT initial array state (same creation-RNG position) and the
EXACT write-seed, but into a FRESH array, so any difference vs the reuse run is due to the
array's prior (reuse) state — i.e. cross-contamination.
"""
import pathlib
import sys
import random
import time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seq_reprogram_dryrun import (  # noqa: E402
    simulator_api, patch_rows, drive, TILES, FILTERS, RANK, GATES,
    WEIGHTS_NPY, WEIGHTS_SHA, SPLIT_DIGEST, MASTER_SEED, SEED_BASE,
    sha, require, atomic_json)

OUT = HERE / 'fresh_control'
OUT.mkdir(exist_ok=True)


def run_fresh(fid, rank, weights, cv, cfg, BoardSimulator, create_crossbar_array,
              HardCore, val_patches, array_serial):
    w = weights[fid].ravel().astype(float)
    scale = float(np.max(np.abs(w)))
    target = np.stack((.03 + .02 * w / scale, .03 - .02 * w / scale), axis=1)

    create_crossbar_array(array_serial, 32, 8)
    b = BoardSimulator(); b.config = cfg
    require(b.connect(array_serial) == 1, 'connect failed')
    c = HardCore(b); c.set_mvm()

    random.seed(SEED_BASE + 1000 * rank); np.random.seed(SEED_BASE + 1000 * rank)
    for line in b.crossbar:
        for mem in line:
            mem.noise_flag = 1

    # explicit erase (same as reuse run) so both start from a fixed low state
    for col in range(6):
        for row in range(32):
            c.apply_voltage(3.0, 0, row, col)
            c.apply_voltage(3.0, 0, row, col)

    for tile, (lo, hi) in enumerate(TILES):
        col0, col1 = 2 * tile, 2 * tile + 1
        for row, tap in enumerate(range(lo, hi)):
            c.write_weight(row, col0, float(target[tap, 0]))
            c.write_weight(row, col1, float(target[tap, 1]))

    measured = np.zeros((75, 2))
    for tile, (lo, hi) in enumerate(TILES):
        for row, tap in enumerate(range(lo, hi)):
            for j, col in enumerate((2 * tile, 2 * tile + 1)):
                r = [c.read_one_weight(row, col) for _ in range(3)]
                measured[tap, j] = float(np.mean([z[0] for z in r]))
    w_hat = (measured[:, 0] - measured[:, 1]) * scale / .04
    w_rmse = float(np.sqrt(np.mean((w_hat - w) ** 2)))
    w_corr = float(np.corrcoef(w_hat, w)[0, 1])

    val_ideal = [float(x @ w) for _, x in val_patches]
    val_sim = []
    for _, x in val_patches:
        for _read in range(3):
            s = 0.0
            for tile, (lo, hi) in enumerate(TILES):
                z = x[lo:hi]; pad = np.pad(z, (0, 32 - len(z)))
                cols = []
                for col in (2 * tile, 2 * tile + 1):
                    mv, _ = drive(b, c, cv, pad, col, [])
                    cols.append(mv)
                s += (cols[0] - cols[1]) * scale / (.04 * .3)
            val_sim.append(s)
    val_ideal_arr = np.repeat(np.asarray(val_ideal), 3)
    val_sim_arr = np.asarray(val_sim)
    ideal_sd = float(np.std(np.asarray(val_ideal)))
    bound = max(GATES['rmse_over_ideal_sd_max'] * ideal_sd, GATES['rmse_scale_floor'] * scale)
    val_rmse = float(np.sqrt(np.mean((val_ideal_arr - val_sim_arr) ** 2)))
    val_corr = float(np.corrcoef(val_ideal_arr, val_sim_arr)[0, 1])
    return dict(filter=fid, maxabs_w=scale, w_rmse=w_rmse, w_corr=w_corr,
                val_rmse=val_rmse, val_corr=val_corr, bound=bound, ideal_sd=ideal_sd)


def main():
    require(sha(WEIGHTS_NPY) == WEIGHTS_SHA, 'weights changed')
    weights = np.load(WEIGHTS_NPY)
    rows, digest = patch_rows()
    require(digest == SPLIT_DIGEST, 'digest mismatch')
    val_patches = [(m, x) for m, x in rows if m['split'] == 'val']
    cv, cfg, BoardSimulator, create_crossbar_array, HardCore = simulator_api()

    results = []
    for fid in [63, 41, 32, 43, 1]:
        rank = RANK[fid]
        is_even = rank % 2 == 0
        random.seed(MASTER_SEED); np.random.seed(MASTER_SEED)
        if not is_even:
            create_crossbar_array(str(OUT / '_dummy'), 32, 8)  # consume array_A RNG
        serial = str(OUT / f'fresh_{fid:02d}')
        t0 = time.monotonic()
        r = run_fresh(fid, rank, weights, cv, cfg, BoardSimulator,
                      create_crossbar_array, HardCore, val_patches, serial)
        r['wall_s'] = round(time.monotonic() - t0, 1)
        results.append(r)
        print(f"FRESH fid={fid:2d} val_rmse={r['val_rmse']:.4f} bound={r['bound']:.4f} "
              f"w_rmse={r['w_rmse']:.4f} w_corr={r['w_corr']:.4f} | {r['wall_s']}s", flush=True)

    atomic_json(OUT / 'fresh_results.json', dict(
        note='Fresh-array seed-matched control: array created at the same MASTER_SEED RNG '
             'position as the reuse array (array_A=1st create, array_B=2nd create), write-seed '
             '90000+1000*rank identical to reuse run.',
        results=results))


if __name__ == '__main__':
    main()
