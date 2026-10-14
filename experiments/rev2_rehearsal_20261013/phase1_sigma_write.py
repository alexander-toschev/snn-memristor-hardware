"""Phase 1 — σ_write characterization (BoardSimulator, software-only).

Sweeps write_weight targets across the differential mapping window 0.01–0.05
(8 cells) plus a few in the docstring's 0.07–0.33 nominal range, measuring
(a) readback absolute error -> σ_write, and (b) cycle-to-cycle variance by
re-writing the SAME cell N=5 times to the same target and measuring readback
spread. Uses erase-before-write so every write starts from a fixed low state
(same discipline as Phase 2). Noise ON (noise_flag=1, noise_rate=0.03).

Output: phase1_result.json
"""
import json
import pathlib
import random
import time

import numpy as np

import rev2_common as C

OUT = pathlib.Path(__file__).resolve().parent
ARR = OUT / 'arrays' / 'phase1_char'

MAPPING_TARGETS = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050]
DOCSTRING_TARGETS = [0.07, 0.10, 0.15, 0.20, 0.25, 0.33]
CYCLE_TARGETS = [0.02, 0.03, 0.04, 0.10]
N_CYCLE = 5


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / 'arrays').mkdir(exist_ok=True)
    t0 = time.monotonic()

    cv, cfg, BoardSimulator, create_crossbar_array, HardCore = C.simulator_api()

    random.seed(C.MASTER_SEED); np.random.seed(C.MASTER_SEED)
    create_crossbar_array(str(ARR), 32, 8)
    b = BoardSimulator(); b.config = cfg
    C.require(b.connect(str(ARR)) == 1, 'connect failed')
    core = HardCore(b); core.set_mvm()
    C.require((core.ROW_NUM, core.COL_NUM) == (32, 8), 'orientation wrong')
    for line in b.crossbar:
        for mem in line:
            mem.noise_flag = 1  # noise ON (VTEAM noise_rate=0.03)

    def write_and_read(row, col, target):
        C.erase_cell(core, row, col, pulses=2)
        final, hist, pulses = core.write_weight(row, col, float(target))
        rb = C.read_avg(core, row, col, n=3)
        return dict(target=float(target), final_write=float(final), readback=rb,
                    error=float(rb - target), abs_error=float(abs(rb - target)),
                    pulses=len(pulses))

    # (a) conductance sweep
    sweep = []
    row = 0
    for tgt in MAPPING_TARGETS + DOCSTRING_TARGETS:
        rec = write_and_read(row, 0, tgt)
        rec['group'] = 'mapping_window' if tgt <= 0.05 else 'docstring_range'
        rec['row'] = row
        sweep.append(rec)
        row += 1

    map_errs = np.array([r['abs_error'] for r in sweep if r['group'] == 'mapping_window'])
    doc_errs = np.array([r['abs_error'] for r in sweep if r['group'] == 'docstring_range'])
    all_errs = np.array([r['abs_error'] for r in sweep])
    sigma_write = dict(
        mapping_window=dict(n=len(map_errs), mean_abs_error=float(map_errs.mean()),
                            std_error=float(map_errs.std()), rmse=float(np.sqrt(np.mean(map_errs**2)))),
        docstring_range=dict(n=len(doc_errs), mean_abs_error=float(doc_errs.mean()),
                             std_error=float(doc_errs.std()), rmse=float(np.sqrt(np.mean(doc_errs**2)))),
        all=dict(n=len(all_errs), mean_abs_error=float(all_errs.mean()),
                 std_error=float(all_errs.std()), rmse=float(np.sqrt(np.mean(all_errs**2)))),
    )

    # (b) cycle-to-cycle variance (same cell, N rewrites to same target)
    cycle = []
    for j, tgt in enumerate(CYCLE_TARGETS):
        col = 2
        reads = []
        for k in range(N_CYCLE):
            rec = write_and_read(4 + j, col, tgt)
            reads.append(rec['readback'])
        arr = np.array(reads)
        cycle.append(dict(target=float(tgt), cell_row=4 + j, cell_col=col, n=N_CYCLE,
                          readbacks=reads, mean=float(arr.mean()), std=float(arr.std()),
                          spread=float(arr.max() - arr.min())))
    cycle_std = np.array([c['std'] for c in cycle])
    cycle_overall = dict(mean_cycle_to_cycle_std=float(cycle_std.mean()),
                         max_cycle_to_cycle_std=float(cycle_std.max()))

    result = dict(
        phase=1,
        noise=dict(noise_flag=1, noise_rate=0.03, model='VTEAM'),
        erase_before_write=dict(pulses=2, voltage=3.0, rev=0),
        sigma_write_absolute=sigma_write,
        cycle_to_cycle=dict(per_target=cycle, overall=cycle_overall),
        wall_seconds=round(time.monotonic() - t0, 1),
    )
    C.atomic_json(OUT / 'phase1_result.json', result)
    print('[phase1] sigma_write (mapping window): mean_abs=%.5f std=%.5f rmse=%.5f'
          % (map_errs.mean(), map_errs.std(), np.sqrt(np.mean(map_errs**2))))
    print('[phase1] sigma_write (docstring range): mean_abs=%.5f std=%.5f rmse=%.5f'
          % (doc_errs.mean(), doc_errs.std(), np.sqrt(np.mean(doc_errs**2))))
    for c in cycle:
        print('[phase1] cycle-to-cycle target=%.3f mean=%.5f std=%.5f spread=%.5f'
              % (c['target'], c['mean'], c['std'], c['spread']))
    print('[phase1] wall_s=%.1f' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
