"""DRIFT test on the MemriBoard BoardSimulator (software only).

QUESTION: does repeated erase->write on the SAME cell produce a systematic
drift (accumulating bias) or just zero-mean noise? This informs the REV2
two-crossbar plan, where 2 chips are reused 14-64x.

Protocol (per the task):
  1. ONE fresh 32x8 array; 16 cells = rows 0..15, column 0.
  2. Fixed target conductance 0.03 (the differential common-mode the REV2
     mapping programs). For each cell, N=50 cycles of:
       erase  = 2x core.apply_voltage(3.0, 0, row, col)  (positive 3V RESET)
       program= core.write_weight(row, col, 0.03)
       read   = core.read_one_weight(row, col) -> record achieved conductance
     => 16 cells x 50 cycles = 800 cycles; per-cycle per-cell readback.
  3. Track write-verify pulse count (len of write_weight's vol_history) per cycle.

Uses the SAME headless adapter as the frozen REV2 dry-run (verified byte-identical
source hashes): converters.py + FakeApplication stub + simulator/src.py BoardSimulator
+ MemriNeurons/cores.py HardCore, 32x8, native noise (noise_flag=1, noise_rate=0.03).

Read-only w.r.t. prior artifacts: creates its own array under this .tmp dir; does NOT
touch prior MemriBoard artifacts, the champion 2x2, or the test set.
"""
import configparser
import hashlib
import importlib.util
import json
import pathlib
import random
import sys
import time
import types
import numpy as np

UP = pathlib.Path('/mnt/d/projects/memristor-aware-twd-snn/memristor_hw/MemriBoard')
OUT = pathlib.Path('/mnt/d/projects/snn-mnist/.tmp/drift_test_20261013')
ARRAY_PATH = OUT / 'drift_array'

TARGET = 0.03          # differential common-mode conductance
N_CYCLES = 50          # erase->write->read cycles per cell
CELLS = [(r, 0) for r in range(16)]   # 16 cells: rows 0..15, column 0
MASTER_SEED = 20261013

# frozen source hashes (from REV2 dry-run protocol 2eae1ff8)
HASHES = {
    'MemriNeurons/cores.py':
        '22973863b5688090d2ab87ea28c611c4275d44cdc17115e8ce9655987c42b716',
    'simulator/src.py':
        '5b751d5925b7af05db8ed751dde73ac3f9a31262a0514c5a6c9948e96c8a96d4',
    'manager/service/converters.py':
        '28c6495c56afa225501dd6b0dea999215f08b73f30e016ff7ea34d6c5412701c',
    'simulator/simulator_settings.ini':
        '9ef3229595cbbba0ccc7d7450981d7a63266092792f4573d405beb2f364d3ddb',
}


def sha(p):
    h = hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


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
    assert cfg.read(UP / 'simulator/simulator_settings.ini'), 'simulator config missing'

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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    # frozen integrity
    for rel, h in HASHES.items():
        got = sha(UP / rel)
        assert got == h, f'{rel} hash mismatch: {got}'
    print('[ok] frozen source hashes verified byte-identical', flush=True)

    cv, cfg, BoardSimulator, create_crossbar_array, HardCore = simulator_api()

    random.seed(MASTER_SEED); np.random.seed(MASTER_SEED)
    create_crossbar_array(str(ARRAY_PATH), 32, 8)

    board = BoardSimulator(); board.config = cfg
    assert board.connect(str(ARRAY_PATH)) == 1, 'connect failed'
    core = HardCore(board); core.set_mvm()
    assert (core.ROW_NUM, core.COL_NUM) == (32, 8), 'orientation wrong'

    # explicit native noise (class defaults are already flag=1 rate=0.03; set for clarity)
    n_noisy = 0
    for line in board.crossbar:
        for mem in line:
            mem.noise_flag = 1
            mem.noise_rate = 0.03
            n_noisy += 1
    print(f'[ok] array created 32x8, cells={n_noisy}, noise_flag=1 noise_rate=0.03', flush=True)

    raw_csv = OUT / 'raw_readback.csv'
    raw_json = OUT / 'raw_readback.json'
    records = []
    csv_hdr = 'cell,row,col,cycle,erased_weight,erased_res,final_weight,final_res,write_return_weight,n_pulses\n'
    with raw_csv.open('w') as f:
        f.write(csv_hdr)
        f.flush()

    t_cell0 = time.monotonic()
    for ci, (row, col) in enumerate(CELLS):
        for cycle in range(N_CYCLES):
            # ---- erase: 2x positive 3V RESET ----
            core.apply_voltage(3.0, 0, row, col)
            core.apply_voltage(3.0, 0, row, col)
            # ---- (optional) read the erased floor ----
            ew, eres = core.read_one_weight(row, col)
            # ---- program to target ----
            wret, whist, vhist = core.write_weight(row, col, TARGET)
            # ---- read back achieved conductance ----
            fw, fres = core.read_one_weight(row, col)
            rec = dict(cell=ci, row=row, col=col, cycle=cycle,
                       erased_weight=float(ew), erased_res=float(eres),
                       final_weight=float(fw), final_res=float(fres),
                       write_return_weight=float(wret), n_pulses=int(len(vhist)))
            records.append(rec)
            with raw_csv.open('a') as f:
                f.write(f"{ci},{row},{col},{cycle},{ew:.8g},{eres:.8g},"
                        f"{fw:.8g},{fres:.8g},{wret:.8g},{len(vhist)}\n")
                f.flush()
        if ci == 0:
            print(f'[ok] cell0 (50 cycles) wall={time.monotonic()-t_cell0:.1f}s '
                  f'=> est total ~{(time.monotonic()-t_cell0)*len(CELLS):.0f}s', flush=True)

    wall = time.monotonic() - t0
    meta = dict(
        target=TARGET, n_cycles=N_CYCLES, cells=[list(c) for c in CELLS],
        master_seed=MASTER_SEED, noise_flag=1, noise_rate=0.03,
        n_records=len(records), wall_seconds=round(wall, 1),
        source_hashes=HASHES,
        note='readback weight = read_one_weight in mvm mode (sum_gain/res); '
             'erase = 2x apply_voltage(3.0, 0, r, c); program = write_weight(r, c, 0.03); '
             'n_pulses = len(write_weight vol_history) = write-verify pulse count.')
    with (OUT / 'meta.json').open('w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    with raw_json.open('w') as f:
        json.dump(dict(meta=meta, records=records), f, indent=2)
    print(f'DONE wall={wall:.1f}s records={len(records)}', flush=True)


if __name__ == '__main__':
    main()
