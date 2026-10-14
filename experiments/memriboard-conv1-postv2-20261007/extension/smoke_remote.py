"""Headless BoardSimulator smoke test for remote env. Mirrors simulator_api() exactly.
Creates a 32x8 array, connects, HardCore set_mvm, writes one cell, reads it back,
and runs one dot_product. No physical board, no test set. Exits 0 only if all pass."""
import configparser, importlib.util, json, pathlib, sys, types, tempfile
import numpy as np

UP = pathlib.Path('/home/alex/memristor/MemriBoard')

spec = importlib.util.spec_from_file_location('smoke_converters', UP / 'manager/service/converters.py')
cv = importlib.util.module_from_spec(spec); spec.loader.exec_module(cv)
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
        for k in ('dac_bit', 'adc_bit', 'res_load', 'sum_gain'): setattr(self, k, int(cfg['board'][k]))
        for k in ('vol_ref_dac', 'vol_ref_adc', 'vol_read', 'res_switches', 'gain', 'soft_cc'):
            setattr(self, k, float(cfg['board'][k]))
app.Application = FakeApplication
sys.modules.update({'manager': manager, 'manager.app': app, 'manager.service': service})
sys.path.insert(0, str(UP))
from simulator.src import BoardSimulator, create_crossbar_array
from MemriNeurons.cores import HardCore

with tempfile.TemporaryDirectory() as td:
    serial = str(pathlib.Path(td) / 'smoke_array')
    create_crossbar_array(serial, 32, 8)
    board = BoardSimulator(); board.config = cfg
    assert board.connect(serial) == 1, 'connect failed'
    core = HardCore(board); board.core = core; core.set_mvm()
    assert (core.ROW_NUM, core.COL_NUM) == (32, 8), 'orientation wrong'
    final, hist, pulses = core.write_weight(0, 0, 0.03)
    assert np.isfinite(float(final)), 'write returned non-finite'
    r = [core.read_one_weight(0, 0) for _ in range(3)]
    assert all(np.isfinite(float(z[0])) and np.isfinite(float(z[1])) for z in r), 'read non-finite'
    # One guarded-style dot_product with scaled voltage (0.3*active).
    x = np.zeros(32); x[0] = 0.3
    out, adc = core.dot_product(x.tolist(), 0)
    assert np.isfinite(float(out)), 'dot_product non-finite'

print(json.dumps({
    'status': 'SMOKE_OK',
    'python': sys.version.split()[0],
    'numpy': np.__version__,
    'dac_bit': core.dac_bit, 'adc_bit': core.adc_bit,
    'write_final': float(final), 'read_weight_mean': float(np.mean([z[0] for z in r])),
    'dot_product_out': float(out), 'adc': int(adc),
    'n_pulses': len(pulses) if hasattr(pulses, '__len__') else None
}, sort_keys=True))
