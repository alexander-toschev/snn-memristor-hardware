import configparser, importlib.util, sys, types, os, random, time
import numpy as np
UP = '/mnt/d/projects/memristor-aware-twd-snn/memristor_hw/MemriBoard'
spec = importlib.util.spec_from_file_location('extension_converters', UP + '/manager/service/converters.py')
cv = importlib.util.module_from_spec(spec); spec.loader.exec_module(cv)
manager = types.ModuleType('manager'); manager.__path__ = []
app = types.ModuleType('manager.app'); service = types.ModuleType('manager.service')
for a, b in [('d2v','convert_dac_to_volt'), ('v2d','convert_volt_to_dac'),
             ('a2r','convert_adc_to_res'), ('r2a','convert_res_to_adc'),
             ('r2w','convert_res_to_weight'), ('w2r','convert_weight_to_res'),
             ('a2v','convert_adc_to_volt'), ('a2c','convert_adc_to_current')]:
    setattr(service, a, getattr(cv, b))
cfg = configparser.ConfigParser(); assert cfg.read(UP + '/simulator/simulator_settings.ini')
class FakeApplication:
    def __init__(self):
        for k in ('dac_bit','adc_bit','res_load','sum_gain'): setattr(self, k, int(cfg['board'][k]))
        for k in ('vol_ref_dac','vol_ref_adc','vol_read','res_switches','gain','soft_cc'): setattr(self, k, float(cfg['board'][k]))
app.Application = FakeApplication
sys.modules.update({'manager': manager, 'manager.app': app, 'manager.service': service})
sys.path.insert(0, UP)
from simulator.src import BoardSimulator, create_crossbar_array
from MemriNeurons.cores import HardCore
print("imports OK; numpy", np.__version__)

random.seed(1234); np.random.seed(1234)
serial = '/mnt/d/projects/snn-mnist/.tmp/seq_reprogram_dryrun_20261013/_smoke'
create_crossbar_array(serial, 32, 8)
b = BoardSimulator(); b.config = cfg
assert b.connect(serial) == 1
core = HardCore(b); core.set_mvm()
print("ROW,COL =", core.ROW_NUM, core.COL_NUM)

# write one cell
t0=time.monotonic(); final, hist, pulses = core.write_weight(0,0,0.03); t1=time.monotonic()
w, r = core.read_one_weight(0,0)
print(f"write_weight -> final={final:.4f} readback_w={w:.4f} res={r:.1f} pulses={len(pulses)} dt={(t1-t0)*1000:.1f}ms")

# read one cell timing (3 reads)
t0=time.monotonic()
for _ in range(3): core.read_one_weight(0,0)
t1=time.monotonic(); print(f"3x read_one_weight dt={(t1-t0)*1000:.1f}ms")

# dot_product timing
x = [0.3 if i%2==0 else 0.0 for i in range(32)]
t0=time.monotonic(); out, adc = core.dot_product(x, 0); t1=time.monotonic()
print(f"dot_product -> out={out:.5f} adc={adc} dt={(t1-t0)*1000:.1f}ms")

# reprogram test (decrease + increase)
core.write_weight(0,0,0.05); w1,_=core.read_one_weight(0,0)
core.write_weight(0,0,0.015); w2,_=core.read_one_weight(0,0)
print(f"reprogram 0.05->{w1:.4f} then 0.015->{w2:.4f} (targets 0.05, 0.015)")
