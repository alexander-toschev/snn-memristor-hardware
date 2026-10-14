import configparser, importlib.util, sys, types, random, time
import numpy as np
UP = '/mnt/d/projects/memristor-aware-twd-snn/memristor_hw/MemriBoard'
spec = importlib.util.spec_from_file_location('extension_converters', UP + '/manager/service/converters.py')
cv = importlib.util.module_from_spec(spec); spec.loader.exec_module(cv)
manager = types.ModuleType('manager'); manager.__path__ = []
app = types.ModuleType('manager.app'); service = types.ModuleType('manager.service')
for a, b in [('d2v','convert_dac_to_volt'),('v2d','convert_volt_to_dac'),
             ('a2r','convert_adc_to_res'),('r2a','convert_res_to_adc'),
             ('r2w','convert_res_to_weight'),('w2r','convert_weight_to_res'),
             ('a2v','convert_adc_to_volt'),('a2c','convert_adc_to_current')]:
    setattr(service, a, getattr(cv, b))
cfg = configparser.ConfigParser(); cfg.read(UP + '/simulator/simulator_settings.ini')
class FakeApplication:
    def __init__(self):
        for k in ('dac_bit','adc_bit','res_load','sum_gain'): setattr(self, k, int(cfg['board'][k]))
        for k in ('vol_ref_dac','vol_ref_adc','vol_read','res_switches','gain','soft_cc'): setattr(self, k, float(cfg['board'][k]))
app.Application = FakeApplication
sys.modules.update({'manager': manager, 'manager.app': app, 'manager.service': service})
sys.path.insert(0, UP)
from simulator.src import BoardSimulator, create_crossbar_array
from MemriNeurons.cores import HardCore

random.seed(1); np.random.seed(1)
serial = '/mnt/d/projects/snn-mnist/.tmp/seq_reprogram_dryrun_20261013/_erase_test'
create_crossbar_array(serial, 32, 8)
b = BoardSimulator(); b.config = cfg; b.connect(serial)
c = HardCore(b); c.set_mvm()
for line in b.crossbar:
    for mem in line: mem.noise_flag = 1

# program to a HIGH weight (0.05) simulating prior filter's residual
c.write_weight(0,0,0.05); w0,_ = c.read_one_weight(0,0)
# now test erase: N RESET pulses
for N in (1,2,3,5,8,12):
    c.write_weight(0,0,0.05)  # reset to 0.05 each time
    for _ in range(N):
        c.apply_voltage(3.0, 0, 0, 0)  # RESET (rev=0, positive)
    wN,_ = c.read_one_weight(0,0)
    print(f"after {N:2d} RESET pulses from 0.05 -> weight={wN:.5f}")

# also test: write_weight to a low target acts as erase
c.write_weight(0,0,0.05)
c.write_weight(0,0,0.005)
wE,_ = c.read_one_weight(0,0)
print(f"write_weight(0.005) as erase from 0.05 -> {wE:.5f}")
