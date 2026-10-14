"""Local synthetic check of revised gate logic: absolute-only stage-32 + scale-aware floor full-75."""
import sys
sys.path.insert(0, '/mnt/d/projects/snn-mnist/.tmp/openclaw-spikes/memriboard-conv1-stratified-20261008/extension')
import numpy as np
from extension_core import validation_gate, metric, summarize, assemble

gates = dict(pearson_min=0.8, rmse_over_ideal_sd_max=0.6, rmse_scale_floor=0.05)

# Case A: low-variance filter (ideal_sd small). 0.6*SD < 0.05*maxabs_w => floor binds.
# filter 63: ideal_sd_val 0.106, max_abs 0.4417. floor = 0.05*0.4417 = 0.0221; 0.6*0.106=0.0636.
scale = 0.4417
a = np.linspace(-0.2, 0.2, 384)  # ideal, sd ~ 0.116
z = a + 0.03 * np.random.default_rng(0).standard_normal(384)  # sim rmse ~0.03
m = metric(a, a + 0.01*np.random.default_rng(1).standard_normal(384), z)
print('Case A metric: corr=%.4f ideal_sd=%.4f rmse=%.4f' % (m['corr'], m['ideal_sd'], m['rmse']))
print('  bound=max(%.4f, %.4f)=%.4f -> passed=%s' % (
    0.6*m['ideal_sd'], 0.05*scale, max(0.6*m['ideal_sd'], 0.05*scale), validation_gate(m, gates, scale)))

# Case B: filter-1 analog. Full-75 ideal_sd ~0.136, RMSE 0.063 (prior rep-2 readback RMSE),
# high correlation. Under OLD 32-tap gate this failed (0.6*0.0816=0.049); under revised
# full-75 gate bound = max(0.6*0.136, 0.05*0.6056) = max(0.0815, 0.0303) = 0.0815, so passes.
scale1 = 0.6056
rng = np.random.default_rng(2)
a1 = np.concatenate([np.linspace(-0.3, -0.02, 192), np.linspace(0.02, 0.3, 192)])
a1 = a1[np.random.default_rng(9).permutation(384)]
z1 = a1 + 0.063 * rng.standard_normal(384)  # rmse ~0.063, corr stays high
m1 = metric(a1, a1, z1)
print('Case B: corr=%.4f ideal_sd=%.4f rmse=%.4f bound=max(%.4f,%.4f)=%.4f passed=%s' % (
    m1['corr'], m1['ideal_sd'], m1['rmse'], 0.6*m1['ideal_sd'], 0.05*scale1,
    max(0.6*m1['ideal_sd'], 0.05*scale1), validation_gate(m1, gates, scale1)))

# Case C: genuinely bad fidelity must still fail (rmse >> bound, corr low).
a2 = np.linspace(-0.5, 0.5, 384)
z2 = a2 + np.random.default_rng(3).standard_normal(384) * 0.3
m2 = metric(a2, a2, z2)
print('Case C: corr=%.4f ideal_sd=%.4f rmse=%.4f passed=%s (expect False)' % (
    m2['corr'], m2['ideal_sd'], m2['rmse'], validation_gate(m2, gates, 1.0)))

assert validation_gate(m, gates, scale) is True, 'A should pass'
assert validation_gate(m1, gates, scale1) is True, 'B should pass (floor rescues low-variance)'
assert validation_gate(m2, gates, 1.0) is False, 'C should fail'
print('ALL_GATE_LOGIC_OK')
