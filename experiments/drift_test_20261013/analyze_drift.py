"""Analysis of the DRIFT test raw readback.

Inputs: raw_readback.json (produced by drift_test.py).
Outputs: summary.json, summary.csv, and prints answers to the 4 protocol questions.
"""
import json
import pathlib
import numpy as np

OUT = pathlib.Path('/mnt/d/projects/snn-mnist/.tmp/drift_test_20261013')
d = json.load(open(OUT / 'raw_readback.json'))
recs = d['records']
N_CYCLES = 50
cells = sorted(set(r['cell'] for r in recs))
NC = len(cells)

# reshape to arrays [cell, cycle]
W = np.zeros((NC, N_CYCLES))
E = np.zeros((NC, N_CYCLES))   # erased weight
P = np.zeros((NC, N_CYCLES))   # n_pulses
for r in recs:
    W[r['cell'], r['cycle']] = r['final_weight']
    E[r['cell'], r['cycle']] = r['erased_weight']
    P[r['cell'], r['cycle']] = r['n_pulses']

cycle = np.arange(N_CYCLES, dtype=float)
Sxx = float(np.sum((cycle - cycle.mean()) ** 2))  # 10412.5 for n=50

def linfit(y):
    """Linear regression y = a + b*x. Returns a, b, resid_std, se_b, t."""
    n = len(y)
    b = float(np.sum((cycle - cycle.mean()) * (y - y.mean())) / Sxx)
    a = float(y.mean() - b * cycle.mean())
    resid = y - (a + b * cycle)
    sse = float(np.sum(resid ** 2))
    dof = n - 2
    resid_std = float(np.sqrt(sse / dof))
    se_b = resid_std / np.sqrt(Sxx)
    t = b / se_b if se_b > 0 else 0.0
    return a, b, resid_std, se_b, t

TARGET = d['meta']['target']

rows = []
print(f"Target conductance = {TARGET}; N = {N_CYCLES} cycles, {NC} cells")
print(f"{'cell':>4} {'mean':>9} {'slope_b':>10} {'resid_std':>10} {'raw_std':>9} {'|b*N|':>9} {'t_b':>7} {'signif?':>8}")
signif = []
slopes = []
for c in range(NC):
    y = W[c]
    a, b, resid_std, se_b, t = linfit(y)
    raw_std = float(y.std())
    bN = abs(b * (N_CYCLES - 1))
    # significance: |t| > 2 (approx 95% for n=50)
    sig = abs(t) > 2.0
    signif.append(sig)
    slopes.append(b)
    rows.append(dict(cell=c, row=int(c), col=0,
                     mean=float(y.mean()), slope=float(b), intercept=float(a),
                     resid_std=resid_std, raw_std=raw_std,
                     abs_bN=bN, t=float(t), significant_2sig=bool(sig)))
    print(f"{c:>4} {y.mean():>9.5f} {b:>10.2e} {resid_std:>10.2e} {raw_std:>9.5f} "
          f"{bN:>9.2e} {t:>7.2f} {'YES' if sig else 'no':>8}")

slopes = np.array(slopes)
print()
print(f"slope across cells: mean={slopes.mean():.3e} std={slopes.std():.3e} "
      f"min={slopes.min():.3e} max={slopes.max():.3e} "
      f"n_pos={int((slopes>0).sum())}/{NC} n_neg={int((slopes<0).sum())}/{NC}")
print(f"n significant (|t|>2): {sum(signif)}/{NC}")

# pooled analysis: fixed-effect per cell, slope on cycle
# center by cell mean to get within-cell trend
W_center = W - W.mean(axis=1, keepdims=True)
# pooled slope across all cells (assume common slope)
flat_cycle = np.tile(cycle, NC)
flat_y = W.ravel()
b_pool = float(np.sum((flat_cycle - flat_cycle.mean()) * (flat_y - flat_y.mean())) /
               np.sum((flat_cycle - flat_cycle.mean()) ** 2))
print(f"\npooled common slope (all 800 pts): {b_pool:.3e} / cycle")

# global mean & std
print(f"global mean final_weight = {W.mean():.5f}  (target {TARGET}, bias {W.mean()-TARGET:+.5f})")
print(f"global std  final_weight = {W.std():.5f}")

# erased floor trend
print(f"\nerased weight: global mean={E.mean():.5f} std={E.std():.5f} "
      f"min={E.min():.5f} max={E.max():.5f}")
aE, bE, _, _, tE = linfit(E.mean(axis=0))
print(f"erased-floor mean vs cycle: slope={bE:.3e}, t={tE:.2f}")

# n_pulses (write effort) trend
print(f"\nn_pulses: mean={P.mean():.1f} std={P.std():.1f} min={P.min()} max={P.max()}")
aP, bP, _, _, tP = linfit(P.mean(axis=0))
print(f"n_pulses mean vs cycle: slope={bP:.3f}, t={tP:.2f}")

# ---- variance trend: block std ----
print("\n--- variance trend (block std of final_weight, 5 blocks of 10) ---")
block_means = []
for k in range(5):
    blk = W[:, k*10:(k+1)*10]
    print(f"  cycles {k*10:2d}-{k*10+9:2d}: cross-cell std of per-cell mean="
          f"{blk.mean(axis=1).std():.5f}  within-cell pooled std={blk.std():.5f}")
    block_means.append(blk.mean())

# per-cell std in first half vs second half
std_first = W[:, :25].std(axis=1)
std_second = W[:, 25:].std(axis=1)
print(f"per-cell std first-half mean={std_first.mean():.5f}, "
      f"second-half mean={std_second.mean():.5f} (ratio {std_second.mean()/std_first.mean():.3f})")

# save summary
summary = dict(
    target=TARGET, n_cycles=N_CYCLES, n_cells=NC,
    global_mean_final_weight=float(W.mean()),
    global_std_final_weight=float(W.std()),
    global_bias_vs_target=float(W.mean() - TARGET),
    pooled_common_slope=float(b_pool),
    slopes=dict(mean=float(slopes.mean()), std=float(slopes.std()),
                min=float(slopes.min()), max=float(slopes.max()),
                n_positive=int((slopes > 0).sum()), n_negative=int((slopes < 0).sum()),
                n_significant_2sig=int(sum(signif))),
    per_cell=rows,
    erased_weight=dict(mean=float(E.mean()), std=float(E.std()),
                       slope_vs_cycle=float(bE), t=float(tE)),
    n_pulses=dict(mean=float(P.mean()), std=float(P.std()),
                  slope_vs_cycle=float(bP), t=float(tP)),
    variance_trend=dict(per_cell_std_first_half_mean=float(std_first.mean()),
                        per_cell_std_second_half_mean=float(std_second.mean()),
                        ratio=float(std_second.mean() / std_first.mean())),
)
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=2, sort_keys=True)

# per-cell CSV
with open(OUT / 'summary.csv', 'w') as f:
    f.write('cell,row,col,mean,slope_b,intercept,resid_std,raw_std,abs_bN,t,significant_2sig\n')
    for r in rows:
        f.write(f"{r['cell']},{r['row']},{r['col']},{r['mean']:.8g},{r['slope']:.8g},"
                f"{r['intercept']:.8g},{r['resid_std']:.8g},{r['raw_std']:.8g},"
                f"{r['abs_bN']:.8g},{r['t']:.4g},{int(r['significant_2sig'])}\n")
print("\n[ok] wrote summary.json + summary.csv")
