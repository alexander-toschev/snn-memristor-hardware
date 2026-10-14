"""Pure policy/aggregation functions; no simulator imports and no measurement side effects."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

TILES = ((0, 32), (32, 64), (64, 75))
FIELDS = ('ideal', 'quantized_ideal', 'programmed_readback', 'quantized_programmed', 'sim_mvm')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def predrive(requested, decoded, observed, actual):
    """Independent checks *before* either HardCore.dot_product or BoardSimulator.mode_mvm.

    Reject invalid source vectors rather than clipping. Both resistance views are mandatory.
    """
    v = np.asarray(requested, dtype=float)
    d = np.asarray(decoded, dtype=float)
    require(v.shape == d.shape == (32,), 'voltage vector must have 32 rows')
    require(np.all(np.isfinite(v)) and np.all(v >= 0) and np.all(v <= .3), 'requested voltage invalid or >0.3V')
    require(np.all(np.isfinite(d)) and np.all(d >= 0) and np.all(d <= .3), 'decoded voltage invalid or >0.3V')
    currents = []
    for name, raw in (('observed', observed), ('actual', actual)):
        resistance = np.asarray(raw, dtype=float)
        require(resistance.shape == (32,) and np.all(np.isfinite(resistance)) and np.all(resistance > 0), name + ' resistance invalid')
        current = float(np.sum(d / resistance))
        require(math.isfinite(current), name + ' predicted current nonfinite')
        currents.append(current)
    peak = max(currents)
    return dict(requested_max_v=float(v.max()), decoded_max_v=float(d.max()),
                observed_current_a=currents[0], actual_current_a=currents[1],
                current_a=peak, margin_a=.015 - peak, permitted=peak < .015)


def metric(ideal, readback, sim):
    a, b, c = map(lambda x: np.asarray(x, dtype=float), (ideal, readback, sim))
    require(a.shape == b.shape == c.shape and a.size > 0, 'empty or unmatched metrics')
    sd = float(np.std(a))
    return dict(count=int(a.size), ideal_sd=sd,
                rmse=float(np.sqrt(np.mean((a-c)**2))),
                mvm_readback_rmse=float(np.sqrt(np.mean((b-c)**2))),
                corr=float(np.corrcoef(a,c)[0,1]) if sd > 0 and np.std(c) > 0 else None)


def validation_gate(m, gates, scale):
    """REVISED variance-aware full-75 functional gate.

    val Pearson(ideal, sim_mvm) >= pearson_min AND
    val RMSE(sim_mvm, ideal) <= max(rmse_over_ideal_sd_max * SD(ideal_full_75),
                                    rmse_scale_floor * maxabs_w(filter)).
    `scale` is maxabs_w = max |Conv1 weight| of this filter. Applied ONLY to the full
    75-tap aggregated output; never to the 32-tap partial stage.
    """
    bound = max(gates['rmse_over_ideal_sd_max'] * m['ideal_sd'],
                gates['rmse_scale_floor'] * scale)
    return bool(m['corr'] is not None and m['corr'] >= gates['pearson_min'] and
                m['ideal_sd'] > 0 and m['rmse'] <= bound)


def assemble(records, filter_id, replicates=3):
    """Reject duplicate/missing tiles and enforce exact 384 train + 384 val samples/replicate."""
    groups = {}
    for r in records:
        if r['noise_flag'] != 1 or r['split'] not in ('train', 'val'):
            continue
        require(r['filter'] == filter_id, 'cross-filter record')
        key = tuple(r[k] for k in ('replicate', 'split', 'split_position', 'top', 'left', 't', 'read'))
        tiles = groups.setdefault(key, {})
        tile = r['tile']
        require(tile in range(3) and tile not in tiles, 'duplicate/invalid patch tile')
        tiles[tile] = r
    out = []
    for key, tiles in groups.items():
        require(set(tiles) == {0,1,2}, 'incomplete three-tile patch')
        out.append(dict(filter=filter_id, replicate=key[0], split=key[1], position=key[2],
                        top=key[3], left=key[4], t=key[5], read=key[6],
                        **{field:float(sum(tiles[t][field] for t in range(3))) for field in FIELDS}))
    out.sort(key=lambda r: (r['replicate'],r['split'],r['position'],r['top'],r['left'],r['t'],r['read']))
    require(len(out) == replicates*768, 'full75 assembled patch count mismatch')
    for rep in range(replicates):
        for split in ('train','val'):
            require(sum(r['replicate']==rep and r['split']==split for r in out)==384,
                    'full75 per-replicate split count mismatch')
    return out


def summarize(assembled, gates, scale, replicates=3):
    reports={}
    for rep in range(replicates):
        train=[r for r in assembled if r['replicate']==rep and r['split']=='train']
        val=[r for r in assembled if r['replicate']==rep and r['split']=='val']
        require(len(train)==len(val)==384, 'surrogate train/val mismatch')
        x=np.asarray([r['sim_mvm'] for r in train]);y=np.asarray([r['ideal'] for r in train])
        slope,intercept=np.linalg.lstsq(np.stack((x,np.ones(len(x))),axis=1),y,rcond=None)[0]
        a=np.asarray([r['ideal'] for r in val]);z=np.asarray([r['sim_mvm'] for r in val])
        m=metric(a,[r['programmed_readback'] for r in val],z)
        surrogate=intercept+slope*z
        reports[str(rep)]={'validation':m,'passed':validation_gate(m,gates,scale),
                           'train_affine_slope':float(slope),'train_affine_intercept':float(intercept),
                           'surrogate_val_rmse':float(np.sqrt(np.mean((a-surrogate)**2))),
                           'surrogate_val_corr':float(np.corrcoef(a,surrogate)[0,1]) if np.std(a)>0 and np.std(surrogate)>0 else None,
                           'other_val':{field:metric(a,[r['programmed_readback'] for r in val],
                                             [r[field] for r in val]) for field in FIELDS if field!='sim_mvm'}}
    return {'replicates':reports,'all_passed':all(r['passed'] for r in reports.values())}
