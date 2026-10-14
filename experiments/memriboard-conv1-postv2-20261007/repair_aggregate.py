"""Read-only repair for the frozen v2 aggregate_full() postprocessing failure.

Never imports/runs pilot.py or BoardSimulator. Outputs are written beside this file,
not in the v2 directory. The v2 runner/protocol/raw artifacts remain frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / 'memriboard-conv1-v2-20261007'
FIELDS = ('ideal', 'quantized_ideal', 'programmed_readback', 'quantized_programmed', 'sim_mvm')
EXPECTED = {'PROTOCOL.json':'8a9fedebead3b1d58dff1b840e5e02fd520d86815681ce6959935c2e04874937',
            'pilot.py':'2d4365b92efedf8f5e8c341a0a6bc8d9686a67a9f70c002c409fb22125bdac98',
            'patches.jsonl':'2d3f6dff38c13eb7b513155a2cd9c5607c57fbc0cbf0f41e51afa42a533330f1'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

def aggregate(records, replicates=3):
    """The frozen v2 aggregation formula using the actual flat raw-record schema."""
    groups={}
    for r in records:
        if r['noise_flag']!=1 or r['split'] not in ('train','val'): continue
        key=(r['replicate'],r['split'],r['split_position'],r['top'],r['left'],r['t'],r['read'])
        tiles=groups.setdefault(key,{})
        tile=r['tile']
        if tile in tiles: raise ValueError(f'duplicate patch tile {key} {tile}')
        tiles[tile]=r
    out=[]
    for key,tiles in groups.items():
        if set(tiles)!={0,1,2}: raise ValueError(f'incomplete three-tile patch: {key}: {set(tiles)}')
        out.append(dict(replicate=key[0],split=key[1],position=key[2],top=key[3],left=key[4],time=key[5],read=key[6],
                        **{field:sum(tiles[t][field] for t in (0,1,2)) for field in FIELDS}))
    out.sort(key=lambda x:(x['replicate'],x['split'],x['position'],x['top'],x['left'],x['time'],x['read']))
    if len(out)!=replicates*768: raise ValueError(f'expected {replicates*768} assembled patches; got {len(out)}')
    summary={}
    for rep in range(replicates):
        train=[x for x in out if x['replicate']==rep and x['split']=='train']
        val=[x for x in out if x['replicate']==rep and x['split']=='val']
        if len(train)!=384 or len(val)!=384: raise ValueError(f'expected 384 train and 384 val, replicate {rep}')
        x=np.asarray([r['sim_mvm'] for r in train]);y=np.asarray([r['ideal'] for r in train])
        slope,intercept=np.linalg.lstsq(np.stack((x,np.ones(len(x))),axis=1),y,rcond=None)[0]
        ideal=np.asarray([r['ideal'] for r in val]);sd=float(np.std(ideal))
        pred=intercept+slope*np.asarray([r['sim_mvm'] for r in val])
        corr=lambda z: float(np.corrcoef(ideal,z)[0,1]) if sd>0 and np.std(z)>0 else None
        report={'train_affine_intercept':float(intercept),'train_affine_slope':float(slope),'val_count':len(val),'ideal_sd':sd,
                'surrogate_val_rmse':float(np.sqrt(np.mean((pred-ideal)**2))),'surrogate_val_corr':corr(pred)}
        for field in ('programmed_readback','quantized_ideal','quantized_programmed','sim_mvm'):
            z=np.asarray([r[field] for r in val]);report[field]={'rmse':float(np.sqrt(np.mean((z-ideal)**2))),'corr':corr(z)}
        report['validation_gate_passed']=bool(report['sim_mvm']['corr'] is not None and report['sim_mvm']['corr']>=.8 and report['sim_mvm']['rmse']<=.6*sd)
        summary[str(rep)]=report
    return out, {'replicates':summary,'all_full75_gates_passed':all(r['validation_gate_passed'] for r in summary.values())}

def main():
    for filename,digest in EXPECTED.items():
        if sha(V2/filename)!=digest: raise ValueError(f'frozen input hash mismatch {filename}')
    protocol=json.loads((V2/'PROTOCOL.json').read_text())
    if protocol['runner_sha256']!=EXPECTED['pilot.py']:raise ValueError('frozen runner mismatch')
    records=[json.loads(s) for s in (V2/'patches.jsonl').read_text().splitlines()]
    if len(records)!=7011: raise ValueError(f'expected 7011 raw patch records, got {len(records)}')
    patches, summary=aggregate(records)
    summary.update(raw_records=len(records),assembled_patches=len(patches),source_sha256=EXPECTED)
    (HERE/'full75_patches_repaired.json').write_text(json.dumps(patches,indent=1,allow_nan=False)+'\n')
    (HERE/'full75_summary_repaired.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
