"""Clean PyTorch evaluation of four archived confirmatory checkpoints; no training."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.deps-classifier'),str(ROOT/'.deps'),str(ROOT)]
import numpy as np
import torch
from experiments.cifar_confirmatory_20261020.temporal_snn.model import TemporalCSNN
from experiments.compact_cifar_20261020.data import load_dataset,CLASSES,PINS
from experiments.cifar_resumable_20261019.durable import sha,atomic_json
from experiments.unn_rev3_20261014.paths import repo_path

HERE=Path(__file__).resolve().parent
SEEDS=(0,1,2,42)
EXPECTED_FS={0:.725,1:.739,2:.741,42:.754}
EXPECTED_TWD={0:.764,1:.770,2:.767,42:.782}


def encode(images):
    """Same float32 ToTensor/floor/clamp/x_min semantics as original LatencyEncoder."""
    x=torch.from_numpy(np.array(images,copy=True)).to(torch.float32)/255.
    t=torch.floor((1-x)*11).long().clamp(0,11)
    return ((torch.arange(12)[:,None,None,None,None]==t[None]) & (x[None]>=.1)).float()


def assets(seed):
    folder=HERE/'checkpoints'/('seed'+str(seed))
    cfg=json.loads((folder/'cfg.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    split=json.loads((folder/'split_manifest.json').read_text())
    if sha(folder/'best_fs.pt')!=summary['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from archived summary')
    if (cfg['seed']!=seed or cfg['use_twd'] is not True or cfg['twd_weight']!=.1 or cfg['memristor_sigma']!=0
        or cfg['T']!=12 or cfg['x_min']!=.1 or cfg['n_conv_layers']!=3 or
        [cfg[k] for k in ('c1_out','c2_out','c3_out','fc1_out')]!=[64,128,256,512] or
        cfg['dataset']!='cifar100:'+','.join(map(str,CLASSES)) or cfg['use_bntt'] is not True):
        raise ValueError('Wrong confirmatory architecture')
    if (split['split_seed']!=20261006 or len(split['train_indices'])!=4500 or len(split['val_indices'])!=500 or
        sorted(split['train_indices']+split['val_indices'])!=list(range(5000))):
        raise ValueError('Archived split differs')
    if summary['final_test_acc_fs_raw']!=EXPECTED_FS[seed] or summary['final_test_acc_twd_raw']!=EXPECTED_TWD[seed]:
        raise ValueError('Archived expected metrics differ')
    return folder,cfg,summary,split


def model_for(cfg,state):
    arguments={name:cfg[name] for name in inspect.signature(TemporalCSNN).parameters if name in cfg}
    arguments.update(in_channels=3,T_for_init=cfg['T'])
    model=TemporalCSNN(**arguments)
    model.load_state_dict(state,strict=True)
    return model.eval()


def scores(spikes,twd,labels):
    ticks=torch.arange(len(spikes))[:,None,None]
    first=torch.where(spikes>.5,ticks,torch.full_like(spikes,len(spikes))).min(0).values
    fs=first.argmin(1);tp=twd.argmax(1)
    silent=spikes.sum((0,2))==0
    fs_tie=(first==first.min(1).values[:,None]).sum(1)>1
    twd_tie=(torch.abs(twd-twd.max(1).values[:,None])<=1e-12).sum(1)>1
    rows=[]
    for i in range(len(labels)):
        rows.append(dict(label=int(labels[i]),first_spike_prediction=int(fs[i]),twd_prediction=int(tp[i]),
            silent=bool(silent[i]),first_spike_tied=bool(fs_tie[i]),twd_tied=bool(twd_tie[i]),
            first_spike_correct=bool(fs[i]==labels[i]),twd_correct=bool(tp[i]==labels[i]),
            strict_first_spike_correct=bool(fs[i]==labels[i] and not silent[i] and not fs_tie[i]),
            strict_twd_correct=bool(tp[i]==labels[i] and not silent[i] and not twd_tie[i]),
            output_spikes=spikes[:,i].to(torch.int8).tolist(),twd_scores=twd[i].tolist()))
    return rows


def evaluate(seeds,batch_size,output):
    if output.exists():raise FileExistsError('Choose a new output directory')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    images,labels=load_dataset('test')
    ids=np.flatnonzero(np.isin(labels,CLASSES))
    if len(ids)!=1000:raise ValueError('Expected exactly1000 official test examples')
    remap={c:i for i,c in enumerate(CLASSES)}
    test_y=torch.tensor([remap[int(labels[i])] for i in ids])
    output.mkdir(parents=True)
    manifest=dict(schema='confirmatory-clean-pytorch-eval-v1',source='clean_pytorch',
        native_memriboard=False,hardware=False,training=False,seeds=seeds,batch_size=batch_size,
        torch_version=torch.__version__,numpy_version=np.__version__,cpu_threads=2,
        test_dataset_sha256=PINS['test'],original_test_indices=ids.tolist(),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in HERE.rglob('*.py')},
        reference_metric='historical conventional FS/TWD; strict silent/tied-abstention reported separately',
        results=[])
    atomic_json(output/'start_manifest.json',manifest)
    for seed in seeds:
        folder,cfg,old,split=assets(seed);begin=time.monotonic()
        state=torch.load(folder/'best_fs.pt',map_location='cpu',weights_only=True)
        model=model_for(cfg,state);rows=[]
        with torch.inference_mode():
            for start in range(0,len(ids),batch_size):
                end=min(start+batch_size,len(ids))
                _,spikes,twd=model(encode(images[ids[start:end]]))
                rows.extend(scores(spikes,twd,test_y[start:end]))
        for row,original in zip(rows,ids):row['original_test_index']=int(original)
        n=len(rows)
        result=dict(seed=seed,source='clean_pytorch',count=n,wall_seconds=time.monotonic()-begin,
            checkpoint_sha256=sha(folder/'best_fs.pt'),cfg_sha256=sha(folder/'cfg.json'),
            split_sha256=sha(folder/'split_manifest.json'),historical_fs=EXPECTED_FS[seed],historical_twd=EXPECTED_TWD[seed],
            fs=sum(r['first_spike_correct'] for r in rows)/n,twd=sum(r['twd_correct'] for r in rows)/n,
            strict_fs=sum(r['strict_first_spike_correct'] for r in rows)/n,
            strict_twd=sum(r['strict_twd_correct'] for r in rows)/n,
            silent=sum(r['silent'] for r in rows),fs_tied=sum(r['first_spike_tied'] for r in rows),
            twd_tied=sum(r['twd_tied'] for r in rows),predictions=rows)
        result['historical_counts_match']=result['fs']==EXPECTED_FS[seed] and result['twd']==EXPECTED_TWD[seed]
        atomic_json(output/('seed'+str(seed)+'.json'),result)
        manifest['results'].append({k:v for k,v in result.items() if k!='predictions'})
        print(json.dumps(manifest['results'][-1]),flush=True)
    manifest['status']='completed'
    manifest['result_sha256']={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    atomic_json(output/'manifest.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS),choices=SEEDS)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--output-dir')
    a=p.parse_args()
    if not 1<=a.batch_size<=32 or len(set(a.seeds))!=len(a.seeds):raise ValueError('Invalid batch/seeds')
    if a.dry_run:
        result=dict(status='ready_no_forward_executed',source='clean_pytorch',training=False,
            assets=[dict(seed=s,checkpoint_sha256=sha(assets(s)[0]/'best_fs.pt')) for s in a.seeds],
            dataset_present=(ROOT/'data/cifar-100-python/test').is_file(),test_count_per_seed=1000,
            native_memriboard=False)
        print(json.dumps(result))
    else:
        output=repo_path(a.output_dir) if a.output_dir else ROOT/'results/cifar_confirmatory_20261020'/str(uuid.uuid4())
        evaluate(a.seeds,a.batch_size,output)

if __name__=='__main__':main()
