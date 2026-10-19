"""Offline two-session MNIST recovery audit; never runs an SNN or optimizer."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.mnist_stdp_finish_20261019 import qualification as q

APPROVED_ENTRY_SHA256=frozenset({'09cfab90a2ba56e7f2a3165370b4d020e1056baaa17a671c61748b9a1ac566b4'})
FULL_AUDITOR_SHA='1f41fb863642b482bbefe49c6524ba20320b7b6ad2c93e88351c00aedb535abb'


def schedule(rows,epochs=25,size=10000):
    it=iter(rows)
    def take(kind):
        value=next(it,None);q.require(value is not None and value['kind']==kind,'Recovery expected event '+kind);return value
    inherited=take('inherited_prefix_qualified');head=[]
    for i in range(epochs):
        row=take('head_epoch')
        q.require(row['epoch']==i+1 and row['updates']==59*(i+1) and row['train_examples']==30000 and row['test_access'] is False,'Recovery head coverage')
        head.append(row)
    frozen=take('head_frozen_before_test');restored=take('restored_new_weight_network');features=[]
    for start in range(0,size,128):
        row=take('new_weight_features')
        q.require((row['split'],row['start'],row['end'])==('test',start,min(start+128,size)),'Recovery test coverage')
        features.append(row)
    split=take('feature_split_complete')
    q.require(split==dict(kind='feature_split_complete',split='test',rows=size,new_weights=True),'Recovery test completion')
    evaluation=take('test_complete');q.require(next(it,None) is None,'Unexpected recovery events')
    return dict(inherited=inherited,head=head,frozen=frozen,restored=restored,features=features,evaluation=evaluation)


def audit(session):
    import numpy as np
    import torch
    q.require(torch.__version__=='2.4.1+cu124' and np.__version__=='2.0.2','Use the pinned isolated MNIST environment')
    torch.set_num_threads(2)
    path=(ROOT/session).resolve();q.require(not Path(session).is_absolute() and path.parent==ROOT/'results/mnist_stdp_finish_20261019','Wrong recovery session path')
    entry=ROOT/'experiments/mnist_stdp_finish_20261019/experiment.py'
    q.require(q.sha(entry) in APPROVED_ENTRY_SHA256,'Recovery entrypoint has not been independently registered for audit')
    from experiments.mnist_stdp_finish_20261019 import experiment as e
    _,recovery,helper_pins=e.helpers()
    qualified,prefix,source_start,_=q.qualify(recovery)
    m=json.loads((path/'manifest.json').read_text());start=json.loads((path/'start_manifest.json').read_text())
    q.require(m['status']=='completed' and m['scope']=='inherited_stdp30000_fresh_readout25_test10000','Only completed recovery can be evaluated')
    q.require(m['entrypoint_sha256']==q.sha(entry) and m['spec']==e.SPEC and m['spec_sha256']==q.digest(e.SPEC)
              and m['spec_sha256'] in e.APPROVED_SPEC_SHA256,'Recovery source/spec binding')
    q.require(m['environment']['full_installed_versions']==source_start['environment']['full_installed_versions'],'Environment continuity')
    for key,value in start.items():
        if key not in ('status','network_executed'):q.require(m[key]==value,'Recovery start/final mismatch:'+key)
    q.require(start['network_executed'] is False and m['network_executed'] is True,'Execution-state marker')
    expected_sources=dict(source_start['source_sha256']);expected_sources.update(helper_pins)
    expected_sources.update({file.relative_to(ROOT).as_posix():q.sha(file) for file in entry.parent.iterdir()
                             if file.is_file() and file.suffix in ('.py','.md','.json')})
    q.require(m['source_sha256']==expected_sources,'Exact recovery source inventory differs')
    for name,pin in m['source_sha256'].items():
        source=(ROOT/name).resolve();q.require(source.is_relative_to(ROOT) and q.sha(source)==pin,'Recovery source bytes')
    q.require({p.name for p in path.iterdir()}==set(m['result_sha256'])|{'manifest.json'},'Recovery evidence inventory')
    for name,pin in m['result_sha256'].items():
        item=(path/name).resolve();q.require(item.parent==path and q.sha(item)==pin,'Recovery evidence SHA')
    q.require(json.loads((path/'source_qualification.json').read_text())==qualified,'Inherited qualification changed')
    with (path/'events.jsonl').open() as stream:ev=schedule(json.loads(line) for line in stream)
    q.require(ev['inherited']['qualification']==qualified and m['source_session']==q.SOURCE.relative_to(ROOT).as_posix()
              and m['input_snapshot_sha256']==q.SNAPSHOT_SHA and m['inherited']==qualified['inherited'],'Two-session lineage')
    expected_actual=dict(new_stdp_images=0,new_train_feature_rows=0,fresh_head_epochs=25,fresh_head_updates=1475,
                         test_feature_rows=10000,test_feature_batches=79,test_evaluations=1)
    q.require(m['confirmed_execution']==expected_actual and m['fresh_head_updates']==1475 and m['fresh_head_prefix_reproduced'] is True,'Actual recovery counters')
    load=lambda name:np.load(path/name,mmap_mode='r',allow_pickle=False)
    for name in ('scaler_mean.npy','scaler_sd.npy','trained_model.pt','trained_weights.npy','trained_threshold.npy','head_initial_weight.npy','head_initial_bias.npy'):
        q.require(q.sha(path/name)==q.sha(q.SOURCE/name),'Recovered/source artifact differs:'+name)
    # Reproduce each DataLoader shuffle without training; validate its RNG transitions.
    from torch.utils.data import DataLoader
    for i,row in enumerate(ev['head']):
        order=load('epoch%02d_train_order.npy'%i)
        q.require(order.dtype==np.int64 and np.array_equal(np.sort(order),np.arange(30000)) and q.array_sha(order)==row['order_sha256'],'Head permutation evidence')
        rng=np.load(path/('shuffle_rng_before_epoch%02d.npy'%i),allow_pickle=False)
        generator=torch.Generator();generator.set_state(torch.from_numpy(rng))
        replay=[]
        for ids in DataLoader(range(30000),batch_size=512,shuffle=True,num_workers=0,drop_last=False,generator=generator):replay.extend(ids.tolist())
        q.require(np.array_equal(order,np.asarray(replay,np.int64)),'Readout shuffle replay differs')
        next_name='shuffle_rng_final.npy' if i==24 else 'shuffle_rng_before_epoch%02d.npy'%(i+1)
        q.require(np.array_equal(generator.get_state().numpy(),load(next_name)),'Shuffle RNG transition differs')
        if i<5:
            e.validate_prefix_epoch(row,prefix['heads'][i])
            for name in ('epoch%02d_train_order.npy'%i,'shuffle_rng_before_epoch%02d.npy'%i):q.require(q.sha(path/name)==q.sha(q.SOURCE/name),'Prior5epoch replay differs')
    q.require(q.sha(path/'shuffle_rng_before_epoch05.npy')==q.sha(q.SOURCE/'shuffle_rng_before_epoch05.npy'),'Prior epoch6-start RNG differs')
    q.require(ev['restored']==dict(kind='restored_new_weight_network',frozen_state=qualified['frozen_state'],learning=False,norm_none=True),'Restored frozen network')
    fixed={'head_weight.npy','head_bias.npy','scaler_mean.npy','scaler_sd.npy','trained_model.pt'}
    q.require(set(ev['frozen']['sha256'])==fixed,'Frozen artifact inventory')
    for name,pin in ev['frozen']['sha256'].items():q.require(q.sha(path/name)==pin,'Head/scaler changed after test')
    counts=load('test_counts.npy');labels=load('test_labels.npy');ids=load('test_ids.npy')
    q.require(counts.shape==(10000,36864) and counts.dtype==np.float32 and labels.dtype==ids.dtype==np.int64
              and np.array_equal(ids,np.arange(10000)),'Test shapes/IDs')
    import gzip,struct
    with gzip.open(ROOT/'experiments/mnist_baseline_20261019/data/MNIST/raw/t10k-labels-idx1-ubyte.gz','rb') as stream:
        q.require(struct.unpack('>II',stream.read(8))==(2049,10000),'Test IDX header')
        q.require(np.array_equal(labels,np.frombuffer(stream.read(),np.uint8).astype(np.int64)),'Official test labels')
    for row in ev['features']:
        block=counts[row['start']:row['end']]
        q.require(np.isfinite(block).all() and ((block>=0)&(block<=100)&(block==np.round(block))).all(),'Invalid test counts')
        q.require(q.array_sha(block)==row['counts_sha256'] and row['frozen_state']==qualified['frozen_state'],'New-weight test feature binding')
    q.require(q.sha(ROOT/'tools/audit_mnist_stdp_full.py')==FULL_AUDITOR_SHA,'Reviewed CPU arithmetic audit helper changed')
    from tools import audit_mnist_stdp_full as arithmetic
    result=arithmetic.predictions(load('test_logits.npy'),load('test_predictions.npy'),labels,load('test_confusion.npy'))
    replay=arithmetic.linear_replay(counts,load('scaler_mean.npy'),load('scaler_sd.npy'),load('head_weight.npy'),load('head_bias.npy'),load('test_logits.npy'))
    expected=dict(**result,test_evaluations=1,selected_epoch=25,hardware_accuracy=False,reconstruction=True)
    q.require(ev['evaluation']==dict(kind='test_complete',**expected) and all(m[k]==v for k,v in expected.items()),'Final recovered test metrics')
    return dict(schema='mnist-two-session-recovery-audit-v1',status='pass' if replay['passed'] else 'review_required',
        auditor_sha256=q.sha(__file__),recovery_manifest_sha256=q.sha(path/'manifest.json'),source_snapshot_sha256=q.SNAPSHOT_SHA,
        source_session=q.SOURCE.relative_to(ROOT).as_posix(),recovery_session=path.relative_to(ROOT).as_posix(),
        inherited=qualified['inherited'],actual_new_execution=expected_actual,discarded_source_head_epochs=5,
        prefix_head_replay_verified=True,all25_shuffle_transitions_replayed=True,linear_readout_replay=replay,**result,
        physical_memristor=False,reconstruction=True,
        limits=['No STDP or feature-SNN forward replay; source trajectory and feature telemetry verified.',
                'Fresh head25 from seed42, not optimizer resume from epoch6.',
                'Shuffle and linear predictions replayed; gradient updates not independently retrained.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();target=(ROOT/args.output).resolve()
    q.require(not Path(args.output).is_absolute() and target.is_relative_to(ROOT/'reports') and not target.exists(),'New repository report required')
    result=audit(args.session);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result));return 0 if result['status']=='pass' else 3


if __name__=='__main__':raise SystemExit(main())
