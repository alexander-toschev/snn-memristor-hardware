"""Finish an interrupted local MNIST run: fresh head25, frozen new-W test once."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.mnist_stdp_finish_20261019 import qualification as q

QUALIFICATION_SHA = 'fd211da4f6f0140a34949bf70866cb61fd182bbf4d03da552e7ee5b21725ad71'
SPEC = dict(schema='mnist-qualified-interrupted-readout-finish-v1',
    input_snapshot_sha256='321bceaf0e7fc390883c70ec43206526b066de15ce5cce6cf41bfb2b3e7f5fee',
    qualification_source_sha256=QUALIFICATION_SHA,
    independent_prefix_audit_sha256='36b20afb4043ba7f9f26fb94f8eaef960c9120ef95d2de352cc52ada6c646641',
    inherited_training_images=30000, inherited_train_feature_rows=30000,
    new_stdp_images=0, new_train_feature_rows=0, fresh_head_epochs=25, head_updates=1475,
    head_seed=42, shuffle_seed=42, replay_completed_prefix_epochs=5,
    prefix_train_loss_absolute_tolerance=1e-6, prefix_train_loss_relative_tolerance=1e-5,
    test_samples=10000, test_feature_batches=79, test_evaluations=1, test_timesteps=100,
    frozen_new_weights=True, physical_memristor=False, reconstruction=True,
    maximum_wall_seconds=7200, maximum_output_bytes=3*1024**3, initial_disk_bytes=8*1024**3)
APPROVED_SPEC_SHA256 = frozenset({'a54858f977c06f6adfeb7a1f77a76dec686ed6378df2a68e9e17a5734af62fda'})


def merge_sources(*inventories):
    result={}
    for inventory in inventories:
        for name,pin in inventory.items():
            q.require(name not in result or result[name]==pin,'Conflicting source pins: '+name)
            result[name]=pin
    return result


def helpers():
    q.require(q.sha(q.__file__) == QUALIFICATION_SHA, 'Qualification source changed')
    paths = {'experiments/mnist_stdp_retraining_20261019/experiment.py': q.TRAINING_SHA,
             'experiments/mnist_full_recovery_20261019/experiment.py': q.RECOVERY_SHA,
             'experiments/mnist_stdp_retraining_20261019/run_full.py': q.FULL_SHA}
    for name, pin in paths.items(): q.require(q.sha(ROOT/name) == pin, 'Frozen helper changed')
    from experiments.mnist_stdp_retraining_20261019 import experiment as training
    from experiments.mnist_full_recovery_20261019 import experiment as recovery
    return training, recovery, paths


class Budget:
    def __init__(self, output): self.output, self.started = output, time.monotonic()
    def check(self):
        q.require(time.monotonic()-self.started <= SPEC['maximum_wall_seconds'], 'Recovery wall budget exhausted')
        q.require(sum(p.stat().st_size for p in self.output.iterdir() if p.is_file()) <= SPEC['maximum_output_bytes'], 'Recovery output budget exhausted')
        q.require(shutil.disk_usage(ROOT).free >= 2*1024**3, 'Recovery disk reserve')


def validate_prefix_epoch(row, original):
    for key in ('epoch', 'updates', 'train_examples', 'order_sha256', 'test_access'):
        q.require(row[key] == original[key], 'Reinitialized head differs from prior prefix: ' + key)
    q.require(math.isclose(row['train_loss'], original['train_loss'], abs_tol=SPEC['prefix_train_loss_absolute_tolerance'],
                          rel_tol=SPEC['prefix_train_loss_relative_tolerance']), 'Reinitialized head train loss differs beyond fixed tolerance')


def restore_frozen_network(p, ref, checkpoint, expected):
    import torch
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    cfg = ref.CSCfg(**checkpoint['cfg']); cfg.device = 'cuda'
    net, _, lif, conn = ref.build_csnn(cfg)
    # Never load the historical model.pt: these are the newly trained source-session weights.
    with torch.no_grad():
        conn.w.copy_(checkpoint['W'].to('cuda'))
        threshold = getattr(lif, 'v_thresh', getattr(lif, 'thresh', None))
        threshold.copy_(checkpoint['v_thresh'].to('cuda'))
    net.train(False)
    conn.update_rule.nu = (torch.tensor(0., device='cuda'), torch.tensor(0., device='cuda'))
    q.require(conn.norm is None and conn.Dales_rule is None, 'Frozen extractor has native transformation')
    q.require(p.state_hashes(net) == expected, 'Restored network differs from all inherited train feature batches')
    net.reset_state_variables()
    return net, lif, ref.build_encoder_from_cfg(cfg)


def execute(output, training, recovery, p, ref, qualified, prefix, checkpoint, manifest):
    import numpy as np
    import torch
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.set_num_threads(2)
    budget=Budget(output); manifest['network_executed']=True
    counts=np.load(q.SOURCE/'train_counts.npy',mmap_mode='r',allow_pickle=False)
    labels=np.load(q.SOURCE/'train_labels.npy',allow_pickle=False)
    mean=torch.from_numpy(np.load(q.SOURCE/'scaler_mean.npy',allow_pickle=False))
    sd=torch.from_numpy(np.load(q.SOURCE/'scaler_sd.npy',allow_pickle=False))
    for name in ('scaler_mean.npy','scaler_sd.npy','trained_model.pt','trained_weights.npy','trained_threshold.npy'):
        shutil.copyfile(q.SOURCE/name, output/name)
    with (output/'events.jsonl').open('x',encoding='utf-8') as events:
        def log(row):
            if row['kind']=='head_epoch' and row['epoch']<=5:
                i=row['epoch']-1; validate_prefix_epoch(row,prefix['heads'][i])
                for name in ['head_initial_weight.npy','head_initial_bias.npy','epoch%02d_train_order.npy'%i,
                             'shuffle_rng_before_epoch%02d.npy'%i]:
                    q.require(q.sha(output/name)==q.sha(q.SOURCE/name),'Fresh head/RNG/prefix artifact mismatch: '+name)
            events.write(json.dumps(row,allow_nan=False)+'\n'); events.flush()
        log(dict(kind='inherited_prefix_qualified',qualification=qualified))
        head,updates=recovery.train_head(counts,labels,mean,sd,output,log,budget)
        q.require(updates==1475,'Wrong fresh-head update count')
        # This RNG is saved before the interrupted sixth epoch, so it must also match.
        q.require(q.sha(output/'shuffle_rng_before_epoch05.npy')==q.sha(q.SOURCE/'shuffle_rng_before_epoch05.npy'),'Post-prefix shuffle state mismatch')
        manifest['fresh_head_prefix_reproduced']=True; manifest['fresh_head_updates']=updates
        fixed={name:q.sha(output/name) for name in ('head_weight.npy','head_bias.npy','scaler_mean.npy','scaler_sd.npy','trained_model.pt')}
        log(dict(kind='head_frozen_before_test',sha256=fixed))
        net,lif,encoder=restore_frozen_network(p,ref,checkpoint,qualified['frozen_state'])
        log(dict(kind='restored_new_weight_network',frozen_state=p.state_hashes(net),learning=False,norm_none=True))
        path,test_labels=training.collect_new_weights(p,net,lif,encoder,qualified['frozen_state'],'test',output,log,budget)
        test=np.load(path,mmap_mode='r',allow_pickle=False)
        result=recovery.evaluate_once(head,test,test_labels,mean,sd,output,budget)
        q.require(p.state_hashes(net)==qualified['frozen_state'],'Frozen network changed')
        q.require(all(q.sha(output/name)==pin for name,pin in fixed.items()),'Head/scaler/checkpoint changed during test')
        q.verify_snapshot()  # Original interrupted evidence must still be byte-identical.
        log(dict(kind='test_complete',**result))
        manifest.update(result,status='completed',scope='inherited_stdp30000_fresh_readout25_test10000',
            confirmed_execution=dict(new_stdp_images=0,new_train_feature_rows=0,fresh_head_epochs=25,fresh_head_updates=1475,
                                     test_feature_rows=10000,test_feature_batches=79,test_evaluations=1))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--run',action='store_true');args=parser.parse_args(argv)
    spec_sha=q.digest(SPEC)
    if args.run:q.require(spec_sha in APPROVED_SPEC_SHA256,'Recovery requires independent source review and registration')
    q.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG',':4096:8')==':4096:8','Unreviewed determinism setting')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    training,recovery,pins=helpers();p=recovery.helpers()
    checked=recovery.preflight(p);ref,_=training.inputs(p)
    import torch
    torch.set_num_threads(2)  # Match the saved train-only CPU scaler reductions.
    q.require(shutil.disk_usage(ROOT).free>=SPEC['initial_disk_bytes'],'Recovery needs8GiB free disk')
    qualified,prefix,start,checkpoint=q.qualify(recovery)
    q.require(checked['environment']['full_installed_versions']==start['environment']['full_installed_versions'],'Recovery package environment changed')
    q.require(set(checked['source_sha256'])<=set(start['source_sha256']),'Preflight added unreviewed source inventory')
    pins=merge_sources(pins,checked['source_sha256'],start['source_sha256'],
        {path.relative_to(ROOT).as_posix():q.sha(path) for path in Path(__file__).parent.iterdir()
         if path.is_file() and path.suffix in ('.py','.md','.json')})
    if args.preflight:
        print(json.dumps(dict(spec=SPEC,spec_sha256=spec_sha,qualification=qualified,network_executed=False,source_sha256=pins)));return 0
    output=ROOT/'results/mnist_stdp_finish_20261019'/str(uuid.uuid4());output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='failed',started_utc=p.now(),spec=SPEC,spec_sha256=spec_sha,source_sha256=pins,
        environment=checked['environment'],entrypoint_sha256=q.sha(__file__),physical_memristor=False,network_executed=False,
        source_session=q.SOURCE.relative_to(ROOT).as_posix(),input_snapshot_sha256=q.SNAPSHOT_SHA,
        inherited=qualified['inherited'],planned=dict(fresh_head_epochs=25,fresh_head_updates=1475,test_rows=10000,test_evaluations=1))
    p.write_json(output/'start_manifest.json',manifest);p.write_json(output/'source_qualification.json',qualified)
    try: execute(output,training,recovery,p,ref,qualified,prefix,checkpoint,manifest)
    except (Exception,KeyboardInterrupt) as exc:manifest['error']=type(exc).__name__+': '+str(exc)
    finally:
        manifest['finished_utc']=p.now();manifest['result_sha256']={path.name:q.sha(path) for path in output.iterdir() if path.is_file()}
        p.write_json(output/'manifest.json',manifest)
    print(json.dumps(dict(status=manifest['status'],output=output.relative_to(ROOT).as_posix(),accuracy=manifest.get('accuracy'),error=manifest.get('error'))))
    return 0 if manifest['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
