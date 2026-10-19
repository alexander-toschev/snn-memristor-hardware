"""Fresh MNIST native-model STDP, then evaluation of its new weights. No hardware."""
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.mnist_full_recovery_20261019 import experiment as recovery

RECOVERY_SHA='06b370e89135981ce756442b715e04ed280498de150cc42f43dfd39b4d4ce394'
SUMMARY=ROOT/'experiments/mnist_baseline_20261019/evidence/training/summary.json'
SUMMARY_SHA='182dc87cbcb9ab1a497b6c6290f9abea0602e657a30081faf79df62e79677e71'
SPEC=dict(schema='mnist-native-model-stdp-retraining-v1',dataset='MNIST',training_ids='first consecutive train IDs',
          full_training_samples=30000,smoke_samples=128,training_batch=1,timesteps=100,seed=42,
          initialization='fresh archived uniform[0,.3] then perfilterL1=78.4; no checkpoint load',
          rule='native WeightDependentPostPre',nu_pre=.001,nu_post=.0001,wmin=0.,wmax=1.,
          hook='preserve undefined connection; E/I and postwindow renorm skipped',
          learning=True,one_step=False,reset_after_each_image=True,
          readout_spec_sha256=recovery.digest(recovery.SPEC),evaluation='only newly trained weights',
          smoke_maximum_seconds=30*60,full_maximum_seconds=8*60*60,maximum_output_bytes=8*1024**3,
          native_model=True,physical_memristor=False,reconstruction=True)
APPROVED_SMOKE_SPEC_SHA256=frozenset({'db6a90ce72a25a37144971f405c6a23097de83c9583dd0c9e713848c50e205bd'})
APPROVED_FULL_SPEC_SHA256=frozenset()


def inputs(p):
    if recovery.sha(recovery.__file__)!=RECOVERY_SHA:raise ValueError('Readout helper source changed')
    if recovery.sha(SUMMARY)!=SUMMARY_SHA:raise ValueError('Parent training summary changed')
    source=json.loads(SUMMARY.read_text(encoding='utf-8'))
    overrides=dict(source['cfg']);overrides.pop('arch')
    expected=dict(N=30000,time=100,device='cuda',c1_out=64,c1_kernel=5,c1_stride=1,c1_pad=0,
                  nu_minus=.001,nu_plus=.0001,poisson_deterministic=True,poisson_rate_scale=.0015,
                  encoder_rate_boost=1.,thresh_init=.38,w_norm_enable=True,w_norm_target=78.4,
                  ei_enable=True,ei_exc=22.5,ei_inh=120)
    if any(overrides.get(k)!=v for k,v in expected.items()) or overrides.get('resume_checkpoint'):
        raise ValueError('Unexpected parent training configuration')
    sys.path.insert(0,str(p.REFERENCE));ref=importlib.import_module('csnn_mnist_net')
    if Path(ref.__file__).resolve()!=p.REFERENCE/'csnn_mnist_net.py' or hasattr(ref,'connection'):
        raise ValueError('Reference semantics changed')
    cfg=ref.CSCfg(**overrides)
    if cfg.seed!=42 or cfg.poisson_base_seed!=123 or cfg.resume_checkpoint is not None:raise ValueError('Unexpected defaults')
    return ref,cfg


def threshold_hashes(p,net):
    import torch
    result={}
    for name,layer in net.layers.items():
        value=getattr(layer,'v_thresh',getattr(layer,'thresh',None))
        if torch.is_tensor(value):result[name]=p.tensor_sha(value)
    return result


def check_learning(net,conn):
    import torch
    if set(net.connections)!={('Input','Conv1')} or not net.learning or not net.training:
        raise ValueError('Wrong learning connection/network state')
    if any(not layer.learning or not layer.training for layer in net.layers.values()):raise ValueError('Layer learning disabled')
    if type(conn.update_rule).__name__!='WeightDependentPostPre':raise ValueError('Wrong native STDP rule')
    rates=torch.as_tensor(conn.update_rule.nu).detach().cpu()
    if not torch.equal(rates,torch.tensor([.001,.0001],dtype=torch.float32)):raise ValueError('Wrong STDP rates/sign')
    if conn.norm is not None or conn.Dales_rule is not None:raise ValueError('Unexpected native normalization/transform')
    if float(conn.wmin)!=0 or float(conn.wmax)!=1 or conn.update_rule.weight_decay!=0:raise ValueError('Wrong native bounds/decay')
    if tuple(conn.b.shape)!=(64,) or not torch.all(conn.b==0):raise ValueError('Unexpected bias')


def assert_clipping_trace(initial,trace,final):
    import torch
    if tuple(trace.shape)!=(100,64,1,5,5):raise ValueError('First-window weight monitor incomplete')
    if not torch.equal(trace[0],initial.clamp(0,1)):
        raise ValueError('First synchronous zero-state update was not pure clipping')
    if not torch.equal(trace[-1],final):raise ValueError('Post-window/native normalization changed final weights')
    return dict(first_step_clipping_verified=True,initial_out_of_bounds=int(((initial<0)|(initial>1)).sum()),
                clipping_l1=float((trace[0]-initial).abs().sum()),
                post_first_step_changed_coefficients=int((trace[-1]!=trace[0]).sum()),
                post_first_step_l1=float((trace[-1]-trace[0]).abs().sum()),
                monitor_semantics='native post-update[t], before post-window hook/normalization')


class Budget(recovery.Budget):
    def __init__(self,output,smoke):
        super().__init__(output);self.seconds=SPEC['smoke_maximum_seconds' if smoke else 'full_maximum_seconds']
    def check(self):
        # This protocol has its own fixed wall budget; preserve storage checks.
        import shutil
        if time.monotonic()-self.start>self.seconds:raise RuntimeError('Fixed protocol wall budget reached')
        if sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())>SPEC['maximum_output_bytes']:raise RuntimeError('Output budget exceeded')
        if shutil.disk_usage(ROOT).free<2*1024**3:raise RuntimeError('Disk reserve reached')


def training(p,ref,cfg,size,output,log,budget):
    import numpy as np
    import torch
    from bindsnet.network.monitors import Monitor
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);random.seed(42)
    net,_,lif,conn=ref.build_csnn(cfg);net.train(True);check_learning(net,conn)
    initial=conn.w.detach().cpu().clone()
    if tuple(initial.shape)!=(64,1,5,5) or not torch.isfinite(initial).all() or (initial<0).any():raise ValueError('Invalid initial weights')
    if not torch.allclose(initial.sum((1,2,3)),torch.full((64,),78.4),atol=1e-5,rtol=1e-6):raise ValueError('Initial norm differs')
    np.save(output/'initial_weights.npy',initial.numpy(),allow_pickle=False)
    np.save(output/'construction_rng_cpu.npy',torch.get_rng_state().numpy(),allow_pickle=False)
    np.save(output/'construction_rng_cuda.npy',torch.cuda.get_rng_state().cpu().numpy(),allow_pickle=False)
    threshold_before=threshold_hashes(p,net);bias_sha=p.tensor_sha(conn.b)
    encoder=ref.build_encoder_from_cfg(cfg)
    spike_monitor=Monitor(lif,state_vars=['s'],time=100);net.add_monitor(spike_monitor,name='stdp_output')
    weight_monitor=Monitor(conn,state_vars=['w'],time=100);net.add_monitor(weight_monitor,name='first_window_weights')
    image_bytes=p.idx_prefix(p.DATA/'train-images-idx3-ubyte.gz',True,60000,size)
    images=np.frombuffer(image_bytes,dtype=np.uint8).copy().reshape(size,1,28,28)
    np.save(output/'stdp_train_ids.npy',np.arange(size,dtype=np.int64),allow_pickle=False)
    trajectory=np.lib.format.open_memmap(output/'stdp_end_of_image_weights.npy',mode='w+',dtype=np.float32,shape=(size,64,1,5,5))
    total_spikes=0;changed_after_first=0;first_evidence=None;started=time.perf_counter()
    previous=initial;torch.cuda.reset_peak_memory_stats()
    try:
        for index in range(size):
            budget.check();check_learning(net,conn)
            if threshold_hashes(p,net)!=threshold_before or p.tensor_sha(conn.b)!=bias_sha:raise ValueError('Threshold/bias drift')
            begin=time.perf_counter()
            with torch.no_grad():
                x=torch.from_numpy(images[index]).to(device='cuda',dtype=torch.float32).div_(255.)
                spikes=encoder(x)
                if tuple(spikes.shape)!=(100,1,1,28,28) or not ((spikes==0)|(spikes==1)).all():raise ValueError('Invalid encoded single image')
                input_sha=p.tensor_sha(spikes)
                net.run(inputs={'Input':spikes},time=100)
                torch.cuda.synchronize()
                recorded=spike_monitor.get('s')
                if tuple(recorded.shape)!=(100,1,64,24,24) or not ((recorded==0)|(recorded==1)).all():raise ValueError('Invalid native spike evidence')
                current=conn.w.detach().cpu().clone()
                if not torch.isfinite(current).all() or ((current<0)|(current>1)).any():raise ValueError('Post-update weights invalid')
                if index==0:
                    trace=weight_monitor.get('w');first_evidence=assert_clipping_trace(initial,trace,current)
                    np.save(output/'first_image_postupdate_weights.npy',trace.numpy(),allow_pickle=False)
                    log(dict(kind='first_image_clipping_and_learning',**first_evidence))
                    net.monitors.pop('first_window_weights');del trace,weight_monitor
                elif not torch.equal(current,previous):changed_after_first+=1
                if threshold_hashes(p,net)!=threshold_before or p.tensor_sha(conn.b)!=bias_sha:raise ValueError('Training changed fixed thresholds/bias')
                trajectory[index]=current.numpy();trajectory.flush()
                count=int(recorded.sum().item());total_spikes+=count
                log(dict(kind='stdp_image',image_id=index,steps=100,input_sha256=input_sha,input_spikes=int(spikes.sum().item()),
                    output_spikes=count,weight_sha256=p.tensor_sha(current),weight_min=float(current.min()),weight_max=float(current.max()),
                    changed_coefficients=int((current!=previous).sum()),delta_l1=float((current-previous).abs().sum()),
                    threshold_sha256=threshold_before,bias_sha256=bias_sha,native_norm_none=conn.norm is None,
                    seconds=time.perf_counter()-begin,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            previous=current;del x,spikes,recorded
            net.reset_state_variables()
            if (index+1)%32==0 or index+1==size:print(json.dumps(dict(stage='stdp',done=index+1,total=size)),flush=True)
        final=conn.w.detach().cpu().clone();threshold=getattr(lif,'v_thresh',getattr(lif,'thresh',None)).detach().cpu().clone()
        np.save(output/'trained_weights.npy',final.numpy(),allow_pickle=False);np.save(output/'trained_threshold.npy',threshold.numpy(),allow_pickle=False)
        checkpoint=dict(W=final,v_thresh=threshold,cfg=vars(cfg),trained_samples=size,initial_weight_sha256=p.tensor_sha(initial),
                        final_weight_sha256=p.tensor_sha(final),new_reconstruction=True,source_protocol_sha256=recovery.digest(SPEC))
        torch.save(checkpoint,output/'trained_model.pt')
        restored=torch.load(output/'trained_model.pt',weights_only=True,map_location='cpu')
        if not torch.equal(restored['W'],final) or not torch.equal(restored['v_thresh'],threshold):raise ValueError('New checkpoint roundtrip mismatch')
        summary=dict(training_samples=size,timesteps=size*100,training_seconds=time.perf_counter()-started,output_spikes=total_spikes,
            images_with_change_after_initial_image=changed_after_first,initial_weight_sha256=p.tensor_sha(initial),
            final_weight_sha256=p.tensor_sha(final),clipping=first_evidence,
            learning_beyond_initial_clipping_observed=bool(changed_after_first or first_evidence['post_first_step_changed_coefficients']))
        log(dict(kind='stdp_training_complete',**summary))
        net.monitors.pop('stdp_output');net.reset_state_variables()
        return net,lif,conn,encoder,summary
    finally:trajectory.flush();del trajectory


def collect_new_weights(p,net,lif,encoder,frozen,split,output,log,budget):
    import numpy as np
    import torch
    from bindsnet.network.monitors import Monitor
    size=30000 if split=='train' else 10000;prefix='train' if split=='train' else 't10k'
    source_count=60000 if split=='train' else 10000
    images=np.frombuffer(p.idx_prefix(p.DATA/(prefix+'-images-idx3-ubyte.gz'),True,source_count,size),dtype=np.uint8).copy().reshape(size,1,28,28)
    labels=np.frombuffer(p.idx_prefix(p.DATA/(prefix+'-labels-idx1-ubyte.gz'),False,source_count,size),dtype=np.uint8).astype(np.int64)
    np.save(output/(split+'_ids.npy'),np.arange(size,dtype=np.int64),allow_pickle=False);np.save(output/(split+'_labels.npy'),labels,allow_pickle=False)
    path=output/(split+'_counts.npy');counts=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(size,36864))
    monitor=Monitor(lif,state_vars=['s'],time=100);net.add_monitor(monitor,name='new_weight_features')
    try:
        for start,end in recovery.batches(size,128):
            budget.check();net.reset_state_variables()
            if p.state_hashes(net)!=frozen:raise ValueError('New trained weights changed before feature batch')
            begin=time.perf_counter()
            with torch.no_grad():
                x=torch.from_numpy(images[start:end]).to(device='cuda',dtype=torch.float32).div_(255.)
                spikes=encoder(x)
                if tuple(spikes.shape)!=(100,end-start,1,28,28) or not ((spikes==0)|(spikes==1)).all():raise ValueError('Feature input invalid')
                net.run(inputs={'Input':spikes},time=100);torch.cuda.synchronize()
                recorded=monitor.get('s')
                if tuple(recorded.shape)!=(100,end-start,64,24,24) or not ((recorded==0)|(recorded==1)).all():raise ValueError('Feature output invalid')
                values=recorded.to(torch.float32).sum(0).reshape(end-start,36864);recovery.validate_counts(values)
                if p.state_hashes(net)!=frozen:raise ValueError('Learning/threshold drift during frozen extraction')
                counts[start:end]=values.numpy();counts.flush()
                log(dict(kind='new_weight_features',split=split,start=start,end=end,input_sha256=p.tensor_sha(spikes),
                    counts_sha256=p.tensor_sha(values),output_spikes=int(values.sum()),frozen_state=frozen,seconds=time.perf_counter()-begin))
            del x,spikes,recorded,values;net.reset_state_variables()
            print(json.dumps(dict(stage='features',split=split,done=end,total=size)),flush=True)
        log(dict(kind='feature_split_complete',split=split,rows=size,new_weights=True))
    finally:counts.flush();del counts;net.monitors.pop('new_weight_features');net.reset_state_variables()
    return path,labels


def execute(output,p,ref,cfg,smoke,manifest):
    import numpy as np
    import torch
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_num_threads(2)
    budget=Budget(output,smoke);manifest['network_executed']=True
    with (output/'events.jsonl').open('x',encoding='utf-8') as events:
        def log(value):events.write(json.dumps(value,allow_nan=False)+'\n');events.flush()
        net,lif,conn,encoder,summary=training(p,ref,cfg,128 if smoke else 30000,output,log,budget)
        manifest['training']=summary
        if smoke:
            manifest.update(status='completed',scope='train_only128_smoke',accuracy=None,test_samples=0,readout_trained=False);return
        net.train(False);conn.update_rule.nu=(torch.tensor(0.,device='cuda'),torch.tensor(0.,device='cuda'))
        frozen=p.state_hashes(net)
        if p.tensor_sha(conn.w)!=summary['final_weight_sha256']:raise ValueError('Wrong new weights at freeze')
        train_path,train_labels=collect_new_weights(p,net,lif,encoder,frozen,'train',output,log,budget)
        train=np.load(train_path,mmap_mode='r',allow_pickle=False);mean,sd=recovery.fit_scaler(train)
        np.save(output/'scaler_mean.npy',mean.numpy(),allow_pickle=False);np.save(output/'scaler_sd.npy',sd.numpy(),allow_pickle=False)
        log(dict(kind='scaler_complete',train_rows=30000,test_access=False,formula=recovery.SPEC['scaler']))
        head,updates=recovery.train_head(train,train_labels,mean,sd,output,log,budget);del train
        frozen_files={name:recovery.sha(output/name) for name in ['head_weight.npy','head_bias.npy','scaler_mean.npy','scaler_sd.npy','trained_model.pt']}
        log(dict(kind='head_frozen_before_test',sha256=frozen_files))
        test_path,test_labels=collect_new_weights(p,net,lif,encoder,frozen,'test',output,log,budget)
        test=np.load(test_path,mmap_mode='r',allow_pickle=False)
        result=recovery.evaluate_once(head,test,test_labels,mean,sd,output,budget)
        if any(recovery.sha(output/name)!=pin for name,pin in frozen_files.items()):raise ValueError('Trained checkpoint/head/scaler changed during test')
        if p.state_hashes(net)!=frozen:raise ValueError('Frozen final state changed')
        manifest.update(result,status='completed',scope='full_stdp30000_then_new_weight_readout',head_updates=updates)
        log(dict(kind='test_complete',**result))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--smoke-128',action='store_true');mode.add_argument('--run',action='store_true')
    args=parser.parse_args(argv);spec_sha=recovery.digest(SPEC)
    if args.smoke_128 and spec_sha not in APPROVED_SMOKE_SPEC_SHA256:raise ValueError('STDP smoke source/scientific review required')
    if args.run and spec_sha not in APPROVED_FULL_SPEC_SHA256:raise ValueError('Full STDP source/scientific review required')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG',':4096:8')!=':4096:8':raise ValueError('Unexpected determinism setting')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    p=recovery.helpers();checked=recovery.preflight(p);ref,cfg=inputs(p)
    checked['source_sha256'][SUMMARY.relative_to(ROOT).as_posix()]=SUMMARY_SHA
    checked['source_sha256'][Path(recovery.__file__).relative_to(ROOT).as_posix()]=RECOVERY_SHA
    if args.preflight:print(json.dumps(dict(spec=SPEC,spec_sha256=spec_sha,effective_cfg=vars(cfg),**checked)));return 0
    output=ROOT/'results/mnist_stdp_retraining_20261019'/str(uuid.uuid4());output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='failed',started_utc=p.now(),spec=SPEC,spec_sha256=spec_sha,effective_cfg=vars(cfg),
                  entrypoint_sha256=recovery.sha(__file__),physical_memristor=False,**checked)
    p.write_json(output/'start_manifest.json',manifest)
    try:execute(output,p,ref,cfg,args.smoke_128,manifest)
    except (Exception,KeyboardInterrupt) as exc:manifest['error']=type(exc).__name__+': '+str(exc)
    finally:
        manifest['finished_utc']=p.now();manifest['result_sha256']={f.name:recovery.sha(f) for f in output.iterdir() if f.is_file()}
        p.write_json(output/'manifest.json',manifest)
    print(json.dumps(dict(status=manifest['status'],output=output.relative_to(ROOT).as_posix(),accuracy=manifest.get('accuracy'),error=manifest.get('error'))))
    return 0 if manifest['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
