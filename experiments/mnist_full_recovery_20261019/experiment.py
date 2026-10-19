"""Fixed 30k/10k MNIST reconstruction. Preparation only until reviewed registration."""
import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
PILOT=ROOT/'experiments/mnist_feature_pilot_20261019/experiment.py'
PILOT_SHA='51f89824e9e9f56e9c29a1e3506f9d398a8e0d536cd79a6876dfd75fe32cfa9e'
TEST_PINS={'t10k-images-idx3-ubyte.gz':'8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6',
           't10k-labels-idx1-ubyte.gz':'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6'}
FIRST_COUNTS_SHA='7de63f26ad0152e9073642d468cfbe99e76794a4a1234ffc217c214ba6eaf904'
FIRST_INPUT_SHA='fc1e7c6cd8c660de0b10c57abde229458a470967847cdd3bc7a7d33e14ce6239'
SPEC=dict(schema='mnist-full-reconstruction-v1',train_samples=30000,test_samples=10000,
          sample_order='first consecutive official split IDs',features=36864,feature_batch=128,timesteps=100,
          extractor='SHA-pinned successful frozen128 semantics',feature_seed=42,head_seed=42,shuffle_seed=42,
          feature_device='cuda',readout_device='cuda',scaler='CPU float32 log1p; streamed float64 Chan sample SD; cast float32; floor1e-6',
          scaler_fit_split='train_only',head='Linear(36864,10,bias=True)',epochs=25,readout_batch=512,
          optimizer='SGD',lr=.001,weight_decay=.0001,momentum=0.,dampening=0.,nesterov=False,
          loss='CrossEntropyLoss(mean)',drop_last=False,epoch_selection='fixed final25',test_evaluations=1,
          stdp=False,board=False,reconstruction=True,minimum_free_disk_bytes=18*1024**3,
          maximum_output_bytes=8*1024**3,maximum_seconds=4*60*60,minimum_free_host_bytes=8*1024**3)
APPROVED_SPEC_SHA256=frozenset()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def helpers():
    if sha(PILOT)!=PILOT_SHA:raise ValueError('Frozen successful pilot source changed')
    sys.path.insert(0,str(ROOT))
    module=importlib.import_module('experiments.mnist_feature_pilot_20261019.experiment')
    if Path(module.__file__).resolve()!=PILOT:raise ValueError('Wrong pilot helper import')
    return module


def batches(size,batch):
    if size<=0 or batch<=0:raise ValueError('Invalid batch bounds')
    return [(start,min(start+batch,size)) for start in range(0,size,batch)]


def available_host_bytes():
    if os.name=='nt':
        import ctypes
        class Memory(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
                ['total','available','page_total','page_available','virtual_total','virtual_available','extended']]
        status=Memory();status.length=ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):raise OSError('Memory preflight failed')
        return status.available
    values={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    return int(values['MemAvailable'].split()[0])*1024


def preflight(p):
    pins=p.verify_sources()
    for name,pin in TEST_PINS.items():
        if sha(p.DATA/name)!=pin:raise ValueError('Test dataset SHA mismatch')
        pins[(p.DATA/name).relative_to(ROOT).as_posix()]=pin
    pins[PILOT.relative_to(ROOT).as_posix()]=PILOT_SHA
    env=p.environment('cuda')
    disk=shutil.disk_usage(ROOT).free;host=available_host_bytes()
    if disk<SPEC['minimum_free_disk_bytes']:raise ValueError('Need18GiB disk including later LFS duplication')
    if host<SPEC['minimum_free_host_bytes']:raise ValueError('Need8GiB available host memory for monitor/reduction')
    return dict(source_sha256=pins,environment=env,free_disk_bytes=disk,free_host_bytes=host,
                feature_batches=dict(train=235,test=79),head_updates=1475,counts_bytes=5898240000,
                reference_accuracy_not_guaranteed=.6965,network_executed=False)


def validate_counts(values):
    import torch
    if not torch.isfinite(values).all() or not ((values>=0)&(values<=100)&(values==values.round())).all():
        raise ValueError('Invalid integer spike counts')


def fit_scaler(counts,batch_size=128):
    """Train-only Chan variance over float32 log1p; no test argument exists."""
    import numpy as np
    import torch
    total=0;mean=None;m2=None
    for start,end in batches(len(counts),batch_size):
        chunk=torch.from_numpy(np.array(counts[start:end],dtype=np.float32,copy=True))
        validate_counts(chunk);values=torch.log1p(chunk).to(torch.float64)
        n=end-start;local_mean=values.mean(0);local_m2=((values-local_mean)**2).sum(0)
        if total==0:mean=local_mean;m2=local_m2
        else:
            delta=local_mean-mean
            m2+=local_m2+delta.square()*(total*n/(total+n))
            mean+=delta*(n/(total+n))
        total+=n
    if total<2:raise ValueError('Need sample variance')
    sd=(m2/(total-1)).sqrt().to(torch.float32).clamp_min(1e-6)
    mean=mean.to(torch.float32)
    if not torch.isfinite(mean).all() or not torch.isfinite(sd).all():raise ValueError('Nonfinite scaler')
    return mean,sd


class Budget:
    def __init__(self,output):self.output=output;self.start=time.monotonic()
    def check(self):
        if time.monotonic()-self.start>SPEC['maximum_seconds']:raise RuntimeError('Fixed4h wall budget reached')
        if sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())>SPEC['maximum_output_bytes']:
            raise RuntimeError('Output budget exceeded')
        if shutil.disk_usage(ROOT).free<2*1024**3:raise RuntimeError('Disk reserve reached')


def build_frozen(p):
    import torch
    import numpy as np
    from bindsnet.network.monitors import Monitor
    sys.path.insert(0,str(p.REFERENCE))
    reference=importlib.import_module('csnn_mnist_net')
    if Path(reference.__file__).resolve()!=p.REFERENCE/'csnn_mnist_net.py' or hasattr(reference,'connection'):
        raise ValueError('Historical reference hook changed')
    data=p.load_checkpoint();cfg=reference.CSCfg(**data['cfg']);cfg.device='cuda'
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);random.seed(42);np.random.seed(42)
    net,_,lif,conn=reference.build_csnn(cfg)
    with torch.no_grad():
        conn.w.copy_(data['W'].to('cuda'))
        threshold=getattr(lif,'v_thresh',getattr(lif,'thresh',None));threshold.copy_(data['v_thresh'].to('cuda'))
    if p.tensor_sha(conn.w)!=p.WEIGHT_SHA or p.tensor_sha(threshold)!=p.tensor_sha(data['v_thresh']):raise ValueError('Checkpoint copy changed')
    if conn.norm is not None or conn.Dales_rule is not None or tuple(conn.b.shape)!=(64,) or not torch.all(conn.b==0):
        raise ValueError('Unexpected normalization/transform/bias')
    net.train(False);conn.update_rule.nu=(torch.tensor(0.,device='cuda'),torch.tensor(0.,device='cuda'))
    frozen=p.state_hashes(net)
    monitor=Monitor(lif,state_vars=['s'],time=100);net.add_monitor(monitor,name='reconstruction_counts')
    return net,monitor,reference.build_encoder_from_cfg(cfg),frozen


def collect(p,split,output,log,budget):
    import numpy as np
    import torch
    size=30000 if split=='train' else 10000;dataset_size=60000 if split=='train' else 10000
    prefix='train' if split=='train' else 't10k'
    images=np.frombuffer(p.idx_prefix(p.DATA/(prefix+'-images-idx3-ubyte.gz'),True,dataset_size,size),dtype=np.uint8).copy().reshape(size,1,28,28)
    labels=np.frombuffer(p.idx_prefix(p.DATA/(prefix+'-labels-idx1-ubyte.gz'),False,dataset_size,size),dtype=np.uint8).astype(np.int64)
    np.save(output/(split+'_ids.npy'),np.arange(size,dtype=np.int64),allow_pickle=False)
    np.save(output/(split+'_labels.npy'),labels,allow_pickle=False)
    path=output/(split+'_counts.npy')
    counts=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(size,36864))
    net,monitor,encoder,frozen=build_frozen(p);written=0;spike_total=0
    torch.cuda.reset_peak_memory_stats()
    try:
        for start,end in batches(size,128):
            budget.check();net.reset_state_variables()
            if p.state_hashes(net)!=frozen:raise ValueError('Frozen state changed before batch')
            torch.cuda.synchronize();begin=time.perf_counter()
            with torch.no_grad():
                x=torch.from_numpy(images[start:end]).to(device='cuda',dtype=torch.float32).div_(255.)
                spikes=encoder(x)
                if tuple(spikes.shape)!=(100,end-start,1,28,28) or not ((spikes==0)|(spikes==1)).all():raise ValueError('Input shape/content')
                input_sha=p.tensor_sha(spikes)
                net.run(inputs={'Input':spikes},time=100)
                torch.cuda.synchronize()  # Complete nonblocking CPU Monitor copies.
                recorded=monitor.get('s')
                if tuple(recorded.shape)!=(100,end-start,64,24,24) or not ((recorded==0)|(recorded==1)).all():raise ValueError('Monitor shape/content')
                values=recorded.to(torch.float32).sum(0).reshape(end-start,36864)
                validate_counts(values)
                if p.state_hashes(net)!=frozen:raise ValueError('Frozen weights/thresholds changed')
                count_sha=p.tensor_sha(values)
                if split=='train' and start==0 and (count_sha!=FIRST_COUNTS_SHA or input_sha!=FIRST_INPUT_SHA):
                    raise ValueError('First batch differs from reviewed128 pilot')
                counts[start:end]=values.numpy();counts.flush();written=end
                total=int(values.sum().item());spike_total+=total
                event=dict(kind='feature_batch',split=split,start=start,end=end,input_sha256=input_sha,counts_sha256=count_sha,
                    input_spikes=int(spikes.sum().item()),output_spikes=total,seconds=time.perf_counter()-begin,
                    frozen_state=frozen,monitor_device=str(recorded.device),monitor_record_bytes=recorded.numel()*recorded.element_size(),
                    host_float32_reduction_bytes=recorded.numel()*4,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                log(event)
            del x,spikes,recorded,values
            net.reset_state_variables()
            print(json.dumps(dict(stage='features',split=split,done=end,total=size)),flush=True)
        if written!=size:raise ValueError('Incomplete feature coverage')
        log(dict(kind='feature_split_complete',split=split,rows=written,output_spikes=spike_total,all_silent=spike_total==0))
    finally:
        counts.flush();del counts,net,monitor,encoder;gc.collect();torch.cuda.empty_cache()
    return path,labels


def train_head(train_counts,labels,mean,sd,output,log,budget):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,Dataset
    class Rows(Dataset):
        def __len__(self):return len(labels)
        def __getitem__(self,index):
            raw=torch.from_numpy(np.array(train_counts[index],dtype=np.float32,copy=True))
            return (torch.log1p(raw)-mean)/sd,int(labels[index]),index
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    generator=torch.Generator().manual_seed(42)
    head=torch.nn.Linear(36864,10,bias=True).to('cuda')
    opt=torch.optim.SGD(head.parameters(),lr=.001,weight_decay=.0001,momentum=0.,dampening=0.,nesterov=False)
    criterion=torch.nn.CrossEntropyLoss(reduction='mean')
    loader=DataLoader(Rows(),batch_size=512,shuffle=True,num_workers=0,pin_memory=True,drop_last=False,generator=generator)
    np.save(output/'head_initial_weight.npy',head.weight.detach().cpu().numpy(),allow_pickle=False)
    np.save(output/'head_initial_bias.npy',head.bias.detach().cpu().numpy(),allow_pickle=False)
    updates=0;head.train()
    for epoch in range(25):
        budget.check();order=[];loss_sum=0.;seen=0;started=time.perf_counter()
        np.save(output/('shuffle_rng_before_epoch%02d.npy'%epoch),generator.get_state().numpy(),allow_pickle=False)
        for xb,yb,ids in loader:
            budget.check();order.extend(ids.tolist());xb=xb.to('cuda',non_blocking=True);yb=yb.to('cuda',non_blocking=True)
            opt.zero_grad(set_to_none=True);loss=criterion(head(xb),yb)
            if not torch.isfinite(loss):raise ValueError('Nonfinite train loss')
            loss.backward();opt.step();updates+=1;loss_sum+=float(loss.item())*len(yb);seen+=len(yb)
        if seen!=30000 or sorted(order)!=list(range(30000)):raise ValueError('Train permutation incomplete')
        order=np.asarray(order,dtype=np.int64);np.save(output/('epoch%02d_train_order.npy'%epoch),order,allow_pickle=False)
        log(dict(kind='head_epoch',epoch=epoch+1,updates=updates,train_examples=seen,train_loss=loss_sum/seen,
                 order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),seconds=time.perf_counter()-started,test_access=False))
        print(json.dumps(dict(stage='head',epoch=epoch+1,epochs=25,train_loss=loss_sum/seen)),flush=True)
    if updates!=1475:raise ValueError('Wrong optimizer budget')
    for value in head.parameters():
        if not torch.isfinite(value).all():raise ValueError('Nonfinite final head')
    np.save(output/'head_weight.npy',head.weight.detach().cpu().numpy(),allow_pickle=False)
    np.save(output/'head_bias.npy',head.bias.detach().cpu().numpy(),allow_pickle=False)
    np.save(output/'shuffle_rng_final.npy',generator.get_state().numpy(),allow_pickle=False)
    return head,updates


def evaluate_once(head,test_counts,labels,mean,sd,output,budget):
    import numpy as np
    import torch
    head.eval();scores=np.empty((10000,10),dtype=np.float32)
    with torch.no_grad():
        for start,end in batches(10000,512):
            budget.check();raw=torch.from_numpy(np.array(test_counts[start:end],dtype=np.float32,copy=True))
            validate_counts(raw);values=(torch.log1p(raw)-mean)/sd
            logits=head(values.to('cuda')).cpu().numpy()
            if not np.isfinite(logits).all():raise ValueError('Nonfinite test logits')
            scores[start:end]=logits
    predictions=scores.argmax(1).astype(np.int64)
    confusion=np.zeros((10,10),dtype=np.int64);np.add.at(confusion,(labels,predictions),1)
    np.save(output/'test_logits.npy',scores,allow_pickle=False);np.save(output/'test_predictions.npy',predictions,allow_pickle=False)
    np.save(output/'test_confusion.npy',confusion,allow_pickle=False)
    return dict(correct=int((predictions==labels).sum()),total=10000,accuracy=float((predictions==labels).mean()),
                test_evaluations=1,selected_epoch=25,hardware_accuracy=False,reconstruction=True)


def run(output,p,manifest):
    import numpy as np
    import torch
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.set_num_threads(2)
    manifest['network_executed']=True
    budget=Budget(output)
    with (output/'events.jsonl').open('x',encoding='utf-8') as events:
        def log(value):events.write(json.dumps(value,allow_nan=False)+'\n');events.flush()
        train_path,train_labels=collect(p,'train',output,log,budget)
        train=np.load(train_path,mmap_mode='r',allow_pickle=False)
        mean,sd=fit_scaler(train);np.save(output/'scaler_mean.npy',mean.numpy(),allow_pickle=False);np.save(output/'scaler_sd.npy',sd.numpy(),allow_pickle=False)
        log(dict(kind='scaler_complete',fit_rows=30000,test_access=False,formula=SPEC['scaler']))
        head,updates=train_head(train,train_labels,mean,sd,output,log,budget)
        del train
        # Freeze and save the final head before accessing any test examples.
        final_head_sha={name:sha(output/name) for name in ['head_weight.npy','head_bias.npy','scaler_mean.npy','scaler_sd.npy']}
        log(dict(kind='head_frozen_before_test',sha256=final_head_sha))
        test_path,test_labels=collect(p,'test',output,log,budget)
        test=np.load(test_path,mmap_mode='r',allow_pickle=False)
        result=evaluate_once(head,test,test_labels,mean,sd,output,budget)
        if any(sha(output/name)!=pin for name,pin in final_head_sha.items()):raise ValueError('Head/scaler changed during test')
        log(dict(kind='test_complete',**result));manifest.update(result,status='completed',head_updates=updates)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--run',action='store_true');args=parser.parse_args(argv)
    if args.run and digest(SPEC) not in APPROVED_SPEC_SHA256:raise ValueError('Full reconstruction source/scientific review required')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG',':4096:8')!=':4096:8':raise ValueError('Unreviewed CUDA determinism setting')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    p=helpers();checked=preflight(p)
    if args.preflight:print(json.dumps(dict(spec=SPEC,spec_sha256=digest(SPEC),**checked)));return 0
    output=ROOT/'results/mnist_full_recovery_20261019'/str(uuid.uuid4());output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='failed',started_utc=p.now(),spec=SPEC,spec_sha256=digest(SPEC),entrypoint_sha256=sha(__file__),**checked)
    p.write_json(output/'start_manifest.json',manifest)
    try:run(output,p,manifest)
    except (Exception,KeyboardInterrupt) as exc:manifest['error']=type(exc).__name__+': '+str(exc)
    finally:
        manifest['finished_utc']=p.now();manifest['result_sha256']={path.name:sha(path) for path in output.iterdir() if path.is_file()}
        p.write_json(output/'manifest.json',manifest)
    print(json.dumps(dict(status=manifest['status'],output=output.relative_to(ROOT).as_posix(),accuracy=manifest.get('accuracy'),error=manifest.get('error'))))
    return 0 if manifest['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
