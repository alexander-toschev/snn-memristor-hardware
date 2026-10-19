"""Train-only128 feature pilot; new frozen-weight reconstruction, no head or board."""
import argparse
import ast
from datetime import datetime,timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import struct
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'experiments/mnist_baseline_20261019/reference_source'
DATA=ROOT/'experiments/mnist_baseline_20261019/data/MNIST/raw'
CHECKPOINT=ROOT/'weights/mnist_baseline_20261019/model.pt'
CHECKPOINT_SHA='43f7d6237b0ec8405956a4f9ef34c2ea0c8a3a592c4c76467e1acc00beee47de'
WEIGHT_SHA='89cbe275beac267f0842fc83ff5abb805b7c85ae2153a4495c80f0933041dad4'
PROVENANCE_SHA='62308dc3c8bb4779006bf5a222ab04596f6e3ca2a605b577276208aa2dc9c954'
DATA_PINS={'train-images-idx3-ubyte.gz':'440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609',
           'train-labels-idx1-ubyte.gz':'3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c'}
VERSIONS={'torch':'2.4.1+cu124','torchvision':'0.19.1+cu124','torchaudio':'2.4.1+cu124',
          'numpy':'2.0.2','bindsnet':'0.3.3'}
BINDSNET_MANIFEST_SHA='6ff7dafcb2369dfbcd11e5ae8f492585a1fa6265a2e43595adf105711dd4beb0'
INSTALL_PROVENANCE_SHA='91c9a4007116df86383d12988f41ef3a8bb8e4fee07f18dfb3a442c0029c722a'
SPEC=dict(schema='mnist-frozen128-pilot-v1',train_ids=list(range(128)),batch_size=128,
          timesteps=100,repeats=2,seed=42,poisson_seed=123,rate_scale=.0015,boost=1.,
          learning=False,reference_hook='unchanged undefined-connection exception skips postwindow norm/EI',
          new_reconstruction=True,readout_training=False,test_split_access=False,board_access=False)
APPROVED_SPEC_SHA256=frozenset({'5a75bb6ecbc20e41b1ec9b22cbc6ca1f0a4c834bc3fe6521a50067ef4771a1f4'})


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def verify_sources():
    provenance=REFERENCE/'PROVENANCE.json'
    if sha(provenance)!=PROVENANCE_SHA:raise ValueError('Reference manifest mismatch')
    pins={provenance.relative_to(ROOT).as_posix():PROVENANCE_SHA}
    for item in json.loads(provenance.read_text(encoding='utf-8-sig'))['files']:
        path=REFERENCE/item['path']
        if not path.resolve().is_relative_to(REFERENCE.resolve()) or sha(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:
            raise ValueError('Reference source changed')
        pins[path.relative_to(ROOT).as_posix()]=item['sha256']
    if sha(CHECKPOINT)!=CHECKPOINT_SHA:raise ValueError('Checkpoint SHA mismatch')
    pins[CHECKPOINT.relative_to(ROOT).as_posix()]=CHECKPOINT_SHA
    for name,pin in DATA_PINS.items():
        if sha(DATA/name)!=pin:raise ValueError('Train IDX SHA mismatch')
        pins[(DATA/name).relative_to(ROOT).as_posix()]=pin
    tree=ast.parse((REFERENCE/'csnn_mnist_net.py').read_text(encoding='utf-8'))
    builder=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_csnn')
    stores={n.id for n in ast.walk(builder) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
    if 'connection' in stores or not any(isinstance(n,ast.Name) and n.id=='connection' and isinstance(n.ctx,ast.Load) for n in ast.walk(builder)):
        raise ValueError('Historical hook semantics changed')
    return pins


def idx_prefix(path,images,expected_count=60000,take=128):
    if not 0<take<=expected_count:raise ValueError('Invalid prefix')
    with gzip.open(path,'rb') as stream:
        fields=struct.unpack('>IIII' if images else '>II',stream.read(16 if images else 8))
        if fields[:2]!=(2051 if images else 2049,expected_count) or (images and fields[2:]!=(28,28)):
            raise ValueError('Wrong IDX header')
        size=take*(784 if images else 1);prefix=stream.read(size)
        if len(prefix)!=size:raise ValueError('Truncated prefix')
        remainder=0
        for block in iter(lambda:stream.read(1024**2),b''):remainder+=len(block)
        if size+remainder!=expected_count*(784 if images else 1):raise ValueError('Wrong IDX length')
    if not images and any(v>9 for v in prefix):raise ValueError('Wrong MNIST label')
    return prefix


def environment(device):
    if Path(sys.prefix).resolve()!=(ROOT/'.venv-mnist-pilot').resolve():
        raise ValueError('Use separate repository-local .venv-mnist-pilot interpreter')
    versions={name:importlib.metadata.version(name) for name in VERSIONS}
    if versions!=VERSIONS:raise ValueError('Unreviewed dependency versions: '+repr(versions))
    provenance_path=Path(__file__).with_name('INSTALL_PROVENANCE.json')
    if sha(provenance_path)!=INSTALL_PROVENANCE_SHA:raise ValueError('Installation provenance changed')
    provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
    canonical=lambda name:name.lower().replace('_','-').replace('.','-')
    installed={canonical(d.metadata['Name']):d.version for d in importlib.metadata.distributions()}
    if installed!=provenance['installed_versions'] or platform.python_version()!=provenance['python']:
        raise ValueError('Full dependency lock or Python version changed')
    for name,pin in provenance['lock_sha256'].items():
        if sha(Path(__file__).with_name(name))!=pin:raise ValueError('Dependency artifact lock changed')
    import numpy as np
    import torch
    import torchvision
    import bindsnet
    for module in (np,torch,torchvision,bindsnet):
        if not Path(module.__file__).resolve().is_relative_to((ROOT/'.venv-mnist-pilot').resolve()):
            raise ValueError('Dependency escaped isolated environment')
    manifest_path=Path(__file__).with_name('BINDSNET_SOURCE.json')
    if sha(manifest_path)!=BINDSNET_MANIFEST_SHA:raise ValueError('Official BindsNET source manifest changed')
    expected=json.loads(manifest_path.read_text())['source_files']
    package_root=Path(bindsnet.__file__).resolve().parent
    actual={p.relative_to(package_root.parent).as_posix():sha(p) for p in package_root.rglob('*.py')}
    if actual!=expected:raise ValueError('Installed BindsNET source differs from official pinned archive')
    value=np.array([1,2,3],dtype=np.float32)
    if not np.array_equal(torch.from_numpy(value).numpy(),value):raise ValueError('Torch NumPy interop failed')
    gpu=None
    if device=='cuda':
        if not torch.cuda.is_available():raise ValueError('CUDA unavailable; no silent CPU fallback')
        # A tiny dependency/interop check only, never the MNIST network.
        if not np.array_equal(torch.from_numpy(value).to('cuda').cpu().numpy(),value):raise ValueError('CUDA interop failed')
        free,total=torch.cuda.mem_get_info()
        gpu=dict(name=torch.cuda.get_device_name(0),free_bytes=free,total_bytes=total,
                 estimated_minimum_free_bytes=4*1024**3,arch_list=torch.cuda.get_arch_list())
        if free<4*1024**3:raise ValueError('Need4GiB free CUDA memory for fixed128/T100 pilot; no batch fallback')
    return dict(python=sys.version,platform=platform.platform(),versions=versions,device=device,gpu=gpu,
                full_installed_versions=installed,install_provenance_sha256=INSTALL_PROVENANCE_SHA,
                dependency_lock_sha256=provenance['lock_sha256'],
                numpy_interop_verified=True,cuda_version=torch.version.cuda,
                bindsnet_source_manifest_sha256=BINDSNET_MANIFEST_SHA,bindsnet_python_files_verified=len(actual))


def tensor_sha(tensor):return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hashes(net):
    import torch
    if set(net.connections)!={('Input','Conv1')}:raise ValueError('Unexpected additional connection')
    if net.learning or net.training or any(layer.learning or layer.training for layer in net.layers.values()):
        raise ValueError('Network or layer learning enabled')
    weights={str(key):tensor_sha(conn.w) for key,conn in net.connections.items()}
    biases={str(key):tensor_sha(conn.b) for key,conn in net.connections.items()}
    thresholds={}
    for name,layer in net.layers.items():
        value=getattr(layer,'v_thresh',getattr(layer,'thresh',None))
        if torch.is_tensor(value):thresholds[name]=tensor_sha(value)
    return dict(weights=weights,biases=biases,thresholds=thresholds)


def load_checkpoint():
    import torch
    if sha(CHECKPOINT)!=CHECKPOINT_SHA:raise ValueError('Checkpoint SHA mismatch')
    data=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    if not isinstance(data,dict) or not isinstance(data.get('cfg'),dict):raise ValueError('Malformed checkpoint')
    w=data['W'];threshold=data['v_thresh']
    if w.dtype!=torch.float32 or tuple(w.shape)!=(64,1,5,5) or not torch.isfinite(w).all() or tensor_sha(w)!=WEIGHT_SHA:
        raise ValueError('Invalid frozen weights')
    if threshold.dtype!=torch.float32 or threshold.ndim!=0 or not torch.isfinite(threshold):raise ValueError('Invalid threshold')
    expected={'time':100,'seed':42,'encoder':'poisson','poisson_rate_scale':.0015,'poisson_base_seed':123,
              'poisson_deterministic':True,'encoder_rate_boost':1.,'c1_out':64,'c1_kernel':5,'c1_stride':1,'c1_pad':0,
              'w_norm_enable':True,'ei_enable':True,'wta_enable':False,'local_inhib_enable':False,'adapt_thresh_enable':False}
    if any(data['cfg'].get(k)!=v for k,v in expected.items()):raise ValueError('Unexpected archived configuration')
    return data


def run_pilot(output,device,manifest):
    import numpy as np
    import torch
    from bindsnet.network.monitors import Monitor
    sys.path.insert(0,str(REFERENCE))
    import csnn_mnist_net as reference
    import encoders
    if Path(reference.__file__).resolve()!=REFERENCE/'csnn_mnist_net.py' or Path(encoders.__file__).resolve()!=REFERENCE/'encoders.py':
        raise ValueError('Wrong reference module import')
    if hasattr(reference,'connection'):raise ValueError('Unexpected global would repair historical broken hook')
    data=load_checkpoint();cfg=reference.CSCfg(**data['cfg']);cfg.device=device
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.set_num_threads(2)
    if device=='cuda':torch.cuda.manual_seed_all(42);torch.cuda.reset_peak_memory_stats()
    net,input_layer,lif,connection=reference.build_csnn(cfg)
    with torch.no_grad():
        connection.w.copy_(data['W'].to(device))
        threshold=getattr(lif,'v_thresh',getattr(lif,'thresh',None))
        if not torch.is_tensor(threshold) or threshold.shape!=data['v_thresh'].shape:raise ValueError('Threshold interface differs')
        threshold.copy_(data['v_thresh'].to(device))
    if tensor_sha(threshold)!=tensor_sha(data['v_thresh']) or tensor_sha(connection.w)!=WEIGHT_SHA:
        raise ValueError('Checkpoint copy changed bytes')
    if tuple(connection.b.shape)!=(64,) or not torch.all(connection.b==0):raise ValueError('Unexpected Conv1 bias')
    if connection.norm is not None or connection.Dales_rule is not None:raise ValueError('Unexpected automatic weight transform')
    net.train(False)
    connection.update_rule.nu=(torch.tensor(0.,device=device),torch.tensor(0.,device=device))
    if net.learning or net.training or connection.training:raise ValueError('Learning freeze failed')
    frozen_state=state_hashes(net)
    encoder=reference.build_encoder_from_cfg(cfg)
    monitor=Monitor(lif,state_vars=['s'],time=100);net.add_monitor(monitor,name='frozen_pilot_counts')
    images=np.frombuffer(idx_prefix(DATA/'train-images-idx3-ubyte.gz',True),dtype=np.uint8).copy().reshape(128,1,28,28)
    labels=np.frombuffer(idx_prefix(DATA/'train-labels-idx1-ubyte.gz',False),dtype=np.uint8).copy()
    np.save(output/'train_ids.npy',np.arange(128,dtype=np.int64),allow_pickle=False)
    np.save(output/'train_images_uint8.npy',images,allow_pickle=False);np.save(output/'train_labels.npy',labels,allow_pickle=False)
    np.save(output/'frozen_weights.npy',data['W'].numpy(),allow_pickle=False)
    x=torch.from_numpy(images).to(device=device,dtype=torch.float32).div_(255.)
    manifest.update(archived_cfg=data['cfg'],effective_cfg=vars(cfg),checkpoint_loaded_weights_only=True,
                    weight_sha256=WEIGHT_SHA,learning_explicitly_disabled=True,source_attested_for_original_run=False,
                    hook_effect='post-window undefined connection error preserved; no effective E/I hook',
                    archived_freeze_difference='explicit learning=False instead of historical zero-nu-only',
                    synchronous_one_step=False,frozen_state=frozen_state,passes=[])
    expected_input_sha=None;expected_counts_sha=None;threshold_sha=tensor_sha(threshold)
    for repeat in range(2):
        net.reset_state_variables()
        before=tensor_sha(connection.w)
        if before!=WEIGHT_SHA or state_hashes(net)!=frozen_state:raise ValueError('State changed before pass')
        if device=='cuda':torch.cuda.synchronize()
        started=time.perf_counter()
        # Stateful BindsNET mutates/reset its runtime tensors outside this block.
        # no_grad preserves ordinary mutable tensors; inference_mode would not.
        with torch.no_grad():
            spikes=encoder(x)
            if tuple(spikes.shape)!=(100,128,1,28,28) or not torch.all((spikes==0)|(spikes==1)):raise ValueError('Invalid encoded input')
            input_sha=tensor_sha(spikes)
            if expected_input_sha is not None and input_sha!=expected_input_sha:raise ValueError('Encoder repeat mismatch')
            net.run(inputs={'Input':spikes},time=100)
            # Monitor defaults to CPU and records nonblocking device-to-host copies.
            # Complete them before reading or reducing recorded CPU tensors.
            if device=='cuda':torch.cuda.synchronize()
            s=monitor.get('s')
            if tuple(s.shape)!=(100,128,64,24,24) or not torch.all((s==0)|(s==1)):raise ValueError('Invalid spike monitor')
            counts=s.to(torch.float32).sum(0).reshape(128,36864)
            after=tensor_sha(connection.w);counts_sha=tensor_sha(counts)
            if after!=before or state_hashes(net)!=frozen_state:raise ValueError('Frozen weights/threshold changed')
            if expected_counts_sha is not None and counts_sha!=expected_counts_sha:raise ValueError('Feature repeat mismatch')
            if repeat==0:np.save(output/'input_spikes_uint8.npy',spikes.cpu().numpy().astype(np.uint8),allow_pickle=False)
            np.save(output/('counts_repeat%d.npy'%repeat),counts.cpu().numpy(),allow_pickle=False)
            if device=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            manifest['passes'].append(dict(repeat=repeat,seconds=elapsed,input_spikes=int(spikes.sum().item()),
                output_spikes=int(counts.sum().item()),active_features=int((counts.sum(0)>0).sum().item()),
                weight_before_sha256=before,weight_after_sha256=after,threshold_sha256=threshold_sha,
                input_spikes_sha256=input_sha,counts_sha256=counts_sha,
                monitor_device='cpu',monitor_record_dtype=str(s.dtype),
                monitor_record_bytes=s.numel()*s.element_size(),
                host_float32_reduction_input_bytes=s.numel()*4,
                timing_includes_device_to_host_copies_and_cpu_aggregation=True,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None))
            expected_input_sha=input_sha;expected_counts_sha=counts_sha
        del s,counts,spikes
        net.reset_state_variables()
    return dict(status='completed',unique_train_samples=128,completed_passes=2,test_samples=0,
                weights_unchanged=True,repeat_counts_identical=True,readout_trained=False,accuracy=None,
                uninformative_all_silent=all(p['output_spikes']==0 for p in manifest['passes']))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--run-pilot',action='store_true')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args(argv)
    if args.run_pilot and (args.device!='cuda' or digest(SPEC) not in APPROVED_SPEC_SHA256):
        raise ValueError('Pilot source/scientific review required for this specification/device')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG',':4096:8')!=':4096:8':raise ValueError('Unexpected CUDA determinism setting')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    pins=verify_sources();env=environment(args.device)
    if args.preflight:
        print(json.dumps(dict(spec=SPEC,spec_sha256=digest(SPEC),environment=env,source_sha256=pins,network_executed=False)))
        return 0
    if shutil.disk_usage(ROOT).free<128*1024**2:raise ValueError('Need128MiB free storage for pilot artifacts')
    output=ROOT/'results/mnist_feature_pilot_20261019'/str(uuid.uuid4());output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='failed',started_utc=now(),spec=SPEC,spec_sha256=digest(SPEC),environment=env,
                  source_sha256=pins,entrypoint_sha256=sha(__file__),board_access=False,new_reconstruction=True)
    write_json(output/'start_manifest.json',manifest)
    try:manifest.update(run_pilot(output,args.device,manifest))
    except (Exception,KeyboardInterrupt) as exc:manifest['error']=type(exc).__name__+': '+str(exc)
    finally:
        manifest['finished_utc']=now();manifest['result_sha256']={p.name:sha(p) for p in output.iterdir() if p.is_file()}
        write_json(output/'manifest.json',manifest)
    print(json.dumps(dict(output=output.relative_to(ROOT).as_posix(),status=manifest['status'],error=manifest.get('error'))))
    return 0 if manifest['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
