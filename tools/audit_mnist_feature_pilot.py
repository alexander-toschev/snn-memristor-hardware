"""Read-only MNIST128 result audit. No network construction, forward or CUDA call."""
from datetime import datetime
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/2026-10-19/mnist-feature-pilot-review'


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_sha(array):return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def audit(session):
    import numpy as np
    import torch  # CPU weights_only checkpoint read, never build/run a model.
    manifest=json.loads((session/'manifest.json').read_text());start=json.loads((session/'start_manifest.json').read_text())
    freeze=json.loads((REPORT/'SOURCE_FREEZE.json').read_text())
    require(manifest['status']=='completed','Session not completed')
    for key in ('spec','spec_sha256','source_sha256','entrypoint_sha256','environment','board_access','new_reconstruction'):
        require(manifest[key]==start[key],'Start/final mismatch: '+key)
    require(manifest['spec']==freeze['spec'] and manifest['spec_sha256']==freeze['spec_sha256'],'Unreviewed specification')
    require(hashlib.sha256(json.dumps(manifest['spec'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==freeze['spec_sha256'],'Canonical SPEC hash mismatch')
    require(manifest['source_sha256']==freeze['source_sha256'],'Source inventory differs')
    for name,pin in manifest['source_sha256'].items():require(sha(ROOT/name)==pin,'Source SHA differs: '+name)
    exp=ROOT/'experiments/mnist_feature_pilot_20261019'
    for name,pin in freeze['experiment_file_sha256'].items():require(sha(exp/name)==pin,'Reviewed pilot source differs: '+name)
    require(manifest['entrypoint_sha256']==freeze['entrypoint_sha256']==sha(exp/'experiment.py'),'Entrypoint differs')
    env=manifest['environment'];require(env['device']=='cuda','Unreviewed device')
    require(env['full_installed_versions']==freeze['dependency_versions'],'Dependency inventory differs')
    for name,pin in env['dependency_lock_sha256'].items():require(sha(exp/name)==pin,'Dependency lock differs')
    require(env['install_provenance_sha256']==sha(exp/'INSTALL_PROVENANCE.json') and env['bindsnet_source_manifest_sha256']==sha(exp/'BINDSNET_SOURCE.json'),'Dependency provenance hash differs')
    binds=json.loads((exp/'BINDSNET_SOURCE.json').read_text())
    for name,pin in binds['source_files'].items():require(sha(ROOT/'.venv-mnist-pilot/Lib/site-packages'/name)==pin,'BindsNET source differs')
    actual={p.name for p in session.iterdir() if p.is_file()}-{'manifest.json'}
    require(actual==set(manifest['result_sha256']),'Result file inventory differs')
    for name,pin in manifest['result_sha256'].items():require(sha(session/name)==pin,'Result SHA differs: '+name)
    require(manifest['unique_train_samples']==128 and manifest['completed_passes']==2 and manifest['test_samples']==0,'Sample scope differs')
    require(manifest['readout_trained'] is False and manifest['board_access'] is False and manifest['accuracy'] is None,'Out-of-scope claim')
    load=lambda name:np.load(session/name,allow_pickle=False)
    ids=load('train_ids.npy');images=load('train_images_uint8.npy');labels=load('train_labels.npy');spikes=load('input_spikes_uint8.npy');weights=load('frozen_weights.npy')
    require(ids.dtype==np.int64 and np.array_equal(ids,np.arange(128)),'Train IDs differ')
    require(images.dtype==np.uint8 and images.shape==(128,1,28,28),'Image format differs')
    require(labels.dtype==np.uint8 and labels.shape==(128,) and (labels<10).all(),'Label format differs')
    data=ROOT/'experiments/mnist_baseline_20261019/data/MNIST/raw'
    with gzip.open(data/'train-images-idx3-ubyte.gz','rb') as stream:
        require(struct.unpack('>IIII',stream.read(16))==(2051,60000,28,28),'Train image IDX header differs')
        require(stream.read(128*784)==images.tobytes(),'Train images not exact first128')
    with gzip.open(data/'train-labels-idx1-ubyte.gz','rb') as stream:
        require(struct.unpack('>II',stream.read(8))==(2049,60000),'Train labels IDX header differs')
        require(stream.read(128)==labels.tobytes(),'Train labels not exact first128')
    checkpoint=torch.load(ROOT/'weights/mnist_baseline_20261019/model.pt',map_location='cpu',weights_only=True)
    require(weights.dtype==np.float32 and weights.shape==(64,1,5,5) and np.isfinite(weights).all(),'Saved weights invalid')
    require(np.array_equal(weights,checkpoint['W'].numpy()) and tensor_sha(weights)==freeze['weight_sha256'],'Checkpoint weight content differs')
    threshold_sha=tensor_sha(checkpoint['v_thresh'].numpy());zero_bias_sha=tensor_sha(np.zeros(64,dtype=np.float32))
    require(manifest['archived_cfg']==checkpoint['cfg'] and manifest['effective_cfg']==checkpoint['cfg'],'Archived/effective cfg differs')
    expected_state=dict(weights={"('Input', 'Conv1')":tensor_sha(weights)},biases={"('Input', 'Conv1')":zero_bias_sha},thresholds={'Conv1':threshold_sha,'Inhib1':threshold_sha})
    require(manifest['frozen_state']==expected_state,'Frozen state metadata differs from checkpoint and zero bias')
    require(spikes.dtype==np.uint8 and spikes.shape==(100,128,1,28,28) and ((spikes==0)|(spikes==1)).all(),'Input spike data invalid')
    input_float_sha=tensor_sha(spikes.astype(np.float32))
    counts=[load('counts_repeat%d.npy'%i) for i in range(2)]
    require(np.array_equal(*counts) and sha(session/'counts_repeat0.npy')==sha(session/'counts_repeat1.npy'),'Counts repeats differ')
    require(len(manifest['passes'])==2,'Pass count differs')
    for i,(c,p) in enumerate(zip(counts,manifest['passes'])):
        require(c.dtype==np.float32 and c.shape==(128,36864) and np.isfinite(c).all() and ((c>=0)&(c<=100)&(c==np.floor(c))).all(),'Counts format/bounds invalid')
        require(p['repeat']==i and p['counts_sha256']==tensor_sha(c) and p['input_spikes_sha256']==input_float_sha,'Pass numeric hash differs')
        require(p['input_spikes']==int(spikes.sum()) and p['output_spikes']==int(c.sum()) and p['active_features']==int((c.sum(0)>0).sum()),'Pass counters differ')
        require(p['weight_before_sha256']==p['weight_after_sha256']==tensor_sha(weights) and p['threshold_sha256']==threshold_sha,'Pass invariant telemetry differs')
        require(p['seconds']>0 and p['monitor_device']=='cpu' and p['monitor_record_dtype']=='torch.bool','Timing/monitor metadata invalid')
        require(p['monitor_record_bytes']==100*128*64*24*24 and p['host_float32_reduction_input_bytes']==100*128*64*24*24*4,'Monitor memory size differs')
        require(0<p['cuda_peak_allocated_bytes']<=p['cuda_peak_reserved_bytes']<=env['gpu']['total_bytes'],'CUDA memory metadata invalid')
    require(manifest['weights_unchanged'] and manifest['repeat_counts_identical'] and manifest['learning_explicitly_disabled'],'Runtime invariant checks absent')
    require(manifest['uninformative_all_silent']==(int(counts[0].sum())==0),'Silence flag differs')
    # Independent arithmetic replay of the deterministic input encoder only.
    # No BindsNET/Conv/LIF or archived encoder function is invoked.
    x=images.reshape(128,784).astype(np.float32)/np.float32(255)
    q=(x*np.float32(255)).astype(np.uint64);idx=np.arange(784,dtype=np.uint64)[None,:]
    seed=(np.uint64(123)^(q.sum(1)*np.uint64(1315423911))^((q*idx).sum(1)*np.uint64(2654435761)))&np.uint64(0xffffffff)
    probability=np.float32(1)-np.exp(-x*np.float32(.0015));mismatches=0
    for t in range(100):
        h=(seed[:,None]+np.uint64(t)*np.uint64(2246822519)+idx*np.uint64(3266489917))&np.uint64(0xffffffff)
        h^=h>>np.uint64(16);h=(h*np.uint64(0x7feb352d))&np.uint64(0xffffffff)
        h^=h>>np.uint64(15);h=(h*np.uint64(0x846ca68b))&np.uint64(0xffffffff);h^=h>>np.uint64(16)
        regenerated=(h.astype(np.float32)/np.float32(4294967296)<probability)
        mismatches+=int(np.count_nonzero(regenerated!=spikes[t].reshape(128,784)))
    require(mismatches==0,'Independent input encoder replay differs')
    c=counts[0];input_per=spikes.sum(axis=(0,2,3,4));output_per=c.sum(1)
    return dict(status='pass',session=session.relative_to(ROOT).as_posix(),auditor_sha256=sha(__file__),manifest_sha256=sha(session/'manifest.json'),source_count=len(manifest['source_sha256']),result_file_count=len(actual),entrypoint_sha256=manifest['entrypoint_sha256'],spec_sha256=manifest['spec_sha256'],source_sha256=manifest['source_sha256'],result_sha256=manifest['result_sha256'],dependency_versions_verified=63,bindsnet_source_files_verified=len(binds['source_files']),train_ids_exact=True,train_images_labels_match_prefix=True,counts_repeats_byte_identical=True,input_saved_float32_sha256=input_float_sha,second_input_evidence='Only repeat0 input tensor saved; repeat1 equality is runtime assertion plus matching per-pass SHA, not independent second-file comparison',independent_numpy_input_replay_mismatches=mismatches,invariant_evidence='Saved W matches checkpoint; runtime assertions/per-pass SHA verify unchanged W/threshold/bias. Final live state tensors and per-step output spikes are not separately saved; no model re-execution in audit.',input_spikes=int(spikes.sum()),output_spikes=int(c.sum()),active_features=int((c.sum(0)>0).sum()),total_features=c.shape[1],counts_min=float(c.min()),counts_max=float(c.max()),nonzero_count_fraction=float(np.count_nonzero(c)/c.size),input_per_sample=dict(min=int(input_per.min()),max=int(input_per.max()),mean=float(input_per.mean()),zero_samples=int((input_per==0).sum())),output_per_sample=dict(min=float(output_per.min()),max=float(output_per.max()),mean=float(output_per.mean()),zero_samples=int((output_per==0).sum())),all_silent=bool((output_per==0).all()),last_timestep_input_spikes=int(spikes[-1].sum()),passes=manifest['passes'],session_wall_seconds=(datetime.fromisoformat(manifest['finished_utc'])-datetime.fromisoformat(manifest['started_utc'])).total_seconds(),total_session_bytes=sum(p.stat().st_size for p in session.iterdir() if p.is_file()),recorded_git_provenance=False,operator_reported_runtime_commit='c9d39ed4cd854c529f01806d05afb256b3e49230',operator_reported_clean_tracked=True,full_extraction='Technically suitable to prepare bounded chunked train extraction in the same reconstruction/environment; current pilot gate does not authorize expanded sample scope or accuracy claims.',accuracy=None,hardware_access=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    session=(ROOT/args.session).resolve();output=(ROOT/args.output).resolve()
    require(session.is_relative_to(ROOT/'results/mnist_feature_pilot_20261019'),'Session path outside pilot results')
    require(output.is_relative_to(ROOT/'reports') and not output.exists(),'Output must be a new report')
    result=audit(session);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ('status','input_spikes','output_spikes','active_features','independent_numpy_input_replay_mismatches','counts_max')}))
