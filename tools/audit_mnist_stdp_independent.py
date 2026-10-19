"""Independent saved-evidence audit: CPU arrays/input arithmetic only, no SNN run."""
import gzip
import hashlib
import json
from pathlib import Path
import struct
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/2026-10-19/mnist-stdp-training-review'
SESSION=ROOT/'results/mnist_stdp_retraining_20261019/440fc54b-a3bc-48f9-ae3c-43196809b96d'


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(value):return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def audit():
    import numpy as np
    import torch  # Restricted CPU checkpoint decoding only.
    session=SESSION
    m=json.loads((session/'manifest.json').read_text());start=json.loads((session/'start_manifest.json').read_text())
    freeze=json.loads((REPORT/'SOURCE_FREEZE.json').read_text())
    require(m['status']=='completed' and m['scope']=='train_only128_smoke','Unexpected result/scope')
    for key,value in start.items():
        if key not in ('status','network_executed'):require(m[key]==value,'Start/final provenance mismatch: '+key)
    require(start['network_executed'] is False and m['network_executed'] is True,'Execution state transition differs')
    require(m['spec']==freeze['spec'] and m['spec_sha256']==freeze['spec_sha256'],'Unreviewed specification')
    require(hashlib.sha256(json.dumps(m['spec'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==m['spec_sha256'],'Canonical spec hash differs')
    expected=dict(freeze['source_sha256'])
    expected.update({
        'experiments/mnist_baseline_20261019/data/MNIST/raw/t10k-images-idx3-ubyte.gz':'8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6',
        'experiments/mnist_baseline_20261019/data/MNIST/raw/t10k-labels-idx1-ubyte.gz':'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6'})
    require(m['source_sha256']==expected,'Source inventory differs')
    for name,pin in expected.items():require(sha(ROOT/name)==pin,'Source hash differs: '+name)
    for name,pin in freeze['experiment_file_sha256'].items():require(sha(ROOT/'experiments/mnist_stdp_retraining_20261019'/name)==pin,'Frozen experiment differs: '+name)
    require(m['entrypoint_sha256']==freeze['entrypoint_sha256'],'Entrypoint differs')
    actual={p.name for p in session.iterdir() if p.is_file()}-{'manifest.json'}
    require(actual==set(m['result_sha256']),'Result inventory differs')
    for name,pin in m['result_sha256'].items():require(sha(session/name)==pin,'Result hash differs: '+name)
    env=m['environment'];pilot=ROOT/'experiments/mnist_feature_pilot_20261019'
    oldfreeze=json.loads((ROOT/'reports/2026-10-19/mnist-feature-pilot-review/SOURCE_FREEZE.json').read_text())
    require(env['full_installed_versions']==oldfreeze['dependency_versions'],'Dependency versions differ')
    for name,pin in env['dependency_lock_sha256'].items():require(sha(pilot/name)==pin,'Dependency lock differs')
    require(sha(pilot/'BINDSNET_SOURCE.json')==env['bindsnet_source_manifest_sha256'],'Native source manifest differs')
    binds=json.loads((pilot/'BINDSNET_SOURCE.json').read_text())
    for name,pin in binds['source_files'].items():require(sha(ROOT/'.venv-mnist-pilot/Lib/site-packages'/name)==pin,'BindsNET source differs')
    rows=[json.loads(line) for line in (session/'events.jsonl').read_text().splitlines()]
    require([r['kind'] for r in rows]==['first_image_clipping_and_learning']+['stdp_image']*128+['stdp_training_complete'],'Event schedule differs')
    images=rows[1:-1];require([r['image_id'] for r in images]==list(range(128)),'Image coverage differs')
    load=lambda name:np.load(session/name,allow_pickle=False)
    initial=load('initial_weights.npy');trace=load('first_image_postupdate_weights.npy')
    weights=load('stdp_end_of_image_weights.npy');final=load('trained_weights.npy');threshold=load('trained_threshold.npy')
    require(np.array_equal(load('stdp_train_ids.npy'),np.arange(128)),'Train IDs differ')
    require(initial.shape==(64,1,5,5) and trace.shape==(100,64,1,5,5) and weights.shape==(128,64,1,5,5) and final.shape==initial.shape,'Array shape differs')
    for value in (initial,trace,weights,final,threshold):require(value.dtype==np.float32 and np.isfinite(value).all(),'Invalid weight/threshold array')
    require(np.allclose(initial.sum((1,2,3)),78.4,rtol=1e-6,atol=1e-5),'Initial normalization differs')
    clipped=np.clip(initial,0,1)
    require(np.array_equal(trace[0],clipped) and np.array_equal(trace[-1],weights[0]) and np.array_equal(weights[-1],final),'Clipping/weight trajectory links differ')
    require(((trace>=0)&(trace<=1)).all() and ((weights>=0)&(weights<=1)).all(),'Weights outside native bounds')
    require(threshold.shape==() and threshold.item()==np.float32(.38).item(),'Threshold changed')
    bias_sha=array_sha(np.zeros(64,dtype=np.float32));threshold_sha=array_sha(threshold)
    changes=[];deltas=[]
    for i,r in enumerate(images):
        require(r['steps']==100 and r['native_norm_none'] is True,'Native update scope differs')
        require(r['weight_sha256']==array_sha(weights[i]),'Per-image weight hash differs')
        require(r['threshold_sha256']==dict(Conv1=threshold_sha,Inhib1=threshold_sha) and r['bias_sha256']==bias_sha,'Fixed parameter telemetry differs')
        delta=weights[i]-(initial if i==0 else weights[i-1]);changed=int(np.count_nonzero(delta));l1=float(np.abs(delta).sum(dtype=np.float64))
        require(changed==r['changed_coefficients'] and np.isclose(l1,r['delta_l1'],rtol=2e-6,atol=1e-6),'Weight change telemetry differs')
        require(float(weights[i].min())==r['weight_min'] and float(weights[i].max())==r['weight_max'],'Weight bounds telemetry differs')
        changes.append(changed);deltas.append(l1)
    # Independently regenerate the deterministic input without invoking archived encoder/model.
    data=ROOT/'experiments/mnist_baseline_20261019/data/MNIST/raw/train-images-idx3-ubyte.gz'
    with gzip.open(data,'rb') as stream:
        require(struct.unpack('>IIII',stream.read(16))==(2051,60000,28,28),'IDX header differs')
        raw=np.frombuffer(stream.read(128*784),dtype=np.uint8).reshape(128,784)
    x=raw.astype(np.float32)/np.float32(255);q=(x*np.float32(255)).astype(np.uint64);idx=np.arange(784,dtype=np.uint64)[None,:]
    seed=(np.uint64(123)^(q.sum(1)*np.uint64(1315423911))^((q*idx).sum(1)*np.uint64(2654435761)))&np.uint64(0xffffffff)
    probability=np.float32(1)-np.exp(-x*np.float32(.0015));encoded=np.empty((100,128,784),dtype=np.float32)
    for t in range(100):
        h=(seed[:,None]+np.uint64(t)*np.uint64(2246822519)+idx*np.uint64(3266489917))&np.uint64(0xffffffff)
        h^=h>>np.uint64(16);h=(h*np.uint64(0x7feb352d))&np.uint64(0xffffffff)
        h^=h>>np.uint64(15);h=(h*np.uint64(0x846ca68b))&np.uint64(0xffffffff);h^=h>>np.uint64(16)
        encoded[t]=h.astype(np.float32)/np.float32(4294967296)<probability
    for i,r in enumerate(images):require(array_sha(encoded[:,i])==r['input_sha256'] and int(encoded[:,i].sum())==r['input_spikes'],'Independent input replay differs at image '+str(i))
    checkpoint=torch.load(session/'trained_model.pt',map_location='cpu',weights_only=True)
    require(np.array_equal(checkpoint['W'].numpy(),final) and np.array_equal(checkpoint['v_thresh'].numpy(),threshold),'Checkpoint payload differs')
    require(checkpoint['trained_samples']==128 and checkpoint['new_reconstruction'] is True and checkpoint['cfg']==m['effective_cfg'],'Checkpoint provenance differs')
    require(checkpoint['initial_weight_sha256']==array_sha(initial) and checkpoint['final_weight_sha256']==array_sha(final),'Checkpoint weight hashes differ')
    summary=m['training'];require(rows[-1]==dict(kind='stdp_training_complete',**summary),'Final summary event differs')
    require(summary['training_samples']==128 and summary['timesteps']==12800 and summary['output_spikes']==sum(r['output_spikes'] for r in images),'Training totals differ')
    require(summary['initial_weight_sha256']==array_sha(initial) and summary['final_weight_sha256']==array_sha(final),'Summary weights differ')
    require(m['physical_memristor'] is False and m['accuracy'] is None and m['test_samples']==0 and m['readout_trained'] is False,'Out-of-scope result')
    seconds=summary['training_seconds'];steady=sum(r['seconds'] for r in images[1:])/127
    require(0<seconds<1800,'Smoke runtime budget differs')
    return dict(status='pass',schema='independent-mnist-stdp-smoke-v1',auditor_sha256=sha(__file__),
        session=session.relative_to(ROOT).as_posix(),manifest_sha256=sha(session/'manifest.json'),entrypoint_sha256=m['entrypoint_sha256'],spec_sha256=m['spec_sha256'],
        sources_verified=len(expected),results_verified=len(actual),dependency_versions_verified=len(env['full_installed_versions']),bindsnet_sources_verified=len(binds['source_files']),
        input_replay=dict(images=128,exact_input_hashes=128,input_spikes=int(encoded.sum()),method='Independent NumPy arithmetic on raw train0:128; no archived encoder or SNN execution'),
        initial_out_of_bounds=int(((initial<0)|(initial>1)).sum()),first_step_equals_clamp=True,
        post_clipping_first_window_changed=int(np.count_nonzero(trace[-1]-clipped)),post_clipping_first_window_l1=float(np.abs(trace[-1]-clipped).sum(dtype=np.float64)),
        subsequent_changed_images=sum(c>0 for c in changes[1:]),subsequent_changed_coefficients_min=min(changes[1:]),
        final_weight_min=float(final.min()),final_weight_max=float(final.max()),final_weights_at_bounds=int(((final==0)|(final==1)).sum()),
        final_above_clipped=int((final>clipped).sum()),final_below_clipped=int((final<clipped).sum()),output_spikes=sum(r['output_spikes'] for r in images),silent_images=sum(r['output_spikes']==0 for r in images),
        fixed_threshold_and_bias_telemetry_match=True,training_seconds=seconds,wall_seconds=(datetime.fromisoformat(m['finished_utc'])-datetime.fromisoformat(m['started_utc'])).total_seconds(),
        cuda_peak_allocated_bytes=max(r['cuda_peak_allocated_bytes'] for r in images),cuda_peak_reserved_bytes=max(r['cuda_peak_reserved_bytes'] for r in images),
        full30000_stdp_planning_hours=[steady*30000/3600,seconds/128*30000/3600],full_feature_count_bytes=40000*36864*4,
        full_training_conditionally_qualified=True,full_gate_registration_required=True,physical_memristor=False,accuracy=None,
        limits=['Saved per-step weights only for first image; no independent full STDP forward replay.',
                'Per-image weights and pinned executed native code support learning beyond clipping; no isolated LTD/LTP event identification.',
                'Input hashes are independently reproduced; output spikes are runtime telemetry, not saved per-step spikes.',
                'Fresh seed/dependencies reconstruction, broken postwindow E/I hook preserved; not confirmation of historical 69.65%.',
                'Full30k must freshly initialize, use newly trained W for readout, freeze head/scaler before one test evaluation.',
                'Runtime is prefix extrapolation, not guaranteed; retain8h/8GiB bounds and fresh memory/disk preflight.'])


if __name__=='__main__':
    target=REPORT/'SMOKE128_INDEPENDENT_AUDIT.json'
    require(not target.exists(),'Preserve existing audit report')
    result=audit();target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
