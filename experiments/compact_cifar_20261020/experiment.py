"""One fixed CPU training run and sealed pilot100 evaluation. No board access."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
for directory in (ROOT,ROOT/'.deps-classifier',ROOT/'.deps'):
    if directory.is_dir():sys.path.insert(0,str(directory))
import numpy as np
import torch
from experiments.compact_cifar_20261020 import data,model

SPEC=dict(schema='compact-rgb48-lif8-cifar10-cpu-v1',seed=20261020,source='CPU',
    classes=data.CLASSES,T=12,leak=.5,threshold=1.,reset='hard_zero_detached_mask',
    surrogate='1/(1+5*abs(u-1))**2',epochs=50,batch_size=128,updates_per_epoch=36,
    train_count=4500,validation_count=500,test_count=100,
    optimizer=dict(name='Adam',lr=.001,betas=[.9,.999],eps=1e-8,weight_decay=0.,amsgrad=False),
    W_projection=[-1.,1.],train_shuffle=True,drop_last=False,threads=2,
    checkpoint_selection='minimum validation mean CE; exact ties earliest epoch',
    test_selection='first10/class official order excluding10 historical fixtures; sorted index',
    test_tuning=False,hardware_accuracy=False)


def now():return datetime.now(timezone.utc).isoformat()


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def tensor_sha(value):return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def initialize():
    random.seed(SPEC['seed']);np.random.seed(SPEC['seed']);torch.manual_seed(SPEC['seed'])
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    return model.Network().float().cpu()


def prepare_artifacts(out,assets):
    sums,times,spikes=data.encode(assets['images'])
    test_sums,test_times,test_spikes=data.encode(assets['test_images'])
    entries=[]
    train_set=set(assets['train_positions'].tolist())
    for i,(image,original,label) in enumerate(zip(assets['images'],assets['official_train_ids'],assets['labels'])):
        entries.append(dict(filtered_position=i,official_train_index=int(original),label=int(label),
            split='train' if i in train_set else 'validation',image_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
    save(out/'train_validation_manifest.json',dict(entries=entries,split_sha256=data.SPLIT_SHA))
    acquisition=[];evaluation=[]
    for n,(image,index) in enumerate(zip(assets['test_images'],assets['official_test_ids'])):
        acquisition.append(dict(official_test_index=int(index),image_chw_uint8=image.tolist(),
            image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),pool_sums=test_sums[n].tolist(),
            ttfs_times=test_times[n].tolist(),schedule_sha256=hashlib.sha256(test_spikes[n].astype(np.uint8).tobytes()).hexdigest()))
        evaluation.append(dict(official_test_index=int(index),label=int(assets['test_labels'][n])))
    save(out/'acquisition_fixture.json',dict(schema='compact-cifar-pilot100-input-v1',cases=acquisition,
        source_dataset_sha256=data.PINS['test'],input_order='RGB c-major4x4',T=12))
    save(out/'sealed_evaluation.json',dict(classes=data.CLASSES,cases=evaluation))
    np.save(out/'train_validation_times.npy',times,allow_pickle=False)
    save(out/'input_freeze.json',dict(created_utc=now(),before_training=True,
        acquisition_fixture_sha256=data.sha(out/'acquisition_fixture.json'),
        sealed_evaluation_sha256=data.sha(out/'sealed_evaluation.json'),
        train_validation_manifest_sha256=data.sha(out/'train_validation_manifest.json')))
    return torch.from_numpy(spikes),torch.from_numpy(assets['labels']),torch.from_numpy(test_spikes)


def wilson(correct,total):
    p=correct/total;z=1.959963984540054;denom=1+z*z/total
    center=(p+z*z/(2*total))/denom
    half=z*((p*(1-p)/total+z*z/(4*total*total))**.5)/denom
    return [center-half,center+half]


def evaluate_selected(out,network,inputs,labels,official_ids):
    freeze=json.loads((out/'checkpoint_freeze.json').read_text(encoding='utf-8'))
    if data.sha(out/'selected.pt')!=freeze['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from before-test freeze')
    # Reload the frozen selected epoch; never evaluate the final epoch accidentally.
    network.load_state_dict(torch.load(out/'selected.pt',map_location='cpu',weights_only=True))
    network.eval()
    with torch.no_grad():logits,z,current,v=network(inputs,return_trace=True)
    scores=logits.numpy();spikes=z.numpy().astype(np.uint8)
    predicted=scores.argmax(axis=1);truth=np.asarray(labels)
    sorted_scores=np.sort(scores,axis=1);margin=sorted_scores[:,-1]-sorted_scores[:,-2]
    silent=spikes.sum(axis=(1,2))==0;ties=margin==0;near=margin<=1e-6
    correct=predicted==truth;informative=~(silent|near)
    confusion=np.zeros((10,10),dtype=np.int64)
    for a,b in zip(truth,predicted):confusion[a,b]+=1
    cases=[]
    for i in range(len(truth)):
        first=[int(np.flatnonzero(spikes[i,:,j])[0]) if spikes[i,:,j].any() else None for j in range(8)]
        cases.append(dict(official_test_index=int(official_ids[i]),label=int(truth[i]),
            prediction=int(predicted[i]),correct=bool(correct[i]),logits=scores[i].tolist(),
            spike_counts=spikes[i].sum(axis=0).tolist(),first_spikes=first,
            all_eight_silent=bool(silent[i]),exact_tie=bool(ties[i]),near_tie=bool(near[i]),
            margin=float(margin[i]),informative=bool(informative[i])))
    report=dict(source='CPU digital new compact model; not MemriBoard',hardware_accuracy=False,
        numerator=int(correct.sum()),denominator=len(truth),accuracy=float(correct.mean()),
        informative_correct=int((correct&informative).sum()),informative_coverage=int(informative.sum()),
        all_silent=int(silent.sum()),exact_ties=int(ties.sum()),near_ties=int(near.sum()),
        confusion=confusion.tolist(),per_class_correct=confusion.diagonal().tolist(),
        per_class_count=confusion.sum(axis=1).tolist(),wilson95_descriptive=wilson(int(correct.sum()),len(truth)),
        interval_limit='Deterministic stratified pilot, not IID random sample or population confidence guarantee',
        checkpoint_freeze_sha256=data.sha(out/'checkpoint_freeze.json'),cases=cases)
    save(out/'CPU_PILOT100.json',report)
    np.save(out/'cpu_currents.npy',current.numpy(),allow_pickle=False)
    np.save(out/'cpu_spikes.npy',spikes,allow_pickle=False)
    np.save(out/'cpu_membranes.npy',v.numpy(),allow_pickle=False)
    return report


def run():
    network=initialize()
    out=ROOT/'results/compact_cifar_20261020'/str(uuid.uuid4());out.mkdir(parents=True)
    manifest=dict(status='failed',started_utc=now(),spec=SPEC,spec_sha256=digest(SPEC),
        source_sha256={p.relative_to(ROOT).as_posix():data.sha(p) for p in Path(__file__).parent.iterdir() if p.is_file()},
        dataset_sha256=data.PINS,environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,
        device='cpu',threads=torch.get_num_threads(),deterministic=torch.are_deterministic_algorithms_enabled()))
    save(out/'start_manifest.json',manifest);save(out/'protocol.json',SPEC)
    try:
        assets=data.prepare();data.load_dataset('meta')
        inputs,labels,test_inputs=prepare_artifacts(out,assets)
        torch.save(network.state_dict(),out/'initial.pt')
        train=torch.as_tensor(assets['train_positions']);val=torch.as_tensor(assets['val_positions'])
        generator=torch.Generator(device='cpu').manual_seed(SPEC['seed'])
        optimizer=torch.optim.Adam(network.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,
                                   weight_decay=0.,amsgrad=False)
        epochs=[];permutations=[];best=float('inf');selected_epoch=None;updates=0
        started=time.monotonic()
        for epoch in range(1,51):
            network.train();permutation=torch.randperm(4500,generator=generator);permutations.append(permutation.numpy())
            losses=0.;correct=0
            for offset in range(0,4500,128):
                ids=train[permutation[offset:offset+128]]
                optimizer.zero_grad(set_to_none=True)
                logits=network(inputs[ids]);loss=torch.nn.functional.cross_entropy(logits,labels[ids],reduction='mean')
                loss.backward();optimizer.step()
                with torch.no_grad():network.W.clamp_(-1.,1.)
                losses+=float(loss.detach())*len(ids);correct+=int((logits.argmax(1)==labels[ids]).sum());updates+=1
            network.eval()
            with torch.no_grad():
                logits=network(inputs[val])
                val_loss=float(torch.nn.functional.cross_entropy(logits,labels[val],reduction='sum'))/500
                val_correct=int((logits.argmax(1)==labels[val]).sum())
            row=dict(epoch=epoch,train_mean_ce=losses/4500,train_correct=correct,
                     validation_mean_ce=val_loss,validation_correct=val_correct,
                     permutation_sha256=tensor_sha(permutation),updates=updates)
            epochs.append(row)
            if val_loss<best:
                best=val_loss;selected_epoch=epoch
                torch.save(network.state_dict(),out/'selected.pt')
            save(out/'epochs.json',epochs)
            save(out/'progress.json',dict(epoch=epoch,updates=updates,selected_epoch=selected_epoch,
                                         elapsed_s=time.monotonic()-started))
        if updates!=1800:raise ValueError('Unexpected optimizer update count')
        np.save(out/'permutations.npy',np.stack(permutations),allow_pickle=False)
        torch.save(dict(model=network.state_dict(),optimizer=optimizer.state_dict(),
                        torch_rng=torch.get_rng_state(),permutation_rng=generator.get_state()),out/'final_training_state.pt')
        numpy_rng=np.random.get_state()
        save(out/'rng.json',dict(python=random.getstate(),numpy=[numpy_rng[0],numpy_rng[1].tolist(),*numpy_rng[2:]]))
        network.load_state_dict(torch.load(out/'selected.pt',map_location='cpu',weights_only=True))
        for name in ('W','A','b'):np.save(out/(name+'.npy'),getattr(network,name).detach().numpy(),allow_pickle=False)
        save(out/'gain_plan.json',dict(status='physical_mapping_pending_actual_stable_map',
            amplitude_v=.05,maximum_active_bl=4,W_shape=[8,48],signed_coefficient_count=384,
            voltage_per_weight=None,physical_programming_performed=False,test_responses_for_gain=False,
            calibration_rule='fresh singleton additivity; no per-weight CPU substitution'))
        save(out/'checkpoint_freeze.json',dict(created_utc=now(),selected_epoch=selected_epoch,
            validation_mean_ce=best,checkpoint_sha256=data.sha(out/'selected.pt'),test_forward_called=False,
            input_freeze_sha256=data.sha(out/'input_freeze.json'),
            W_sha256=data.sha(out/'W.npy'),A_sha256=data.sha(out/'A.npy'),b_sha256=data.sha(out/'b.npy')))
        report=evaluate_selected(out,network,test_inputs,assets['test_labels'],assets['official_test_ids'])
        manifest.update(status='completed',selected_epoch=selected_epoch,updates=updates,
                        cpu_pilot_correct=report['numerator'],cpu_pilot_total=100)
    except (Exception,KeyboardInterrupt) as exc:manifest.update(error=type(exc).__name__+': '+str(exc))
    finally:
        manifest['finished_utc']=now()
        manifest['result_sha256']={p.relative_to(out).as_posix():data.sha(p) for p in out.rglob('*') if p.is_file()}
        save(out/'manifest.json',manifest)
    print(json.dumps(dict(session=out.relative_to(ROOT).as_posix(),status=manifest['status'],
                         correct=manifest.get('cpu_pilot_correct'),denominator=100,error=manifest.get('error'))))
    return 0 if manifest['status']=='completed' else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args();raise SystemExit(run())
