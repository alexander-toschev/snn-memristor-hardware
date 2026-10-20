"""Offline raw replay then fixed ridge readout. No native model or board calls."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
import numpy as np
from tools import audit_mnist_conductance as numeric
from experiments.mnist_accuracy_20261020 import inputs,replay
from experiments.cifar_resumable_20261019.durable import atomic_json,sha
from experiments.unn_rev3_20261014.paths import repo_path


def audit(session):
    m=json.loads((session/'manifest.json').read_text())
    start=json.loads((session/'start_manifest.json').read_text())
    numeric.require(m['schema']=='mnist-frozen-accuracy-acquisition-v1' and m['status']=='completed','Completed accuracy acquisition required')
    numeric.require(m['session_id']==session.name and m['source'] in ('hardware','simulator'),'Session identity differs')
    numeric.require(not any(m.get(k) for k in ('error','cleanup_error','raw_finish_error')),'Acquisition error')
    for k,v in start.items():
        if k!='status':numeric.same(m[k],v,'start.'+k)
    names={p.relative_to(session).as_posix() for p in session.rglob('*') if p.is_file()}-{'manifest.json'}
    numeric.require(names==set(m['result_sha256']),'Inventory differs')
    for name,pin in m['result_sha256'].items():
        numeric.require(sha(numeric.local(name,session))==pin,'Result bytes differ: '+name)
    for name,pin in m['source_sha256'].items():
        if name.startswith('vendor/MemriBoard/MemriCORE/'):continue
        numeric.require(sha(repo_path(name))==pin,'Public source differs: '+name)
    fixture=json.loads((session/'fixture.json').read_text());inputs.verify(fixture)
    numeric.require(sha(session/'fixture.json')==m['fixture_sha256']==sha(inputs.HERE/'fixture.json'),'Frozen fixture differs')
    protocol=json.loads((session/'protocol.json').read_text())
    # Declarative protocol checked independently from the executed Collector.
    numeric.require(protocol['schema']=='mnist-frozen-post-stdp-accuracy128-100-v1' and
                    protocol['groups']==inputs.GROUPS and protocol['maximum_pulses']==0 and
                    protocol['amplitude_v']==.05 and protocol['maximum_mask']==4 and
                    protocol['maximum_impacts']==128064 and protocol['checkpoints']==['start','end'] and
                    protocol['lif_leak']==.5 and protocol['lif_threshold']==1.,
                    'Read-only protocol differs')
    numeric.require(inputs.digest(protocol)==m['protocol_sha256'],'Protocol SHA differs')
    numeric.require(sha(session/'parent_manifest.json')==m['parent']['manifest_sha256'] and
                    sha(session/'parent_observations.json')==m['parent']['observations_sha256'],'Parent binding differs')
    parent=json.loads((session/'parent_manifest.json').read_text())
    parent_obs=json.loads((session/'parent_observations.json').read_text())
    numeric.require(parent['status']=='completed' and parent['source']==m['source'] and
                    parent['session_id']==m['parent']['session_id'] and
                    parent['gain']==parent_obs['gain']==m['gain']==m['parent']['gain'],'Frozen parent state/gain differs')
    numeric.require(parent['result_sha256']['observations.json']==m['parent']['observations_sha256'],'Parent inventory binding differs')
    numeric.require(sha(session/'board.ini')==m['source_sha256']['configs/raspberry_board.ini'],'Board INI differs')
    if m['source']=='simulator':
        numeric.require(sha(session/'array_initial.cb')==parent['result_sha256']['array.cb']==m['parent']['final_array_sha256'],
                        'SIM must load parent final array, not initial or a new array')
        numeric.require(sha(session/'simulator_settings.ini')==sha(ROOT/'vendor/MemriBoard/simulator/simulator_settings.ini'),'SIM INI differs')
    records=numeric.durable_records(session)
    checker=replay.Replay(records,fixture,m['source'],m['gain'],m['parent'])
    try:actual=checker.run(protocol)
    finally:records.close()
    numeric.same(actual,json.loads((session/'observations.json').read_text()),'observations raw replay')
    for k in ('impacts','pulses','decisions','measured_images','gain'):numeric.same(actual[k],m[k],'manifest.'+k)
    numeric.require(actual['pulses']==0 and m['labels_loaded'] is False,'Write/label leakage')
    gs=np.asarray(actual['checkpoints'][0]['mean_inverse_r'])
    ge=np.asarray(actual['checkpoints'][1]['mean_inverse_r'])
    gp=np.asarray(parent_obs['checkpoints'][-1]['mean_inverse_r'])
    def stats(delta):
        return dict(mean_s=float(delta.mean()),maximum_absolute_s=float(np.abs(delta).max()),
                    changed_cells=int(np.count_nonzero(delta)),per_wl_mean_s=delta.mean(axis=1).tolist())
    report=dict(status='PASS',source=m['source'],session=session.relative_to(ROOT).as_posix(),
        manifest_sha256=sha(session/'manifest.json'),protocol_sha256=m['protocol_sha256'],
        analyzer_sha256=sha(__file__),replay_sha256=sha(Path(replay.__file__)),
        native_impacts=checker.calls,pulses=0,records=checker.records,images=228,gain=m['gain'],
        source_private_status='private_runtime_not_independently_reverified',
        parent_to_start_g=stats(gs-gp),start_to_end_g=stats(ge-gs),
        image_cell_participation=checker.participation,mode_counts=dict(checker.mode_counts),
        state_note='No write commands; native readout drift/disturbance remains possible. HW parent state cannot be restored from software.')
    return report,actual


def fit_ridge(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=int)
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale==0]=1.
    z=(x-mean)/scale;a=np.column_stack((z,np.ones(len(z))))
    penalty=np.eye(a.shape[1]);penalty[-1,-1]=0.
    coef=np.linalg.solve(a.T@a+penalty,a.T@np.eye(10)[y])
    return dict(mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist(),lambda_value=1.,bias_penalized=False)


def predict(model,x):
    a=np.column_stack(((np.asarray(x)-model['mean'])/model['scale'],np.ones(len(x))))
    return a@np.asarray(model['coef'])


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
    return [center-radius,center+radius]


def evaluate(scores,features,labels):
    scores=np.asarray(scores);features=np.asarray(features);labels=np.asarray(labels)
    predicted=scores.argmax(axis=1)
    ties=(np.abs(scores-scores.max(axis=1)[:,None])<=1e-12).sum(axis=1)>1
    silent=np.count_nonzero(features,axis=1)==0
    accepted=~(ties|silent);correct=predicted==labels;strict=correct&accepted
    confusion=np.zeros((10,11),dtype=int)
    for label,pred,ok in zip(labels,predicted,accepted):confusion[label,pred if ok else 10]+=1
    n=len(labels);k=int(strict.sum());covered=int(accepted.sum())
    return dict(total=n,strict_correct=k,strict_accuracy=k/n,conventional_argmax_correct=int(correct.sum()),
        conventional_argmax_accuracy=float(correct.mean()),coverage=covered/n,
        conditional_accuracy=float(correct[accepted].mean()) if covered else None,
        silent=int(silent.sum()),tied=int(ties.sum()),abstained=int((~accepted).sum()),
        strict_wilson95=wilson(k,n),confusion_rows_true_columns_pred0to9_then_abstain=confusion.tolist(),
        predictions=[dict(test_index=i,label=int(labels[i]),argmax=int(predicted[i]),
            strict_prediction=int(predicted[i]) if accepted[i] else None,silent=bool(silent[i]),
            tied=bool(ties[i]),strict_correct=bool(strict[i]),feature_spikes=int(features[i].sum()),
            scores=scores[i].tolist()) for i in range(n)])


def analyze(session,output):
    if output.exists():raise FileExistsError('Choose a new analysis directory')
    report,actual=audit(session)
    output.mkdir(parents=True);atomic_json(output/'AUDIT.json',report)
    outcomes=actual['outcomes']
    result=dict(source=report['source'],session=report['session'],test_selection='official test prefix0..99',
                training_selection='official train prefix0..127, freshly measured at frozen post-STDP state',
                gains=actual['gain'],protocol='ridge1/train-only standardization/128rasterfeatures',
                interval_limit='Wilson is descriptive; fixed nonrandom prefix is not a full10000 population accuracy guarantee',
                drift=dict(parent_to_start=report['parent_to_start_g'],start_to_end=report['start_to_end_g']),
                primary_group='learning',groups={})
    train_y=inputs.labels('train')
    # Both readouts are fitted and saved before opening official test labels.
    matrices={};models={}
    for name,wls in [('learning',range(4)),('control',range(4,8))]:
        features=np.asarray([[s for wl in wls for s in row['neurons'][wl]['posts']] for row in outcomes],dtype=float)
        matrices[name]=features
        models[name]=fit_ridge(features[:128],train_y)
        atomic_json(output/(name+'_READOUT.json'),models[name])
        np.save(output/(name+'_features.npy'),features,allow_pickle=False)
    test_y=inputs.labels('test')
    for name,features in matrices.items():
        result['groups'][name]=evaluate(predict(models[name],features[128:]),features[128:],test_y)
        result['groups'][name]['readout_sha256']=sha(output/(name+'_READOUT.json'))
        result['groups'][name]['train_spikes']=int(features[:128].sum())
        result['groups'][name]['test_spikes']=int(features[128:].sum())
    atomic_json(output/'ACCURACY.json',result)
    return {k:result['groups']['learning'][k] for k in ('strict_accuracy','conventional_argmax_accuracy','silent','tied','coverage')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();print(json.dumps(analyze(repo_path(a.session),repo_path(a.output_dir))))
