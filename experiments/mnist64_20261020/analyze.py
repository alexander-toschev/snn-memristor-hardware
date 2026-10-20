"""Offline independent native replay and fixed readout verification."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
import numpy as np
from tools import audit_mnist_conductance as numeric
from experiments.mnist64_20261020 import inputs,replay,readout
from experiments.cifar_resumable_20261019.durable import atomic_json,sha
from experiments.unn_rev3_20261014.paths import repo_path


def audit(session,output):
    if output.exists():raise FileExistsError('Choose fresh output directory')
    m=json.loads((session/'manifest.json').read_text());start=json.loads((session/'start_manifest.json').read_text())
    numeric.require(m['schema']=='mnist64-pipeline-stage-v1' and m['status'] in ('completed','uninformative_calibration'),'Terminal successful semantic outcome required')
    numeric.require(m['session_id']==session.name and not any(m.get(k) for k in ('error','cleanup_error','raw_finish_error')),'Identity/error')
    for k,v in start.items():
        if k!='status':numeric.same(m[k],v,'start.'+k)
    names={p.relative_to(session).as_posix() for p in session.rglob('*') if p.is_file()}-{'manifest.json'}
    numeric.require(names==set(m['result_sha256']),'Inventory differs')
    for n,pin in m['result_sha256'].items():numeric.require(sha(numeric.local(n,session))==pin,'Result bytes differ:'+n)
    for n,pin in m['source_sha256'].items():
        if n.startswith('vendor/MemriBoard/MemriCORE/'):continue
        numeric.require(sha(repo_path(n))==pin,'Public source differs:'+n)
    fixture=json.loads((session/'fixture.json').read_text());inputs.verify(fixture)
    numeric.require(sha(session/'fixture.json')==m['fixture_sha256'],'Fixture hash')
    protocol=json.loads((session/'protocol.json').read_text());numeric.require(inputs.digest(protocol)==m['protocol_sha256'],'Protocol hash')
    numeric.require(m['physical_pulse_duration_s'] is None and m['labels_in_acquisition'] is False,'Duration/labels scope')
    parent=m['parent'];gain=m['initial_gain']
    if parent:
        numeric.require(sha(session/'parent_manifest.json')==parent['manifest_sha256'] and sha(session/'parent_observations.json')==parent['observations_sha256'],'Parent snapshots')
        pm=json.loads((session/'parent_manifest.json').read_text());po=json.loads((session/'parent_observations.json').read_text())
        numeric.require(pm['status']=='completed' and pm['source']==m['source'] and pm['session_id']==parent['session_id'] and pm['gain']==po['gain']==gain==parent['gain'],'Parent gain/state binding')
        numeric.require(pm['result_sha256']['observations.json']==parent['observations_sha256'],'Parent observations binding')
        if m['source']=='simulator':numeric.require(sha(session/'array_initial.cb')==pm['result_sha256']['array.cb']==parent['final_array_sha256'],'Final parent array not loaded')
    if m['source']=='simulator':numeric.require(sha(session/'array_initial.cb')==m['array_initial_sha256'],'Initial snapshot')
    rows=numeric.durable_records(session);checker=replay.Replay(rows,fixture,m['source'],gain,parent)
    try:actual=checker.run(protocol)
    finally:rows.close()
    numeric.same(actual,json.loads((session/'observations.json').read_text()),'independent raw replay')
    for k in ('status','impacts','pulses','decisions','measured_images','gain'):numeric.same(actual[k],m[k],'manifest.'+k)
    report=dict(status='PASS',source=m['source'],scope=m['scope'],session=session.relative_to(ROOT).as_posix(),
        impacts=checker.calls,pulses=checker.pulses,records=checker.records,gain=checker.gain,
        manifest_sha256=sha(session/'manifest.json'),replay_sha256=sha(Path(replay.__file__)),
        private_runtime_status='private_runtime_not_independently_reverified',separate_control_bank=False)
    output.mkdir(parents=True)
    if m['scope']=='accuracy':
        x=readout.features(actual);model=readout.fit_ridge(x[:128],inputs.labels('train'))
        numeric.same(model,json.loads((session/'READOUT.json').read_text()),'train-only ridge replay')
        score=readout.evaluate(readout.predict(model,x[128:]),x[128:],inputs.labels('test'))
        saved=json.loads((session/'ACCURACY.json').read_text())
        numeric.same(score,{k:saved[k] for k in score},'accuracy replay')
        numeric.require(np.array_equal(x,np.load(session/'features.npy',allow_pickle=False)),'Features differ')
        report['accuracy']={k:score[k] for k in ('strict_accuracy','conventional_argmax_accuracy','coverage','silent','tied','strict_wilson95')}
        delta=np.asarray(actual['checkpoints'][-1]['mean_inverse_r'])-np.asarray(actual['checkpoints'][0]['mean_inverse_r'])
        report['read_only_drift_g_s']=dict(maximum_absolute=float(np.abs(delta).max()),changed_cells=int(np.count_nonzero(delta)))
    atomic_json(output/'AUDIT.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();print(json.dumps(audit(repo_path(a.session),repo_path(a.output_dir))))
