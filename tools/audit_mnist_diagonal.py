"""Independent diagonal-gain MNIST128 replay; unchanged physical I/O checks."""
import argparse
from configparser import ConfigParser
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
from tools import audit_mnist_conductance as numeric
from experiments.mnist_diagonal_20261020 import replay as diagonal
from experiments.unn_rev3_20261014.paths import repo_path
from read_all import EXPECTED_BOARD


def verify_bundle(session):
    require=numeric.require
    m=json.loads((session/'manifest.json').read_text());start=json.loads((session/'start_manifest.json').read_text())
    protocol=json.loads((session/'protocol.json').read_text())
    require(m['schema']=='mnist128-diagonal-launch-v1' and m['source'] in ('simulator','hardware'),'Wrong portable hardware scope')
    require(m['status'] in ('completed','uninformative_calibration'),'Complete/no-gain terminal outcome required')
    require(not any(m.get(k) for k in ('error','cleanup_error','raw_finish_error')),'Recorded acquisition error')
    require(m['reference'] is None and m['session_id']==session.name,'Hardware main session identity differs')
    for key in ('schema','source','session_id','protocol_sha256','source_sha256','fixture_sha256','reference','invocation','environment','started_utc','private_runtime_status'):
        numeric.same(m[key],start[key],'immutable manifest.'+key)
    numeric.unknown_physical_duration(m,start)
    numeric.same(protocol,diagonal.expected_protocol(),'unchanged protocol')
    require(m['protocol_sha256']==numeric.digest(protocol),'Protocol hash differs')
    require(m['private_runtime_status']=='private_runtime_not_independently_reverified','Private runtime status omitted')
    paths=m['source_sha256']
    required={'experiments/mnist_conductance_20261019/core.py','experiments/mnist_diagonal_20261020/experiment.py','experiments/mnist_diagonal_20261020/core.py',
        'experiments/cifar_resumable_20261019/durable.py','vendor/MemriBoard/manager/board.py',
        'vendor/MemriBoard/manager/service/converters.py','configs/raspberry_board.ini',numeric.FIXTURE_PATH}
    require(required<=set(paths),'Missing recorded public dependencies')
    for name,pin in paths.items():
        if name.startswith('vendor/MemriBoard/MemriCORE/'):continue
        require(numeric.sha(repo_path(name))==pin,'Recorded public source differs: '+name)
    require(numeric.sha(session/'board.ini')==paths['configs/raspberry_board.ini'],'INI snapshot differs')
    config=ConfigParser();config.read_string((session/'board.ini').read_text(encoding='utf-8-sig'))
    require(config.get('board','board_type')=='rp5_fpga_python','Unexpected INI driver')
    for key,value in EXPECTED_BOARD.items():require(config.getfloat('board',key)==value,'INI setting differs: '+key)
    require(m['fixture_sha256']==numeric.FIXTURE_SHA==numeric.sha(session/'fixture.json')==numeric.sha(ROOT/numeric.FIXTURE_PATH),'Fixture bytes differ')
    fixture=json.loads((session/'fixture.json').read_text());numeric.verify_fixture(fixture)
    if m['source']=='simulator':
        require(numeric.sha(session/'array_initial.cb')==m['array_initial_sha256']==start['array_initial_sha256'],'Initial simulator array differs')
        require((session/'simulator_settings.ini').read_bytes()==(ROOT/'vendor/MemriBoard/simulator/simulator_settings.ini').read_bytes(),'SIM settings differ')
    actual={p.relative_to(session).as_posix() for p in session.rglob('*') if p.is_file()}-{'manifest.json'}
    require(actual==set(m['result_sha256']),'Result inventory differs')
    for name,pin in m['result_sha256'].items():require(numeric.sha(numeric.local(name,session))==pin,'Result bytes differ: '+name)
    require(m['classification_accuracy_measured'] is False and m['full10k_executed'] is False,'Pilot scope overclaim')
    return m,protocol,fixture


def audit(session):
    session=Path(session).resolve()
    numeric.require(session.parent==ROOT/'results/mnist_diagonal_20261020','Wrong results directory')
    m,protocol,fixture=verify_bundle(session)
    records=numeric.durable_records(session)
    replay=diagonal.Replay(records,fixture,m['source'],reference=None)
    try:actual=replay.run(protocol)
    finally:records.close()
    numeric.same(json.loads((session/'observations.json').read_text()),actual,'observations semantic replay')
    for key in ('status','impacts','pulses','decisions','measured_images','gain'):
        numeric.same(m[key],actual[key],'final manifest.'+key)
    input_total=[sum(counts[bl] for counts in replay.input_events.values()) for bl in range(32)]
    return dict(schema='mnist128-diagonal-local-audit-v1',status='PASS',source=m['source'],outcome=actual['status'],
        session=session.relative_to(ROOT).as_posix(),analyzer_sha256=numeric.sha(__file__),
        numeric_replay_sha256=numeric.sha(Path(numeric.__file__)),diagonal_replay_sha256=numeric.sha(Path(diagonal.__file__)),manifest_sha256=numeric.sha(session/'manifest.json'),
        protocol_sha256=numeric.digest(protocol),fixture_sha256=numeric.FIXTURE_SHA,
        private_runtime_status='private_runtime_not_independently_reverified',private_receipt_required=False,
        native_impacts=replay.calls,pulses=replay.pulses,decisions=replay.decisions,measured_images=replay.ordinal,
        selected_gain=actual['gain'],records=replay.records,checkpoint_count=len(replay.checkpoints),
        checkpoint_cells=256,learning_eligible_cells=128,control_cells=128,
        mode_counts=dict(replay.mode_counts),maximum_predicted_mvm_current_a=replay.max_mvm_current,
        input_events_by_bl=input_total,no_image_event_bl=[i for i,n in enumerate(input_total) if n==0],
        image_cell_participation=replay.participation,diagnostics=actual['diagnostics'],
        classification_accuracy_measured=False,full10k_executed=False,model_or_connector_opened_by_auditor=False,
        limitations=['Integrity/mechanism qualification does not demonstrate learning benefit or accuracy.',
                    'No private driver attestation; Git does not verify untracked runtime bytes.',
                    'Digital checkpoints cannot restore physical conductances after interruption.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=repo_path(args.output)
    if out.exists():raise FileExistsError('Audit output already exists')
    result=audit(repo_path(args.session));out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8',newline='\n') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','source','outcome','native_impacts','pulses','selected_gain')}))

if __name__=='__main__':main()
