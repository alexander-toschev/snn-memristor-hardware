"""Review-gated label-free MNIST conductance pilot; public MemriBoard/INI only."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import random
import shutil
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
from experiments.mnist_conductance_20261019 import core
from experiments.cifar_resumable_20261019 import durable
from tools import prepare_mnist_conductance_fixture as inputs

FIXTURE='experiments/mnist_conductance_20261019/fixture.json'
FIXTURE_SHA='02a75724f9048d451f1a82aa313c225310ff0ce4bf85bd1de9e90ae660323356'
DESIGN='docs/plans/MNIST_PHYSICAL_PILOT128_20261019.md'
DESIGN_SHA='3a2e4b6a3fce91f4fb57701d0bc200977746bbc54e0012176ea4e8d2ad1a2455'
APPROVED_SIM_SPEC_SHA256=frozenset({'a9cc77abce4cdde7e62a4bb8924d3eaf8b795509207dff3f262b4bbb56597e95'})
REVIEWED_NEW_SOURCE_SHA256={  # Entry self-SHA lives in external CODE_FREEZE.
    'experiments/mnist_conductance_20261019/core.py':'9541bae666adc847705360f65d919ff044719d015b87512bfdc449b6cd707d53',
    'experiments/mnist_conductance_20261019/README.md':'e5de69ad710986e5535938af3a3ce882e7c813413137adb2ed7aabe932150691'}
PINS={
    'experiments/unn_rev3_20261014/paths.py':'2c841b4a446a767595c68aad243abd966cee94ffd91f74c3c2823998fdc9ec1c',
    'tools/prepare_mnist_conductance_fixture.py':'9f195d33cb27309c3ed98403fe33eac4bb945d6d4e7b434f81142f98343d8215',
    'experiments/cifar_resumable_20261019/durable.py':'b00c094da70af933a7318a119fe5439b40d957c7aaabb39c752170a1dbf2abb3',
    'experiments/conductance_stdp_20261019/core.py':'005586394be54aad0ac7d116c729d7b733628161ef7a442975ee9b94209aa86f',
    'cell_calibration.py':'eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8',
    'read_all.py':'b30ae2f4dd3cd166dbc5cefb07b558127de2bbdba1f6674c40b1b45f4dd74d5f',
    'native_conversions.py':'2ad2e9c057bb14979b083a6e846e8c417c080a6c5273d848e31f893b7970253e',
    'experiment_hardware.py':'059dc4f827943a9df631631f52e776b57182afc9d36db0c1ad5246427f73548c'}


def sha(path):
    return durable.sha(path)


def load_fixture():
    path=ROOT/FIXTURE
    if sha(path)!=FIXTURE_SHA:raise ValueError('Reviewed input fixture changed or is a LFS pointer')
    data=json.loads(path.read_text(encoding='utf-8'))
    if data['schema']!='mnist-conductance-label-free176-v1' or data['groups']!=inputs.GROUPS:
        raise ValueError('Fixture split contract differs')
    if data['source_gzip_sha256']!=inputs.SOURCE_SHA or data['extractor_sha256']!=PINS['tools/prepare_mnist_conductance_fixture.py']:
        raise ValueError('Fixture source provenance differs')
    expected=[i for group in inputs.GROUPS.values() for i in group]
    if [row['train_index'] for row in data['images']]!=expected or data['labels_included'] is not False:
        raise ValueError('Wrong fixture coverage/labels')
    joined=[]
    for row in data['images']:
        pixels=bytes(row['pixels_uint8']); derived=inputs.pooled_schedule(pixels)
        if row['shape']!=[28,28] or hashlib.sha256(pixels).hexdigest()!=row['image_sha256']:
            raise ValueError('Image pixels differ')
        if any(row[key]!=value for key,value in derived.items()):raise ValueError('Integer pooling/TTFS differs')
        joined.append(pixels)
    if hashlib.sha256(b''.join(joined)).hexdigest()!=data['images_sha256']:raise ValueError('Aggregate image bytes differ')
    return data


def preflight(source):
    for path,pin in {**PINS,DESIGN:DESIGN_SHA,**REVIEWED_NEW_SOURCE_SHA256}.items():
        if sha(ROOT/path)!=pin:raise ValueError('Reviewed source changed: '+path)
    fixture=load_fixture()
    from cell_calibration import check_sources
    config,snapshot,pins=check_sources(source)  # Only files/INI; no Connector construction.
    pins.update(PINS); pins[DESIGN]=DESIGN_SHA; pins[FIXTURE]=FIXTURE_SHA
    for path in Path(__file__).parent.iterdir():
        if path.is_file() and path.suffix in ('.py','.md'):
            pins[path.relative_to(ROOT).as_posix()]=sha(path)
    free=shutil.disk_usage(ROOT).free
    if free<core.SPEC['initial_minimum_free_bytes']:raise ValueError('Need6GiB free for bounded raw/LFS/reserve')
    return config,snapshot,pins,fixture,free


def reference_main(name,pins):
    session=inputs.local(name)
    if session.parent!=ROOT/'results/mnist_conductance_20261019':raise ValueError('Reference main must be a pilot session')
    manifest=json.loads((session/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='completed' or manifest.get('source')!='simulator' or manifest.get('reference') is not None:
        raise ValueError('Reference requires completed main native SIM')
    if manifest['protocol_sha256']!=core.digest(core.SPEC) or manifest['source_sha256']!=pins or manifest['fixture_sha256']!=FIXTURE_SHA:
        raise ValueError('Reference main source/input/protocol differs')
    actual={p.relative_to(session).as_posix() for p in session.rglob('*') if p.is_file() and p.name!='manifest.json'}
    if actual!=set(manifest['result_sha256']):raise ValueError('Reference main artifact inventory differs')
    for relative,pin in manifest['result_sha256'].items():
        path=(session/relative).resolve()
        if not path.is_relative_to(session.resolve()) or sha(path)!=pin:raise ValueError('Reference main artifact changed')
    if sha(session/'array_initial.cb')!=manifest['array_initial_sha256']:
        raise ValueError('Reference main immutable initial array differs')
    observations=json.loads((session/'observations.json').read_text(encoding='utf-8'))
    gain=observations['gain']
    if gain not in core.GAINS or observations['status']!='completed' or observations['calibration']['selected_gain']!=gain:
        raise ValueError('Reference main gain not qualified')
    return dict(session=name,manifest_sha256=sha(session/'manifest.json'),
        observations_sha256=sha(session/'observations.json'),gain=gain,
        array_initial_sha256=manifest['array_initial_sha256'],fit_raw_index_sha256=sha(session/'raw/index.json'))


def preserve_initial_array(serial,output,reference=None):
    """Native SIM mutates array.cb on writes; retain pre-Connector bytes separately."""
    initial=output/'array_initial.cb'
    with initial.open('xb') as target:
        target.write(serial.with_suffix('.cb').read_bytes())
    value=sha(initial)
    if reference is not None and value!=reference['array_initial_sha256']:
        raise ValueError('Reference native SIM initial array differs from main')
    return value


def save(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--run-simulator',action='store_true')
    mode.add_argument('--run-hardware',action='store_true')
    parser.add_argument('--preflight-source',choices=('hardware','simulator'),default='simulator')
    parser.add_argument('--reference-main',help='SIM-only completed main session, fixed gain/initial array binding')
    args=parser.parse_args(argv)
    source=args.preflight_source if args.dry_run else 'hardware' if args.run_hardware else 'simulator'
    if args.reference_main and source!='simulator':raise ValueError('No-update reference is SIM only')
    protocol_sha=core.digest(core.SPEC)
    if args.run_hardware:raise ValueError('Hardware launch gate closed; no HW grant exists for this new pilot')
    if not args.dry_run and protocol_sha not in APPROVED_SIM_SPEC_SHA256:
        raise ValueError('Native SIM requires independent source review and registered protocol')
    config,snapshot,pins,fixture,free=preflight(source)
    reference=reference_main(args.reference_main,pins) if args.reference_main else None
    if args.dry_run:
        print(json.dumps(dict(source=source,protocol_sha256=protocol_sha,protocol=core.SPEC,
            fixture_sha256=FIXTURE_SHA,source_sha256=pins,reference=reference,disk_free_bytes=free,
            registered_sim=protocol_sha in APPROVED_SIM_SPEC_SHA256,connector_created=False,board_access=False)))
        return 0
    from read_all import now
    from experiment_hardware import launch_provenance
    sid=str(uuid.uuid4());output=ROOT/'results/mnist_conductance_20261019'/sid;output.mkdir(parents=True,exist_ok=False)
    save(output/'protocol.json',core.SPEC);(output/'board.ini').write_bytes(snapshot)
    (output/'fixture.json').write_bytes((ROOT/FIXTURE).read_bytes())
    packages={}
    for package in ('numpy','SQLAlchemy','greenlet'):
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]=None
    manifest=dict(session_id=sid,source=source,status='failed',started_utc=now(),
        protocol_sha256=protocol_sha,source_sha256=pins,fixture_sha256=FIXTURE_SHA,
        reference=reference,invocation=launch_provenance(argv),
        environment=dict(python=sys.version,platform=platform.platform(),packages=packages),
        physical_pulse_duration_s=None,classification_accuracy_measured=False,full10k_executed=False)
    writer=connector=runner=None; lifecycle_assumed=False
    try:
        writer=durable.Writer(output/'raw',limit=core.SPEC['raw_chunk_bytes'],
            maximum_expanded_bytes=core.SPEC['maximum_raw_expanded_bytes'],minimum_free_bytes=core.SPEC['raw_minimum_free_bytes'])
        from experiments.unn_rev3_20261014.paths import use_local_dependencies
        use_local_dependencies()
        import numpy as np
        from manager.board import Connector
        from manager.service import converters
        from simulator.src import create_crossbar_array
        from cell_calibration import SIM_INI
        random.seed(core.SPEC['seed']);np.random.seed(core.SPEC['seed'])
        serial=output/'array';create_crossbar_array(str(serial),32,8)
        (output/'simulator_settings.ini').write_bytes((ROOT/SIM_INI).read_bytes())
        manifest['array_initial_sha256']=preserve_initial_array(serial,output,reference)
        save(output/'start_manifest.json',manifest)
        connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
            board_type=config.get('board','board_type'),task_format='SI',config=config,crossbar_serial=str(serial))
        runner=core.Collector(connector,converters,writer,source,fixture,reference=reference)
        lifecycle_assumed=True
        result=runner.run();save(output/'observations.json',result)
        manifest.update(status=result['status'],impacts=result['impacts'],pulses=result['pulses'],
                        decisions=result['decisions'],measured_images=result['measured_images'],gain=result['gain'])
    except (Exception,KeyboardInterrupt) as exc:
        manifest['error']=repr(exc)
    finally:
        if connector is not None and not lifecycle_assumed:
            try:
                closed=connector.close_port()
                if writer:writer.write(dict(event='pre_run_public_close',result=core.safe(closed)))
                if not closed:manifest['cleanup_error']='Public close returned false'
            except Exception as exc:manifest['cleanup_error']=repr(exc)
        if writer:
            try:writer.finish()
            except Exception as exc:manifest.update(status='failed',raw_finish_error=repr(exc))
        if runner:
            manifest.update(impacts=runner.calls,pulses=runner.pulses,decisions=runner.decisions,
                            measured_images=runner.ordinal,gain=runner.gain)
            if manifest['status']=='failed':save(output/'partial_observations.json',runner.summary('failed'))
        manifest['finished_utc']=now()
        manifest['result_sha256']={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}
        save(output/'manifest.json',manifest)
    print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'],error=manifest.get('error'))))
    return 0 if manifest['status']=='completed' else 3 if manifest['status']=='uninformative_calibration' else 2


if __name__=='__main__':raise SystemExit(main())
