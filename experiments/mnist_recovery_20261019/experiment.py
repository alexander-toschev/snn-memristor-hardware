"""Offline MNIST archive inspection. No training, network execution or board imports."""
import argparse
import ast
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path, PureWindowsPath
import struct
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE='experiments/mnist_baseline_20261019'


def local_path(value):
    path,windows=Path(value),PureWindowsPath(value)
    if not value or path.is_absolute() or windows.drive or windows.root or '..' in path.parts or '..' in windows.parts:
        raise ValueError('Use a repository-relative path')
    resolved=(ROOT/path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError('Path escapes repository')
    return resolved


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def check_idx(path,expected_count,images):
    with gzip.open(path,'rb') as stream:
        header=stream.read(16 if images else 8)
        if len(header)!=(16 if images else 8):raise ValueError('Truncated IDX header')
        fields=struct.unpack('>IIII' if images else '>II',header)
        if fields[0]!=(2051 if images else 2049) or fields[1]!=expected_count:
            raise ValueError('Unexpected IDX magic or sample count')
        if images and fields[2:]!=(28,28):raise ValueError('Unexpected image dimensions')
        size=0;minimum=255;maximum=0
        for block in iter(lambda:stream.read(1024*1024),b''):
            size+=len(block);minimum=min(minimum,min(block));maximum=max(maximum,max(block))
        if size!=expected_count*(784 if images else 1):raise ValueError('IDX payload length mismatch')
        if not images and not (0<=minimum<=maximum<=9):raise ValueError('MNIST label outside 0..9')
    return dict(samples=expected_count,dimensions=[28,28] if images else [],payload_bytes=size,
                minimum=minimum,maximum=maximum,gzip_crc_checked=True)


def checkpoint_archive(path):
    # Inspect container metadata/storage bytes without unpickling any object.
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise ValueError('Checkpoint ZIP CRC failure')
        entries=[]
        for info in archive.infolist():
            payload=archive.read(info)
            entries.append(dict(name=info.filename,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
    return dict(format='torch ZIP, no pickle execution',entries=entries,weights_loaded=False)


def source_semantics(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    builder=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='build_csnn')
    assigned={n.id for n in ast.walk(builder) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
    used_connection=[n.lineno for n in ast.walk(builder) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id=='connection']
    return dict(connection_name_bound_inside_builder='connection' in assigned,
        connection_load_lines=used_connection,
        historical_hook_warning=bool(used_connection and 'connection' not in assigned),
        interpretation='Reference snapshot only: undefined connection in post-run normalization hook is caught before E/I. Initial conn normalization exists. Actual run source SHA not attested.')


def inspect():
    manifest=json.loads(local_path(ARCHIVE+'/MANIFEST.json').read_text(encoding='utf-8-sig'))
    source=json.loads(local_path(ARCHIVE+'/reference_source/PROVENANCE.json').read_text(encoding='utf-8-sig'))
    checks=[]
    for entry in manifest['files']:
        path=local_path(entry['repository_path'])
        actual=digest(path) if path.is_file() else None
        checks.append(dict(path=entry['repository_path'],expected_sha256=entry['sha256'],actual_sha256=actual,
                           matches=actual==entry['sha256'] and path.stat().st_size==entry['bytes']))
    for entry in source['files']:
        name=ARCHIVE+'/reference_source/'+entry['path'];path=local_path(name)
        actual=digest(path) if path.is_file() else None
        checks.append(dict(path=name,expected_sha256=entry['sha256'],actual_sha256=actual,
                           matches=actual==entry['sha256'] and path.stat().st_size==entry['bytes']))
    result=dict(schema='mnist-recovery-preflight-v1',archive_hash_checks=checks,
        archive_integrity_valid=all(c['matches'] for c in checks),training_executed=False,
        network_executed=False,board_access=False,native_simulator_executed=False,
        historical_accuracy=manifest['reported_accuracy'],historical_accuracy_reproduced=False,
        source_attested_for_run=False,readout_saved=False,scaler_saved=False)
    if not result['archive_integrity_valid']:
        result['status']='archive_integrity_failed';return result
    datasets={}
    for split,count in (('train',60000),('t10k',10000)):
        for kind,suffix in (('images','idx3'),('labels','idx1')):
            name=f'{ARCHIVE}/data/MNIST/raw/{split}-{kind}-{suffix}-ubyte.gz'
            datasets[name]=check_idx(local_path(name),count,kind=='images')
    result['datasets']=datasets
    result['checkpoint_containers']={name:checkpoint_archive(local_path('weights/mnist_baseline_20261019/'+name))
                                     for name in ('model.pt','parent_model_after_train.pt','label_map.pt')}
    result['source_semantics']=source_semantics(local_path(ARCHIVE+'/reference_source/csnn_mnist_net.py'))
    result['readout_recovery_spec']=dict(features=36864,classes=10,train_samples=30000,test_samples=10000,
        feature_pipeline='spike counts -> log1p -> train-only z-score (sample std, floor1e-6)',
        historical_linear=dict(epochs=25,batch_size=512,optimizer='SGD',lr=.001,weight_decay=.0001),
        count_collection=dict(batch_size=128,timesteps=100,rate_scale=.0015,poisson_base_seed=123,deterministic=True),
        stored_raw_float32_features_bytes=40000*36864*4,
        required_new_artifacts=['readout state_dict','train-only mean/std','train/test sample IDs',
                                'encoder and feature-runtime hashes','RNG seeds and states','dependency lock','predictions and labels'])
    versions={}
    for name in ('torch','torchvision','bindsnet','numpy','tqdm','scipy'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    result['current_python_environment']=dict(python=sys.version,distributions=versions,
        imports_verified=False,meaning='Metadata only; not proof of compatible imports or archived run versions')
    result['status']='archive_verified_recovery_not_ready'
    result['blockers']=['Trained Linear head and z-score statistics missing',
        'Archived execution dependencies not pinned; current source environment.yml was absent from reference commit',
        'Feature-extractor semantics must be frozen: post-window hook has undefined connection and swallows exceptions',
        'Freeze/verify actual Conv1 weights during counts: zero STDP learning rates alone do not attest all update/normalization behavior',
        'No validated MemriBoard MNIST collector/mapper or native SIM equivalence yet']
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect',action='store_true')
    parser.add_argument('--output',help='New JSON path relative to repository')
    args=parser.parse_args(argv)
    if not args.inspect:parser.print_help();return 0
    # Local dependency metadata only; do not import model libraries.
    for name in ('.deps','.deps-cifar'):
        path=local_path(name)
        if path.is_dir():sys.path.insert(0,str(path))
    result=inspect()
    text=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if args.output:
        path=local_path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as stream:stream.write(text)
        print(json.dumps(dict(output=args.output,status=result['status'])))
    else:print(text)
    return 0 if result['archive_integrity_valid'] else 2


if __name__=='__main__':raise SystemExit(main())
