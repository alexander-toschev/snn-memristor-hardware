"""Local readout adaptation over frozen SIM MVM and CPU LIF; no Connector instance.

Training images discover input masks without labels. All their responses are
reconstructed from already measured partial operators; no ideal W@x fallback.
Train-only validation selects epochs; final checkpoint precedes test labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from typing import cast
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiment_common.runtime import use_local_dependencies
use_local_dependencies(prefer_environment=True)
import numpy as np
from numpy.typing import NDArray
import torch
from experiment_common.durable import atomic_json, sha
from experiment_common.exploratory_transfer import measured_lut
from experiment_common.lif import extract
from experiment_common.mnist_inputs import images, labels, inventory
from experiment_common.mnist_data import SOURCES
from experiment_common.readout import evaluate
from experiment_common.readout_adaptation import adapt, verify_checkpoint, verified_file
from experiment_common.schema import MemmapFactory
from experiment_common.selective_metrology import object_dict, number
from experiment_common.tiled_transfer import profiles
from manager.service import converters

BASE = ROOT/'results/mnist_readout_adaptation_20261115'
SOURCE_BASE = ROOT/'results/mnist_expanded_exploratory_20261114'
ACQUISITION = '605c6b02-4865-44a3-8c85-14d7f22d03e4'
BASELINE = '9fc7ac47-e0ed-413f-b3a9-4c55a99499d4'
ASSETS = ROOT/'experiments/mnist_tiled_transfer_20261022/assets'
PROTOCOL: dict[str, object] = dict(
    acquisition=ACQUISITION,baseline_evaluation=BASELINE,baseline_method='bracket_centered',
    train_images=10000,fit_images=8000,validation_images=2000,test_images=10000,seed=42,
    split='stratified largest remainder; deterministic seed 42',lif_threshold=.38,
    optimizer='SGD',learning_rate=.001,weight_decay=.0001,batch_size=512,max_epochs=25,
    selection='highest validation STRICT accuracy; earliest epoch on ties',
    refit='fresh scaler and head on all 10000 train for selected epochs; seed 42',
    scaler='log1p; sample SD correction=1, floor=1e-6; fit partition only during selection',
    test='one evaluation after final checkpoint seal; test previously observed, exploratory',
    hardware_calls=0,new_stdp_training=False,physical_writes=0,cpu_threads=4,gpu=False,
    ideal_weight_fallback=False,test_based_parameter_selection=False)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def terminal_manifest(sid: str) -> dict[str, object]:
    folder = SOURCE_BASE/sid
    receipt = object_dict(json.loads((SOURCE_BASE/'archives'/f'{sid}.json').read_text(encoding='utf-8')))
    expected = object_dict(object_dict(receipt['files'])['manifest.json'])['sha256']
    if sha(folder/'manifest.json') != expected:
        raise ValueError('Source terminal manifest differs from verified archive receipt')
    manifest = object_dict(json.loads((folder/'manifest.json').read_text(encoding='utf-8')))
    if manifest['status'] != 'completed' or manifest['source'] != 'simulator':
        raise ValueError('Completed SIM evidence required')
    return manifest


def run() -> tuple[dict[str, object], int]:
    torch.set_num_threads(4)
    started = time.monotonic()
    sid = str(uuid.uuid4())
    out = BASE/sid
    out.mkdir(parents=True)
    source = SOURCE_BASE/ACQUISITION
    baseline = SOURCE_BASE/BASELINE
    files = [Path(__file__), ROOT/'experiment_common/readout_adaptation.py',
             ROOT/'experiment_common/readout.py',ROOT/'experiment_common/lif.py',
             ROOT/'experiment_common/mnist_inputs.py',ROOT/'experiment_common/poisson.py',
             ROOT/'experiment_common/exploratory_transfer.py',ROOT/'experiment_common/tiled_transfer.py',
             ROOT/'vendor/MemriBoard/manager/service/converters.py']
    manifest: dict[str, object] = dict(session_id=sid,status='failed',source='simulator_replay_cpu',
        started_utc=now(),protocol=PROTOCOL,hardware_calls=0,new_stdp_training=False,
        error=None,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        packages={name:importlib.metadata.version(name) for name in ('torch','numpy','bindsnet')},
        python=sys.version,official_dataset_sha256=SOURCES)
    atomic_json(out/'start_manifest.json',manifest)
    atomic_json(out/'protocol.json',PROTOCOL)
    print(f'Local readout session {sid}',flush=True)
    def progress(message: str) -> None:
        atomic_json(out/'progress.json',dict(message=message,updated_utc=now(),elapsed_s=time.monotonic()-started))
        print(message,flush=True)
    result: dict[str, object] = dict(session_id=sid,status='failed',error=None,accuracy=None)
    try:
        acquisition = terminal_manifest(ACQUISITION)
        prior = terminal_manifest(BASELINE)
        if prior['source_session'] != ACQUISITION or prior['baseline_method'] != 'bracket_centered':
            raise ValueError('Baseline source/policy mismatch')
        if prior['source_manifest_sha256'] != sha(source/'manifest.json'):
            raise ValueError('Baseline acquisition checksum mismatch')
        summary = object_dict(json.loads(verified_file(source,'summary.json',acquisition).read_text(encoding='utf-8')))
        if summary['completed_loads'] != summary['planned_loads'] or summary['pilot'] is not False:
            raise ValueError('Full acquisition required')
        # The prior test counts are only reusable with the same measured and
        # digital transformation. Verify actual files, not descriptive labels.
        old_hashes = object_dict(acquisition['source_sha256'])
        prior_hashes = object_dict(prior['source_sha256'])
        critical = ('lif.py','mnist_inputs.py','poisson.py','exploratory_transfer.py','tiled_transfer.py')
        for name in critical:
            relative = 'experiment_common/'+name
            if sha(ROOT/relative) != old_hashes[relative] or sha(ROOT/relative) != prior_hashes[relative]:
                raise ValueError('Frozen digital transformation changed: '+relative)
        converter_path = 'vendor/MemriBoard/manager/service/converters.py'
        if sha(ROOT/converter_path) != old_hashes[converter_path]:
            raise ValueError('Native converter changed')
        for path in ASSETS.iterdir():
            if path.is_file() and sha(path) != old_hashes[path.relative_to(ROOT).as_posix()]:
                raise ValueError('Frozen inference asset changed')
        map_path = verified_file(source,'physical_map.json',acquisition)
        if sha(map_path) != object_dict(acquisition['protocol'])['map_sha256']:
            raise ValueError('Source map hash mismatch')
        selected = profiles(map_path)
        test_counts_path = verified_file(baseline,'test_counts.npy',prior)
        old_lut_path = verified_file(baseline,'inference_lut.npy',prior)
        original_lut_path = verified_file(source,'partial_lut.npy',acquisition)
        if prior['source_lut_sha256'] != sha(original_lut_path):
            raise ValueError('Baseline acquisition LUT checksum mismatch')
        old_keys: NDArray[np.uint32] = np.load(ASSETS/'input_masks.npy',allow_pickle=False).astype(np.uint32)
        old_lut: NDArray[np.float32] = np.load(old_lut_path,allow_pickle=False)
        # No train labels, and no test images, enter mask discovery.
        data = images('train',10000)
        train_keys, encoding_hashes = inventory({'train':data},8,progress)
        keys = np.union1d(old_keys,np.asarray(train_keys,dtype=np.uint32)).astype(np.uint32)
        if np.any(keys >= 1<<25):
            raise ValueError('Input mask outside frozen 5x5 filter')
        payload = object_dict(json.loads((ASSETS/'learned_conductances.json').read_text(encoding='utf-8')))
        lut = measured_lut(source,keys,selected,number(payload['span_g_s']),converters,'bracket_centered')
        if not np.array_equal(lut[np.searchsorted(keys,old_keys)],old_lut):
            raise ValueError('Old test-mask LUT differs after extension')
        if not np.isfinite(lut).all():
            raise ValueError('Nonfinite measured response')
        np.save(out/'input_masks.npy',keys)
        np.save(out/'inference_lut.npy',lut)
        atomic_json(out/'evidence.json',dict(acquisition=ACQUISITION,baseline=BASELINE,
            acquisition_manifest_sha256=sha(source/'manifest.json'),baseline_manifest_sha256=sha(baseline/'manifest.json'),
            test_counts_sha256=sha(test_counts_path),original_lut_sha256=sha(old_lut_path),
            map_sha256=sha(map_path),frozen_asset_sha256={p.name:sha(p) for p in ASSETS.iterdir() if p.is_file()},
            original_key_count=len(old_keys),train_key_count=len(train_keys),extended_key_count=len(keys),
            old_test_lut_bit_identical=True,train_encoding_sha256=encoding_hashes,
            baseline_strict_accuracy=.538,baseline_argmax_accuracy=.5382,labels_used_for_masks=False))
        factory = cast(MemmapFactory,np.lib.format.open_memmap)
        counts = factory(out/'train_counts.npy',mode='w+',dtype=np.uint8,shape=(10000,64*24*24))
        extract(data,keys,lut,.38,counts,8,progress,'train')
        counts.flush()
        del data
        model,epochs = adapt(counts,labels('train',10000),out,progress)
        test_counts: NDArray[np.uint8] = np.load(test_counts_path,mmap_mode='r',allow_pickle=False)
        if test_counts.shape != (10000,64*24*24) or test_counts.dtype != np.uint8:
            raise ValueError('Frozen test count shape/dtype mismatch')
        test_reads = 0
        def test_labels(split: str,n: int) -> NDArray[np.int64]:
            nonlocal test_reads
            if split != 'test' or n != 10000 or test_reads:
                raise ValueError('Exactly one final test-label access allowed')
            verify_checkpoint(out)
            test_reads += 1
            atomic_json(out/'test_access.json',dict(count=test_reads,at_utc=now(),
                checkpoint_seal_sha256=sha(out/'checkpoint_before_test.json'),selection_frozen=True))
            return labels(split,n)
        accuracy = evaluate(model,test_counts,out,label_reader=test_labels)
        verify_checkpoint(out)
        atomic_json(out/'accuracy.json',dict(accuracy))
        result.update(status='completed',accuracy=dict(accuracy),selected_epochs=epochs,
            baseline_strict_accuracy=.538,baseline_argmax_accuracy=.5382,
            strict_delta_percentage_points=100*(accuracy['strict_accuracy']-.538),
            test_label_reads=test_reads,exploratory_previously_seen_test=True,
            new_stdp_training=False,hardware_calls=0,scope='readout adaptation over frozen measured SIM operators and CPU LIF')
    except (Exception,KeyboardInterrupt) as exc:
        result.update(status='failed',error=type(exc).__name__+': '+str(exc))
    result['elapsed_seconds'] = time.monotonic()-started
    atomic_json(out/'summary.json',result)
    (out/'map.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>MNIST readout adaptation</title>'
        '<h1>MNIST readout adaptation</h1><pre>'
        + html.escape(json.dumps(result,ensure_ascii=False,indent=2)) + '</pre>',
        encoding='utf-8')
    manifest.update(status=result['status'],error=result['error'],finished_utc=now(),
        result_sha256={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    atomic_json(out/'manifest.json',manifest)
    return result,0 if result['status']=='completed' else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-local',action='store_true',help='Run the fixed CPU-only adaptation protocol')
    args = parser.parse_args()
    if not args.run_local:
        print(json.dumps(PROTOCOL,ensure_ascii=False,indent=2))
        return 0
    result,code = run()
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
