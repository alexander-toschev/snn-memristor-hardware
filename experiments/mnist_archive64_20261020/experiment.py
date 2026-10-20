"""Архивная MNIST64: сохранённые STDP-веса → native SIM MVM → CPU LIF/readout.

Шаги читаются последовательно в execute(): без нового STDP, без физической
платы и без идеальной свёртки вместо MemriBoard. --pilot явно128/100; основной
режим30000/10000. Hardware-параметра у этого entrypoint нет.
"""
from __future__ import annotations
import argparse
import contextlib
import importlib.metadata
import json
import logging
from pathlib import Path
import random
import shutil
import sys
import uuid
from typing import cast


ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
# Сначала именно NumPy/Torch выбранного venv. Legacy path helper далее добавит
# локальные зависимости MemriBoard, но не должен подменить уже выбранный Torch.
import numpy as np
import torch
from experiments.mnist_archive64_20261020 import inputs,lif,native,readout
from experiments.mnist_archive64_20261020.schema import Accuracy,RunResult,MemmapFactory,Manifest
from experiments.mnist_feature_pilot_20261019.experiment import load_checkpoint,CHECKPOINT,CHECKPOINT_SHA
from experiments.stable_cell_calibration_20261020 import experiment as calibration
from experiments.cifar_resumable_20261019.durable import Writer,atomic_json,sha
from experiments.unn_rev3_20261014.paths import use_local_dependencies
from experiment_hardware import launch_provenance
from read_all import now

BASE=ROOT/'results/mnist_archive64_20261020'
CHARACTERIZATION=Path(__file__).parent/'calibration'
SPEC: dict[str,object]=dict(schema='mnist-archive64-native-fullmask-lut-v1',checkpoint_sha256=CHECKPOINT_SHA,
    source='native_memriboard_simulator',physical_hardware=False,new_stdp_training=False,
    archive_filters=64,taps=25,spatial=[24,24],timesteps=100,poisson_rate=.0015,poisson_seed=123,
    poisson_content_deterministic=True,encoder_boost=1.,synchronous_input_delay=1,
    lif_tau_ms=150.,lif_threshold=.38,lif_refrac_ms=2,rest_reset=0.,dt_ms=1.,
    mvm_voltage_v=.05,maximum_active_bl=25,digital_microvector_addition=False,
    differential_pairs=[[0,1],[2,3],[4,5],[6,7]],stages=16,
    mapping='Gminus=L;Gplus=L+W*span/maxWfilter; L/span prior label-free native characterization',
    reset_ramp_v=list(native.LEVELS),reset_ramp_duration_s=.000125,
    feedback_steps_per_leg=96,feedback_set_v=1.7,feedback_reset_v=1.4,feedback_duration_s=.00001,
    target_tolerance_fraction_span=.02,maximum_programming_pulses=native.MAX_PULSES,
    maximum_programming_calls=native.PROGRAM_CALL_CAP,post_read_calls=native.POST_READ_CAP,
    read_repeats=3,set_limit_a=.002,reset_limit_a=.04,mvm_limit_a=.005,mvm_predicted_output_limit_v=3.3,
    cache='only exact native ADC/V three passes; freshR guards each request; no interpolation; per-stage invalidation',
    alpha='positive per-filter dot(originalW,nativeSingletonDifference)/sum(nativeSingletonDifference^2); no labels/intercept',
    readout='log1p counts; train-only sample-zscore; Linear+bias; SGD25 lr.001 wd.0001 batch512 seed42; final epoch',
    conventional_and_strict_accuracy=True,test_tuning=False,private_runtime_status='private_runtime_not_independently_reverified')


def mapping() -> tuple[float,float,dict[str,object]]:
    # Только ранее измеренная характеристика SIM без изображений и классов.
    m=json.loads((CHARACTERIZATION/'manifest.json').read_text(encoding='utf-8'))
    path=CHARACTERIZATION/'observations.json'
    if m['status']!='completed' or m['kind']!='characterize' or sha(path)!=m['result_sha256']['observations.json']:
        raise ValueError('Completed immutable native characterization required')
    measured=json.loads(path.read_text(encoding='utf-8'))['mapping']
    low=float(measured['low_g_s']);span=float(measured['span_g_s'])
    if low<=0 or span<=0:raise ValueError('Invalid conductance mapping')
    return low,span,dict(session=m['session_id'],local_copy=CHARACTERIZATION.relative_to(ROOT).as_posix(),manifest_sha256=sha(CHARACTERIZATION/'manifest.json'),
                         observations_sha256=sha(path),mapping=measured)


def execute(pilot: bool,batch_size: int,argv: list[str]) -> RunResult:
    if not 1<=batch_size<=16:raise ValueError('Small exact encoder batch1..16 required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    config,hashes=calibration.preflight('simulator')
    checkpoint=load_checkpoint();weights=checkpoint['W'].numpy().astype(np.float64).reshape(64,25)
    threshold=float(checkpoint['v_thresh'])
    if abs(threshold-.38)>1e-7:raise ValueError('Archived LIF threshold differs')
    low,span,binding=mapping()
    for folder in (Path(__file__).parent,ROOT/'experiments/mnist_baseline_20261019/reference_source'):
        for path in folder.glob('*.py'):hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    hashes[CHECKPOINT.relative_to(ROOT).as_posix()]=sha(CHECKPOINT)
    for path in CHARACTERIZATION.glob('*.json'):hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in ('experiments/mnist_feature_pilot_20261019/experiment.py','experiments/mnist_accuracy_20261020/inputs.py',
                 'tools/prepare_mnist_conductance_fixture.py','experiment_hardware.py'):
        hashes[name]=sha(ROOT/name)
    ntrain,ntest=(128,100) if pilot else (30000,10000)
    sid=str(uuid.uuid4());out=BASE/sid;out.mkdir(parents=True)
    def progress(message: str) -> None:
        print(message,flush=True)
    manifest: Manifest=dict(session_id=sid,source='simulator',scope='pilot128-100' if pilot else 'archive30000-10000',
        status='failed',started_utc=now(),protocol=SPEC,source_sha256=hashes,invocation=launch_provenance(argv),
        checkpoint_sha256=CHECKPOINT_SHA,characterization=binding,seed=20261019,batch_size=batch_size,
        train_images=ntrain,test_images=ntest,physical_hardware=False,new_stdp_training=False,
        packages={name:importlib.metadata.version(name) for name in ('torch','numpy','bindsnet')})
    runner=None;writer=None;connector=None;opened=False
    status='failed';error: str|None=None;metrics: Accuracy|None=None
    progress('MNIST archive64 native SIM session '+sid)
    try:
        if shutil.disk_usage(ROOT).free<8*1024**3:raise ValueError('Need8GiB free disk for evidence/counts')
        atomic_json(out/'protocol.json',SPEC);np.save(out/'original_weights.npy',checkpoint['W'].numpy(),allow_pickle=False)
        # 1. CPU: официальные изображения, архивный Poisson и бинарные patches.
        # Test pixels участвуют только в перечне входов LUT; labels не читаются.
        data={s:inputs.images(s,n) for s,n in (('train',ntrain),('test',ntest))}
        keys,encoding_hashes=inputs.inventory(data,batch_size,progress)
        native_masks=sorted(set(keys)|{0}|{1<<b for b in range(25)})
        np.save(out/'mask_keys.npy',np.asarray(keys,dtype=np.uint32),allow_pickle=False)
        lut_call_cap=16*8*3*sum(1+m.bit_count() for m in native_masks)
        cap=native.PROGRAM_CALL_CAP+native.POST_READ_CAP+lut_call_cap
        manifest.update(Manifest(encoding_sha256=encoding_hashes,unique_masks=len(keys),native_masks=len(native_masks),
            max_popcount=max(m.bit_count() for m in keys),native_lut_call_cap=lut_call_cap,maximum_impacts=cap,
            split_masks=0,digital_microvector_addition=False,cached_logical_filter_mvm=0))
        atomic_json(out/'input_inventory.json',dict(keys=keys,native_masks=native_masks,encoding_sha256=encoding_hashes,
            max_popcount=manifest['max_popcount'],labels_used=False,image_sources=inputs.IMAGE_SOURCES))
        # Узкая типовая граница untyped NumPy2.0; shape/dtype проверены ниже.
        open_counts=cast(MemmapFactory,np.lib.format.open_memmap)
        train=open_counts(out/'train_counts.npy',mode='w+',dtype=np.uint8,shape=(ntrain,36864))
        test=open_counts(out/'test_counts.npy',mode='w+',dtype=np.uint8,shape=(ntest,36864))
        if train.dtype!=np.uint8 or train.shape!=(ntrain,36864) or test.dtype!=np.uint8 or test.shape!=(ntest,36864):
            raise ValueError('Unexpected count-map dtype/shape')
        train[:]=0;test[:]=0
        measured=np.full((64,25,2),np.nan);logical=0;stage_luts=[]
        shutil.copyfile(ROOT/calibration.INI,out/'board.ini');shutil.copyfile(ROOT/calibration.SIM_INI,out/'simulator_settings.ini')
        with (out/'lifecycle_output.txt').open('w',encoding='utf-8',newline='\n') as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                use_local_dependencies()
                from manager.board import Connector
                from manager.service import converters
                from simulator.src import create_crossbar_array
                random.seed(20261019);np.random.seed(20261019)
                create_crossbar_array(str(out/'array'),32,8)
                shutil.copyfile(out/'array.cb',out/'array_initial.cb')
                writer=Writer(out/'raw',limit=8*1024**2,maximum_expanded_bytes=8*1024**3,minimum_free_bytes=2*1024**3)
                connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                    board_type=config.get('board','board_type'),task_format='SI',config=config,crossbar_serial=str(out/'array'))
                # Progress пишет в сохранённый внешний stdout, native noise идёт
                # отдельно в lifecycle/raw. Пользователь видит этапы и фильтры.
                visible=sys.__stdout__
                if visible is None:raise ValueError('Progress stdout unavailable')
                def live(message: str) -> None:
                    visible.write(message+'\n');visible.flush()
                runner=native.Collector(connector,converters,writer,cap,live)
                manifest['initial_array_sha256']=sha(out/'array_initial.cb');atomic_json(out/'start_manifest.json',manifest)
                opened=True
                result=connector.open_port();runner.emit('port_open',result=safe_native(result))
                if not result:raise ValueError('Public Connector.open_port failed')
                for stage in range(16):
                    # 2. SIM: записываем4 архивных фильтра в8 дифференциальных WL.
                    block=weights[stage*4:stage*4+4]
                    measured[stage*4:stage*4+4]=runner.program(block,low,span,stage,out)
                    # 3. SIM: полный mode10 для каждого mask и повторные проверки.
                    lut=runner.measure_lut(stage,keys,native_masks,block,out)
                    stage_luts.append(lut)
                    runner.emit('stage_complete',stage=stage)
                    writer.checkpoint(stage=stage,impacts=runner.calls,pulses=runner.pulses)
                result=connector.close_port();runner.emit('port_close',result=safe_native(result));opened=False
                if not result:raise ValueError('Public Connector.close_port failed')
        writer.finish();writer=None
        # 4. CPU: объединяем64 уже измеренных фильтра после16 отдельных
        # программирований. Не утверждаем, что64filters одновременно помещались
        # на32x8. LIF нейроны независимы; один проход равен16 группам по4,
        # но Poisson кодируется один раз. Каждая LUTячейка получена nativeMVM.
        combined=np.concatenate(stage_luts,axis=1)
        np.save(out/'native_currents_lut.npy',combined,allow_pickle=False)
        for split,target in (('train',train),('test',test)):
            stats=lif.extract(data[split],np.asarray(keys,dtype=np.uint32),combined,threshold,target,batch_size,progress,split)
            logical+=stats['cached_logical_filter_mvm'];target.flush()
        np.save(out/'measured_conductance_s.npy',measured,allow_pickle=False)
        # Readback-weight metric uses .3V/nativeR mapping only for diagnostics;
        # actual forward already used .05V native MVM and singleton alpha.
        effective=(measured[:,:,0]-measured[:,:,1])*weights.max(axis=1)[:,None]/span
        np.save(out/'readback_weights.npy',effective,allow_pickle=False)
        manifest.update(Manifest(cached_logical_filter_mvm=logical,readback_weight_rmse=float(np.sqrt(np.mean((effective-weights)**2))),
                        train_spikes=int(train.sum()),test_spikes=int(test.sum())))
        # 5. CPU: train-only scaler/head. Test labels открываются толькоevaluate.
        head=readout.fit(train,out,progress)
        metrics=readout.evaluate(head,test,out);atomic_json(out/'accuracy.json',metrics)
        status='completed';manifest.update(Manifest(status=status,accuracy=metrics))
        progress('Accuracy: argmax=%.4f strict=%.4f; no physical hardware'%(metrics['conventional_argmax_accuracy'],metrics['strict_accuracy']))
    except (Exception,KeyboardInterrupt) as exc:
        error=type(exc).__name__+': '+str(exc);status='failed'
        manifest.update(Manifest(status=status,error=error))
    finally:
        if opened and connector is not None:
            try:
                closed=connector.close_port()
                if not closed:raise ValueError('Public close failed')
            except Exception as exc:
                status='failed';manifest.update(Manifest(status=status,cleanup_error=repr(exc)))
        if runner is not None:manifest.update(Manifest(impacts=runner.calls,pulses=runner.pulses,programming_decisions=runner.decisions))
        if writer is not None:
            try:writer.finish()
            except Exception as exc:
                status='failed';manifest.update(Manifest(status=status,raw_finish_error=repr(exc)))
        manifest['finished_utc']=now()
        manifest['result_sha256']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
        atomic_json(out/'manifest.json',manifest)
    return dict(session=out.relative_to(ROOT).as_posix(),status=status,error=error,accuracy=metrics)


def safe_native(value: object) -> object:
    from experiments.stable_cell_calibration_20261020.core import safe
    return safe(value)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-simulator',action='store_true');parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    if not args.run_simulator:
        # Preflight не открывает Connector и не выполняет сеть.
        config,_=calibration.preflight('simulator');load_checkpoint();low,span,binding=mapping()
        print(json.dumps(dict(status='ready',mode='native SIM only',protocol=SPEC,mapping=binding,batch_size=args.batch_size),ensure_ascii=False))
        return 0
    result=execute(args.pilot,args.batch_size,sys.argv);print(json.dumps(result,ensure_ascii=False))
    return 0 if result['status']=='completed' else 2

if __name__=='__main__':raise SystemExit(main())
