"""Online pulse-quantized STDP через native MemriBoard Simulator.

--run-simulator --smoke: только block0/image0,100ticks и pre/post frozenprobe.
Без --smoke: 16последовательных блоков ×10000train, frozen evaluation,
новый readout и10000test. Плату этот entrypoint не открывает.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import logging
from pathlib import Path
import random
import shutil
import sys
import time
import uuid
from typing import cast

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from numpy.typing import NDArray
import torch
from experiments.mnist_online_stdp64_20261021.logic import DigitalStep,initialization
from experiments.mnist_online_stdp64_20261021.collector import Collector,INIT_CALLS,INIT_PULSES,DEADBAND
from experiment_common import mnist_inputs as inputs,lif,readout
from experiment_common.schema import MemmapFactory
from experiment_common import runtime as calibration
from experiment_common.durable import Writer,atomic_json,sha
from experiment_common.runtime import use_local_dependencies,conductance_mapping,query_cost
from experiment_hardware import launch_provenance
from read_all import now

BASE=ROOT/'results/mnist_online_stdp64_20261021'
SPEC: dict[str,object]=dict(schema='mnist-online-native-timestep-stdp-v1',source='native_memriboard_simulator',
    new_stdp_training=True,pretrained_weights_used=False,physical_hardware=False,filters=64,block_filters=4,
    stages=16,train_images=10000,test_images=10000,smoke_images=1,smoke_stages=1,timesteps=100,
    kernel=5,stride=1,spatial=[24,24],poisson_rate=.0015,poisson_seed=123,initial_weight_seed=42,sim_seed=20261019,
    initial_weight_norm=78.4,first_zero_step_clip=[0,1],lif_tau=150.,lif_threshold=.38,lif_refrac=2,
    trace_tau=20.,trace_additive=False,nu_pre=.001,nu_post=.0001,weight_dependent_nominal_clip=[0,1],
    update_after_each_timestep=True,shared_spatial_reduction=576,mvm_previous_input=True,stdp_current_input=True,
    conductance_deadband_fraction=DEADBAND,maximum_learning_pulses_per_coefficient_per_tick=1,
    learning_set_v=1.7,learning_reset_v=1.4,learning_duration_s=.00001,
    mvm_v=.05,current_limits_a=dict(set=.002,reset=.04,mvm=.005),mvm_headroom_v=3.3,
    gain='fixed1/(330*decodedDAC(.05)*span)',gain_refit=False,shadow_ideal_weights=False,
    baseline_legs_written_only_during_initialization=True,readback_repeats=3,
    fresh_legs_per_tick=200,zero_offset_measured_each_query_group=True,exact_mvm_passes=3,
    training_labels_used=False,test_tuning=False,raw_expanded_cap_bytes=256*1024**3,
    readout='archive log1p/train-only sampleSD floor1e-6/LinearSGD25epochs seed42',
    private_runtime_status='private_runtime_not_independently_reverified')






def inventory(data: dict[str,NDArray[np.uint8]]) -> tuple[list[int],int,dict[str,str]]:
    seen={0};training_calls=0;hashes: dict[str,str]={}
    for split,pixels in data.items():
        digest=hashlib.sha256()
        for start,masks in inputs.batches(pixels,8):
            seen.update(map(int,np.unique(masks)));digest.update(masks.tobytes())
            if split=='train':
                for image in range(masks.shape[1]):
                    training_calls+=query_cost([0])
                    for tick in range(99):training_calls+=query_cost(list(map(int,np.unique(masks[tick,image]))))
            if start%256==0:print('Input-only inventory %s %d/%d'%(split,start+len(masks[0]),len(pixels)),flush=True)
        hashes[split]=digest.hexdigest()
    return sorted(seen),training_calls,hashes


def online(runner: Collector,pixels: NDArray[np.uint8],stage: int,out: Path) -> None:
    digital=DigitalStep();spike_totals=[]
    for image in range(len(pixels)):
        spikes=inputs.encode(pixels[image:image+1]);masks=inputs.patch_masks(spikes);digital.reset()
        image_spikes=0
        for tick in range(100):
            conductance=runner.read_array(stage=stage,image=image,tick=tick,phase='online_pre_tick')
            raw=(conductance[:,:,0]-conductance[:,:,1])/runner.span
            previous=np.zeros((24,24),dtype=np.uint32) if tick==0 else masks[tick-1,0]
            keys=list(map(int,np.unique(previous)))
            measured=runner.measure(keys,stage,'online',image=image,tick=tick)
            # CPU раскладывает ОДИН измеренный отклик каждого mask по всем
            # повторным spatial позициям. Не суммирует идеальныеW и входы.
            current=torch.from_numpy(measured[np.searchsorted(keys,previous)].transpose(2,0,1).copy()).unsqueeze(0)
            post,target=digital.step(current,spikes[tick],torch.from_numpy(raw.astype(np.float32)))
            image_spikes+=int(post.sum())
            runner.emit('digital_tick',stage=stage,image=image,tick=tick,raw_nominal=raw.tolist(),
                clipped_nominal=np.clip(raw,0,1).tolist(),target_nominal=target.tolist(),
                post_spike_indices=torch.nonzero(post.flatten(),as_tuple=False).flatten().tolist(),
                pre_spike_indices=torch.nonzero(spikes[tick].flatten(),as_tuple=False).flatten().tolist())
            runner.update(conductance,target.numpy(),stage,image,tick)
        spike_totals.append(image_spikes)
        runner.writer.checkpoint(stage=stage,image=image,impacts=runner.calls,pulses=runner.pulses)
        if image%8==0 or image+1==len(pixels):runner.progress('Online STDP block%d image%d/%d calls=%d pulses=%d'%(stage,image+1,len(pixels),runner.calls,runner.pulses))
    atomic_json(out/('training_%02d.json'%stage),dict(images=len(pixels),spikes_per_image=spike_totals,cumulative_learning_stats=runner.learning_stats))


def run(smoke: bool,argv: list[str]) -> dict[str,object]:
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    config,hashes=calibration.preflight('simulator');low,span=conductance_mapping()
    n=1 if smoke else 10000;stages=1 if smoke else 16
    data={'train':inputs.images('train',n)}
    if not smoke:data['test']=inputs.images('test',10000)
    keys,training_mvm_calls,encoding_hashes=inventory(data)
    probe_masks=inputs.patch_masks(inputs.encode(data['train'][:1]))
    probe_keys=sorted(set(map(int,np.unique(probe_masks)))|{0})
    # 600freshR + максимум100*(guard+write+3postR)=1100calls/tick.
    # Инициализация считается один раз на блок, не на каждую картинку.
    cap=stages*(INIT_CALLS+n*100*1100+training_mvm_calls+2*query_cost(probe_keys)+(0 if smoke else query_cost(keys)))
    pulse_cap=stages*(INIT_PULSES+n*100*100)
    sid=str(uuid.uuid4());out=BASE/sid;out.mkdir(parents=True)
    for path in Path(__file__).parent.glob('*.py'):hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    manifest: dict[str,object]=dict(session_id=sid,status='failed',scope='block0-one-image-100ticks' if smoke else 'online64-train10000-test10000',
        source='simulator',physical_hardware=False,protocol=SPEC,source_sha256=hashes,invocation=launch_provenance(argv),
        started_utc=now(),train_images=n,stages=stages,maximum_impacts=cap,maximum_pulses=pulse_cap,
        encoding_sha256=encoding_hashes,low_g_s=low,span_g_s=span,phases=[])
    phases: list[dict[str,object]]=[];runner: Collector|None=None;writer: Writer|None=None
    opened=False;connector=None;status='failed';error: str|None=None
    print('Online SIM session '+sid,flush=True)
    def mark(name: str,stage: int,start: tuple[float,int,int]) -> None:
        if runner is None:raise ValueError('Missing collector')
        record=dict(name=name,stage=stage,seconds=time.monotonic()-start[0],calls=runner.calls-start[1],pulses=runner.pulses-start[2])
        phases.append(record);runner.emit('phase_complete',**record);atomic_json(out/'phase_progress.json',phases)
    def snapshot() -> tuple[float,int,int]:
        if runner is None:raise ValueError('Missing collector')
        return time.monotonic(),runner.calls,runner.pulses
    try:
        if shutil.disk_usage(ROOT).free<8*1024**3:raise ValueError('Need8GiB free evidence space')
        raw,clipped=initialization()
        np.save(out/'fresh_initial_weights.npy',raw.numpy(),allow_pickle=False)
        np.save(out/'initial_postclip_weights.npy',clipped.numpy(),allow_pickle=False)
        atomic_json(out/'protocol.json',SPEC);atomic_json(out/'start_manifest.json',manifest)
        np.save(out/'mask_keys.npy',np.asarray(keys,dtype=np.uint32),allow_pickle=False)
        open_counts=cast(MemmapFactory,np.lib.format.open_memmap)
        train=open_counts(out/'train_counts.npy',mode='w+',dtype=np.uint8,shape=(n,2304 if smoke else 36864))
        test=None if smoke else open_counts(out/'test_counts.npy',mode='w+',dtype=np.uint8,shape=(10000,36864))
        train[:]=0
        if test is not None:test[:]=0
        shutil.copyfile(ROOT/calibration.INI,out/'board.ini');shutil.copyfile(ROOT/calibration.SIM_INI,out/'simulator_settings.ini')
        with (out/'lifecycle_output.txt').open('w',encoding='utf-8',newline='\n') as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                use_local_dependencies()
                from manager.board import Connector
                from manager.service import converters
                from simulator.src import create_crossbar_array
                random.seed(20261019);np.random.seed(20261019)
                create_crossbar_array(str(out/'array'),32,8);shutil.copyfile(out/'array.cb',out/'array_initial.cb')
                writer=Writer(out/'raw',limit=8*1024**2,maximum_expanded_bytes=256*1024**3,minimum_free_bytes=2*1024**3)
                connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                    board_type=config.get('board','board_type'),task_format='SI',config=config,crossbar_serial=str(out/'array'))
                visible=sys.__stdout__
                if visible is None:raise ValueError('Progress stdout missing')
                def progress(message: str) -> None:visible.write(message+'\n');visible.flush()
                runner=Collector(connector,converters,writer,cap,pulse_cap,span,progress)
                manifest.update(gain=runner.gain,decoded_input_v=runner.decoded_input_v)
                opened=True
                if not connector.open_port():raise ValueError('Public Connector.open_port failed')
                runner.emit('port_open')
                for stage in range(stages):
                    begin=snapshot();runner.initialize(clipped[stage*4:stage*4+4].reshape(4,25).numpy().astype(np.float64),low,stage,out);mark('initialization',stage,begin)
                    begin=snapshot();before=runner.measure(probe_keys,stage,'preprobe')
                    pre_counts=lif.counts_for_masks(probe_masks,np.asarray(probe_keys,dtype=np.uint32),before,.38)
                    np.save(out/('preprobe_counts_%02d.npy'%stage),pre_counts,allow_pickle=False);mark('preprobe',stage,begin)
                    begin=snapshot();online(runner,data['train'],stage,out);mark('online_stdp',stage,begin)
                    begin=snapshot();after=runner.measure(probe_keys,stage,'postprobe')
                    post_counts=lif.counts_for_masks(probe_masks,np.asarray(probe_keys,dtype=np.uint32),after,.38)
                    np.save(out/('postprobe_counts_%02d.npy'%stage),post_counts,allow_pickle=False);mark('postprobe',stage,begin)
                    if smoke:train[:]=post_counts.reshape(1,2304)
                    else:
                        begin=snapshot();frozen=runner.measure(keys,stage,'frozen_evaluation')
                        np.save(out/('frozen_lut_%02d.npy'%stage),frozen,allow_pickle=False)
                        mark('frozen_lut',stage,begin)
                        begin=snapshot()
                        for split,target in (('train',train),('test',test)):
                            if target is None:raise ValueError('Missing full feature storage')
                            lif.extract(data[split],np.asarray(keys,dtype=np.uint32),frozen,.38,
                                target[:,stage*2304:(stage+1)*2304],8,progress,'block%d-%s'%(stage,split));target.flush()
                        mark('frozen_cpu_features',stage,begin)
                    shutil.copyfile(out/'array.cb',out/('array_stage_%02d.cb'%stage))
                    runner.writer.checkpoint(stage=stage,completed=True,impacts=runner.calls,pulses=runner.pulses)
                if not connector.close_port():raise ValueError('Public Connector.close_port failed')
                opened=False;runner.emit('port_close')
        writer.finish();writer=None;train.flush()
        if not smoke:
            if test is None:raise ValueError('Missing test counts')
            head=readout.fit(train,out,lambda text:print(text,flush=True))
            accuracy=readout.evaluate(head,test,out);atomic_json(out/'accuracy.json',accuracy);manifest['accuracy']=accuracy
        status='completed'
    except (Exception,KeyboardInterrupt) as exc:error=type(exc).__name__+': '+str(exc)
    finally:
        if opened and connector is not None:
            try:
                if not connector.close_port():raise ValueError('Public close failed')
            except Exception as exc:status='failed';manifest['cleanup_error']=repr(exc)
        if writer is not None:
            try:writer.finish()
            except Exception as exc:status='failed';manifest['raw_finish_error']=repr(exc)
        manifest.update(status=status,error=error,finished_utc=now(),phases=phases)
        if runner is not None:manifest.update(impacts=runner.calls,pulses=runner.pulses,decisions=runner.decisions,learning_stats=runner.learning_stats)
        manifest['result_sha256']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
        atomic_json(out/'manifest.json',manifest)
    return dict(session=out.relative_to(ROOT).as_posix(),status=status,error=error,accuracy=manifest.get('accuracy'))


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-simulator',action='store_true');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not args.run_simulator:
        calibration.preflight('simulator');low,span=conductance_mapping()
        n=1 if args.smoke else 10000;blocks=1 if args.smoke else 16
        # Быстрый верхний algebraic estimate, без кодирования/labels/Connector.
        # Actual передopen вычислит меньший exactcap по бинарным входам.
        print(json.dumps(dict(status='preflight_only',protocol=SPEC,low_g_s=low,span_g_s=span,
            training_fresh_R_calls=blocks*n*100*600,maximum_learning_pulses=blocks*n*100*100,
            initialization_call_cap=blocks*INIT_CALLS,initialization_pulse_cap=blocks*INIT_PULSES,
            runtime_estimate='Measure smoke phase timings; no full-run ETA assumed'),ensure_ascii=False))
        return 0
    result=run(args.smoke,sys.argv);print(json.dumps(result,ensure_ascii=False))
    return 0 if result['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
