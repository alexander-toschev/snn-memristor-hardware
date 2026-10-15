"""Read-only, fixed-amplitude eight-output MVM calibration. Public Connector only."""
import argparse
from collections import Counter
import hashlib
import json
import logging
import math
import random
from statistics import mean, stdev
import time
import uuid

from cell_calibration import check_sources, SIM_INI, LOCK
from read_all import ROOT, EXPECTED_BOARD, local_path, native_connector, now, sha256, save_json, json_value
from native_conversions import voltage, readback, mvm_output
from experiment_hardware import launch_provenance

CONFIG = 'configs/calibrated_mvm.json'
PINS={'cell_calibration.py':'eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8',
      'read_all.py':'b30ae2f4dd3cd166dbc5cefb07b558127de2bbdba1f6674c40b1b45f4dd74d5f',
      'native_conversions.py':'2ad2e9c057bb14979b083a6e846e8c417c080a6c5273d848e31f893b7970253e',
      'experiment_hardware.py':'059dc4f827943a9df631631f52e776b57182afc9d36db0c1ad5246427f73548c'}
SPEC = dict(protocol_id='all256-preparation-mvm-lif-v2', nominal_v=.1, repeats=5, read_repeats=3,
    seed=20261016, densities=[2,4,8,16,24], mixed_counts=dict(fit=80,selection=40,test=81),
    synthetic_groups=[list(range(a,a+8)) for a in (0,8,16,24)],
    synthetic_patterns=dict(dispersed=[[i] for i in range(8)]+[[] for _ in range(4)],
        coincident=[[0,1],[],[2,3],[],[4,5],[],[6,7],[],[],[],[],[]]),timesteps=12,
    path_probe_anchors=[8,31],path_probe_columns=list(range(16,31)),
    sparse_split_coverage='balanced density4 partitions cover all32 BL in fit/selection/test',
    lif_scale='1.25/maxabs(no-reset tau.5 reference over both group patterns); basis only',
    maximum_impacts=18576, maximum_programming_pulses=0, modes=[9,10],
    guard=dict(max_active=4, basis_factor=2., basis_sd_multiplier=3., current_model_a=.005,
        gui_model_v=3.3, headroom_fraction=.8, heuristic_not_physical_bound=True),
    calibration=dict(candidates=['identity','diagonal','full'], ridge=[0.,.0001,.01,1.],
        maximum_gain=10., maximum_condition=1000000., minimum_snr=5., maximum_cv=.05,
        relative_error=.05, absolute_error_adc=2., maximum_membrane_error=.05,
        selection_required_sparse_masks=16,test_required_sparse_masks=32),
    source_states_matched=False, trained_weights_programmed=False, trained_bn_applied=False)


def validate(plan):
    if plan != SPEC:
        raise ValueError('Protocol differs from reviewed all256 specification')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def schedule(plan):
    """One vector entry means eight successive WL impacts; skipped entries have none."""
    validate(plan)
    probes=[sorted([anchor,bl]) for anchor in plan['path_probe_anchors'] for bl in plan['path_probe_columns']]
    rng=random.Random(plan['seed']); used={tuple(m) for m in probes}; groups={}
    for role,count in plan['mixed_counts'].items():
        masks=[]
        for density in plan['densities']:
            if density==4:
                # Independent balanced partitions ensure all32 BL occur in each
                # sparse split, before any measured data or candidate selection.
                for _ in range(((count-(role=='test'))//5)//8):
                    while True:
                        columns=list(range(32));rng.shuffle(columns)
                        partition=[tuple(sorted(columns[i:i+4])) for i in range(0,32,4)]
                        if all(mask not in used for mask in partition):break
                    used.update(partition);masks.extend(map(list,partition))
                continue
            for _ in range((count-(role=='test'))//5):
                while True:
                    mask=tuple(sorted(rng.sample(range(32),density)))
                    if mask not in used:break
                used.add(mask);masks.append(list(mask))
        if role=='test':masks.append(list(range(32)))
        groups[role]=masks
    entries=[]
    def add(role,masks):
        # Density ascending; repeats interleave masks within each density.
        for density in sorted({len(m) for m in masks}):
            for repeat in range(5):
                for index,mask in enumerate(masks):
                    if len(mask)==density:entries.append(dict(phase=role,mask=mask,mask_index=index,repeat=repeat))
    anchors=[[]]+[[bl] for bl in range(32)]
    add('basis',anchors)
    add('fit',groups['fit'])
    add('selection',groups['selection']);add('selection_anchors',anchors)
    add('test',groups['test']);add('test_anchors',anchors)
    add('input_path_probe',probes)
    for group_id,columns in enumerate(plan['synthetic_groups']):
        for name,pattern in plan['synthetic_patterns'].items():
            for repeat in range(5):
                for t,active in enumerate(pattern):
                    entries.append(dict(phase='synthetic_lif',mask=[columns[j] for j in active],
                        group_id=group_id,pattern=name,repeat=repeat,t=t))
    for index,entry in enumerate(entries):entry['vector_id']=index+1
    if len(entries)*8+1536!=18576:raise ValueError('Frozen schedule count mismatch')
    return entries


class Acquisition:
    def __init__(self,connector,converters,output,sid,plan,source,entries):
        validate(plan)
        if source not in ('hardware','simulator') or connector.cb_type!=('real' if source=='hardware' else 'simulator'):
            raise ValueError('Backend/source mismatch')
        self.connector,self.converters,self.output,self.sid=connector,converters,output,sid
        self.plan,self.source,self.entries=plan,source,entries
        self.sequence=0;self.reads=[];self.vectors=[];self.skips=[];self.g={};self.basis={}
        self.log=(output/'raw.jsonl').open('x',encoding='utf-8',newline='\n')
        self.summary=dict(status='failed',source=source,protocol_id=plan['protocol_id'],programming_pulses=0,
            observed_counts={},skipped_counts={},analog_lif_tested=False,trained_bn_applied=False,
            trained_weights_programmed=False,address_coverage_is_not_qualification=True)

    def emit(self,event):
        self.log.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n');self.log.flush()

    def impact(self,meta):
        mode,wl=meta['mode'],meta['wl']
        if mode not in (9,10) or wl not in range(8) or self.sequence>=18576:raise ValueError('Command bound')
        mask=meta.get('mask',[])
        if mode==10 and (len(set(mask))!=len(mask) or any(b not in range(32) for b in mask) or len(mask)>4):
            raise ValueError('Unreviewed MVM mask')
        value=.1 if mode==9 else [.1 if bl in mask else 0. for bl in range(32)]
        self.sequence+=1
        command=dict(mode_flag=9,vol=value,pulse_width=0.,sign=0,wl=wl,bl=meta['bl'],id=self.sequence) if mode==9 else dict(mode_flag=10,vol=list(value),wl=wl,id=0)
        conversion=voltage(value,EXPECTED_BOARD,self.converters) if mode==9 else [voltage(v,EXPECTED_BOARD,self.converters) for v in value]
        row=dict(session_id=self.sid,source=self.source,task_id=self.sequence,timestamp_utc=now(),
            monotonic_time_s=time.monotonic(),command_si=command,native_voltage_conversion=conversion,**meta)
        self.emit(dict(event='request',**row));started=time.monotonic()
        try:
            result=self.connector.impact({**command,'vol':list(value) if mode==10 else value})
            row['native_result']=json_value(result)
            if not isinstance(result,(tuple,list)) or len(result)!=3:raise ValueError('Expected R,id,ADC tuple')
            native_r,returned_id,adc=result
            row.update(native_r=json_value(native_r),returned_id=json_value(returned_id),raw_adc=json_value(adc))
            if type(returned_id) is not int or returned_id!=0:raise ValueError('Native mode9/10 ID must be zero')
            if mode==9:
                if not isinstance(native_r,(int,float)) or not math.isfinite(native_r) or native_r<=0:
                    raise ValueError('Invalid native mode9 resistance')
                row.update(readback(adc,EXPECTED_BOARD,self.converters,read_voltage=.1))
                row['native_r_interpretation']='Tuple uses INI .3; G uses explicit native a2r at nominal .1'
            else:
                row['output_voltage_v']=mvm_output(adc,EXPECTED_BOARD,self.converters)
                if row['output_voltage_v']>4.:raise ValueError('Observed output exceeds 80% reference')
            row['status']='ok'
            if mode==9:self.reads.append({k:v for k,v in row.items() if k not in ('command_si','native_voltage_conversion','native_result')})
            return row
        except (Exception,KeyboardInterrupt) as exc:
            row.update(status='error',error=f'{type(exc).__name__}: {exc}');raise
        finally:self.emit(dict(event='result',latency_s=time.monotonic()-started,**row))

    def read_array(self,phase):
        for wl in range(8):
            for bl in range(32):
                for repeat in range(3):self.impact(dict(mode=9,phase=phase,wl=wl,bl=bl,repeat=repeat))

    def make_basis(self):
        for index in range(33):
            rows=[v['adc'] for v in self.vectors if v['phase']=='basis' and v['mask_index']==index]
            if len(rows)!=5:continue
            self.basis[index]=[(mean(r[wl] for r in rows),stdev(r[wl] for r in rows)) for wl in range(8)]

    def guard(self,meta):
        mask=meta['mask'];reasons=[];predictions=[];limits=self.plan['guard']
        if len(mask)>limits['max_active']:reasons.append('predeclared_max_active_4')
        decoded=voltage(.1,EXPECTED_BOARD,self.converters)['native_decoded_v']
        for wl in range(8):
            current=decoded*sum(self.g[wl,bl] for bl in mask);model=330*current
            gui=sum(.1*round(330*self.g[wl,bl],4) for bl in mask)
            p=dict(wl=wl,mode9_model_current_a=current,mode9_model_output_v=model,gui_model_output_v=gui)
            if not all(math.isfinite(v) for v in (current,model,gui)) or current>.005 or model>4 or gui>=3.3:
                reasons.append(f'wl{wl}_mode9_model_guard')
            if meta['phase']!='basis':
                if 0 not in self.basis or any(bl+1 not in self.basis for bl in mask):
                    reasons.append(f'wl{wl}_basis_missing')
                else:
                    zero,sd0=self.basis[0][wl]
                    upper=zero+3*sd0+2*sum(max(0,self.basis[bl+1][wl][0]-zero+3*math.hypot(self.basis[bl+1][wl][1],sd0)) for bl in mask)
                    p['basis_heuristic_upper_adc']=upper
                    if upper>.8*(2**EXPECTED_BOARD['adc_bit']-1):reasons.append(f'wl{wl}_fresh_basis_headroom')
            predictions.append(p)
        return dict(allowed=not reasons,reasons=reasons,predictions=predictions,
            upper_bound_certified=False,physical_current_compliance=False)

    def run(self):
        try:
            if not self.connector.open_port():raise ConnectionError('open_port returned false')
            self.read_array('pre')
            self.g={(wl,bl):mean(r['conductance_siemens'] for r in self.reads if r['wl']==wl and r['bl']==bl) for wl in range(8) for bl in range(32)}
            previous=''
            for meta in self.entries:
                if meta['phase']!=previous:
                    if previous=='basis':self.make_basis()
                    print(f"{self.source}: {meta['phase']}",flush=True);previous=meta['phase']
                decision=self.guard(meta)
                self.emit(dict(event='vector_guard',session_id=self.sid,source=self.source,**meta,**decision))
                key=f"{meta['phase']}/density{len(meta['mask'])}"
                if not decision['allowed']:
                    self.skips.append(dict(**meta,reasons=decision['reasons']))
                    self.summary['skipped_counts'][key]=self.summary['skipped_counts'].get(key,0)+1
                    continue
                results=[self.impact(dict(mode=10,wl=wl,**meta)) for wl in range(8)]
                self.vectors.append(dict(**meta,adc=[r['raw_adc'] for r in results],task_ids=[r['task_id'] for r in results]))
                self.summary['observed_counts'][key]=self.summary['observed_counts'].get(key,0)+1
            self.read_array('post')
            if len(self.reads)!=1536 or len(self.vectors)+len(self.skips)!=len(self.entries):raise ValueError('Coverage ledger mismatch')
            self.summary['status']='completed'
        except (Exception,KeyboardInterrupt) as exc:self.summary['error']=f'{type(exc).__name__}: {exc}'
        finally:
            try:
                if not self.connector.close_port():raise ConnectionError('close_port returned false')
            except Exception as exc:self.summary.update(status='failed',close_error=f'{type(exc).__name__}: {exc}')
            self.log.close();self.summary.update(impact_attempts=self.sequence,valid_read_responses=len(self.reads),
                complete_vectors=len(self.vectors),skipped_vectors=len(self.skips),finished_utc=now())
            save_json(self.output/'observations.json',dict(vectors=self.vectors,readbacks=self.reads,skips=self.skips))
            save_json(self.output/'summary.json',self.summary)
        return self.summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    for flag in ('run-simulator','run-hardware','dry-run'):mode.add_argument('--'+flag,action='store_true')
    args=parser.parse_args(argv)
    if not(args.run_simulator or args.run_hardware or args.dry_run):parser.print_help();return 0
    plan_bytes=local_path(CONFIG).read_bytes();plan=json.loads(plan_bytes);validate(plan)
    for name,pin in PINS.items():
        if sha256(local_path(name))!=pin:raise ValueError('Pinned helper changed: '+name)
    entries=schedule(plan)
    source='simulator' if args.run_simulator else 'hardware'
    config,snapshot,hashes=check_sources(source)
    for name in (*PINS,CONFIG,'all256_mvm.py','mvm_calibration.py','experiment.py'):
        hashes[name]=sha256(local_path(name))
    description=dict(source=source,protocol_id=plan['protocol_id'],maximum_impacts=18576,
        maximum_programming_pulses=0,source_sha256=hashes,schedule_sha256=digest(entries),
        hardware_access=source=='hardware' and not args.dry_run,source_states_matched=False,
        trained_weights_programmed=False,cifar_inputs_replayed=False,physical_voltage_calibrated=False,
        dense_masks_predeclared_unmeasured=True,basis_guard_is_heuristic=True)
    if args.dry_run:print(json.dumps(description));return 0
    locked=False;lock=local_path(LOCK)
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as stream:stream.write(f'Calibrated MVM {now()}\n')
        locked=True
    try:
        sid=str(uuid.uuid4());output=local_path('results/calibrated_mvm')/sid;output.mkdir(parents=True)
        (output/'protocol.json').write_bytes(plan_bytes);(output/'board.ini').write_bytes(snapshot)
        save_json(output/'schedule.json',entries)
        manifest=dict(session_id=sid,status='failed',started_utc=now(),invocation=launch_provenance(argv),**description)
        save_json(output/'start_manifest.json',manifest);serial=None
        try:
            if source=='hardware':connector=native_connector(config)
            else:
                from experiments.unn_rev3_20261014.paths import use_local_dependencies
                use_local_dependencies()
                import numpy as np
                from manager.board import Connector
                from simulator.src import create_crossbar_array
                random.seed(plan['seed']);np.random.seed(plan['seed'])
                serial=output/'array';create_crossbar_array(str(serial),32,8)
                (output/'simulator_settings.ini').write_bytes(local_path(SIM_INI).read_bytes())
                manifest['array_initial_sha256']=sha256(serial.with_suffix('.cb'))
                connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                    board_type='rp5_fpga_python',task_format='SI',config=config,crossbar_serial=str(serial))
            from manager.service import converters
            result=Acquisition(connector,converters,output,sid,plan,source,entries).run()
            manifest.update(status=result['status'],impact_attempts=result['impact_attempts'],error=result.get('error'))
        except (Exception,KeyboardInterrupt) as exc:manifest['error']=f'{type(exc).__name__}: {exc}'
        finally:
            if serial is not None and serial.with_suffix('.cb').exists():manifest['array_final_file_sha256']=sha256(serial.with_suffix('.cb'))
            manifest['array_file_hash_interpretation']='Persisted bytes; native in-memory read changes may not be saved'
            manifest['finished_utc']=now();manifest['result_sha256']={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
            save_json(output/'manifest.json',manifest)
        print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'])))
        return 0 if manifest['status']=='completed' else 2
    finally:
        if locked:lock.unlink()


if __name__=='__main__':raise SystemExit(main())
