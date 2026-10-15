"""Confirm fixed routing with NEW sequential one-hot measurements; no weight writes."""
import argparse
from collections import Counter
import copy
import json
import logging
import math
import random
from statistics import mean, stdev
import time
import uuid

from all256_mvm import Acquisition, SPEC as BASE_SPEC, PINS as BASE_PINS, digest
from cell_calibration import check_sources, SIM_INI, LOCK
from read_all import ROOT, EXPECTED_BOARD, local_path, native_connector, now, sha256, save_json
from experiment_hardware import launch_provenance
from native_conversions import voltage
from mapped_sequential_analysis import pattern, centered_block, freeze_reference, evaluate

CONFIG='configs/mapped_sequential_mvm.json'
PINS={**BASE_PINS,'all256_mvm.py':'c8391bb9125bc9f0d280f91d1ca56b4cf98293d702531805bac13e2836c168f0'}
SPEC=dict(protocol_id='mapped-sequential-mvm-lif-v1',nominal_v=.1,repeats=5,read_repeats=3,seed=20261017,
    pairs=[dict(wl=[0,1],positive_bl=[0,1,2,3,4,5,6,7,8,9,10,11,13,31],negative_bl=[0,2,3,4,6,7,15,5,8,9,12,14,31,10]),
        dict(wl=[2,3],positive_bl=[0,5,6,9,11,12,14,15,31],negative_bl=[2,3,5,7,8,10,11,15,9]),
        dict(wl=[4,5],positive_bl=[0,1,2,4,6,10,11,13,14,15,31],negative_bl=[0,9,7,5,10,1,12,4,8,31,14]),
        dict(wl=[6,7],positive_bl=[0,3,4,5,6,7,8,9,10,12,13,14,31],negative_bl=[0,1,5,7,3,8,2,11,10,6,31,4,9])],
    patterns=['dispersed','grouped','dense_all'],timesteps=12,maximum_impacts=10706,
    phase_counts=dict(pre=768,basis=1320,selection=1320,evaluation=5210,final=1320,post=768),
    modes=[9,10],maximum_programming_pulses=0,physical_active_inputs_maximum=1,
    minimum_snr=5.,maximum_cv=.05,relative_error=.05,absolute_error_adc=2.,maximum_membrane_error=.05,
    tau=.5,threshold=1.,threshold_inclusive=True,reset=0.,negative_membrane_preserved=True,
    scale='basis only:1.25/maxabs(no-reset reference over all three patterns), common per pair',
    guard=dict(current_model_a=.005,gui_model_v=3.3,headroom_v=4.,basis_factor=2.,basis_sd_multiplier=3.,heuristic_only=True),
    mapping_is_posthoc_proposal=True,mapping_changes_during_run=False,source_states_matched=False,
    trained_weights_programmed=False,cifar_replay=False)


def validate(plan):
    if plan!=SPEC:raise ValueError('Protocol differs from reviewed mapped-sequential specification')


def schedule(plan):
    validate(plan);entries=[];block=0
    def add(**meta):entries.append(dict(schedule_id=len(entries)+1,**meta))
    def reads(phase):
        for wl in range(8):
            for bl in range(32):
                for repeat in range(3):add(mode=9,phase=phase,wl=wl,bl=bl,repeat=repeat)
    def anchors(phase):
        for repeat in range(5):
            for mask in [[]]+[[b] for b in range(32)]:
                for wl in range(8):add(mode=10,phase=phase,wl=wl,mask=mask,repeat=repeat,role='anchor')
    reads('pre');anchors('basis');anchors('selection')
    for pair_id,pair in enumerate(plan['pairs']):
        for name in plan['patterns']:
            for repeat in range(5):
                for t,active in enumerate(pattern(name,len(pair['positive_bl']))):
                    for wl,columns in zip(pair['wl'],(pair['positive_bl'],pair['negative_bl'])):
                        block+=1;common=dict(mode=10,phase='evaluation',pair_id=pair_id,pattern=name,repeat=repeat,
                            t=t,wl=wl,block_id=block,active_logical_indices=active)
                        add(**common,mask=[],role='zero_before')
                        for j in active or [None]:
                            add(**common,mask=[columns[j]] if j is not None else [],
                                logical_index=j,role='partial' if j is not None else 'zero_signal')
                        add(**common,mask=[],role='zero_after')
    anchors('final');reads('post')
    if dict(Counter(e['phase'] for e in entries))!=plan['phase_counts']:raise AssertionError('Budget mismatch')
    return entries


class SequentialRun(Acquisition):
    """Reuse pinned raw transport validation, not the old coupled-MVM algorithm."""
    def __init__(self,connector,converters,output,sid,plan,source,entries):
        validate(plan)
        super().__init__(connector,converters,output,sid,copy.deepcopy(BASE_SPEC),source,[])
        self.protocol=plan;self.schedule=entries;self.records=[];self.blocks=[];self.skipped=[];self.reference=None
        self.summary.update(protocol_id=plan['protocol_id'],physical_max_active=1,cifar_replay=False,
            mapped_cells=94,mapped_coefficients=47,analog_joint_mvm_tested=False)

    def impact(self,meta):
        if self.sequence>=10706 or len(meta.get('mask',[]))>1:raise ValueError('Sequential command bound')
        row=super().impact(meta)
        if meta['mode']==10:self.records.append(dict(**meta,task_id=row['task_id'],raw_adc=row['raw_adc']))
        return row

    def make_basis(self):
        self.basis={}
        for wl in range(8):
            for mask in [[]]+[[b] for b in range(32)]:
                values=[r['raw_adc'] for r in self.records if r['phase']=='basis' and r['wl']==wl and r['mask']==mask]
                if len(values)==5:self.basis[wl,tuple(mask)]=(mean(values),stdev(values))

    def guard_row(self,meta):
        wl=meta['wl'];mask=meta.get('mask',[]);reasons=[]
        current=voltage(.1,EXPECTED_BOARD,self.converters)['native_decoded_v']*sum(self.g[wl,b] for b in mask)
        model=330*current;gui=sum(.1*round(330*self.g[wl,b],4) for b in mask)
        if not all(math.isfinite(x) for x in (current,model,gui)) or current>.005 or model>4 or gui>=3.3:reasons.append('mode9_model_guard')
        upper=None
        if meta['phase']!='basis':
            zero=self.basis.get((wl,()));single=self.basis.get((wl,tuple(mask)))
            if zero is None or single is None:reasons.append('basis_missing')
            else:
                upper=zero[0]+3*zero[1]+(2*max(0.,single[0]-zero[0]+3*math.hypot(single[1],zero[1])) if mask else 0.)
                if upper>.8*16383:reasons.append('fresh_basis_headroom')
        return dict(allowed=not reasons,reasons=reasons,mode9_model_current_a=current,mode9_model_output_v=model,
            gui_model_output_v=gui,basis_heuristic_upper_adc=upper,upper_bound_certified=False,physical_current_compliance=False)

    def run(self):
        try:
            if not self.connector.open_port():raise ConnectionError('open_port returned false')
            index=0;previous=''
            while index<len(self.schedule):
                meta=self.schedule[index];phase=meta['phase']
                if phase!=previous:
                    if previous=='pre':self.g={(w,b):mean(r['conductance_siemens'] for r in self.reads if r['wl']==w and r['bl']==b) for w in range(8) for b in range(32)}
                    if previous=='basis':self.make_basis()
                    if phase=='evaluation':
                        self.reference=freeze_reference(self.records,self.protocol)
                        save_json(self.output/'frozen_reference.json',self.reference)
                        self.emit(dict(event='reference_frozen',session_id=self.sid,source=self.source,
                            sha256=digest(self.reference),before_task_id=self.sequence+1))
                    print(f'{self.source}: {phase}',flush=True);previous=phase
                if meta['mode']==9:self.impact(meta);index+=1;continue
                group=[meta]
                if phase=='evaluation':
                    group=[]
                    while index+len(group)<len(self.schedule) and self.schedule[index+len(group)].get('block_id')==meta['block_id']:
                        group.append(self.schedule[index+len(group)])
                decisions=[self.guard_row(m) for m in group]
                permitted=all(d['allowed'] for d in decisions)
                self.emit(dict(event='acquisition_guard',session_id=self.sid,source=self.source,
                    schedule_ids=[m['schedule_id'] for m in group],decisions=decisions,allowed=permitted))
                if not permitted:
                    self.skipped.append(dict(schedule_ids=[m['schedule_id'] for m in group],phase=phase,
                        reasons=sorted({reason for d in decisions for reason in d['reasons']})))
                    index+=len(group);continue
                started=time.monotonic();results=[self.impact(m) for m in group]
                if phase=='evaluation':
                    self.blocks.append(dict(pair_id=meta['pair_id'],pattern=meta['pattern'],repeat=meta['repeat'],t=meta['t'],
                        wl=meta['wl'],block_id=meta['block_id'],active_logical_indices=meta['active_logical_indices'],
                        task_ids=[r['task_id'] for r in results],schedule_ids=[m['schedule_id'] for m in group],
                        zero_before_adc=results[0]['raw_adc'],signal_adc=[r['raw_adc'] for r in results[1:-1]],
                        zero_after_adc=results[-1]['raw_adc'],
                        centered_adc=centered_block(results[0]['raw_adc'],[r['raw_adc'] for r in results[1:-1]],results[-1]['raw_adc']),
                        elapsed_s=time.monotonic()-started))
                index+=len(group)
            if self.sequence+sum(len(s['schedule_ids']) for s in self.skipped)!=10706 or len(self.reads)!=1536:raise ValueError('Coverage ledger mismatch')
            self.summary['status']='completed'
        except (Exception,KeyboardInterrupt) as exc:self.summary['error']=f'{type(exc).__name__}: {exc}'
        finally:
            try:
                if not self.connector.close_port():raise ConnectionError('close_port returned false')
            except Exception as exc:self.summary.update(status='failed',close_error=f'{type(exc).__name__}: {exc}')
            self.log.close();self.summary.update(impact_attempts=self.sequence,valid_read_responses=len(self.reads),
                valid_mvm_responses=len(self.records),complete_blocks=len(self.blocks),
                skipped_commands=sum(len(s['schedule_ids']) for s in self.skipped),finished_utc=now())
            save_json(self.output/'observations.json',dict(records=self.records,readbacks=self.reads,blocks=self.blocks,skips=self.skipped))
            if self.reference is not None:
                try:save_json(self.output/'analysis.json',evaluate(self.records,self.reads,self.blocks,self.reference,self.protocol))
                except Exception as exc:self.summary.update(status='failed',analysis_error=f'{type(exc).__name__}: {exc}')
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
    entries=schedule(plan);source='simulator' if args.run_simulator else 'hardware'
    config,snapshot,hashes=check_sources(source)
    for name in (*PINS,CONFIG,'mapped_sequential_mvm.py','mapped_sequential_analysis.py','experiment.py'):hashes[name]=sha256(local_path(name))
    description=dict(source=source,protocol_id=plan['protocol_id'],maximum_impacts=10706,phase_counts=plan['phase_counts'],
        maximum_programming_pulses=0,source_sha256=hashes,schedule_sha256=digest(entries),
        hardware_access=source=='hardware' and not args.dry_run,source_states_matched=False,
        trained_weights_programmed=False,cifar_inputs_replayed=False,physical_voltage_calibrated=False,basis_guard_is_heuristic=True)
    if args.dry_run:print(json.dumps(description));return 0
    locked=False;lock=local_path(LOCK)
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as stream:stream.write(f'Mapped sequential MVM {now()}\n')
        locked=True
    try:
        sid=str(uuid.uuid4());output=local_path('results/mapped_sequential_mvm')/sid;output.mkdir(parents=True)
        (output/'protocol.json').write_bytes(plan_bytes);(output/'board.ini').write_bytes(snapshot);save_json(output/'schedule.json',entries)
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
                random.seed(plan['seed']);np.random.seed(plan['seed']);serial=output/'array';create_crossbar_array(str(serial),32,8)
                (output/'simulator_settings.ini').write_bytes(local_path(SIM_INI).read_bytes())
                manifest['array_initial_sha256']=sha256(serial.with_suffix('.cb'))
                connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                    board_type='rp5_fpga_python',task_format='SI',config=config,crossbar_serial=str(serial))
            from manager.service import converters
            result=SequentialRun(connector,converters,output,sid,plan,source,entries).run()
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
