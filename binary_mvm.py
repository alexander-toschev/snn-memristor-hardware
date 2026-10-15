"""Bounded binary MVM / digital LIF diagnostic. Public Connector only, no programming."""
import argparse
import hashlib
import json
import logging
import math
import random
from statistics import mean
import time
import uuid

from cell_calibration import check_sources, SIM_INI, LOCK
from read_all import ROOT, EXPECTED_BOARD, local_path, native_connector, now, sha256, save_json, json_value
from native_conversions import voltage, readback, mvm_output
from experiment_hardware import launch_provenance
from binary_mvm_analysis import evaluate

CONFIG='configs/binary_mvm.json'
PINS={'cell_calibration.py':'eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8',
      'read_all.py':'b30ae2f4dd3cd166dbc5cefb07b558127de2bbdba1f6674c40b1b45f4dd74d5f',
      'native_conversions.py':'2ad2e9c057bb14979b083a6e846e8c417c080a6c5273d848e31f893b7970253e',
      'experiment_hardware.py':'059dc4f827943a9df631631f52e776b57182afc9d36db0c1ad5246427f73548c'}
SPEC=dict(protocol_id='binary-mvm-digital-lif-v1',pairs=[dict(wl=wl,data_bl=cols) for wl,cols in
    [([0,1],[0,2,3]),([2,3],[5,9,11]),([4,5],[0,1,4]),([6,7],[0,3,4])]],pedestal_bl=31,
    nominal_v=.1,repeats=5,readbacks_per_cell=3,calibration_bits=[0,1,2,4],validation_bits=list(range(8)),
    sequences=dict(dispersed=[0,1,2,4,0,0,0,0,0,0,0,0],coincident=[0,7,0,0,0,0,0,0,0,0,0,0]),
    maximum_impacts=3072,phase_counts=dict(pre=96,calibration=320,validation=640,sequence=1920,post=96),
    modes=[9,10],maximum_programming_pulses=0,current_model_limit_a=.005,gui_prediction_limit_v=3.3,
    adc_headroom_fraction=.8,simulator_seed=20261015,
    thresholds=dict(minimum_snr=5,maximum_cv=.05,additivity_fraction=.05,error_se_multiplier=3,error_floor_adc=2,
        drift_fraction=.05,drift_se_multiplier=3,pedestal_floor_adc=5,
        validation_repeatability_cv=.05,near_zero_signal_adc=5,near_zero_sd_floor_adc=2,
        near_zero_policy='abs(expected)<=max(5,5*calibration_repeat_noise); combined validation SD<=max(2,calibration_repeat_noise)'),
    lif=dict(tau=.5,threshold=1.,threshold_inclusive=True,reset=0.,identity_bn=True,
        normalization='2/sum(abs(calibration_signed_basis))',maximum_membrane_error=.05,
        exact_spike_agreement=True,expected_spike_and_silent_required=True),
    address_selection_map='results/cell_maps/9923f9bc-1ffc-4977-b26f-fd68c6c84e90',
    address_selection_is_current_state_qualification=False)


def validate(plan):
    if plan!=SPEC:raise ValueError('Unreviewed binary protocol; edit code/config together')


def pair_for(wl,plan):
    return next((i,p) for i,p in enumerate(plan['pairs']) if wl in p['wl'])


def vector_for(wl,bits,pedestal,plan):
    _,pair=pair_for(wl,plan)
    active=[bl for i,bl in enumerate(pair['data_bl']) if bits&(1<<i)]
    if pedestal:active.append(plan['pedestal_bl'])
    return [plan['nominal_v'] if i in active else 0. for i in range(32)]


def tasks(plan):
    for phase in ('pre','calibration','validation','sequence','post'):
        if phase in ('pre','post'):
            for wl in range(8):
                pair_id,pair=pair_for(wl,plan)
                for bl in pair['data_bl']+[plan['pedestal_bl']]:
                    for repeat in range(3):yield dict(phase=phase,mode=9,wl=wl,bl=bl,repeat=repeat,pair_id=pair_id)
        else:
            for repeat in range(5):
                for pedestal in (False,True):
                    sequences=plan['sequences'].items() if phase=='sequence' else [('',plan[phase+'_bits'])]
                    for name,bits_list in sequences:
                        for t,bits in enumerate(bits_list):
                            for wl in range(8):
                                yield dict(phase=phase,mode=10,wl=wl,bits=bits,pedestal=pedestal,repeat=repeat,
                                    sequence=name,t=t,pair_id=pair_for(wl,plan)[0])


class BinaryRun:
    def __init__(self,connector,converters,output,session_id,plan,source):
        validate(plan)
        if source not in ('hardware','simulator') or connector.cb_type!=('real' if source=='hardware' else 'simulator'):
            raise ValueError('Backend/source mismatch')
        self.connector,self.converters,self.output,self.plan,self.source=connector,converters,output,plan,source
        self.session_id=session_id;self.sequence=0;self.records=[];self.readbacks=[]
        self.log=(output/'raw.jsonl').open('x',encoding='utf-8',newline='\n')
        self.summary=dict(status='failed',source=source,protocol_id=plan['protocol_id'],phase_counts={},
            programming_pulses=0,analog_lif_tested=False,trained_bn_applied=False,weights_programmed=False)

    def emit(self,row):
        self.log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');self.log.flush()

    def impact(self,meta):
        mode,wl=meta['mode'],meta['wl']
        if mode not in (9,10) or wl not in range(8) or self.sequence>=3072:raise ValueError('Protocol bound')
        value=.1 if mode==9 else vector_for(wl,meta['bits'],meta['pedestal'],self.plan)
        self.sequence+=1
        command=dict(mode_flag=9,vol=value,pulse_width=0.,sign=0,wl=wl,bl=meta['bl'],id=self.sequence) if mode==9 else dict(mode_flag=10,vol=list(value),wl=wl,id=0)
        codes=voltage(value,EXPECTED_BOARD,self.converters) if mode==9 else [voltage(v,EXPECTED_BOARD,self.converters) for v in value]
        row=dict(session_id=self.session_id,source=self.source,task_id=self.sequence,timestamp_utc=now(),
            monotonic_time_s=time.monotonic(),command_si=command,native_voltage_conversion=codes,**meta)
        self.emit(dict(event='request',**row));started=time.monotonic()
        self.summary['phase_counts'][meta['phase']]=self.summary['phase_counts'].get(meta['phase'],0)+1
        try:
            result=self.connector.impact({**command,'vol':list(value) if mode==10 else value})
            row['native_result']=json_value(result)
            if not isinstance(result,(tuple,list)) or len(result)!=3:raise ValueError('Expected native R,id,ADC tuple')
            native_r,returned_id,adc=result
            row.update(native_r=json_value(native_r),returned_id=json_value(returned_id),raw_adc=json_value(adc))
            if type(returned_id) is not int or returned_id!=0:raise ValueError('Expected native ID0 for mode9/10')
            if mode==9:
                row.update(readback(adc,EXPECTED_BOARD,self.converters,read_voltage=.1))
                row['native_r_interpretation']='Native tuple uses INI .3; separate native a2r at commanded .1 used for pre/post G'
            else:
                row['output_voltage_v']=mvm_output(adc,EXPECTED_BOARD,self.converters)
                if row['output_voltage_v']>.8*EXPECTED_BOARD['vol_ref_adc']:raise ValueError('Observed output exceeds 80% ADC reference')
                row['native_r_interpretation']='Not interpreted for mode10'
            row['status']='ok'
            compact={k:v for k,v in row.items() if k not in ('command_si','native_voltage_conversion','native_result')}
            (self.readbacks if mode==9 else self.records).append(compact)
        except (Exception,KeyboardInterrupt) as exc:
            row.update(status='error',error=f'{type(exc).__name__}: {exc}');raise
        finally:self.emit(dict(event='result',latency_s=time.monotonic()-started,**row))

    def check_model(self):
        conductance={(r['wl'],r['bl']):mean(s['conductance_siemens'] for s in self.readbacks if s['wl']==r['wl'] and s['bl']==r['bl']) for r in self.readbacks}
        predictions=[]
        decoded=voltage(.1,EXPECTED_BOARD,self.converters)['native_decoded_v']
        for wl in range(8):
            for pedestal in (False,True):
                for bits in range(8):
                    active=[bl for bl,v in enumerate(vector_for(wl,bits,pedestal,self.plan)) if v]
                    current=decoded*sum(conductance[wl,bl] for bl in active)
                    output=EXPECTED_BOARD['sum_gain']*current
                    gui=sum(.1*round(EXPECTED_BOARD['sum_gain']*conductance[wl,bl],4) for bl in active)
                    predictions.append(dict(wl=wl,pedestal=pedestal,bits=bits,current_a=current,output_v=output,gui_prediction_v=gui))
        self.summary['pre_mvm_model_predictions']=predictions
        if any(not all(math.isfinite(p[k]) for k in ('current_a','output_v','gui_prediction_v')) or
               p['current_a']>.005 or p['output_v']>.8*EXPECTED_BOARD['vol_ref_adc'] or p['gui_prediction_v']>=3.3 for p in predictions):
            self.emit(dict(event='rejected_mvm',session_id=self.session_id,source=self.source,reason='pre_read_model_bounds'))
            raise ValueError('Pre-read model exceeds 5mA / 80% ADC / GUI3.3V; no MVM sent')

    def run(self):
        try:
            if not self.connector.open_port():raise ConnectionError('open_port returned false')
            previous=''
            for meta in tasks(self.plan):
                if meta['phase']!=previous:
                    if meta['phase']=='calibration':self.check_model()
                    print(f"{self.source}: {meta['phase']}",flush=True);previous=meta['phase']
                self.impact(meta)
            if self.summary['phase_counts']!=self.plan['phase_counts']:raise ValueError('Acquisition count mismatch')
            self.summary['analysis']=evaluate(self.records,self.readbacks,self.plan)
            self.summary['status']='completed'
        except (Exception,KeyboardInterrupt) as exc:self.summary['error']=f'{type(exc).__name__}: {exc}'
        finally:
            try:
                if not self.connector.close_port():raise ConnectionError('close_port returned false')
            except Exception as exc:self.summary.update(status='failed',close_error=f'{type(exc).__name__}: {exc}')
            self.log.close();self.summary.update(impact_attempts=self.sequence,valid_responses=len(self.records)+len(self.readbacks),finished_utc=now())
            save_json(self.output/'observations.json',dict(mvm=self.records,readbacks=self.readbacks))
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
    source='simulator' if args.run_simulator else 'hardware'
    config,snapshot,hashes=check_sources(source)
    for name in (*PINS,CONFIG,'binary_mvm.py','binary_mvm_analysis.py','experiment.py'):hashes[name]=sha256(local_path(name))
    schedule=[m for m in tasks(plan)]
    schedule_hash=hashlib.sha256(json.dumps(schedule,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    description=dict(source=source,protocol_id=plan['protocol_id'],maximum_impacts=3072,phase_counts=plan['phase_counts'],
        programming_pulses=0,source_sha256=hashes,schedule_sha256=schedule_hash,hardware_access=source=='hardware' and not args.dry_run,
        source_states_matched=False,physical_voltage_calibrated=False,physical_pulse_duration_known=False)
    if args.dry_run:print(json.dumps(description));return 0
    locked=False;lock=local_path(LOCK)
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as stream:stream.write(f'Binary MVM {now()}\n')
        locked=True
    try:
        sid=str(uuid.uuid4());output=local_path('results/binary_mvm')/sid;output.mkdir(parents=True)
        (output/'protocol.json').write_bytes(plan_bytes);(output/'board.ini').write_bytes(snapshot)
        save_json(output/'schedule.json',schedule)
        manifest=dict(session_id=sid,status='failed',started_utc=now(),invocation=launch_provenance(argv),**description)
        save_json(output/'start_manifest.json',manifest)
        serial=None
        try:
            if source=='hardware':connector=native_connector(config)
            else:
                from experiments.unn_rev3_20261014.paths import use_local_dependencies
                use_local_dependencies()
                import numpy as np
                from manager.board import Connector
                from simulator.src import create_crossbar_array
                random.seed(plan['simulator_seed']);np.random.seed(plan['simulator_seed'])
                serial=output/'array';create_crossbar_array(str(serial),32,8)
                (output/'simulator_settings.ini').write_bytes(local_path(SIM_INI).read_bytes())
                manifest.update(array_initial_sha256=sha256(serial.with_suffix('.cb')),simulator_seed=plan['simulator_seed'])
                connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                    board_type='rp5_fpga_python',task_format='SI',config=config,crossbar_serial=str(serial))
            from manager.service import converters
            result=BinaryRun(connector,converters,output,sid,plan,source).run()
            manifest.update(status=result['status'],impact_attempts=result['impact_attempts'],valid_responses=result['valid_responses'],error=result.get('error'))
        except (Exception,KeyboardInterrupt) as exc:manifest['error']=f'{type(exc).__name__}: {exc}'
        finally:
            if serial is not None and serial.with_suffix('.cb').exists():manifest['array_final_file_sha256']=sha256(serial.with_suffix('.cb'))
            manifest['array_file_hash_interpretation']='Persisted .cb bytes only; no claim that native in-memory state was unchanged or saved'
            manifest['finished_utc']=now();manifest['result_sha256']={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
            save_json(output/'manifest.json',manifest)
        print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'])))
        return 0 if manifest['status']=='completed' else 2
    finally:
        if locked:lock.unlink()


if __name__=='__main__':raise SystemExit(main())
