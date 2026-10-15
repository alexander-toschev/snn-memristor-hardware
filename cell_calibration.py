"""One bounded conductance-discovery cycle, native Connector only. No automatic run."""
import argparse
from configparser import ConfigParser
import json
import logging
import math
import random
import statistics
import time
import uuid

from read_all import ROOT, INI, VENDOR, EXPECTED_BOARD, EXPECTED_SOURCES, local_path, preflight, native_connector, now, sha256, save_json, json_value
from native_conversions import voltage, readback, mvm_output
from experiment_hardware import launch_provenance, READ_ALL_SHA256

CONFIG='configs/cell_calibration.json'
SIM_INI='vendor/MemriBoard/simulator/simulator_settings.ini'
CELLS=tuple((wl,bl) for wl in range(8) for bl in range(32))
LOCK='results/read_all/.hardware.lock'
PROTOCOL_ID='cell-conductance-discovery-v2-gui-write-guard'
WRITE_GUARD=dict(prediction='requested_voltage_over_last_valid_mode7_native_resistance',
    sign0_limit_a=.04,sign1_limit_source='board.soft_cc',invalid_resistance='global_stop',
    skip_policy='advance_ramp_without_impact_or_state_update')


def stats(values):
    return dict(n=len(values),mean=statistics.mean(values),sd=statistics.stdev(values) if len(values)>1 else None,
                minimum=min(values),maximum=max(values))


def validate_plan(plan):
    expected={'protocol_id':PROTOCOL_ID,'programming_guard':WRITE_GUARD,'maximum_programming_decisions':138752,
        'geometry':dict(wl_count=8,bl_count=32,excluded_wl=[]),'cycles':1,
        'ramp':dict(start_v=.3,step_v=.01,stop_v=3.0,maximum_steps=271),'reset_endpoint_q':.005,
        'set_crossing_q':[.01,.02,.03,.04,.05],'tested_window_upper_q':.05,'read_voltage_v':.3,
        'checkpoint_reads':5,'mvm_levels_v':[.025,.05,.075,.1],'mvm_repeats':5,'initial_array_reads':3,'final_array_reads':3,
        'hardware_pulse_width_s':None,'hardware_submitted_pulse_width_s':0,'simulator_pulse_width_s':.000125,
        'current_model_limit_a':.005,'adc_headroom_fraction':.8,'maximum_programming_pulses':138752,
        'maximum_impacts':203008}
    if any(plan.get(k)!=v for k,v in expected.items()):
        raise ValueError('Protocol differs from reviewed bounded discovery; update code/protocol together')


class Discovery:
    def __init__(self,connector,converters,output,session_id,plan,source):
        if source not in ('hardware','simulator'): raise ValueError('Unknown source')
        if connector.cb_type != ('real' if source=='hardware' else 'simulator'): raise ValueError('Backend/source mismatch')
        self.connector,self.converters,self.output=connector,converters,output
        self.session_id,self.plan,self.source=session_id,plan,source
        self.log=(output/'raw.jsonl').open('x',encoding='utf-8',newline='\n')
        self.sequence=self.pulses=0
        self.decisions=self.skipped=0
        self.last_mode7={}
        self.last_write=None
        self.summary=dict(status='failed',source=source,cells=[],initial_array=[],final_array=[],checkpoint_count=0,
            programming_repeatability_qualified=False,baseline_scope=plan['baseline_scope'],protocol_id=plan['protocol_id'])

    def emit(self,row):
        self.log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); self.log.flush()

    def append(self,name,row):
        with (self.output/name).open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()

    def impact(self,wl,bl,mode,vol,phase,sign=0,**metadata):
        if (wl,bl) not in CELLS or mode not in (7,10): raise ValueError('Unsupported address/mode')
        guard={}
        if mode==7:
            if type(vol) not in (float,int) or not math.isfinite(vol) or not 0<=vol<=3 or sign not in (0,1):
                raise ValueError('Invalid mode7 voltage/polarity')
            if vol:
                if self.decisions>=self.plan['maximum_programming_decisions']: raise ValueError('Whole-session programming decision budget exhausted')
                previous=self.last_mode7.get((wl,bl))
                if previous is None or not math.isfinite(previous[0]) or previous[0]<=0:
                    raise ValueError('No valid previous native mode7 resistance for write guard')
                limit=self.plan['programming_guard']['sign0_limit_a'] if sign==0 else EXPECTED_BOARD['soft_cc']
                predicted=vol/previous[0]
                if not math.isfinite(predicted): raise ValueError('Non-finite predicted programming current')
                # Counts admitted attempts and skipped decisions, never invents an impact ID.
                if predicted<=limit and self.pulses>=self.plan['maximum_programming_pulses']:
                    raise ValueError('Whole-session pulse budget exhausted')
                if predicted<=limit and self.sequence>=self.plan['maximum_impacts']:
                    raise ValueError('Whole-session impact budget exhausted')
                self.decisions+=1
                guard=dict(programming_decision_id=self.decisions,previous_native_r_ohm=previous[0],
                    resistance_source_task_id=previous[1],predicted_current_a=predicted,current_limit_a=limit)
                if predicted>limit:
                    self.skipped+=1
                    width=self.plan['simulator_pulse_width_s'] if self.source=='simulator' else 0.0
                    self.emit(dict(event='skipped',session_id=self.session_id,source=self.source,phase=phase,wl=wl,bl=bl,
                        timestamp_utc=now(),monotonic_time_s=time.monotonic(),reason='predicted_write_current_exceeds_limit',
                        planned_command_si=dict(mode_flag=7,vol=vol,pulse_width=width,sign=sign,wl=wl,bl=bl),
                        native_voltage_conversion=voltage(vol,EXPECTED_BOARD,self.converters),**guard,**metadata))
                    return None
                self.pulses+=1
            codes=voltage(vol,EXPECTED_BOARD,self.converters)
        else:
            if (not isinstance(vol,list) or len(vol)!=32 or
                any(type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=.1 for v in vol) or
                sum(v!=0 for v in vol)>1): raise ValueError('Expected single-BL MVM at 0..0.1 V')
            codes=[voltage(v,EXPECTED_BOARD,self.converters) for v in vol]
        if self.sequence>=self.plan['maximum_impacts']: raise ValueError('Whole-session impact budget exhausted')
        self.sequence+=1
        width=self.plan['simulator_pulse_width_s'] if self.source=='simulator' and mode==7 and vol else 0.0
        task={'mode_flag':7,'vol':vol,'pulse_width':width,'sign':sign,'wl':wl,'bl':bl,'id':self.sequence} if mode==7 else {
            'mode_flag':10,'vol':list(vol),'wl':wl,'id':0}
        if mode==7 and vol: self.last_write=self.sequence
        row=dict(session_id=self.session_id,source=self.source,task_id=self.sequence,phase=phase,wl=wl,bl=bl,
            timestamp_utc=now(),monotonic_time_s=time.monotonic(),command_si=task,native_voltage_conversion=codes,
            native_adc_samples=10 if mode==7 and self.source=='hardware' else 1,
            physical_pulse_width_s=None,simulator_duration_steps=125 if width else None,**guard,**metadata)
        self.emit({**row,'event':'request'}); started=time.monotonic()
        try:
            submitted={**task,'vol':list(vol) if mode==10 else vol}
            result=self.connector.impact(submitted); row['native_result']=json_value(result)
            if not isinstance(result,(list,tuple)) or len(result)!=3: raise ValueError('Expected (R,id,ADC)')
            native_r,echoed,adc=result
            row.update(native_r=json_value(native_r),returned_id=json_value(echoed),raw_adc=json_value(adc))
            if type(echoed) is not int or echoed!=task['id']: raise ValueError('Native response ID mismatch')
            if mode==7:
                row.update(readback(adc,EXPECTED_BOARD,self.converters,read_voltage=.3))
                if type(native_r) not in (int,float) or not math.isfinite(native_r) or native_r<=0:
                    raise ValueError('Invalid native mode7 resistance for write guard')
                self.last_mode7[wl,bl]=(native_r,self.sequence)
            else:
                row['output_voltage_v']=mvm_output(adc,EXPECTED_BOARD,self.converters)
                if row['output_voltage_v']>self.plan['adc_headroom_fraction']*EXPECTED_BOARD['vol_ref_adc']:
                    raise ValueError('Observed MVM output exceeds ADC headroom')
            row['status']='ok'; return row
        except (Exception,KeyboardInterrupt) as exc:
            row.update(status='error',error=f'{type(exc).__name__}: {exc}'); raise
        finally: self.emit({**row,'event':'result','latency_s':time.monotonic()-started})

    def checkpoint(self,cell,role,crossed=(),reuse=False):
        if reuse and cell['checkpoints'] and cell['checkpoints'][-1]['last_write_task_id']==self.last_write:
            state=cell['checkpoints'][-1]
            self.append('checkpoint_roles.jsonl',dict(wl=cell['wl'],bl=cell['bl'],role=role,checkpoint_id=state['checkpoint_id'],
                reason='No further nonzero programming command; same checkpoint reused, not a new measured state'))
            return state
        self.summary['checkpoint_count']+=1
        state=dict(checkpoint_id=self.summary['checkpoint_count'],role=role,wl=cell['wl'],bl=cell['bl'],
            last_write_task_id=self.last_write,crossed_targets=list(crossed),status='failed',reads=[],mvm=[])
        cell['checkpoints'].append(state)
        try:
            for repeat in range(5):
                row=self.impact(cell['wl'],cell['bl'],7,0.0,'checkpoint_read',checkpoint_id=state['checkpoint_id'],repeat=repeat)
                state['reads'].append({k:row[k] for k in ('task_id','raw_adc','q','conductance_siemens')})
            state['q']=stats([r['q'] for r in state['reads']])
            state['conductance_siemens']=stats([r['conductance_siemens'] for r in state['reads']])
            state['last_read_q']=state['reads'][-1]['q']
            state['signed_target_errors_q']={str(target):state['q']['mean']-target for target in crossed}
            g=state['conductance_siemens']['mean']
            current=.1*g; output=current*EXPECTED_BOARD['sum_gain']
            state['prediction']=dict(current_a=current,output_v=output,uncalibrated=True)
            if not math.isfinite(g) or g<=0 or current>.005 or output>.8*EXPECTED_BOARD['vol_ref_adc']:
                self.append('rejected_commands.jsonl',dict(checkpoint_id=state['checkpoint_id'],reason='MVM model guard',**state['prediction']))
                raise ValueError('MVM model current/headroom bound; no forced terminal measurement')
            for repeat in range(5):
                for position,level in enumerate([0.0]+self.plan['mvm_levels_v']+[0.0]):
                    volts=[level if i==cell['bl'] else 0.0 for i in range(32)]
                    row=self.impact(cell['wl'],cell['bl'],10,volts,'checkpoint_mvm',
                        checkpoint_id=state['checkpoint_id'],repeat=repeat,position=position,nominal_v=level)
                    state['mvm'].append({k:row[k] for k in ('task_id','raw_adc','output_voltage_v','repeat','position','nominal_v')})
            state['transfer']=[dict(position=p,nominal_v=level,dac=voltage(level,EXPECTED_BOARD,self.converters)['dac_code'],
                native_decoded_v=voltage(level,EXPECTED_BOARD,self.converters)['native_decoded_v'],
                adc=stats([r['raw_adc'] for r in state['mvm'] if r['position']==p]),
                output_v=stats([r['output_voltage_v'] for r in state['mvm'] if r['position']==p]))
                for p,level in enumerate([0.0]+self.plan['mvm_levels_v']+[0.0])]
            state['status']='completed'; return state
        except (Exception,KeyboardInterrupt) as exc:
            state['error']=f'{type(exc).__name__}: {exc}'; raise
        finally: self.append('checkpoints.jsonl',state)

    def discover_cell(self,wl,bl):
        cell=dict(wl=wl,bl=bl,status='failed',reset_pulses=0,set_pulses=0,
            reset_decisions=0,set_decisions=0,reset_skipped_current=0,set_skipped_current=0,
            checkpoints=[],crossed_targets=[],reset_reached=False)
        self.summary['cells'].append(cell); self.last_write=None
        try:
            baseline=self.checkpoint(cell,'local_pre_reset'); q=baseline['last_read_q']
            for step in range(271):
                if q<=.005: break
                before=(self.pulses,self.decisions,self.skipped)
                try: response=self.impact(wl,bl,7,round(.3+.01*step,2),'reset',sign=0,step=step)
                finally:
                    cell['reset_pulses']+=self.pulses-before[0]
                    cell['reset_decisions']+=self.decisions-before[1]
                    cell['reset_skipped_current']+=self.skipped-before[2]
                if response is not None: q=response['q']
            cell.update(reset_reached=q<=.005,reset_decision_q=q)
            reset=self.checkpoint(cell,'post_reset',reuse=True)
            cell['reset_confirmed']=cell['reset_reached'] and reset['q']['mean']<=.005 and reset['last_read_q']<=.005
            cell['reset_confirmation_mean_q']=reset['q']['mean']
            cell['reset_confirmation_last_q']=reset['last_read_q']
            if cell['reset_confirmed']:
                q=reset['last_read_q']
                for step in range(271):
                    if cell['crossed_targets'] and cell['crossed_targets'][-1]>=.05: break
                    before=(self.pulses,self.decisions,self.skipped)
                    try: response=self.impact(wl,bl,7,round(.3+.01*step,2),'set',sign=1,step=step)
                    finally:
                        cell['set_pulses']+=self.pulses-before[0]
                        cell['set_decisions']+=self.decisions-before[1]
                        cell['set_skipped_current']+=self.skipped-before[2]
                    if response is None: continue
                    q=response['q']
                    crossed=[target for target in self.plan['set_crossing_q'] if target not in cell['crossed_targets'] and q>=target]
                    if crossed:
                        cell['crossed_targets'].extend(crossed)
                        self.checkpoint(cell,'set_crossing',crossed)
                cell['set_terminal_response_q']=q
            else: cell['set_skipped_reason']='reset_unconfirmed_after_readback' if cell['reset_reached'] else 'reset_endpoint_not_reached'
            terminal=self.checkpoint(cell,'terminal',reuse=True)
            cell.update(terminal_checkpoint_id=terminal['checkpoint_id'],terminal_g=terminal['conductance_siemens'],
                terminal_q=terminal['q'],unreached_targets=[q for q in self.plan['set_crossing_q'] if q not in cell['crossed_targets']],
                status='discovery_completed',tested_window_upper_reached=.05 in cell['crossed_targets'])
        except (Exception,KeyboardInterrupt) as exc:
            cell['error']=f'{type(exc).__name__}: {exc}'; raise
        finally:
            cell['checkpoints']=[{k:v for k,v in state.items() if k not in ('reads','mvm')} for state in cell['checkpoints']]
            self.append('cell_results.jsonl',cell)

    def run(self,cells=CELLS):
        try:
            if not self.connector.open_port(): raise ConnectionError('Connector.open_port returned false')
            initial={}
            for wl,bl in cells:
                readings=[self.impact(wl,bl,7,0.0,'initial_array_read',repeat=i) for i in range(3)]
                initial[wl,bl]=stats([r['conductance_siemens'] for r in readings])
                row=dict(wl=wl,bl=bl,conductance_siemens=initial[wl,bl])
                self.summary['initial_array'].append(row); self.append('initial_array.jsonl',row)
            for wl,bl in cells:
                self.discover_cell(wl,bl)
                print(f'cell discovery WL{wl}/BL{bl}: bounded cycle complete',flush=True)
            for cell in self.summary['cells']:
                readings=[self.impact(cell['wl'],cell['bl'],7,0.0,'final_array_read',repeat=i) for i in range(3)]
                observed=stats([r['conductance_siemens'] for r in readings]); previous=cell['terminal_g']['mean']
                row=dict(wl=cell['wl'],bl=cell['bl'],conductance_siemens=observed,terminal_mean_siemens=previous,
                    fractional_difference=(observed['mean']-previous)/previous,
                    initial_mean_siemens=initial[cell['wl'],cell['bl']]['mean'],
                    fractional_difference_from_initial=(observed['mean']-initial[cell['wl'],cell['bl']]['mean'])/initial[cell['wl'],cell['bl']]['mean'],
                    interpretation='Observed later difference; not proof of write-disturb cause or retained MVM transfer')
                self.summary['final_array'].append(row); self.append('final_array.jsonl',row)
            self.summary['status']='completed'
        except (Exception,KeyboardInterrupt) as exc: self.summary['error']=f'{type(exc).__name__}: {exc}'
        finally:
            try:
                if not self.connector.close_port(): raise ConnectionError('Connector.close_port returned false')
            except Exception as exc: self.summary.update(status='failed',close_error=f'{type(exc).__name__}: {exc}')
            self.log.close()
            self.summary.update(impact_attempts=self.sequence,programming_pulse_attempts=self.pulses,
                programming_decisions=self.decisions,programming_skipped_current=self.skipped,finished_utc=now())
            save_json(self.output/'summary.json',self.summary)
        return self.summary


def check_sources(source):
    if sha256(local_path('read_all.py'))!=READ_ALL_SHA256: raise ValueError('Unreviewed read_all helper')
    if source=='hardware': return preflight(INI)
    snapshot=local_path(INI).read_bytes()
    config=ConfigParser(); config.read_string(snapshot.decode('utf-8-sig'))
    simulator=ConfigParser(); simulator.read_string(local_path(SIM_INI).read_text(encoding='utf-8-sig'))
    for key,value in EXPECTED_BOARD.items():
        if config.getfloat('board',key)!=value or simulator.getfloat('board',key)!=value:
            raise ValueError(f'Connector/simulator INI mismatch: {key}')
    required={name:value for name,value in EXPECTED_SOURCES.items() if not name.startswith('MemriCORE/')}
    required.update({'simulator/src.py':'d41edc5c0464e15d44a23a1bb628e8025194e518d914635c1a14dd46ea2d4cc2',
        'simulator/memristor.py':'bb9c43d0f254c512c21902a64b5414e05f63f1deecf10016e5b97bd086050aca'})
    hashes={}
    for name,expected in required.items():
        path=f'{VENDOR}/{name}'; hashes[path]=sha256(local_path(path))
        if hashes[path]!=expected: raise ValueError(f'Simulator runtime fingerprint mismatch: {name}')
    for path in local_path(VENDOR).rglob('*.py'):
        hashes[path.relative_to(ROOT).as_posix()]=sha256(path)
    hashes[INI]=sha256(local_path(INI)); hashes[SIM_INI]=sha256(local_path(SIM_INI))
    return config,snapshot,hashes


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--run-simulator',action='store_true')
    mode.add_argument('--run-hardware',action='store_true')
    mode.add_argument('--dry-run',action='store_true',help='Physical preflight and bounds only, no imports/connections')
    args=parser.parse_args(argv)
    if not(args.run_simulator or args.run_hardware or args.dry_run): parser.print_help(); return 0
    source='simulator' if args.run_simulator else 'hardware'
    plan_bytes=local_path(CONFIG).read_bytes(); plan=json.loads(plan_bytes.decode('utf-8-sig')); validate_plan(plan)
    config,snapshot,hashes=check_sources(source)
    for name in ('experiment.py','cell_calibration.py','native_conversions.py','read_all.py','experiment_hardware.py',CONFIG):
        hashes[name]=sha256(local_path(name))
    description=dict(source=source,protocol_id=plan['protocol_id'],programming_guard=plan['programming_guard'],
        maximum_programming_decisions=plan['maximum_programming_decisions'],cells=256,cycles=1,maximum_checkpoints_per_cell=7,
        maximum_programming_pulses=138752,maximum_impacts=203008,
        initial_array_read_impacts=768,final_array_read_impacts=768,
        baseline_scope=plan['baseline_scope'],tested_window_upper_q=.05,cell_maximum_q_inferred=False,
        physical_pulse_width_s=None,simulator_pulse_width_s=.000125 if source=='simulator' else None,
        simulator_duration_steps=125 if source=='simulator' else None,
        native_mode7_adc_count=1 if source=='simulator' else 10,source_sha256=hashes)
    if args.dry_run: print(json.dumps(dict(hardware_access=False,**description))); return 0
    invocation=launch_provenance(argv)
    lock=local_path(LOCK); locked=False
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as stream: stream.write(f'Cell calibration {now()}\n')
        locked=True
    try:
        session_id=str(uuid.uuid4()); output=local_path('results/cell_calibrations')/session_id
        output.mkdir(parents=True,exist_ok=False)
        (output/'protocol.json').write_bytes(plan_bytes); (output/'board.ini').write_bytes(snapshot)
        manifest=dict(session_id=session_id,status='failed',started_utc=now(),invocation=invocation,**description)
        save_json(output/'start_manifest.json',manifest)
        try:
            if source=='hardware': connector=native_connector(config)
            else:
                from experiments.unn_rev3_20261014.paths import use_local_dependencies
                use_local_dependencies()
                import numpy as np
                from manager.board import Connector
                from simulator.src import create_crossbar_array
                random.seed(plan['simulator_seed']); np.random.seed(plan['simulator_seed'])
                serial=output/'array'
                (output/'simulator_settings.ini').write_bytes(local_path(SIM_INI).read_bytes())
                create_crossbar_array(str(serial),32,8)
                manifest.update(array_initial_sha256=sha256(serial.with_suffix('.cb')),simulator_seed=plan['simulator_seed'])
                connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                    board_type='rp5_fpga_python',task_format='SI',config=config,crossbar_serial=str(serial))
            from manager.service import converters
            discovery=Discovery(connector,converters,output,session_id,plan,source)
            result=discovery.run()
            manifest.update(status=result['status'],impact_attempts=result['impact_attempts'],
                programming_pulse_attempts=result['programming_pulse_attempts'],programming_decisions=result['programming_decisions'],
                programming_skipped_current=result['programming_skipped_current'],error=result.get('error'))
        except (Exception,KeyboardInterrupt) as exc: manifest['error']=f'{type(exc).__name__}: {exc}'
        finally:
            manifest['finished_utc']=now()
            manifest['result_sha256']={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
            save_json(output/'manifest.json',manifest)
        print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'])))
        return 0 if manifest['status']=='completed' else 2
    finally:
        if locked: lock.unlink()


if __name__=='__main__': raise SystemExit(main())
