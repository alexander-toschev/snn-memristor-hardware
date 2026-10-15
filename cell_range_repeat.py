"""Bounded three-cell native-R interval experiment. Explicit opt-in, public Connector only."""
import argparse
import json
import logging
import math
import random
import time
import uuid

from cell_calibration import Discovery, check_sources, stats, SIM_INI, LOCK
from read_all import ROOT, EXPECTED_BOARD, VENDOR, local_path, native_connector, now, sha256, save_json
from experiment_hardware import launch_provenance

CONFIG='configs/cell_range_repeat.json'
BASE_CONFIG='configs/cell_calibration.json'
PINS={'cell_calibration.py':'eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8',
      'native_conversions.py':'2ad2e9c057bb14979b083a6e846e8c417c080a6c5273d848e31f893b7970253e',
      'experiment_hardware.py':'059dc4f827943a9df631631f52e776b57182afc9d36db0c1ad5246427f73548c',
      BASE_CONFIG:'ffc2dc1f0c0560b7375ff2d3e1125b06a9dea445546a0cba24605060f266adf1',
      VENDOR+'/manager/terminate.py':'00b385405bf19b6adff5e3d8dfa02e6c80dd10b3f4693476730d3bdcc2c3b779'}
CELLS=[dict(wl=wl,bl=8,targets_q=qs) for wl,qs in ((0,[.04,.045,.05]),(3,[.04,.045,.05]),(5,[.03,.035,.04]))]


def bounds(q):
    return [EXPECTED_BOARD['sum_gain']/(q*1.01),EXPECTED_BOARD['sum_gain']/(q*.99)]


def validate(plan):
    expected=dict(protocol_id='three-cell-native-R-interval-v1',cells=CELLS,cycles_per_cell=3,q_relative_tolerance=.01,
        reset_endpoint_q=.005,ramp=dict(start_v=.3,step_v=.01,stop_v=3.,maximum_steps=271),initial_reads=1,
        reset_confirmation_reads=5,pre_reads=5,post_reads=5,delayed_reads=3,delay_s=1.,final_reads=3,
        maximum_programming_decisions=4878,maximum_programming_pulses=4878,maximum_impacts=6120,
        baseline_scope='per_cell_cycle_before_reset')
    if set(plan)!=set(expected)|{'qualification'} or any(plan.get(k)!=v for k,v in expected.items()):
        raise ValueError('Unreviewed interval protocol; operational overrides are forbidden')


class IntervalRun(Discovery):
    def __init__(self,connector,converters,output,session_id,plan,source,equal_in,wait=time.sleep):
        super().__init__(connector,converters,output,session_id,plan,source)
        self.equal_in,self.wait=equal_in,wait
        self.cycle_id=0;self.latest={}
        self.summary=dict(status='failed',source=source,protocol_id=plan['protocol_id'],cycles=[],continuous_range_qualified=False)

    def emit(self,row):
        # Validate native R before its successful response reaches the state machine.
        if row.get('event')=='result' and row.get('status')=='ok' and row['command_si']['mode_flag']==7:
            if row['native_r']!=row['resistance_ohm']:
                row.update(status='error',error='Native R differs from pinned converter readback R')
                super().emit(row)
                raise ValueError(row['error'])
        super().emit(row)

    def impact(self,*args,**kwargs):
        row=super().impact(*args,cycle_id=self.cycle_id,**kwargs)
        if row is not None and row['command_si']['mode_flag']==7:
            self.latest[row['wl'],row['bl']]=row
        return row

    def reads(self,wl,bl,count,phase,**meta):
        return [self.impact(wl,bl,7,0.,phase,repeat=i,**meta) for i in range(count)]

    def read_summary(self,rows,interval=None):
        return dict(task_ids=[r['task_id'] for r in rows],raw_adc=[r['raw_adc'] for r in rows],
            native_r_ohm=[r['native_r'] for r in rows],q=stats([r['q'] for r in rows]),
            all_inside=all(self.equal_in(interval)([r['native_r']]) for r in rows) if interval else None)

    def hit_checkpoint(self,wl,bl,target):
        key=dict(target_q=target['target_q']); band=target['native_r_interval_ohm']
        target['pre']=self.read_summary(self.reads(wl,bl,5,'interval_pre',**key),band)
        g=stats([1/r for r in target['pre']['native_r_ohm']])['mean']
        predicted=dict(current_a=.1*g,output_v=.1*g*EXPECTED_BOARD['sum_gain'])
        target['mvm_prediction']=predicted
        if not math.isfinite(g) or g<=0 or predicted['current_a']>.005 or predicted['output_v']>.8*EXPECTED_BOARD['vol_ref_adc']:
            self.emit(dict(event='rejected_mvm',cycle_id=self.cycle_id,wl=wl,bl=bl,**key,**predicted))
            raise ValueError('MVM model bound')
        target['mvm']=[]
        for repeat in range(5):
            for position,level in enumerate([0.,.025,.05,.075,.1,0.]):
                row=self.impact(wl,bl,10,[level if i==bl else 0. for i in range(32)],'interval_mvm',
                    repeat=repeat,position=position,nominal_v=level,**key)
                target['mvm'].append({k:row[k] for k in ('task_id','raw_adc','output_voltage_v','repeat','position','nominal_v')})
        target['post']=self.read_summary(self.reads(wl,bl,5,'interval_post',**key),band)
        started=time.monotonic();self.wait(self.plan['delay_s']);elapsed=time.monotonic()-started
        target['delay']=dict(requested_s=self.plan['delay_s'],elapsed_s=elapsed,started_monotonic_s=started,
            delay_met=elapsed>=self.plan['delay_s'])
        self.emit(dict(event='delay',cycle_id=self.cycle_id,wl=wl,bl=bl,**key,**target['delay']))
        target['delayed']=self.read_summary(self.reads(wl,bl,3,'interval_delayed',**key),band)
        target['all_readbacks_inside']=all(target[k]['all_inside'] for k in ('pre','post','delayed'))
        # Same provisional .05/.1 test; no fitted output correction.
        z,l,h=(stats([r['raw_adc'] for r in target['mvm'] if r['position']==p]) for p in (0,2,4))
        sl,sh=l['mean']-z['mean'],h['mean']-z['mean'];dl,dh=math.hypot(l['sd'],z['sd']),math.hypot(h['sd'],z['sd'])
        eps=h['mean']-2*l['mean']+z['mean'];se=math.sqrt((h['sd']**2+4*l['sd']**2+z['sd']**2)/5)
        tolerance=max(.05*abs(sh),3*se,2);cv=dh/sh if sh>0 else None
        target['mvm_screen']=dict(snr_low=sl/max(1,dl),snr_high=sh/max(1,dh),cv_high=cv,
            epsilon_adc=eps,tolerance_adc=tolerance,passes=sl>0 and sh>0 and sl/max(1,dl)>=5 and sh/max(1,dh)>=5
            and cv is not None and cv<=.05 and abs(eps)<=tolerance)
        target['interval_and_mvm_pass']=target['all_readbacks_inside'] and target['mvm_screen']['passes'] and target['delay']['delay_met']

    def run_cycle(self,cell,cycle):
        wl,bl=cell['wl'],cell['bl'];self.cycle_id+=1;self.last_write=None
        record=dict(cycle_id=self.cycle_id,cycle=cycle,wl=wl,bl=bl,status='failed',targets=[])
        self.summary['cycles'].append(record);start=(self.decisions,self.pulses,self.skipped,self.sequence)
        try:
            current=self.reads(wl,bl,1,'cycle_initial')[0]
            for step in range(271):
                if current['q']<=.005:break
                row=self.impact(wl,bl,7,round(.3+.01*step,2),'interval_reset',sign=0,step=step)
                if row is not None:current=row
            record['reset_crossed']=current['q']<=.005
            confirm=self.reads(wl,bl,5,'reset_confirmation')
            record['reset_confirmation']=self.read_summary(confirm)
            record['reset_confirmed']=record['reset_crossed'] and record['reset_confirmation']['q']['mean']<=.005 and confirm[-1]['q']<=.005
            step=0
            for qtarget in cell['targets_q']:
                target=dict(target_q=qtarget,native_r_interval_ohm=bounds(qtarget),status='pending',immediate_hit=False)
                record['targets'].append(target)
                try:
                    if not record['reset_confirmed']:
                        target.update(status='reset_unconfirmed');continue
                    while True:
                        current=self.latest[wl,bl];band=target['native_r_interval_ohm']
                        if self.equal_in(band)([current['native_r']]):
                            origin='programming_response' if current['command_si']['vol'] else 'readback_without_new_write'
                            target.update(status='hit',immediate_hit=True,evidence_task_id=current['task_id'],
                                evidence_native_r_ohm=current['native_r'],evidence_q=current['q'],hit_origin=origin)
                            self.hit_checkpoint(wl,bl,target);break
                        if current['native_r']<band[0]:
                            target.update(status='overshoot',evidence_task_id=current['task_id'],evidence_q=current['q']);break
                        if step>=271:
                            target.update(status='set_ramp_exhausted',evidence_task_id=current['task_id'],evidence_q=current['q']);break
                        self.impact(wl,bl,7,round(.3+.01*step,2),'interval_set',sign=1,step=step,target_q=qtarget)
                        step+=1
                except (Exception,KeyboardInterrupt) as exc:
                    target.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
                finally:self.append('targets.jsonl',dict(cycle_id=self.cycle_id,wl=wl,bl=bl,**target))
            record['final']=self.read_summary(self.reads(wl,bl,3,'cycle_final'));record['status']='completed'
        except (Exception,KeyboardInterrupt) as exc:
            record['error']=f'{type(exc).__name__}: {exc}';raise
        finally:
            record.update(programming_decisions=self.decisions-start[0],programming_pulse_attempts=self.pulses-start[1],
                programming_skipped_current=self.skipped-start[2],impact_attempts=self.sequence-start[3])
            self.append('cycles.jsonl',record)

    def run(self):
        try:
            if not self.connector.open_port():raise ConnectionError('Connector.open_port returned false')
            for cell in self.plan['cells']:
                for cycle in range(1,4):
                    self.run_cycle(cell,cycle)
                    print(f"WL{cell['wl']}/BL{cell['bl']} cycle{cycle}: completed",flush=True)
            self.summary['status']='completed'
        except (Exception,KeyboardInterrupt) as exc:self.summary['error']=f'{type(exc).__name__}: {exc}'
        finally:
            try:
                if not self.connector.close_port():raise ConnectionError('close_port returned false')
            except Exception as exc:self.summary.update(status='failed',close_error=f'{type(exc).__name__}: {exc}')
            self.log.close()
            self.summary.update(impact_attempts=self.sequence,programming_pulse_attempts=self.pulses,
                programming_decisions=self.decisions,programming_skipped_current=self.skipped,finished_utc=now())
            save_json(self.output/'summary.json',self.summary)
        return self.summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    for flag in ('run-simulator','run-hardware','dry-run'):mode.add_argument('--'+flag,action='store_true')
    args=parser.parse_args(argv)
    if not(args.run_simulator or args.run_hardware or args.dry_run):parser.print_help();return 0
    for name,pin in PINS.items():
        if sha256(local_path(name))!=pin:raise ValueError('Pinned dependency changed: '+name)
    spec=json.loads(local_path(CONFIG).read_text(encoding='utf-8'));validate(spec)
    plan=json.loads(local_path(BASE_CONFIG).read_text(encoding='utf-8'));plan.update(spec)
    source='simulator' if args.run_simulator else 'hardware'
    config,snapshot,hashes=check_sources(source)
    for name in (*PINS,CONFIG,'cell_range_repeat.py','experiment.py','native_conversions.py','experiment_hardware.py'):
        hashes[name]=sha256(local_path(name))
    description=dict(source=source,protocol_id=plan['protocol_id'],maximum_impacts=6120,maximum_programming_decisions=4878,
        maximum_programming_pulses=4878,cells=CELLS,cycles_per_cell=3,source_sha256=hashes,
        intervals=[dict(target_q=q,native_r_interval_ohm=bounds(q)) for q in (.03,.035,.04,.045,.05)],
        physical_pulse_width_s=None,hardware_access=source=='hardware' and not args.dry_run)
    if args.dry_run:print(json.dumps(description));return 0
    lock=local_path(LOCK);locked=False
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as stream:stream.write(f'Interval repeat {now()}\n')
        locked=True
    try:
        session_id=str(uuid.uuid4());output=local_path('results/cell_range_repeats')/session_id;output.mkdir(parents=True)
        save_json(output/'protocol.json',plan);(output/'board.ini').write_bytes(snapshot)
        manifest=dict(session_id=session_id,status='failed',started_utc=now(),invocation=launch_provenance(argv),**description)
        save_json(output/'start_manifest.json',manifest)
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
            from manager.terminate import EqualIn
            result=IntervalRun(connector,converters,output,session_id,plan,source,EqualIn).run()
            manifest.update({k:result[k] for k in ('status','impact_attempts','programming_pulse_attempts','programming_decisions','programming_skipped_current')})
            manifest['error']=result.get('error')
        except (Exception,KeyboardInterrupt) as exc:manifest['error']=f'{type(exc).__name__}: {exc}'
        finally:
            manifest['finished_utc']=now();manifest['result_sha256']={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
            save_json(output/'manifest.json',manifest)
        print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'])))
        return 0 if manifest['status']=='completed' else 2
    finally:
        if locked:lock.unlink()


if __name__=='__main__':raise SystemExit(main())
