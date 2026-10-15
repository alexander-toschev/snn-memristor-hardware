"""Reviewed refined stop bands; unchanged outer qualification and native Connector safeguards."""
import argparse
import json
import logging
import random
import uuid

from cell_range_repeat import IntervalRun, PINS as BASE_PINS, BASE_CONFIG
from cell_calibration import check_sources, SIM_INI, LOCK
from read_all import ROOT, EXPECTED_BOARD, VENDOR, local_path, native_connector, now, sha256, save_json
from experiment_hardware import launch_provenance

CONFIG='configs/cell_range_refined.json'
PINS={**BASE_PINS, 'cell_range_repeat.py':'f03860a42755337fcf1a5f71a0edc291bbc8308684a358d7dbbcc386de5b7161'}
CELLS=[dict(wl=wl,bl=8,targets_q=qs) for wl,qs in ((0,[.04,.045,.05]),(3,[.04,.045,.05]),(5,[.031,.032]))]


def stop_tolerance(q):
    return .01 if q in (.045,.05) else .0075


def bounds(q,tolerance=.01):
    return [EXPECTED_BOARD['sum_gain']/(q*(1+tolerance)),EXPECTED_BOARD['sum_gain']/(q*(1-tolerance))]


def target_fields(q):
    tolerance=stop_tolerance(q)
    return dict(target_q=q,native_r_interval_ohm=bounds(q),stop_native_r_interval_ohm=bounds(q,tolerance),
                stop_q_relative_tolerance=tolerance)


def validate(plan):
    expected=dict(protocol_id='three-cell-native-R-interval-v2-refined',cells=CELLS,cycles_per_cell=3,q_relative_tolerance=.01,
        stop_q_relative_tolerance=.0075,control_targets_q=[.045,.05],control_stop_q_relative_tolerance=.01,
        reset_endpoint_q=.005,ramp=dict(start_v=.3,step_v=.01,stop_v=3.,maximum_steps=271),initial_reads=1,
        reset_confirmation_reads=5,pre_reads=5,post_reads=5,delayed_reads=3,delay_s=1.,final_reads=3,
        maximum_programming_decisions=4878,maximum_programming_pulses=4878,maximum_impacts=5991,
        baseline_scope='per_cell_cycle_before_reset')
    if set(plan)!=set(expected)|{'qualification'} or any(plan.get(k)!=v for k,v in expected.items()):
        raise ValueError('Unreviewed refined protocol; operational overrides are forbidden')


class RefinedRun(IntervalRun):
    # Only the bounded target loop differs. All I/O, guards, readback qualification,
    # MVM, delay, error persistence and connection lifecycle remain inherited.
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
                target=dict(**target_fields(qtarget),status='pending',immediate_hit=False)
                record['targets'].append(target)
                try:
                    if not record['reset_confirmed']:
                        target.update(status='reset_unconfirmed');continue
                    while True:
                        current=self.latest[wl,bl];band=target['stop_native_r_interval_ohm']
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
    for name in (*PINS,CONFIG,'cell_range_refined.py','experiment.py','native_conversions.py','experiment_hardware.py'):
        hashes[name]=sha256(local_path(name))
    description=dict(source=source,protocol_id=plan['protocol_id'],maximum_impacts=5991,maximum_programming_decisions=4878,
        maximum_programming_pulses=4878,cells=CELLS,cycles_per_cell=3,source_sha256=hashes,
        intervals=[target_fields(q) for q in (.031,.032,.04,.045,.05)],
        physical_pulse_width_s=None,hardware_access=source=='hardware' and not args.dry_run)
    if args.dry_run:print(json.dumps(description));return 0
    lock=local_path(LOCK);locked=False
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as stream:stream.write(f'Refined interval repeat {now()}\n')
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
            result=RefinedRun(connector,converters,output,session_id,plan,source,EqualIn).run()
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
