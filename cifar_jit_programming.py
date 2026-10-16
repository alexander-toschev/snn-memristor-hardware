"""Bounded C9 just-in-time programming diagnostic. No CIFAR acquisition."""
import argparse
import copy
from datetime import datetime, timezone
import json
import logging
import random
import statistics as st
import time
import uuid

import cifar_feedback_diagnostic as base
from cifar_feedback import (ROOT,local_path,sha256,save_json,now,EXPECTED_BOARD,
    check_sources,LOCK,native_connector,launch_provenance,fidelity_report)
from native_conversions import mvm_output

CONFIG='configs/cifar_jit_programming.json'
ROUTE_SESSION='results/cifar_feedback_remaining_reserves/58ca618b-c37f-454e-be25-db21d1d25f60'
ROUTE_TILE=ROUTE_SESSION+'/tiles/f1_p0_reserve_C/tile.json'
PINS={**base.PINS,
    'experiments/unn_rev3_20261014/paths.py':'2c841b4a446a767595c68aad243abd966cee94ffd91f74c3c2823998fdc9ec1c',
    'cifar_feedback_diagnostic.py':'d4bcc3c00d9fc37552af3c2f3b205fa9bb795ce057cafedd93c864ee33069e62',
    'configs/cifar_feedback_diagnostic.json':'c798a58227d25cc5ca7fe8a6f8b3472efb28fe90dde8759d86840e8b42e92246',
    ROUTE_TILE:'7b262ed74cfebaa8a95622a01f7bf20165b9443cd9cbea84785355f71aeb2b20',
    ROUTE_SESSION+'/manifest.json':'5fb5041cbdba148ff0a02182512a98ca0784fed074e2830d6fc06b230f933171'}
SPEC={key:copy.deepcopy(base.SPEC[key]) for key in ('programming_guard','simulator_pulse_width_s',
    'reset_ramp_v','set_ramp_v','feedback_relative_tolerance','feedback_absolute_adc_tolerance',
    'target_gain_v_per_weight','adc_headroom_fraction','raw_part_maximum_bytes','simulator_seed')}
SPEC['programming_guard']['skip_policy']='global_stop_without_impact_or_state_update'
SPEC.update(protocol_id='cifar-C9-jit-programming-v1',cycles=2,pair_order=list(range(9)),
    sentinel_cell=[3,8],retention_delay_s=1.,cutoff_utc='2026-10-16T18:50:00+00:00',
    maximum_impacts=29934,maximum_programming_pulses=3942,maximum_programming_decisions=3942,
    maximum_pair_decisions=18,maximum_impacts_per_decision=1663,reset_signal_limit_adc=2,
    baseline_scope='fresh_positive_reset_then_negative_reset_then_combined_pair_gate',
    route_source=ROUTE_TILE,route_source_sha256=PINS[ROUTE_TILE],
    pair_gate='five fresh paired brackets after both resets; each leg abs(mean)<=2ADC and sampleSD<=2ADC',
    early_reset_eligibility_is_not_the_pair_gate=True,
    continuation='quiet_gate_failure or poor weight quality continues; current_guard, transport, invalid data, budget, deadline stop all',
    program_stop_policy='inherited one bounded residual-selected SET sweep; no reversal/retry',
    cifar_acquisition=False,matrix_fit=False,source_gain_is_not_cross_backend_physical_equivalence=True)


class DeadlineReached(RuntimeError):pass
class CurrentGuardStop(RuntimeError):pass


def utc_now():return datetime.now(timezone.utc)


def load_inputs():
    for name,pin in PINS.items():
        if sha256(local_path(name))!=pin:raise ValueError('Pinned JIT source/input changed: '+name)
    plan=json.loads(local_path(CONFIG).read_text(encoding='utf-8'))
    if plan!=SPEC:raise ValueError('JIT protocol differs from reviewed config')
    tile=json.loads(local_path(ROUTE_TILE).read_text(encoding='utf-8'))
    manifest=json.loads(local_path(ROUTE_SESSION+'/manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed' or manifest['source']!='hardware' or manifest['result_sha256']['tiles/f1_p0_reserve_C/tile.json']!=PINS[ROUTE_TILE]:
        raise ValueError('Route source manifest mismatch')
    route=copy.deepcopy(tile['effective_route'])
    if not tile['technical_completed'] or [s['tap_index'] for s in route]!=list(range(9)):
        raise ValueError('Incomplete exact C9 route')
    addresses=[tuple(s[side]) for s in route for side in ('positive_cell','negative_cell')]
    if len(set(addresses))!=18 or any(s['padding'] for s in route):raise ValueError('Invalid C9 addresses')
    return plan,route


class JITRun(base.DiagnosticRun):
    def __init__(self,*args,clock=None,monotonic=None,sleep=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.clock=clock or utc_now;self.monotonic=monotonic or time.monotonic;self.sleep=sleep or time.sleep
        self.cutoff=datetime.fromisoformat(self.plan['cutoff_utc'])
        self.monotonic_cutoff=self.monotonic()+max(0.,(self.cutoff-self.clock()).total_seconds())
        self.deadline_recorded=False
        self.pair_start_counts=None

    def check_deadline(self,**meta):
        if self.clock()>=self.cutoff or self.monotonic()>=self.monotonic_cutoff:
            if not self.deadline_recorded:
                self.emit(dict(event='deadline_stop',session_id=self.session_id,source=self.source,
                    timestamp_utc=self.clock().isoformat(),cutoff_utc=self.plan['cutoff_utc'],
                    next_task_id=self.sequence+1,impact_issued=False,**meta))
                self.deadline_recorded=True
            raise DeadlineReached('Fixed UTC/monotonic acquisition cutoff reached')

    def impact(self,wl,bl,mode,vol,phase,sign=0,**meta):
        self.check_deadline(phase=phase,wl=wl,bl=bl)
        if self.pair_start_counts is not None:
            impacts,decisions=self.pair_start_counts
            if self.sequence-impacts>=1663:raise ValueError('Per-pair impact budget exhausted')
            if mode==7 and vol and self.decisions-decisions>=219:raise ValueError('Per-pair programming decision budget exhausted')
        row=super().impact(wl,bl,mode,vol,phase,sign,**meta)
        if row is None:raise CurrentGuardStop('Inherited current guard refused a programming command')
        return row

    def sentinel(self,phase,meta):
        cell=self.plan['sentinel_cell']
        guard=self.read(cell,1,phase+'_guard',**meta)[0]
        result=dict(**meta,phase=phase,cell=cell,guard_task_id=guard['task_id'],
            measurement=self.leg_triple(cell,phase,**meta))
        self.append('sentinel.jsonl',result)
        return result

    def paired_snapshot(self,slot,phase,meta):
        guards=[self.read(slot[side],1,phase+'_guard',guard_side=side,**meta)[0]['task_id']
                for side in ('positive_cell','negative_cell')]
        samples=[]
        for repeat in range(5):
            values,rows=self.block([slot],[slot],phase,dict(**meta,bracket_repeat=repeat))
            samples.append(dict(row_sums_v=values,signed_voltage_v=values[0]-values[1],measured_rows=rows))
        limit=2*mvm_output(1,EXPECTED_BOARD,self.converters)
        means=[st.mean(s['row_sums_v'][i] for s in samples) for i in range(2)]
        sds=[st.stdev(s['row_sums_v'][i] for s in samples) for i in range(2)]
        result=dict(**meta,phase=phase,tap=slot['tap_index'],positive_cell=slot['positive_cell'],negative_cell=slot['negative_cell'],
            guard_task_ids=guards,samples=samples,row_mean_v=means,row_sd_v=sds,
            signed_mean_v=st.mean(s['signed_voltage_v'] for s in samples),
            signed_sd_v=st.stdev(s['signed_voltage_v'] for s in samples),quiet_limit_v=limit,
            both_legs_quiet=all(abs(mean)<=limit and sd<=limit for mean,sd in zip(means,sds)))
        self.append('paired_snapshots.jsonl',result)
        return result

    def retention(self,slot,meta):
        self.check_deadline(phase='retention_wait')
        started=self.monotonic();requested=self.plan['retention_delay_s']
        if started+requested>=self.monotonic_cutoff or (self.cutoff-self.clock()).total_seconds()<=requested:
            self.emit(dict(event='retention_not_started',session_id=self.session_id,**meta,
                requested_delay_s=requested,reason='insufficient_time_before_fixed_cutoff'))
            raise DeadlineReached('Retention cannot finish before cutoff')
        self.sleep(requested);elapsed=self.monotonic()-started
        self.emit(dict(event='retention_wait',session_id=self.session_id,**meta,
            requested_delay_s=requested,started_monotonic_s=started,finished_monotonic_s=self.monotonic(),
            elapsed_monotonic_s=elapsed,delay_met=elapsed>=requested))
        if elapsed<requested:raise RuntimeError('Requested retention delay was not met')
        return self.paired_snapshot(slot,'retention',dict(**meta,requested_delay_s=requested,elapsed_monotonic_s=elapsed))

    def disturbance_map(self,route,meta):
        observed=[self.paired_snapshot(slot,'after_decision_map',dict(**meta,map_pair_index=i)) for i,slot in enumerate(route)]
        repeats=[[observed[i]['samples'][k]['signed_voltage_v'] for i in range(9)] for k in range(5)]
        quality=fidelity_report([s['trained_weight'] for s in route],repeats,
            self.plan['target_gain_v_per_weight'][self.source],mvm_output(1,EXPECTED_BOARD,self.converters))
        record=dict(**meta,observed=observed,physical_fidelity=quality,
            scope='all current C9 states after this decision; later or gate-failed targets may remain unprogrammed')
        self.append('disturbance_maps.jsonl',record)
        return dict(phase='after_decision_map',cycle=meta['cycle'],pair_index=meta['pair_index'],physical_fidelity=quality)

    def collect(self,route):
        summary=dict(status='failed',technical_completed=False,cifar_acquisition=False,
            final_state_fidelity_pass=False,programming_fidelity_pass=False,decisions=[],route=route)
        try:
            for cycle in range(2):
                for index,slot in enumerate(route):
                    meta=dict(cycle=cycle,pair_index=index,pair_tap=slot['tap_index'],filter_id=1)
                    decision=dict(**meta,status='incomplete');summary['decisions'].append(decision)
                    begin=(self.sequence,self.pulses)
                    self.pair_start_counts=(self.sequence,self.decisions)
                    try:
                        decision['positive_reset']=self.reset_candidate(slot['positive_cell'],dict(**meta,reset_leg='positive'))
                        decision['sentinel_after_positive_reset']=self.sentinel('sentinel_after_positive_reset',meta)
                        decision['negative_reset']=self.reset_candidate(slot['negative_cell'],dict(**meta,reset_leg='negative'))
                        decision['sentinel_after_negative_reset']=self.sentinel('sentinel_after_negative_reset',meta)
                        gate=self.paired_snapshot(slot,'fresh_pair_gate',meta);decision['fresh_pair_gate']=gate
                        if gate['both_legs_quiet']:
                            decision['program']=self.program_pair(slot,meta)
                            decision['sentinel_after_program']=self.sentinel('sentinel_after_program',meta)
                            decision['outcome']='programmed_diagnostic'
                        else:
                            decision['outcome']='fresh_quiet_gate_failed_no_SET'
                            decision['no_SET_confirmation']=self.paired_snapshot(slot,'no_SET_confirmation',meta)
                        decision['retention']=self.retention(slot,meta)
                        decision['disturbance_map']=self.disturbance_map(route,meta)
                        decision['status']='completed'
                    finally:
                        decision.update(impact_attempts=self.sequence-begin[0],programming_pulse_attempts=self.pulses-begin[1])
                        self.append('decisions.jsonl',decision)
                    if self.sequence-begin[0]>1663 or self.pulses-begin[1]>219:
                        raise ValueError('Per-pair bounded accounting mismatch')
            final=[d['disturbance_map']['physical_fidelity'] for d in summary['decisions'] if d['pair_index']==8]
            summary.update(status='completed',technical_completed=True,cycle_final_physical_fidelity=final,
                final_state_fidelity_pass=all(q['physical_fidelity_pass'] for q in final),
                programming_fidelity_pass=all(q['physical_fidelity_pass'] for q in final)
                    and all('program' in d and d['fresh_pair_gate']['both_legs_quiet'] for d in summary['decisions']),
                all_fresh_gates_passed=all(d['fresh_pair_gate']['both_legs_quiet'] for d in summary['decisions']),
                programming_repeatability_qualified=False,
                quiet_gate_failures=sum(d['outcome']=='fresh_quiet_gate_failed_no_SET' for d in summary['decisions']))
        except (Exception,KeyboardInterrupt) as exc:
            summary['error']=f'{type(exc).__name__}: {exc}';raise
        finally:
            summary.update(impact_attempts=self.sequence,programming_pulse_attempts=self.pulses,
                programming_decisions=self.decisions,programming_skipped_current=self.skipped)
            self.log.close();save_json(self.output/'summary.json',summary)
        return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    for flag in ('run-simulator','run-hardware','dry-run'):mode.add_argument('--'+flag,action='store_true')
    args=parser.parse_args(argv)
    if not(args.run_simulator or args.run_hardware or args.dry_run):parser.print_help();return 0
    plan,route=load_inputs();source='simulator' if args.run_simulator else 'hardware'
    config,snapshot,hashes=check_sources(source)
    for name in (*PINS,CONFIG,'cifar_jit_programming.py','experiment.py'):hashes[name]=sha256(local_path(name))
    description=dict(protocol_id=plan['protocol_id'],source=source,source_sha256=hashes,
        hardware_access=source=='hardware' and not args.dry_run,maximum_impacts=29934,
        maximum_programming_pulses=3942,maximum_programming_decisions=3942,pair_decisions=18,
        cutoff_utc=plan['cutoff_utc'],cutoff_already_reached=utc_now()>=datetime.fromisoformat(plan['cutoff_utc']),
        route_source=ROUTE_TILE,route_source_sha256=PINS[ROUTE_TILE],physical_pulse_width_s=None,
        cifar_acquisition=False,fresh_native_simulator_array=True)
    if args.dry_run:print(json.dumps(description));return 0
    if description['cutoff_already_reached']:raise DeadlineReached('Do not open Connector after cutoff')
    locked=False;lock=local_path(LOCK)
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as stream:stream.write('C9 JIT '+now()+'\n')
        locked=True
    connector=manifest=output=run=None
    try:
        sid=str(uuid.uuid4());output=local_path('results/cifar_jit_programming')/sid;output.mkdir(parents=True)
        save_json(output/'protocol.json',plan);save_json(output/'route.json',route);(output/'board.ini').write_bytes(snapshot)
        manifest=dict(session_id=sid,status='failed',started_utc=now(),invocation=launch_provenance(argv),**description)
        save_json(output/'start_manifest.json',manifest)
        if utc_now()>=datetime.fromisoformat(plan['cutoff_utc']):raise DeadlineReached('Cutoff before backend initialization')
        if source=='hardware':connector=native_connector(config)
        else:
            from experiments.unn_rev3_20261014.paths import use_local_dependencies
            use_local_dependencies()
            import numpy as np
            from manager.board import Connector
            from simulator.src import create_crossbar_array
            random.seed(plan['simulator_seed']);np.random.seed(plan['simulator_seed'])
            serial=output/'array';create_crossbar_array(str(serial),32,8)
            manifest['array_initial_sha256']=sha256(serial.with_suffix('.cb'))
            connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                board_type='rp5_fpga_python',task_format='SI',config=config,crossbar_serial=str(serial))
        from manager.service import converters
        if utc_now()>=datetime.fromisoformat(plan['cutoff_utc']):raise DeadlineReached('Cutoff before open_port')
        if not connector.open_port():raise ConnectionError('Connector.open_port returned false')
        run=JITRun(connector,converters,output,sid,plan,source)
        summary=run.collect(route)
        manifest.update(status='completed',technical_completed=summary['technical_completed'],
            final_state_fidelity_pass=summary['final_state_fidelity_pass'],programming_fidelity_pass=summary['programming_fidelity_pass'])
    except (Exception,KeyboardInterrupt) as exc:
        if manifest is None:raise
        manifest['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if run is not None:run.log.close()
        if connector is not None:
            try:
                if not connector.close_port():raise ConnectionError('Connector.close_port returned false')
            except Exception as exc:
                if manifest is not None:manifest.update(status='failed',close_error=str(exc))
        if manifest is not None:
            manifest.update(finished_utc=now(),impact_attempts=run.sequence if run else 0,
                programming_pulse_attempts=run.pulses if run else 0,programming_decisions=run.decisions if run else 0,
                result_sha256={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file()})
            save_json(output/'manifest.json',manifest)
        if locked:lock.unlink()
    print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'])))
    return 0 if manifest['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
