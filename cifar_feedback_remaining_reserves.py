"""Bounded B/C acquisition with immutable audited A parents and fixed reserves."""
import argparse
import copy
import json
import logging
import random
import statistics as st
import uuid

import cifar_feedback_adaptive as adaptive
import cifar_feedback_reserve as reserve_A
from cifar_feedback_adaptive import (AdaptiveRun,slots,ROOT,local_path,sha256,save_json,now,
    EXPECTED_BOARD,mvm_output,fixed_calibration,fidelity_report,apply_matrix,decode_measured,
    address_coverage,check_sources,LOCK,native_connector,launch_provenance)
from reserve_remaining_plan import BLOCKS,PLAN,candidate_pools,select_route,require_parent_bindings

CONFIG='configs/cifar_feedback_remaining_reserves.json'
PINS={**reserve_A.PINS,
    'cifar_feedback_reserve.py':'a4e3a822ede1460c905435000912dc91b851b26c8d0608006ceba12708d3f7c9',
    'configs/cifar_feedback_reserve.json':'d042e27faf56a0334f7669689255ea45ad1d43cface8a3fc93ffdf42f4415b9c'}
SPEC={**reserve_A.SPEC,**PLAN,'collection_scope':'280immutable_plus20fresh_BC; no old BC fallback',
    'maximum_programming_pulses':5260,'maximum_programming_decisions':5260,
    'cifar_collection_on_fidelity_failure':'both_fixed_BC_blocks; scientific_capacity_failure_skips_only_its_block; all_other_errors_stop_all',
    'maximum_new_microtiles':2}
for _key in ('target_filter_id','target_parent_tile_id','target_taps','reserved_negative_cell','reserved_tap',
             'positive_priority','candidate_pool_scope','selection','route_gate'):
    SPEC.pop(_key,None)


class RouteCapacityError(ValueError):
    """Only this scientific failure permits the next independent fixed block."""


def preferred_dispositions(block,result):
    output=[];complete=result.get('technical_completed',False)
    candidates={tuple(c['cell']):c for c in result['reset_candidates']}
    route={tuple(s[side]) for s in result.get('effective_route',[]) for side in ('positive_cell','negative_cell')}
    for side in ('positive','negative'):
        for tap,cell in block['preferred_'+side].items():
            candidate=candidates.get(tuple(cell));assigned=tuple(cell) in route
            if candidate is None:status='technical_unassessed'
            elif not candidate['eligible']:status='screened_not_quiet'
            elif result.get('stop_reason')=='whole_route_capacity_failed_before_SET':status='quiet_but_route_capacity_abort'
            elif assigned:status='quiet_actual_CIFAR_assigned' if complete else 'quiet_selected_acquisition_incomplete'
            elif route:status='quiet_not_selected'
            else:status='quiet_route_unassessed'
            attempted=any(p['programming_pulse_attempts']>0 and p[p['active_set_side']]==cell for p in result['pairs'])
            output.append(dict(cell=cell,tap=tap,status=status,quiet=candidate['eligible'] if candidate else None,
                screening_complete=candidate is not None,route_assigned=assigned,
                cifar_complete=complete,technical_stop=result.get('error'),actual_cifar_assigned=complete and assigned,
                SET_attempted=attempted if complete else False if result.get('stop_reason')=='whole_route_capacity_failed_before_SET' else None))
    return output


class RemainingReserveRun(AdaptiveRun):
    def collect_block(self,f,parent,block,fixture):
        pending=slots(parent);micro=0
        active=copy.deepcopy(pending);fid=f['filter_id'];tid=2*parent['tile_id']+micro
        meta=dict(filter_id=fid,tile_id=tid,parent_tile_id=parent['tile_id'],microtile_id=micro,reserve_block=block['name'])
        result=dict(**meta,status='failed',programming=[],partials=[],reset_candidates=[],pairs=[],
            pending_taps_before=[s['tap_index'] for s in pending],original_chunk_taps=[s['tap_index'] for s in slots(parent)])
        try:
            pools=candidate_pools(block);selected={}
            for side,cells in pools.items():
                qualifying=[]
                for index,cell in enumerate(cells):
                    candidate=self.reset_candidate(cell,dict(**meta,side=side,candidate_index=index))
                    result['reset_candidates'].append(dict(side=side,candidate_index=index,**candidate))
                    if candidate['eligible']:qualifying.append(cell)
                selected[side]=qualifying
            decision=select_route(block,selected)
            result['route_decision']=decision
            if not decision['capacity_pass']:
                result['stop_reason']='whole_route_capacity_failed_before_SET'
                raise RouteCapacityError(result['stop_reason'])
            selected=decision['route']
            result['assigned_taps']=[s['tap_index'] for s in active]
            for index,slot in enumerate(active):
                for side in ('positive_cell','negative_cell'):slot[side]=selected[side][index]
            result['effective_route']=active
            save_json(self.output/'frozen_route.json',dict(candidate_pools=pools,slots=active,
                allocation_before_fill=decision['allocation_before_fill'],preferred_disposition=decision['preferred_disposition'],
                mapping_uses_trained_weight_values=False,frozen_before_task_id=self.sequence+1))
            self.emit(dict(event='route_frozen',session_id=self.session_id,**meta,
                sha256=sha256(self.output/'frozen_route.json'),before_task_id=self.sequence+1))
            for slot in active:
                pair=self.program_pair(slot,dict(**meta,tap=slot['tap_index']))
                result['pairs'].append(pair)
                for side in ('positive_cell','negative_cell'):
                    result['programming'].append(dict(cell=slot[side],status='completed',
                        tap=slot['tap_index'],side=side,feedback_pair_stop=pair['stop'],
                        selected_SET_side=side==pair['active_set_side'],
                        participated_in_SET=side==pair['active_set_side'] and pair['programming_pulse_attempts']>0,reset_eligible=True))
            wmax=max(abs(s['trained_weight']) for original in f['tiles'] for s in slots(original))
            result['pre_readback']=[self.read(s[side],3,'tile_pre_readback',tap=s['tap_index'],side=side,**meta)
                for s in active for side in ('positive_cell','negative_cell')]
            anchors=self.anchors(active,'calibration',meta);save_json(self.output/'calibration_anchors.json',anchors)
            correction=self.matrix_calibration(active,anchors,meta);result['matrix']=correction
            differences=[]
            for row in anchors:
                corrected=[apply_matrix(correction['matrix'],[row[j]['singles'][i]['voltage_v']-row[j]['zero']['voltage_v'] for j in range(2)]) for i in range(len(active))]
                differences.append([pair[0]-pair[1] for pair in corrected])
            cal=fixed_calibration([s['trained_weight'] for s in active],differences,self.plan['target_gain_v_per_weight'][self.source],mvm_output(5,EXPECTED_BOARD,self.converters))
            raw_differences=[[row[0]['singles'][i]['voltage_v']-row[0]['zero']['voltage_v']-row[1]['singles'][i]['voltage_v']+row[1]['zero']['voltage_v'] for i in range(len(active))] for row in anchors]
            result['physical_fidelity']=fidelity_report([s['trained_weight'] for s in active],raw_differences,self.plan['target_gain_v_per_weight'][self.source],mvm_output(1,EXPECTED_BOARD,self.converters))
            cal.update(frozen_before_task_id=self.sequence+1,trained_weights=[s['trained_weight'] for s in active],wmax_per_filter=wmax)
            save_json(self.output/'frozen_calibration.json',cal);result['calibration']=cal
            self.emit(dict(event='calibration_frozen',session_id=self.session_id,**meta,
                sha256=sha256(self.output/'frozen_calibration.json'),matrix_sha256=sha256(self.output/'frozen_matrix.json'),before_task_id=self.sequence+1))
            for ci,case in enumerate(fixture['cases']):
                for repeat in range(5):
                    for timestep,x in enumerate(case['spikes_t12x75']):
                        chosen=[s for s in active if x[s['tap_index']]]
                        event=dict(**meta,case=ci,original_test_index=case['original_test_index'],repeat=repeat,t=timestep)
                        row_sums,ids=self.block(active,chosen,'cifar',event)
                        corrected=apply_matrix(correction['matrix'],row_sums);difference=corrected[0]-corrected[1]
                        part=dict(**event,active_taps=[s['tap_index'] for s in chosen],measured_rows=ids,signed_voltage_v=difference,
                            raw_row_sums_v=row_sums,corrected_row_sums_v=corrected,raw_signed_voltage_v=row_sums[0]-row_sums[1],
                            trained_unit_contribution=decode_measured(correction['matrix'],row_sums,cal['decoder_gain_v_per_weight']))
                        result['partials'].append(part);self.append('partials.jsonl',part)
            result['final_anchors']=self.anchors(active,'final_anchors',meta)
            final_differences=[[row[0]['singles'][i]['voltage_v']-row[0]['zero']['voltage_v']-row[1]['singles'][i]['voltage_v']+row[1]['zero']['voltage_v'] for i in range(len(active))] for row in result['final_anchors']]
            result['final_physical_fidelity']=fidelity_report([s['trained_weight'] for s in active],final_differences,self.plan['target_gain_v_per_weight'][self.source],mvm_output(1,EXPECTED_BOARD,self.converters))
            result['physical_weight_quality_pass']=result['physical_fidelity']['physical_fidelity_pass'] and result['final_physical_fidelity']['physical_fidelity_pass']
            result['post_readback']=[self.read(s[side],3,'tile_post_readback',tap=s['tap_index'],side=side,**meta)
                for s in active for side in ('positive_cell','negative_cell')]
            result.update(status='completed',technical_completed=True,decode_usable=cal['decode_usable'])
        except (Exception,KeyboardInterrupt) as exc:
            result.update(technical_completed=False,error=f'{type(exc).__name__}: {exc}');raise
        finally:
            result.update(impact_attempts=self.sequence,programming_pulse_attempts=self.pulses,
                programming_decisions=self.decisions,programming_skipped_current=self.skipped)
            result['preferred_disposition']=preferred_dispositions(block,result)
            self.log.close();save_json(self.output/'tile.json',result)
        return result


def _bound_evidence(binding,source,complete):
    directory=local_path(binding['session']);mp=directory/'manifest.json';ap=local_path(binding['audit_path'])
    if sha256(mp)!=binding['manifest_sha256'] or sha256(ap)!=binding['audit_sha256']:
        raise ValueError('Bound parent/evidence SHA mismatch')
    manifest=json.loads(mp.read_text(encoding='utf-8'));audit=json.loads(ap.read_text(encoding='utf-8'))
    if (manifest['source']!=source or audit['source']!=source or manifest['session_id']!=audit['session_id']
            or audit['manifest_sha256']!=binding['manifest_sha256'] or audit['audit_status']!='passed'
            or audit['errors'] or bool(audit['acquisition_complete'])!=complete
            or (manifest['status']=='completed')!=complete):raise ValueError('Parent/evidence audit identity/status mismatch')
    if complete and audit['full_time_outputs']!=2400:raise ValueError('Incomplete parent decoded coverage')
    if audit['new_counters']!=manifest['new_counters']:raise ValueError('Parent/evidence counters mismatch')
    for name,pin in manifest['result_sha256'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path)!=pin:raise ValueError('Immutable ancestor bytes changed: '+name)
    for name,pin in PINS.items():
        if name in manifest['source_sha256'] and manifest['source_sha256'][name]!=pin:raise ValueError('Ancestor scientific source changed: '+name)
    return directory,manifest,audit


def historical_accounting(parent_session,parent,failed_A=None):
    key='cumulative_impacts_including_all_ancestry' if 'cumulative_impacts_including_all_ancestry' in parent else 'cumulative_impacts_including_all_parent_attempts'
    pulse_key=key.replace('impacts','pulses')
    impacts=parent[key];pulses=parent[pulse_key]
    ledger=[dict(session=parent_session,impact_attempts=impacts,programming_pulse_attempts=pulses,
                 accounting_scope='parent_with_all_ancestry_already_included')]
    if failed_A is not None:
        session,manifest=failed_A
        if session==parent_session or manifest['status']=='completed' or manifest['parent_import']['session']!=parent_session:
            raise ValueError('Failed-A evidence is not a distinct failed child of this baseline')
        # Do not add failed-A cumulative totals: those already include this same baseline.
        counts=manifest['new_counters'];impacts+=counts['impact_attempts'];pulses+=counts['programming_pulse_attempts']
        ledger.append(dict(session=session,impact_attempts=counts['impact_attempts'],
            programming_pulse_attempts=counts['programming_pulse_attempts'],accounting_scope='failed_A_new_attempts_only'))
    return dict(parent_cumulative_impacts=impacts,parent_cumulative_pulses=pulses,historical_attempt_ledger=ledger)


def import_other280(wrappers,parent_session,result_sha):
    imported=[];removed=[]
    for index,original in enumerate(wrappers):
        fid,pid=original['filter_id'],original['parent_tile_id']
        if (fid,pid) in ((2,0),(1,0)):
            removed.extend((fid,s['tap_index']) for s in original['result']['effective_route']);continue
        imported.append({**original,'origin':dict(session=parent_session,path='completed_microtiles.json',
            wrapper_index=index,sha256=result_sha,reused=True,new_measurement=False,
            acquisition_tile_id=original['result']['tile_id'],ancestry_original_origin=original['origin'])})
    expected=[(2,t) for t in range(11)]+[(1,t) for t in range(9)]
    if sorted(removed)!=sorted(expected) or sum(len(c['result']['effective_route']) for c in imported)!=280:
        raise ValueError('Must exclude exactly B11+C9 from complete parent')
    return imported


def verify_parent(source,routing,fixture,plan):
    bindings=require_parent_bindings(plan);binding=bindings[source]
    directory,parent,audit=_bound_evidence(binding,source,True)
    failed=None
    if binding['choice']=='completed_A':
        if binding.get('failed_A'):raise ValueError('Completed-A choice cannot add failed-A attempts')
        if parent['protocol_id']!=reserve_A.SPEC['protocol_id']:raise ValueError('Expected completed reserve-A parent')
        _,proof=reserve_A.verify_parent(parent['parent_import']['session'],source,routing,fixture,reserve_A.SPEC)
        if parent['parent_import']!=proof:raise ValueError('Completed-A baseline ancestry changed')
    else:
        if parent['protocol_id']!=adaptive.SPEC['protocol_id']:raise ValueError('Expected baseline parent')
        _,proof=reserve_A.verify_parent(binding['session'],source,routing,fixture,reserve_A.SPEC)
        failed_binding=binding['failed_A'];_,failed_manifest,_=_bound_evidence(failed_binding,source,False)
        if failed_manifest['protocol_id']!=reserve_A.SPEC['protocol_id'] or failed_manifest['parent_import']!=proof:
            raise ValueError('Failed-A ancestry is not the same bound baseline')
        failed=(failed_binding['session'],failed_manifest)
    wrappers=json.loads((directory/'completed_microtiles.json').read_text(encoding='utf-8'))
    adaptive.aggregate_microtiles(wrappers,routing,fixture)
    chunks=import_other280(wrappers,binding['session'],parent['result_sha256']['completed_microtiles.json'])
    metadata={**binding,'result_sha256':parent['result_sha256'],'imported_trained_coefficients':280,
        'excluded_blocks':[dict(filter_id=2,parent_tile_id=0,taps=list(range(11))),dict(filter_id=1,parent_tile_id=0,taps=list(range(9)))],
        'verified_ancestry':proof,**historical_accounting(binding['session'],parent,failed)}
    return chunks,metadata


def load_inputs():
    require_parent_bindings(SPEC) # reject disabled or unbound plans before any backend work
    for name,pin in PINS.items():
        if sha256(local_path(name))!=pin:raise ValueError('Frozen B/C dependency changed: '+name)
    _,routing,fixture=adaptive.load_inputs()
    plan=json.loads(local_path(CONFIG).read_text(encoding='utf-8'))
    if plan!=json.loads(json.dumps(SPEC)):raise ValueError('B/C protocol differs from reviewed snapshot')
    require_parent_bindings(plan) # unknown parent cannot reach backend selection
    for block in BLOCKS:
        parent=routing['filters'][block['filter_id']]['tiles'][block['parent_tile_id']]
        active=slots(parent);count=sum(map(len,candidate_pools(block).values()))
        if [s['tap_index'] for s in active]!=block['taps']:raise ValueError('B/C trained taps changed')
        if count*244+len(active)*811+adaptive.measurement_tail(active,fixture)!=block['maximum_impacts']:
            raise ValueError('B/C impact bounds changed')
        if count*55+len(active)*109!=block['maximum_writes']:raise ValueError('B/C pulse bounds changed')
    return plan,routing,fixture


def collect_blocks(connector,converters,output,sid,plan,source,routing,fixture,chunks,totals):
    capacity_failures=[]
    for block in BLOCKS:
        f=routing['filters'][block['filter_id']];parent=f['tiles'][block['parent_tile_id']]
        folder=output/'tiles'/f"f{f['filter_id']}_p0_reserve_{block['name']}";folder.mkdir(parents=True)
        per={**plan,'maximum_impacts':min(block['maximum_impacts'],40378-totals['impact_attempts']),
            'maximum_programming_pulses':min(block['maximum_writes'],5260-totals['programming_pulse_attempts']),
            'maximum_programming_decisions':min(block['maximum_writes'],5260-totals['programming_decisions'])}
        run=RemainingReserveRun(connector,converters,folder,sid,per,source)
        try:result=run.collect_block(f,parent,block,fixture)
        except RouteCapacityError as exc:
            capacity_failures.append(dict(block=block['name'],reason=str(exc)))
            continue
        finally:
            run.log.close()
            for key,value in [('impact_attempts',run.sequence),('programming_pulse_attempts',run.pulses),
                              ('programming_decisions',run.decisions),('programming_skipped_current',run.skipped)]:totals[key]+=value
        chunks.append(dict(filter_id=f['filter_id'],parent_tile_id=0,microtile_id=0,result=result,
            origin=dict(session=output.relative_to(ROOT).as_posix(),path=(folder/'tile.json').relative_to(output).as_posix(),
                sha256=sha256(folder/'tile.json'),reused=False,new_measurement=True,acquisition_tile_id=0)))
        checkpoint=output/'completed_microtiles.next.json';save_json(checkpoint,chunks);checkpoint.replace(output/'completed_microtiles.json')
        print(f"reserve {block['name']}: {len(result['effective_route'])} coefficients measured",flush=True)
    if capacity_failures:
        save_json(output/'capacity_failures.json',capacity_failures)
        raise RouteCapacityError('incomplete B/C acquisition: '+','.join(item['block'] for item in capacity_failures))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    for flag in ('run-simulator','run-hardware','dry-run'):mode.add_argument('--'+flag,action='store_true')
    args=parser.parse_args(argv)
    if not(args.run_simulator or args.run_hardware or args.dry_run):parser.print_help();return 0
    source='simulator' if args.run_simulator else 'hardware';plan,routing,fixture=load_inputs()
    chunks,parent_import=verify_parent(source,routing,fixture,plan)
    config,snapshot,hashes=check_sources(source)
    for name in (*PINS,CONFIG,'reserve_remaining_plan.py','cifar_feedback_remaining_reserves.py','experiment.py'):
        hashes[name]=sha256(local_path(name))
    description=dict(protocol_id=plan['protocol_id'],source=source,source_sha256=hashes,parent_import=parent_import,
        hardware_access=source=='hardware' and not args.dry_run,maximum_impacts=40378,maximum_programming_pulses=5260,
        maximum_programming_decisions=5260,new_trained_coefficients=20,imported_trained_coefficients=280,
        physical_pulse_width_s=None,fixed_decoder_gain_v_per_weight=plan['target_gain_v_per_weight'][source],
        simulator_state_policy='fresh native array; imported280 remain historical; no continuous-state equivalence')
    if args.dry_run:print(json.dumps(description));return 0
    lock=local_path(LOCK);locked=False
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as stream:stream.write(f'CIFAR reserves BC {now()}\n')
        locked=True
    connector=manifest=output=None
    totals=dict(impact_attempts=0,programming_pulse_attempts=0,programming_decisions=0,programming_skipped_current=0)
    try:
        sid=str(uuid.uuid4());output=local_path('results/cifar_feedback_remaining_reserves')/sid;output.mkdir(parents=True)
        save_json(output/'protocol.json',plan);(output/'board.ini').write_bytes(snapshot);save_json(output/'imported_microtiles.json',chunks)
        manifest=dict(session_id=sid,status='failed',started_utc=now(),invocation=launch_provenance(argv),**description)
        save_json(output/'start_manifest.json',manifest)
        if source=='hardware':connector=native_connector(config)
        else:
            from experiments.unn_rev3_20261014.paths import use_local_dependencies
            use_local_dependencies()
            import numpy as np
            from manager.board import Connector
            from simulator.src import create_crossbar_array
            random.seed(plan['simulator_seed']);np.random.seed(plan['simulator_seed'])
            serial=output/'array';create_crossbar_array(str(serial),32,8);manifest['array_initial_sha256']=sha256(serial.with_suffix('.cb'))
            connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                board_type='rp5_fpga_python',task_format='SI',config=config,crossbar_serial=str(serial))
        from manager.service import converters
        if not connector.open_port():raise ConnectionError('Connector.open_port returned false')
        collect_blocks(connector,converters,output,sid,plan,source,routing,fixture,chunks,totals)
        summary=adaptive.aggregate_microtiles(chunks,routing,fixture)
        summary.update(status='completed',technical_completed=True,acquisition_complete=True,
            decoded_conv_bn_lif_complete=len(summary['trajectories'])==200,new_counters=totals,parent_import=parent_import,
            physical_weight_quality_pass=all(c['result']['physical_weight_quality_pass'] for c in chunks),
            new_trained_coefficients=20,imported_trained_coefficients=280,fitted_alpha_used_for_decoding=False,
            diagnostic_acquisition_not_accuracy_qualification=True,address_coverage=address_coverage([c['result'] for c in chunks]))
        save_json(output/'summary.json',summary);manifest.update(status='completed',acquisition_complete=True)
    except (Exception,KeyboardInterrupt) as exc:
        if manifest is not None:manifest['error']=f'{type(exc).__name__}: {exc}'
        else:raise
    finally:
        if connector is not None:
            try:
                if not connector.close_port():raise ConnectionError('Connector.close_port returned false')
            except Exception as exc:
                if manifest is not None:manifest.update(status='failed',close_error=str(exc))
        if manifest is not None:
            attempted=[json.loads(p.read_text(encoding='utf-8')) for p in (output/'tiles').glob('*/tile.json')]
            manifest.update(finished_utc=now(),new_counters=totals,new_attempt_address_coverage=address_coverage(attempted),
                cumulative_impacts_including_all_ancestry=totals['impact_attempts']+parent_import['parent_cumulative_impacts'],
                cumulative_pulses_including_all_ancestry=totals['programming_pulse_attempts']+parent_import['parent_cumulative_pulses'],
                result_sha256={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file()})
            save_json(output/'manifest.json',manifest)
        if locked:lock.unlink()
    print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'])))
    return 0 if manifest['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
