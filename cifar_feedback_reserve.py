"""One bounded reserve-cell experiment: fresh 13 coefficients, immutable other 287."""
import argparse
import copy
import json
import logging
import random
import statistics as st
import uuid

import cifar_feedback_adaptive as adaptive
from cifar_feedback_adaptive import (AdaptiveRun,slots,tiles,ROOT,local_path,sha256,save_json,now,
    EXPECTED_BOARD,mvm_output,fixed_calibration,fidelity_report,apply_matrix,decode_measured,
    address_coverage,check_sources,LOCK,native_connector,launch_provenance)

CONFIG='configs/cifar_feedback_reserve.json'
PINS={**adaptive.PINS,
    'cifar_feedback_adaptive.py':'a46eb7b94891d1adf48d10f126227e10ecfa696aba61a98f293199a8dcb9f2ba',
    'configs/cifar_feedback_adaptive.json':'c788a91a1dc51e4a9f6070354c9597249a45b4770decb07847030f683ef09663'}
SPEC={**adaptive.SPEC,'protocol_id':'cifar-feedback-reserve-wl7-bl15-v1',
    'collection_scope':'fresh_filter3_taps52_to64; immutable_other287; original42_ancestry_preserved',
    'cifar_collection_on_fidelity_failure':'collect13_diagnostic_coefficients_after_route_gate; quality_nonblocking',
    'maximum_impacts':24323,'maximum_programming_pulses':3232,'maximum_programming_decisions':3232,
    'target_filter_id':3,'target_parent_tile_id':4,'target_taps':list(range(52,65)),
    'reserved_negative_cell':[7,15],'reserved_tap':64,'positive_priority':[[6,2],[6,11]],
    'candidate_pool_scope':'original_positive16; original_negative16_plus_reserved_negative15',
    'selection':'13quietP_priority_2_11_then_original; 12quiet_originalN_then_reserved15',
    'route_gate':'reserved15_quiet_and_at_least12_original_negative_and13_positive_quiet; otherwise_abort_before_SET',
    'maximum_new_microtiles':1,'automatic_retry':False,
    'parent_bindings':{
      'hardware':dict(session='results/cifar_feedback_adaptive/5caa3d62-acc5-41a2-9c69-bc1a74bbf427',
        manifest_sha256='9415ec477977bc1d3ba493aa4da99f6104e498d4391a1f79b3c8b8d5544e952e',
        audit_path='results/cifar_feedback_reviews/adaptive_hardware_full_20261016/audit.json',
        audit_sha256='84c8a5e299c2893124f447f4128839f38afbdfc9204c17022387afef281f37a4'),
      'simulator':dict(session='results/cifar_feedback_adaptive/64036fb0-2d33-4350-b672-96ef8380acec',
        manifest_sha256='1e474d60766db746016f9ee648f52abbe2c499018141c10659167e5113f5aec7',
        audit_path='results/cifar_feedback_reviews/adaptive_simulator_full_20261016_v2/audit.json',
        audit_sha256='fe3df5205076258e12fcd85e8a33b29cbe3328358d2c807a4b8656f9599299e2')}}
# Remove inherited execution fields belonging only to the older adaptive schedule.
for _name in ('maximum_microtiles_per_original_chunk','prefix42_remaining_maximum_impacts',
              'prefix42_remaining_maximum_programming_pulses','zero_capacity_policy'):
    SPEC.pop(_name,None)


def reserve_pools(parent):
    pools=adaptive.candidate_pools(parent)
    if len(pools['positive_cell'])!=16 or len(pools['negative_cell'])!=16 or [7,15] in pools['negative_cell']:
        raise ValueError('Reviewed original reserve pool differs')
    pools['negative_cell'].append([7,15])
    return pools


def select_route(pools,eligible):
    negative=[c for c in eligible['negative_cell'] if c!=[7,15]]
    if [7,15] not in eligible['negative_cell'] or len(negative)<12 or len(eligible['positive_cell'])<13:
        raise ValueError('reserve_route_gate_failed_before_SET')
    priority=[]
    for cell in [[6,2],[6,11]]+pools['positive_cell']:
        if cell in eligible['positive_cell'] and cell not in priority:priority.append(cell)
    return dict(positive_cell=priority[:13],negative_cell=negative[:12]+[[7,15]])


class ReserveRun(AdaptiveRun):
    def collect_reserve(self,f,parent,fixture):
        pending=slots(parent);micro=0
        active=copy.deepcopy(pending);fid=f['filter_id'];tid=2*parent['tile_id']+micro
        meta=dict(filter_id=fid,tile_id=tid,parent_tile_id=parent['tile_id'],microtile_id=micro)
        result=dict(**meta,status='failed',programming=[],partials=[],reset_candidates=[],pairs=[],
            pending_taps_before=[s['tap_index'] for s in pending],original_chunk_taps=[s['tap_index'] for s in slots(parent)])
        try:
            pools=reserve_pools(parent);selected={}
            for side,cells in pools.items():
                qualifying=[]
                for index,cell in enumerate(cells):
                    candidate=self.reset_candidate(cell,dict(**meta,side=side,candidate_index=index))
                    result['reset_candidates'].append(dict(side=side,candidate_index=index,**candidate))
                    if candidate['eligible']:qualifying.append(cell)
                selected[side]=qualifying
            result['quiet_candidate_counts']={side:len(cells) for side,cells in selected.items()}
            try:selected=select_route(pools,selected)
            except ValueError:
                result['stop_reason']='reserve_route_gate_failed_before_SET'
                raise
            result['assigned_taps']=[s['tap_index'] for s in active]
            for index,slot in enumerate(active):
                for side in ('positive_cell','negative_cell'):slot[side]=selected[side][index]
            result['effective_route']=active
            save_json(self.output/'frozen_route.json',dict(candidate_pools=pools,slots=active,
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
            self.log.close();save_json(self.output/'tile.json',result)
        return result


def verify_parent(session,source,routing,fixture,plan):
    binding=plan['parent_bindings'][source]
    if session!=binding['session']:raise ValueError('Unreviewed reserve parent or backend mismatch')
    folder=local_path(session);mp=folder/'manifest.json';ap=local_path(binding['audit_path'])
    if sha256(mp)!=binding['manifest_sha256'] or sha256(ap)!=binding['audit_sha256']:
        raise ValueError('Reserve parent manifest/audit hash mismatch')
    manifest=json.loads(mp.read_text(encoding='utf-8'));audit=json.loads(ap.read_text(encoding='utf-8'))
    if (manifest['source']!=source or audit['source']!=source or audit['session_id']!=manifest['session_id']
            or audit['manifest_sha256']!=binding['manifest_sha256'] or audit['audit_status']!='passed'
            or audit['errors'] or not audit['acquisition_complete'] or audit['full_time_outputs']!=2400):
        raise ValueError('Reserve parent is not a bound complete independent PASS')
    for name,pin in manifest['result_sha256'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path)!=pin:
            raise ValueError('Reserve parent immutable file mismatch: '+name)
    for name,pin in PINS.items():
        if name in manifest['source_sha256'] and manifest['source_sha256'][name]!=pin:
            raise ValueError('Reserve parent scientific source mismatch: '+name)
    # Recheck the actual ancestral files, not merely the copied metadata for 42 taps.
    _,ancestry=adaptive.verify_parent(manifest['parent_import']['session'],source,routing,adaptive.SPEC)
    if ancestry!=manifest['parent_import']:raise ValueError('Parent ancestry metadata mismatch')
    originals=json.loads((folder/'completed_microtiles.json').read_text(encoding='utf-8'))
    adaptive.aggregate_microtiles(originals,routing,fixture) # exact 300 taps / complete measured ledgers
    chunks=[];removed=[]
    for index,original in enumerate(originals):
        if (original['filter_id'],original['parent_tile_id'])==(3,4):
            removed.extend(s['tap_index'] for s in original['result']['effective_route']);continue
        chunks.append({**original,'origin':dict(session=session,path='completed_microtiles.json',wrapper_index=index,
            sha256=manifest['result_sha256']['completed_microtiles.json'],reused=True,new_measurement=False,
            acquisition_tile_id=original['result']['tile_id'],ancestry_original_origin=original['origin'])})
    if sorted(removed)!=list(range(52,65)) or sum(len(c['result']['effective_route']) for c in chunks)!=287:
        raise ValueError('Reserve replacement must remove exactly all 13 taps once')
    metadata={**binding,'result_sha256':manifest['result_sha256'],'imported_trained_coefficients':287,
        'excluded_filter_id':3,'excluded_parent_tile_id':4,'excluded_taps':list(range(52,65)),
        'parent_new_counters':manifest['new_counters'],'ancestry42':ancestry,
        'parent_cumulative_impacts':manifest['cumulative_impacts_including_all_parent_attempts'],
        'parent_cumulative_pulses':manifest['cumulative_pulses_including_all_parent_attempts']}
    return chunks,metadata


def load_inputs():
    for name,pin in PINS.items():
        if sha256(local_path(name))!=pin:raise ValueError('Pinned reserve dependency changed: '+name)
    _,routing,fixture=adaptive.load_inputs()
    plan=json.loads(local_path(CONFIG).read_text(encoding='utf-8'))
    if plan!=SPEC:raise ValueError('Reserve config differs from reviewed protocol')
    parent=routing['filters'][3]['tiles'][4];active=slots(parent)
    if [s['tap_index'] for s in active]!=list(range(52,65)):raise ValueError('Reserve trained tap range changed')
    pools=reserve_pools(parent);n=len(active);c=sum(map(len,pools.values()))
    impacts=c*244+n*811+adaptive.measurement_tail(active,fixture);writes=c*55+n*109
    if (impacts,writes)!=(plan['maximum_impacts'],plan['maximum_programming_pulses']):raise ValueError('Reserve bounds changed')
    return plan,routing,fixture


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    for flag in ('run-simulator','run-hardware','dry-run'):mode.add_argument('--'+flag,action='store_true')
    parser.add_argument('--resume-session',help='Optional explicit exact parent; defaults to the source-specific pinned parent')
    args=parser.parse_args(argv)
    if not(args.run_simulator or args.run_hardware or args.dry_run):parser.print_help();return 0
    source='simulator' if args.run_simulator else 'hardware';plan,routing,fixture=load_inputs()
    parent_session=args.resume_session or plan['parent_bindings'][source]['session']
    chunks,parent_import=verify_parent(parent_session,source,routing,fixture,plan)
    config,snapshot,hashes=check_sources(source)
    for name in (*PINS,CONFIG,'cifar_feedback_reserve.py','experiment.py'):hashes[name]=sha256(local_path(name))
    description=dict(protocol_id=plan['protocol_id'],source=source,source_sha256=hashes,
        hardware_access=source=='hardware' and not args.dry_run,parent_import=parent_import,
        maximum_impacts=24323,maximum_programming_pulses=3232,maximum_programming_decisions=3232,
        imported_trained_coefficients=287,new_trained_coefficients=13,reset_candidate_count=33,
        fixed_decoder_gain_v_per_weight=plan['target_gain_v_per_weight'][source],physical_pulse_width_s=None,
        simulator_state_policy='fresh native array; imported287 are historical; no continuous-state equivalence')
    if args.dry_run:print(json.dumps(description));return 0
    lock=local_path(LOCK);locked=False
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as stream:stream.write(f'CIFAR reserve {now()}\n')
        locked=True
    connector=manifest=output=run=None
    try:
        sid=str(uuid.uuid4());output=local_path('results/cifar_feedback_reserve')/sid;output.mkdir(parents=True)
        save_json(output/'protocol.json',plan);(output/'board.ini').write_bytes(snapshot)
        save_json(output/'imported_microtiles.json',chunks)
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
            serial=output/'array';create_crossbar_array(str(serial),32,8)
            manifest['array_initial_sha256']=sha256(serial.with_suffix('.cb'))
            connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                board_type='rp5_fpga_python',task_format='SI',config=config,crossbar_serial=str(serial))
        from manager.service import converters
        if not connector.open_port():raise ConnectionError('Connector.open_port returned false')
        f=routing['filters'][3];parent=f['tiles'][4];folder=output/'tiles/f3_p4_reserve';folder.mkdir(parents=True)
        run=ReserveRun(connector,converters,folder,sid,plan,source)
        result=run.collect_reserve(f,parent,fixture)
        chunks.append(dict(filter_id=3,parent_tile_id=4,microtile_id=0,result=result,
            origin=dict(session=output.relative_to(ROOT).as_posix(),path='tiles/f3_p4_reserve/tile.json',
                sha256=sha256(folder/'tile.json'),reused=False,new_measurement=True,acquisition_tile_id=8)))
        save_json(output/'completed_microtiles.json',chunks)
        summary=adaptive.aggregate_microtiles(chunks,routing,fixture)
        summary.update(status='completed',technical_completed=True,acquisition_complete=True,
            decoded_conv_bn_lif_complete=len(summary['trajectories'])==200,
            physical_weight_quality_pass=all(c['result']['physical_weight_quality_pass'] for c in chunks),
            fresh_weight_quality_pass=result['physical_weight_quality_pass'],parent_import=parent_import,
            fixed_decoder_gain_v_per_weight=plan['target_gain_v_per_weight'][source],fitted_alpha_used_for_decoding=False,
            diagnostic_acquisition_not_accuracy_qualification=True,new_trained_coefficients=13,imported_trained_coefficients=287,
            address_coverage=address_coverage([c['result'] for c in chunks]),new_attempt_address_coverage=address_coverage([result]))
        save_json(output/'summary.json',summary);manifest.update(status='completed',acquisition_complete=True)
    except (Exception,KeyboardInterrupt) as exc:
        if manifest is not None:manifest['error']=f'{type(exc).__name__}: {exc}'
        else:raise
    finally:
        if run is not None:run.log.close()
        if connector is not None:
            try:
                if not connector.close_port():raise ConnectionError('Connector.close_port returned false')
            except Exception as exc:
                if manifest is not None:manifest.update(status='failed',close_error=str(exc))
        if manifest is not None:
            counters=dict(impact_attempts=run.sequence if run else 0,programming_pulse_attempts=run.pulses if run else 0,
                programming_decisions=run.decisions if run else 0,programming_skipped_current=run.skipped if run else 0)
            attempted=[json.loads(p.read_text(encoding='utf-8')) for p in (output/'tiles').glob('*/tile.json')]
            manifest.update(finished_utc=now(),new_counters=counters,new_attempt_address_coverage=address_coverage(attempted),
                cumulative_impacts_including_all_ancestry=counters['impact_attempts']+parent_import['parent_cumulative_impacts'],
                cumulative_pulses_including_all_ancestry=counters['programming_pulse_attempts']+parent_import['parent_cumulative_pulses'],
                result_sha256={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file()})
            save_json(output/'manifest.json',manifest)
        if locked:lock.unlink()
    print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'])))
    return 0 if manifest['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
