"""Versioned bounded microtile recovery; no threshold changes or fabricated taps."""
import argparse
import copy
import json
import logging
import math
import random
import statistics as st
import uuid

import cifar_feedback_diagnostic as diagnostic
from cifar_feedback_diagnostic import (DiagnosticRun,fixed_calibration,decode_measured,address_coverage,
    candidate_pools,slots,tiles,ROOT,local_path,sha256,save_json,now,EXPECTED_BOARD,
    fidelity_report,mvm_output,check_sources,LOCK,native_connector,launch_provenance)
from cifar_memristor import matrix_masks
from cifar_memristor_analysis import select_matrix,apply_matrix,aggregate

CONFIG='configs/cifar_feedback_adaptive.json'
PINS={**diagnostic.PINS,
    'cifar_feedback_diagnostic.py':'d4bcc3c00d9fc37552af3c2f3b205fa9bb795ce057cafedd93c864ee33069e62',
    'configs/cifar_feedback_diagnostic.json':'c798a58227d25cc5ca7fe8a6f8b3472efb28fe90dde8759d86840e8b42e92246'}
SPEC={**diagnostic.SPEC,'protocol_id':'cifar-feedback-adaptive-two-microtiles-v1',
    'collection_scope':'all300_taps; optional_exact_audited42_prefix; no_import_of_failed_partial_chunk',
    'cifar_collection_on_fidelity_failure':'all300_via_fresh_microtiles_or_audited42_plus_new258; quality_nonblocking; technical_failures_abort',
    'maximum_microtiles_per_original_chunk':2,'maximum_impacts':868466,
    'maximum_programming_pulses':117180,'maximum_programming_decisions':117180,
    'prefix42_remaining_maximum_impacts':756104,'prefix42_remaining_maximum_programming_pulses':101382,
    'selection':'first_k_pending_taps_and_first_k_quiet_legs; k=min(pending,eligible_positive,eligible_negative)',
    'candidate_pool_scope':'complete_original_chunk_pool_on_each_microtile; never_shrink_to_k_plus_three',
    'zero_capacity_policy':'abort_without_resweep; after_second_microtile_remaining_taps_abort',
    'full_matrix_requires_reference_rank_two':True,'parent_bindings':{
        'hardware':dict(session='results/cifar_feedback_diagnostic/c0f8f6d3-9087-42c7-a586-1656b8d96f5e',
            manifest_sha256='7e6455839b7029cfd48e6f7538587e7a60ff1e6de7ea156d494132a48591eee5',
            audit_path='results/cifar_feedback_reviews/hardware_full_attempt_20261016/audit.json',
            audit_sha256='86bc6b919d8fd0ec0242447d5093babab3cac7472773b4b54fa946519213acff'),
        'simulator':dict(session='results/cifar_feedback_diagnostic/e1e1d0a7-00e2-403b-ac4e-defbf1fe80b8',
            manifest_sha256='8125fd63eb2e2699019c4e3172a5cb512e2cf8faf579fd97b87ea6a956711b86',
            audit_path='results/cifar_feedback_reviews/diagnostic_simulator_full_20261016/audit.json',
            audit_sha256='59c86228b83b606ac13e1a53a90b661229655efd69c8f1cc6b1c248d14d3d585')}}


def measurement_tail(active,fixture):
    return diagnostic.feedback.previous.counts([(None,dict(slots=active))],fixture)['maximum_impacts']-242*len(active)


def adaptive_budget(selected,fixture):
    candidates=coefficients=tail=0
    for _,parent in selected:
        active=slots(parent);n=len(active);coefficients+=n;candidates+=2*(2*n+6)
        alternatives=[measurement_tail(active,fixture)]+[measurement_tail(active[:k],fixture)+measurement_tail(active[k:],fixture) for k in range(1,n)]
        tail+=max(alternatives)
    return dict(maximum_impacts=candidates*244+coefficients*811+tail,
        maximum_programming_pulses=candidates*55+coefficients*109,
        maximum_microtiles=2*len(selected),trained_coefficients=coefficients,reset_candidate_attempts_maximum=candidates)


def select_adaptive_matrix(train,selection,adc_unit):
    import numpy as np
    reference_rank=int(np.linalg.matrix_rank(np.array([r['reference_v'] for r in train],dtype=float)))
    result=select_matrix(train,selection,adc_unit)
    for candidate in result['candidates']:
        if candidate['name'].startswith('full_') and reference_rank<2:
            candidate.update(eligible=False,selection_pass=False,ineligible_reason='reference_design_rank_below_two')
    eligible=[r for r in result['candidates'] if r['eligible']]
    chosen=min(eligible,key=lambda r:(r['selection_score'],r['name']!='identity',r['name'])) if eligible else result['candidates'][0]
    result.update(selected_model=chosen['name'],matrix=chosen['matrix'],selection_pass=chosen['selection_pass'],
        identity_fallback=not eligible,reference_design_rank=reference_rank,
        full_matrix_identifiable_from_reference=reference_rank==2)
    return result


def aggregate_microtiles(chunks,routing,fixture):
    keys=[(c['filter_id'],c['parent_tile_id'],c['microtile_id']) for c in chunks]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate logical microtile')
    dynamic=copy.deepcopy(routing);normalized=[]
    for f in dynamic['filters']:
        expected={s['tap_index']:s['trained_weight'] for t in f['tiles'] for s in slots(t)}
        matching=sorted((c for c in chunks if c['filter_id']==f['filter_id']),key=lambda c:(c['parent_tile_id'],c['microtile_id']))
        seen=[];f['tiles']=[]
        for index,chunk in enumerate(matching):
            result=chunk['result'];active=result['effective_route']
            if chunk['microtile_id'] not in (0,1) or not 0<=chunk['parent_tile_id']<len(routing['filters'][f['filter_id']]['tiles']):
                raise ValueError('Invalid parent/microtile identity')
            if not result['technical_completed']:raise ValueError('Incomplete microtile cannot contribute')
            parent=routing['filters'][f['filter_id']]['tiles'][chunk['parent_tile_id']]
            allowed={s['tap_index'] for s in slots(parent)}
            for slot in active:
                tap=slot['tap_index'];seen.append(tap)
                if tap not in allowed or slot['trained_weight']!=expected[tap]:raise ValueError('Changed tap/weight in microtile')
            # Analysis-only IDs: acquired raw/result identities remain untouched.
            normalized.append({**result,'tile_id':index})
            f['tiles'].append(dict(tile_id=index,slots=active))
        if sorted(seen)!=sorted(expected):raise ValueError('Missing or duplicate trained taps; no zero fill')
    if any(c['filter_id'] not in [f['filter_id'] for f in routing['filters']] for c in chunks):raise ValueError('Unexpected filter')
    return aggregate(normalized,dynamic,fixture)


class AdaptiveRun(DiagnosticRun):
    def matrix_calibration(self,active,anchors,meta):
        h=[[st.mean(row[side]['singles'][i]['voltage_v']-row[side]['zero']['voltage_v'] for row in anchors)
            for i in range(len(active))] for side in range(2)]
        phases={}
        for phase,masks in matrix_masks(len(active)).items():
            observations=[]
            for probe_id,mask in enumerate(masks):
                repeats=[]
                for repeat in range(3):
                    values,rows=self.block(active,[active[i] for i in mask],'matrix_'+phase,
                        dict(**meta,probe_id=probe_id,repeat=repeat,mask_slots=mask))
                    repeats.append(values)
                    self.append('matrix_probes.jsonl',dict(phase=phase,probe_id=probe_id,repeat=repeat,
                        mask_slots=mask,measured_rows=rows,measured_voltage_v=values))
                observations.append(dict(probe_id=probe_id,mask_slots=mask,measured_repeats_v=repeats,
                    measured_mean_v=[st.mean(row[i] for row in repeats) for i in range(2)],
                    reference_v=[sum(h[i][j] for j in mask) for i in range(2)]))
            phases[phase]=observations
        correction=select_adaptive_matrix(phases['train'],phases['selection'],mvm_output(1,EXPECTED_BOARD,self.converters))
        correction.update(reference_h_v=h,observations=phases,
            max_singleton_change_v=max(abs(apply_matrix(correction['matrix'],[h[0][i],h[1][i]])[j]-h[j][i])
                for i in range(len(active)) for j in range(2)))
        save_json(self.output/'frozen_matrix.json',correction)
        return correction

    def collect_microtile(self,f,parent,pending,micro,fixture):
        active=copy.deepcopy(pending);fid=f['filter_id'];tid=2*parent['tile_id']+micro
        meta=dict(filter_id=fid,tile_id=tid,parent_tile_id=parent['tile_id'],microtile_id=micro)
        result=dict(**meta,status='failed',programming=[],partials=[],reset_candidates=[],pairs=[],
            pending_taps_before=[s['tap_index'] for s in pending],original_chunk_taps=[s['tap_index'] for s in slots(parent)])
        try:
            pools=candidate_pools(parent);selected={}
            for side,cells in pools.items():
                qualifying=[]
                for index,cell in enumerate(cells):
                    candidate=self.reset_candidate(cell,dict(**meta,side=side,candidate_index=index))
                    result['reset_candidates'].append(dict(side=side,candidate_index=index,**candidate))
                    if candidate['eligible']:qualifying.append(cell)
                selected[side]=qualifying
            k=min(len(active),*(len(cells) for cells in selected.values()))
            if k==0:
                result['stop_reason']='zero_quiet_pair_capacity'
                raise ValueError(result['stop_reason'])
            active=active[:k];selected={side:cells[:k] for side,cells in selected.items()}
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


def verify_parent(session,source,routing,plan):
    """Accept only an immutable, independently audited, source-specific prefix."""
    binding=plan['parent_bindings'][source]
    if session!=binding['session']:raise ValueError('Unreviewed parent session or backend mismatch')
    folder=local_path(session);manifest_path=folder/'manifest.json';audit_path=local_path(binding['audit_path'])
    if sha256(manifest_path)!=binding['manifest_sha256'] or sha256(audit_path)!=binding['audit_sha256']:
        raise ValueError('Parent manifest/audit hash mismatch')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if (manifest['source']!=source or audit['source']!=source or audit['session_id']!=manifest['session_id']
            or audit['manifest_sha256']!=binding['manifest_sha256'] or audit['audit_status']!='passed' or audit['errors']):
        raise ValueError('Parent independent audit is not a matching PASS')
    for name,pin in PINS.items():
        if name in manifest['source_sha256'] and manifest['source_sha256'][name]!=pin:
            raise ValueError('Parent scientific source differs: '+name)
    for name,pin in manifest['result_sha256'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path)!=pin:
            raise ValueError('Parent immutable file mismatch: '+name)
    chunks=[]
    for f,parent in tiles(routing)[:3]:
        relative=f"tiles/f{f['filter_id']}_t{parent['tile_id']}/tile.json"
        if relative not in manifest['result_sha256']:raise ValueError('Missing bound prefix tile')
        result=json.loads((folder/relative).read_text(encoding='utf-8'))
        expected=slots(parent);actual=result['effective_route']
        if (not result['technical_completed'] or result['filter_id']!=0 or result['tile_id']!=parent['tile_id']
                or [(s['tap_index'],s['trained_weight']) for s in actual]!=[(s['tap_index'],s['trained_weight']) for s in expected]
                or len(actual)!=14):raise ValueError('Parent prefix does not contain the exact 42 trained taps')
        points=[(p['case'],p['repeat'],p['t']) for p in result['partials']]
        if len(points)!=600 or set(points)!={(c,r,t) for c in range(10) for r in range(5) for t in range(12)}:
            raise ValueError('Incomplete or duplicate imported partial ledger')
        chunks.append(dict(filter_id=0,parent_tile_id=parent['tile_id'],microtile_id=0,
            origin=dict(session=session,path=relative,sha256=manifest['result_sha256'][relative],
                reused=True,new_measurement=False,acquisition_tile_id=result['tile_id']),result=result))
    metadata={**binding,'result_sha256':manifest['result_sha256'],
        'parent_total_impacts':audit['counters']['impact_attempts'],
        'parent_total_pulses':audit['counters']['programming_pulse_attempts'],
        'imported_prefix':['f0_t0','f0_t1','f0_t2'],'imported_trained_coefficients':42,
        'failed_parent_tile_contributes':False,'other_parent_tiles_ignored':True}
    return chunks,metadata


def load_inputs():
    for name,pin in PINS.items():
        if sha256(local_path(name))!=pin:raise ValueError('Pinned adaptive dependency changed: '+name)
    _,routing,fixture=diagnostic.load_inputs()
    plan=json.loads(local_path(CONFIG).read_text(encoding='utf-8'))
    if plan!=SPEC:raise ValueError('Adaptive configuration differs from reviewed protocol')
    for selected,expected in ((tiles(routing),(868466,117180)),(tiles(routing)[3:],(756104,101382))):
        budget=adaptive_budget(selected,fixture)
        if (budget['maximum_impacts'],budget['maximum_programming_pulses'])!=expected:
            raise ValueError('Adaptive budget changed')
    return plan,routing,fixture


def collect_parent(connector,converters,output,sid,plan,source,f,parent,fixture,totals,budget,chunks):
    pending=copy.deepcopy(slots(parent))
    for micro in range(2):
        folder=output/'tiles'/f"f{f['filter_id']}_p{parent['tile_id']}_m{micro}";folder.mkdir(parents=True)
        candidate_count=sum(map(len,candidate_pools(parent).values()))
        n=len(pending)
        per=dict(maximum_impacts=candidate_count*244+n*811+measurement_tail(pending,fixture),
                 maximum_programming_pulses=candidate_count*55+n*109)
        # Cap before every command via the inherited journal/impact guard, not only after completion.
        for key,total in (('maximum_impacts','impact_attempts'),('maximum_programming_pulses','programming_pulse_attempts')):
            per[key]=min(per[key],budget[key]-totals[total])
        per['maximum_programming_decisions']=candidate_count*55+n*109
        per['maximum_programming_decisions']=min(per['maximum_programming_decisions'],budget['maximum_programming_pulses']-totals['programming_decisions'])
        run=AdaptiveRun(connector,converters,folder,sid,{**plan,**per},source)
        try:result=run.collect_microtile(f,parent,pending,micro,fixture)
        finally:
            for key,value in (('impact_attempts',run.sequence),('programming_pulse_attempts',run.pulses),
                              ('programming_decisions',run.decisions),('programming_skipped_current',run.skipped)):
                totals[key]+=value
        chunks.append(dict(filter_id=f['filter_id'],parent_tile_id=parent['tile_id'],microtile_id=micro,
            origin=dict(session=output.relative_to(ROOT).as_posix(),path=(folder/'tile.json').relative_to(output).as_posix(),
                sha256=sha256(folder/'tile.json'),reused=False,new_measurement=True,acquisition_tile_id=result['tile_id']),result=result))
        checkpoint=output/'completed_microtiles.next.json'
        save_json(checkpoint,chunks)
        checkpoint.replace(output/'completed_microtiles.json')
        pending=pending[len(result['effective_route']):]
        print(f"filter {f['filter_id']} parent {parent['tile_id']} micro {micro}: {len(result['effective_route'])} taps; pending {len(pending)}",flush=True)
        if not pending:return
    raise ValueError('Two-microtile limit reached with unmeasured trained taps')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    for flag in ('run-simulator','run-hardware','dry-run'):mode.add_argument('--'+flag,action='store_true')
    parser.add_argument('--resume-session',help='Exact reviewed parent; imports only its first three completed tiles')
    args=parser.parse_args(argv)
    if not(args.run_simulator or args.run_hardware or args.dry_run):parser.print_help();return 0
    plan,routing,fixture=load_inputs();source='simulator' if args.run_simulator else 'hardware'
    chunks,parent_import=verify_parent(args.resume_session,source,routing,plan) if args.resume_session else ([],None)
    config,snapshot,hashes=check_sources(source)
    for name in (*PINS,CONFIG,'cifar_feedback_adaptive.py','experiment.py'):hashes[name]=sha256(local_path(name))
    selected=tiles(routing)[3:] if parent_import else tiles(routing);budget=adaptive_budget(selected,fixture)
    description=dict(protocol_id=plan['protocol_id'],source=source,source_sha256=hashes,
        hardware_access=source=='hardware' and not args.dry_run,**budget,parent_import=parent_import,
        fixed_decoder_gain_v_per_weight=plan['target_gain_v_per_weight'][source],physical_pulse_width_s=None,
        task_id_scope='session/filter/acquisition_tile_id/task_id',
        simulator_state_policy='fresh_native_array; imported prefix retains historical measurements; no continuous-state equivalence')
    if args.dry_run:print(json.dumps(description));return 0
    lock=local_path(LOCK);locked=False
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as stream:stream.write(f'CIFAR adaptive {now()}\n')
        locked=True
    connector=manifest=output=None
    totals=dict(impact_attempts=0,programming_pulse_attempts=0,programming_decisions=0,programming_skipped_current=0)
    try:
        sid=str(uuid.uuid4());output=local_path('results/cifar_feedback_adaptive')/sid;output.mkdir(parents=True)
        save_json(output/'protocol.json',plan);(output/'board.ini').write_bytes(snapshot)
        save_json(output/'completed_microtiles.json',chunks)
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
        for f,parent in selected:
            collect_parent(connector,converters,output,sid,plan,source,f,parent,fixture,totals,budget,chunks)
        summary=aggregate_microtiles(chunks,routing,fixture)
        summary.update(status='completed',technical_completed=True,acquisition_complete=True,
            decoded_conv_bn_lif_complete=len(summary['trajectories'])==200,
            physical_weight_quality_pass=all(c['result']['physical_weight_quality_pass'] for c in chunks),
            fixed_decoder_gain_v_per_weight=plan['target_gain_v_per_weight'][source],
            fitted_alpha_used_for_decoding=False,diagnostic_acquisition_not_accuracy_qualification=True,
            new_counters=totals,parent_import=parent_import,microtiles_completed=len(chunks),
            address_coverage_scope='represented_completed_microtiles_including_imported_prefix; excludes_parent_failed_attempt',
            address_coverage=address_coverage([c['result'] for c in chunks]))
        save_json(output/'summary.json',summary);manifest.update(status='completed',acquisition_complete=True)
    except (Exception,KeyboardInterrupt) as exc:
        if manifest is not None:manifest['error']=f'{type(exc).__name__}: {exc}'
        else:raise
    finally:
        if connector is not None:
            try:
                if not connector.close_port():raise ConnectionError('Connector.close_port returned false')
            except Exception as exc:
                if manifest is not None:manifest.update(status='failed',close_error=f'{type(exc).__name__}: {exc}')
        if manifest is not None:
            attempted=[json.loads(p.read_text(encoding='utf-8')) for p in (output/'tiles').glob('*/tile.json')]
            manifest.update(finished_utc=now(),new_counters=totals,microtiles_completed=len(chunks),
                new_attempted_microtiles=len(attempted),new_attempt_address_coverage=address_coverage(attempted),
                cumulative_impacts_including_all_parent_attempts=totals['impact_attempts']+(parent_import['parent_total_impacts'] if parent_import else 0),
                cumulative_pulses_including_all_parent_attempts=totals['programming_pulse_attempts']+(parent_import['parent_total_pulses'] if parent_import else 0),
                result_sha256={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file()})
            save_json(output/'manifest.json',manifest)
        if locked:lock.unlink()
    print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'])))
    return 0 if manifest['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
