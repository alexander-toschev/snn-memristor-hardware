"""New bounded physical-output feedback pilot; historical q-target runtime is immutable."""
import argparse
import logging
import random
import uuid
import copy
import json
import math
import statistics as st

import cifar_memristor as previous
from cifar_memristor import TileRun,slots,tiles,ROOT,local_path,sha256,save_json,now,EXPECTED_BOARD
from cifar_memristor_analysis import scalar_calibration,apply_matrix,aggregate
from native_conversions import mvm_output
from cell_calibration import check_sources,LOCK,SIM_INI
from read_all import native_connector
from experiment_hardware import launch_provenance

CONFIG='configs/cifar_feedback.json'
PINS={**previous.PINS,
    'cifar_memristor.py':'2410b67088f39b6d415ed560bff6d06aeae6b8712ab2906bfb69d7191773eefe',
    'cifar_memristor_analysis.py':'8522b922d9d6cf7d98e4e6f6f3c578453cc3d50d688af1ca4e322c67217785de',
    'configs/cifar_memristor.json':'be859fd8b07c8c7fac3caadcd6eb44fc0ae572ccf99ba96b28e7702d35bb966d'}
SPEC={**previous.SPEC,'protocol_id':'cifar-mode10-feedback-v1',
    'target_gain_v_per_weight':{'hardware':1.,'simulator':.03},
    'target_gain_is_source_specific_provisional':True,
    'reset_ramp_v':[round(.3+.05*i,3) for i in range(55)],
    'set_ramp_v':[round(.3+.025*i,3) for i in range(109)],
    'feedback_relative_tolerance':.05,'feedback_absolute_adc_tolerance':.5,
    'spares_per_wl':3,'candidate_bl_pool':list(range(16))+[31],
    'candidate_order':'original logical-slot order, then ascending unused permitted BL',
    'reset_eligibility':'abs_mode10_centered_mean_le_2ADC_and_sampleSD_le_2ADC_on_five_independent_triples; native_R_valid; q_diagnostic_only',
    'reset_signal_limit_adc':2,
    'active_set_leg':'sign(target_signed_voltage - measured_reset_difference); no reversal',
    'maximum_programming_pulses':74940,'maximum_programming_decisions':74940,
    'maximum_impacts':580210,'pilot_maximum_impacts':25612,'pilot_maximum_programming_pulses':3396,
    'physical_fidelity_gate':{'review_pending':False,'raw_target_nrmse_max':.20,'uncertainty_nrms_max':.20,
        'uncertainty_formula':'sqrt(sampleSD(D)^2+ADC_LSB^2)/target_gain; no sqrt(repeats); engineering not worstcase',
        'resolved_sign_errors_max':0,'target_significant_adc':2,'measurement_sign_resolution':'abs(meanD)>2ADC+2sampleSD(D)'},
    'program_stop_policy':'one_reset_sweep_per_candidate; one_residual-selected_SET_sweep_per_pair; current_guard/hit/overshoot/end; no reversal/retry'}
SPEC.update(cifar_collection_on_fidelity_failure='collect bounded diagnostic pilot; remaining requires eligible pilot; later tile quality failures do not silently drop coefficients or stop diagnostic collection')


def candidate_pools(tile):
    active=slots(tile);out={}
    for side in ('positive_cell','negative_cell'):
        wl=active[0][side][0];ordered=[s[side][1] for s in active]
        if len(set(ordered))!=len(ordered) or any(b not in SPEC['candidate_bl_pool'] for b in ordered):
            raise ValueError('Routing outside reviewed unique address pool')
        spares=[b for b in SPEC['candidate_bl_pool'] if b not in ordered][:3]
        if len(spares)!=3:raise ValueError('Insufficient predeclared spares')
        out[side]=[[wl,b] for b in ordered+spares]
    return out


def counts(selected,fixture):
    old=previous.counts(selected,fixture);n=old['trained_coefficients'];candidates=2*n+6*len(selected)
    return {**old,'reset_candidates':candidates,'maximum_programming_pulses':55*candidates+109*n,
        'maximum_impacts':244*candidates+811*n+(old['maximum_impacts']-242*n)}


def fidelity_report(weights,raw_differences,gain,adc_unit):
    if not weights or len(raw_differences)!=5 or any(len(row)!=len(weights) for row in raw_differences):
        raise ValueError('Incomplete physical-fidelity observations')
    if not all(math.isfinite(v) for row in raw_differences+[weights] for v in row) or not math.isfinite(gain) or gain<=0 or not math.isfinite(adc_unit) or adc_unit<=0:
        raise ValueError('Invalid physical-fidelity values/scales')
    means=[st.mean(row[i] for row in raw_differences) for i in range(len(weights))]
    sds=[st.stdev(row[i] for row in raw_differences) for i in range(len(weights))]
    errors=[d/gain-w for d,w in zip(means,weights)];norm=math.sqrt(st.mean(w*w for w in weights))
    if norm<=0:raise ValueError('All-zero target weight norm')
    rmse=math.sqrt(st.mean(e*e for e in errors));resolution=adc_unit/gain
    significant=[i for i,w in enumerate(weights) if abs(w)>2*resolution]
    resolved=[i for i,d in enumerate(means) if abs(d)>2*adc_unit+2*sds[i]]
    wrong=[i for i in significant if i in resolved and means[i]*weights[i]<0]
    unresolved=[i for i in significant if i not in resolved]
    uncertainties=[math.sqrt(sd*sd+adc_unit*adc_unit)/gain for sd in sds]
    uncertainty_nrms=math.sqrt(st.mean(u*u for u in uncertainties))/norm
    return dict(target_gain_v_per_weight=gain,raw_measured_weight=[d/gain for d in means],
        raw_target_errors=errors,raw_target_rmse=rmse,raw_target_nrmse=rmse/norm,
        one_adc_weight_resolution=resolution,quantization_floor_not_reduced_by_averaging=True,
        observed_weight_sd=[sd/gain for sd in sds],resolved_sign_error_slots=wrong,
        target_significant_slots=significant,measured_sign_unresolved_significant_slots=unresolved,
        unresolved_quantization_slots=[i for i in range(len(weights)) if i not in significant],
        engineering_weight_uncertainty=uncertainties,uncertainty_nrms=uncertainty_nrms,uncertainty_is_worstcase_bound=False,
        physical_fidelity_pass=rmse/norm<=.20 and uncertainty_nrms<=.20 and not wrong and not unresolved,
        uses_fitted_alpha_or_matrix=False)


class FeedbackRun(TileRun):
    def leg_triple(self,cell,phase,**meta):
        before=self.mvm(cell,phase,False,role='zero_before',**meta)
        signal=self.mvm(cell,phase,True,role='signal',**meta)
        after=self.mvm(cell,phase,False,role='zero_after',**meta)
        return dict(zero_before=before,signal=signal,zero_after=after,
            voltage_v=signal['voltage_v']-(before['voltage_v']+after['voltage_v'])/2)

    def reset_candidate(self,cell,meta):
        result=dict(cell=cell,status='failed',eligible=False,eligibility_scope='bounded_reset_policy_only')
        start=(self.sequence,self.pulses,self.decisions,self.skipped)
        try:
            current=self.read(cell,1,'candidate_initial',**meta)[0]
            result['initial_mvm']=self.leg_triple(cell,'candidate_initial_mvm',**meta)
            feedback=result['initial_mvm'];limit=2*mvm_output(1,EXPECTED_BOARD,self.converters)
            stop='reset_ramp_exhausted'
            for step,v in enumerate(self.plan['reset_ramp_v']):
                if abs(feedback['voltage_v'])<=limit:stop='mode10_quiet_crossed';break
                row=self.impact(*cell,7,v,'candidate_reset',sign=0,step=step,**meta)
                if row is None:stop='current_guard';break
                current=row
                feedback=self.leg_triple(cell,'candidate_reset_feedback',step=step,**meta)
                self.append('reset_feedback.jsonl',dict(**meta,cell=cell,step=step,write_task_id=row['task_id'],measurement=feedback))
            crossed=abs(feedback['voltage_v'])<=limit
            if crossed:stop='mode10_quiet_crossed'
            confirm=self.read(cell,5,'candidate_reset_confirmation',**meta)
            observed=[self.leg_triple(cell,'candidate_reset_confirmation_mvm',repeat=i,**meta) for i in range(5)]
            result.update(status='completed',reset_stop=stop,reset_crossed=crossed,
                q_confirmation=[dict(task_id=r['task_id'],q=r['q']) for r in confirm],mvm_confirmation=observed,
                baseline_mvm_mean_v=st.mean(r['voltage_v'] for r in observed),
                baseline_mvm_sd_v=st.stdev(r['voltage_v'] for r in observed),
                mode10_quiet_limit_v=limit,native_q_diagnostic_crossed=current['q']<=.005,
                eligible=crossed and abs(st.mean(r['voltage_v'] for r in observed))<=limit and st.stdev(r['voltage_v'] for r in observed)<=limit,
                physical_failure_cause='unknown; not inferred from this screening')
            return result
        finally:
            result.update(impact_attempts=self.sequence-start[0],programming_pulse_attempts=self.pulses-start[1],
                programming_decisions=self.decisions-start[2],programming_skipped_current=self.skipped-start[3])
            self.append('reset_candidates.jsonl',{**meta,**result})

    def program_pair(self,slot,meta):
        meta={('pair_tap' if key=='tap' else key):value for key,value in meta.items()}
        gain=self.plan['target_gain_v_per_weight'][self.source];target=gain*slot['trained_weight']
        tolerance=max(self.plan['feedback_absolute_adc_tolerance']*mvm_output(1,EXPECTED_BOARD,self.converters),.05*abs(target))
        result=dict(tap=slot['tap_index'],trained_weight=slot['trained_weight'],target_signed_v=target,
            feedback_stop_tolerance_v=tolerance,status='failed',positive_cell=slot['positive_cell'],negative_cell=slot['negative_cell'])
        start=(self.sequence,self.pulses,self.decisions,self.skipped)
        try:
            result['guard_refresh']=[self.read(slot[side],1,'pair_guard_refresh',side=side,**meta)[0]
                for side in ('positive_cell','negative_cell')]
            values,rows=self.block([slot],[slot],'pair_initial_feedback',meta)
            current=values[0]-values[1];result['initial_mvm']=rows;result['initial_signed_v']=current
            direction=1 if target-current>=0 else -1
            side='positive_cell' if direction==1 else 'negative_cell';result['active_set_side']=side
            stop='set_ramp_exhausted'
            for step,v in enumerate(self.plan['set_ramp_v']):
                if abs(current-target)<=tolerance:stop='feedback_hit';break
                if direction*(current-target)>tolerance:stop='feedback_overshoot';break
                row=self.impact(*slot[side],7,v,'pair_set',sign=1,step=step,target_signed_v=target,**meta)
                if row is None:stop='current_guard';break
                values,rows=self.block([slot],[slot],'pair_set_feedback',dict(**meta,step=step))
                current=values[0]-values[1]
                self.append('set_feedback.jsonl',dict(**meta,step=step,write_task_id=row['task_id'],measured_rows=rows,signed_voltage_v=current))
            if abs(current-target)<=tolerance:stop='feedback_hit'
            elif direction*(current-target)>tolerance:stop='feedback_overshoot'
            reads=[self.read(slot[s],5,'pair_post_mode7',side=s,**meta) for s in ('positive_cell','negative_cell')]
            repeats=[]
            for repeat in range(5):
                values,rows=self.block([slot],[slot],'pair_post_feedback',dict(**meta,repeat=repeat))
                repeats.append(dict(measured_rows=rows,signed_voltage_v=values[0]-values[1]))
            result.update(status='completed',stop=stop,endpoint_signed_v=current,post_mode7=reads,post_feedback=repeats,
                post_mean_signed_v=st.mean(r['signed_voltage_v'] for r in repeats),
                post_sd_signed_v=st.stdev(r['signed_voltage_v'] for r in repeats),
                stop_band_confirmed_all_five=all(abs(r['signed_voltage_v']-target)<=tolerance for r in repeats),
                stop_band_does_not_imply_weight_fidelity=True)
            return result
        finally:
            result.update(impact_attempts=self.sequence-start[0],programming_pulse_attempts=self.pulses-start[1],
                programming_decisions=self.decisions-start[2],programming_skipped_current=self.skipped-start[3])
            self.append('pair_programming.jsonl',{**meta,**result})

    def collect_tile(self,f,tile,fixture):
        active=copy.deepcopy(slots(tile));fid=f['filter_id'];tid=tile['tile_id'];meta=dict(filter_id=fid,tile_id=tid)
        result=dict(**meta,status='failed',programming=[],partials=[],reset_candidates=[],pairs=[])
        try:
            pools=candidate_pools(tile);selected={}
            for side,cells in pools.items():
                qualifying=[]
                for index,cell in enumerate(cells):
                    candidate=self.reset_candidate(cell,dict(**meta,side=side,candidate_index=index))
                    result['reset_candidates'].append(dict(side=side,candidate_index=index,**candidate))
                    if candidate['eligible']:qualifying.append(cell)
                selected[side]=qualifying[:len(active)]
            if any(len(cells)!=len(active) for cells in selected.values()):
                result['stop_reason']='insufficient_quiet_reset_candidates_for_fixed_tile'
                raise ValueError(result['stop_reason'])
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
            cal=scalar_calibration([s['trained_weight'] for s in active],differences,mvm_output(5,EXPECTED_BOARD,self.converters))
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
                            trained_unit_contribution=difference/cal['alpha_v_per_weight'] if cal['decode_usable'] else None)
                        result['partials'].append(part);self.append('partials.jsonl',part)
            result['final_anchors']=self.anchors(active,'final_anchors',meta)
            final_differences=[[row[0]['singles'][i]['voltage_v']-row[0]['zero']['voltage_v']-row[1]['singles'][i]['voltage_v']+row[1]['zero']['voltage_v'] for i in range(len(active))] for row in result['final_anchors']]
            result['final_physical_fidelity']=fidelity_report([s['trained_weight'] for s in active],final_differences,self.plan['target_gain_v_per_weight'][self.source],mvm_output(1,EXPECTED_BOARD,self.converters))
            result['continuation_eligible']=cal['decode_usable'] and result['physical_fidelity']['physical_fidelity_pass'] and result['final_physical_fidelity']['physical_fidelity_pass']
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


def load_inputs():
    for name,pin in PINS.items():
        if sha256(local_path(name))!=pin:raise ValueError('Pinned feedback dependency changed: '+name)
    _,routing,fixture=previous.load_inputs()
    plan=json.loads(local_path(CONFIG).read_text(encoding='utf-8'))
    if plan!=SPEC:raise ValueError('Feedback protocol differs from reviewed configuration')
    if counts(tiles(routing),fixture)['maximum_impacts']!=580210 or counts(tiles(routing)[:1],fixture)['maximum_impacts']!=25612:
        raise ValueError('Feedback budget/routing mismatch')
    return plan,routing,fixture


def verify_pilot(name,source,hashes,fixture,routing):
    folder=local_path(name)/'tiles/f0_t0'
    tile=json.loads((folder/'tile.json').read_text(encoding='utf-8'))
    route=json.loads((folder/'frozen_route.json').read_text(encoding='utf-8'))
    original=slots(routing['filters'][0]['tiles'][0]);effective=tile.get('effective_route',[])
    if len(effective)!=14 or route['slots']!=effective:raise ValueError('Incomplete frozen feedback routing')
    pools=candidate_pools(routing['filters'][0]['tiles'][0]);selected={}
    for side,cells in pools.items():
        candidates=[r for r in tile['reset_candidates'] if r['side']==side]
        if len(candidates)!=17 or [r['cell'] for r in candidates]!=cells:raise ValueError('Pilot reset pool coverage changed')
        qualifying=[]
        for candidate in candidates:
            observations=candidate['mvm_confirmation']
            if len(observations)!=5:raise ValueError('Missing reset confirmation')
            values=[r['voltage_v'] for r in observations];limit=candidate['mode10_quiet_limit_v']
            eligible=candidate['reset_crossed'] and abs(st.mean(values))<=limit and st.stdev(values)<=limit
            if candidate['eligible']!=eligible:raise ValueError('Reset eligibility mismatch')
            if eligible:qualifying.append(candidate['cell'])
        selected[side]=qualifying[:14]
        if len(selected[side])!=14:raise ValueError('Pilot lacks complete eligible reset route')
    for i,(before,after) in enumerate(zip(original,effective)):
        if {k:v for k,v in before.items() if k not in ('positive_cell','negative_cell')}!={k:v for k,v in after.items() if k not in ('positive_cell','negative_cell')}:
            raise ValueError('Trained coefficient changed in remap')
        if any(after[side]!=selected[side][i] for side in selected):raise ValueError('Remap is not deterministic reset-qualified order')
    modified=copy.deepcopy(routing);modified['filters'][0]['tiles'][0]['slots']=effective
    imported,parent=previous.verify_pilot(name,source,hashes,fixture,modified)
    if 'tiles/f0_t0/frozen_route.json' not in parent['result_sha256']:raise ValueError('Missing routing commitment hash')
    if not imported.get('continuation_eligible'):raise ValueError('Pilot physical weight fidelity does not support continuation')
    for key in ('physical_fidelity','final_physical_fidelity'):
        report=imported[key]
        if report['raw_target_nrmse']>.20 or report['uncertainty_nrms']>.20 or report['resolved_sign_error_slots'] or report['measured_sign_unresolved_significant_slots']:
            raise ValueError('Physical fidelity gate failed, independent of fitted alpha')
    return imported,parent

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    for flag in ('run-simulator','run-hardware','dry-run'):mode.add_argument('--'+flag,action='store_true')
    parser.add_argument('--stage',choices=('pilot','remaining'),default='pilot');parser.add_argument('--pilot-session')
    args=parser.parse_args(argv)
    if not(args.run_simulator or args.run_hardware or args.dry_run):parser.print_help();return 0
    if (args.stage=='remaining')!=bool(args.pilot_session):parser.error('Remaining requires --pilot-session; pilot forbids it')
    plan,routing,fixture=load_inputs();source='simulator' if args.run_simulator else 'hardware'
    config,snapshot,hashes=check_sources(source)
    for name in (*PINS,CONFIG,'cifar_feedback.py','cifar_memristor.py','cifar_memristor_analysis.py','experiment.py'):
        hashes[name]=sha256(local_path(name))
    selected=tiles(routing)[:1] if args.stage=='pilot' else tiles(routing)[1:]
    budget=counts(selected,fixture);imported=[];parent=None
    if args.pilot_session:
        pilot,parent=verify_pilot(args.pilot_session,source,hashes,fixture,routing);imported=[pilot]
    description=dict(protocol_id=plan['protocol_id'],stage=args.stage,source=source,source_sha256=hashes,
        hardware_access=source=='hardware' and not args.dry_run,**budget,aggregate_maximum_impacts=580210,
        aggregate_maximum_programming_pulses=74940,parent_pilot=parent,physical_pulse_width_s=None,
        task_id_scope='unique within tile; identify by session_id/filter_id/tile_id/task_id',
        simulator_continuation='copy immutable native-persisted pilot array; final read-induced state is not persisted by native close; imported pilot is not remeasured')
    if args.dry_run:print(json.dumps(description));return 0
    lock=local_path(LOCK);locked=False
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as stream:stream.write(f'CIFAR Conv1 {now()}\n')
        locked=True
    connector=None;manifest=None;output=None;results=list(imported)
    try:
        sid=str(uuid.uuid4());output=local_path('results/cifar_feedback')/sid;output.mkdir(parents=True)
        save_json(output/'protocol.json',plan);(output/'board.ini').write_bytes(snapshot)
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
            serial=output/'array'
            if args.pilot_session:
                source_array=local_path(args.pilot_session)/'array.cb'
                if parent['result_sha256'].get('array.cb')!=sha256(source_array):raise ValueError('Pilot simulator array fingerprint mismatch')
                serial.with_suffix('.cb').write_bytes(source_array.read_bytes())
            else:create_crossbar_array(str(serial),32,8)
            manifest['array_initial_sha256']=sha256(serial.with_suffix('.cb'))
            connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                board_type='rp5_fpga_python',task_format='SI',config=config,crossbar_serial=str(serial))
        from manager.service import converters
        if not connector.open_port():raise ConnectionError('Connector.open_port returned false')
        for f,tile in selected:
            folder=output/'tiles'/f"f{f['filter_id']}_t{tile['tile_id']}";folder.mkdir(parents=True)
            per=counts([(f,tile)],fixture);tile_plan={**plan,**per, 'maximum_programming_decisions':per['maximum_programming_pulses']}
            run=FeedbackRun(connector,converters,folder,sid,tile_plan,source)
            result=run.collect_tile(f,tile,fixture);results.append(result)
            print(f"filter {f['filter_id']} tile {tile['tile_id']}: alpha usable={result['decode_usable']}",flush=True)
            if sum(r['impact_attempts'] for r in results)>580210 or sum(r['programming_pulse_attempts'] for r in results)>74940:
                raise ValueError('Combined pilot/remaining budget exceeded')
        summary=aggregate(results,routing,fixture)
        summary.update(status='completed',stage=args.stage,technical_completed=True,
            continuation_eligible=all(r['continuation_eligible'] for r in results),
            decode_usable=all(r['decode_usable'] for r in results),filter_tiles_completed=len(results),
            acquisition_complete=args.stage=='remaining' and len(results)==28,
            decoded_conv_bn_lif_complete=args.stage=='remaining' and len(summary['trajectories'])==200 and all(r['decode_usable'] for r in results),
            pilot_only_partial_convolution=args.stage=='pilot',
            total_impacts_including_pilot=sum(r['impact_attempts'] for r in results),
            total_pulses_including_pilot=sum(r['programming_pulse_attempts'] for r in results),
            tile_status=[dict(filter_id=r['filter_id'],tile_id=r['tile_id'],decode_usable=r['decode_usable'],continuation_eligible=r['continuation_eligible']) for r in results])
        save_json(output/'summary.json',summary);manifest.update(status='completed',decode_usable=summary['decode_usable'])
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
            manifest.update(finished_utc=now(),result_sha256={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file()})
            save_json(output/'manifest.json',manifest)
        if locked:lock.unlink()
    print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'])))
    return 0 if manifest['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
