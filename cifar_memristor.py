"""Bounded real-weight Conv1 experiment. Explicit pilot, then immutable-pilot continuation."""
import argparse
import json
import logging
import math
import random
import statistics as st
import uuid

from cell_calibration import Discovery, WRITE_GUARD, check_sources, LOCK, SIM_INI
from read_all import ROOT, local_path, sha256, save_json, now, native_connector, EXPECTED_BOARD
from experiment_hardware import launch_provenance
from native_conversions import mvm_output
from cifar_memristor_analysis import scalar_calibration, aggregate,select_matrix,apply_matrix

CONFIG='configs/cifar_memristor.json'
ROUTING='weights/candidate_cifar75_routing.json'
FIXTURE='weights/prepared_cifar100_conv1.json'
PINS={
 'cell_calibration.py':'eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8',
 'native_conversions.py':'2ad2e9c057bb14979b083a6e846e8c417c080a6c5273d848e31f893b7970253e',
 'read_all.py':'b30ae2f4dd3cd166dbc5cefb07b558127de2bbdba1f6674c40b1b45f4dd74d5f',
 'experiment_hardware.py':'059dc4f827943a9df631631f52e776b57182afc9d36db0c1ad5246427f73548c',
 'weights/conv1_weights.npy':'8be312b9eafb9de23f8340b17f848e7e0e2e7884293c826a6d9f6cbb5125d440',
 ROUTING:'bbe71222138fd28ef928dcca22b319abc1223ec87b6680edd56b03c63dac0245',
 FIXTURE:'86113ff1d2fdbe155056d2e1377cf380e7395120fd93ccf2e1efc7a0a933167c',
 'weights/best_fs.pt':'ee4fc4f58a3d010218116e9ad4c7ba808787959f9ed7687cd58cbb32d90f62fb',
 'weights/cfg.json':'94ea7d5f57c5fdf6e5f4e319c38b4921783cdc04f5f082fdca59c2e86a14c645',
}
SPEC=dict(protocol_id='cifar-conv1-program-measure-c2-v1',filters=[0,1,2,3],repeats=5,timesteps=12,cases=10,
    ramp_v=[round(.3+.05*i,2) for i in range(55)],q_center=.015,q_delta=.005,q_relative_tolerance=.01,
    reset_q=.005,reset_confirm_reads=5,post_program_reads=5,tile_readback_repeats=3,
    maximum_impacts=222118,maximum_programming_pulses=66000,maximum_programming_decisions=66000,
    pilot_maximum_impacts=9350,pilot_maximum_programming_pulses=3080,simulator_seed=20261016,
    simulator_pulse_width_s=.000125,hardware_pulse_width_s=None,adc_headroom_fraction=.8,
    baseline_scope='per_assignment_before_own_writes',programming_guard=WRITE_GUARD,
    alpha_resolution_sd_multiplier=5,alpha_absolute_floor_adc=5,alpha_cv_diagnostic_limit=.05,
    inference_voltage_v=.1,current_model_limit_a=.005,
    program_stop_policy='first_current_guard_skip_ends_polarity; one_sweep_no_retries; SET_hit_or_overshoot_stops',
    matrix=dict(shape=[2,2],offset_v=[0,0],repeats=3,train='singles_all_even_odd',
        selection='zero_singles_adjacent_pairs',ridge_relative=.01,minimum_singular=.1,maximum_singular=10,
        maximum_condition=1000,selection_relative_error=.05,selection_absolute_adc=2,
        reference='uncorrected_fresh_H_times_mask; no trained weights or CIFAR',selection_policy='minimum_score_among_numerically_eligible; quality_flags_do_not_gate',fallback='identity_if_no_numerically_eligible'))


def tiles(routing):
    return [(f,t) for f in routing['filters'] for t in f['tiles']]


def slots(tile):return [s for s in tile['slots'] if not s['padding']]


def matrix_masks(n):
    return dict(train=[[i] for i in range(n)]+[list(range(n)),list(range(0,n,2)),list(range(1,n,2))],
        selection=[[]]+[[i] for i in range(n)]+[[i,i+1] for i in range(0,n-1,2)])


def counts(selected,fixture):
    n=sum(len(slots(t)) for _,t in selected);cal=sum(10*(len(slots(t))+1) for _,t in selected)
    infer=0
    for _,tile in selected:
        for case in fixture['cases']:
            for x in case['spikes_t12x75']:
                k=sum(x[s['tap_index']] for s in slots(tile))
                infer+=5*(4+2*max(1,int(k)))
    matrix_calls=sum(3*(4+2*max(1,len(mask))) for _,tile in selected for masks in matrix_masks(len(slots(tile))).values() for mask in masks)
    return dict(maximum_programming_pulses=220*n,maximum_impacts=242*n+12*n+2*cal+infer+matrix_calls,
                physical_assignments=2*n,trained_coefficients=n,tiles=len(selected),cifar_impacts=infer,matrix_impacts=matrix_calls)


def load_inputs():
    for name,pin in PINS.items():
        if sha256(local_path(name))!=pin:raise ValueError('Pinned source/input changed: '+name)
    plan=json.loads(local_path(CONFIG).read_text(encoding='utf-8'))
    if plan!=SPEC:raise ValueError('Protocol differs from reviewed specification')
    routing=json.loads(local_path(ROUTING).read_text(encoding='utf-8'))
    fixture=json.loads(local_path(FIXTURE).read_text(encoding='utf-8'))
    all_tiles=tiles(routing)
    if counts(all_tiles,fixture)['maximum_impacts']!=222118 or counts(all_tiles[:1],fixture)['maximum_impacts']!=9350:
        raise ValueError('Fixture/routing budget mismatch')
    return plan,routing,fixture


class TileRun(Discovery):
    def emit(self,row):
        if row.get('event')=='result' and row.get('status')=='ok' and row['command_si']['mode_flag']==7:
            if row['native_r']!=row['resistance_ohm']:
                row.update(status='error',error='Native R differs from pinned converter')
                super().emit(row);raise ValueError(row['error'])
        super().emit(row)

    def read(self,cell,n,phase,**meta):
        return [self.impact(*cell,7,0.,phase,repeat=i,**meta) for i in range(n)]

    def program(self,cell,target,meta):
        result=dict(cell=cell,target_q=target,status='failed',reset_confirmed=False)
        start=(self.pulses,self.decisions,self.skipped)
        try:
            current=self.read(cell,1,'program_initial',**meta)[-1]
            for step,v in enumerate(self.plan['ramp_v']):
                if current['q']<=self.plan['reset_q']:break
                row=self.impact(*cell,7,v,'program_reset',sign=0,step=step,**meta)
                if row is None:result['reset_stop']='current_guard';break
                current=row
            result['reset_crossed']=current['q']<=self.plan['reset_q']
            confirmed=self.read(cell,5,'reset_confirmation',**meta)
            result['reset_confirmation_task_ids']=[r['task_id'] for r in confirmed]
            result['reset_confirmed']=result['reset_crossed'] and st.mean(r['q'] for r in confirmed)<=.005 and confirmed[-1]['q']<=.005
            current=confirmed[-1];lo,hi=target*.99,target*1.01
            reason='reset_unconfirmed'
            if result['reset_confirmed']:
                reason='set_ramp_exhausted'
                for step,v in enumerate(self.plan['ramp_v']):
                    if lo<=current['q']<=hi:reason='hit';break
                    if current['q']>hi:reason='overshoot';break
                    row=self.impact(*cell,7,v,'program_set',sign=1,step=step,target_q=target,**meta)
                    if row is None:reason='current_guard';break
                    current=row
                if lo<=current['q']<=hi:reason='hit'
                elif current['q']>hi:reason='overshoot'
            post=self.read(cell,5,'program_postread',**meta)
            result.update(status='completed',endpoint=reason,endpoint_q=current['q'],post_q_mean=st.mean(r['q'] for r in post),
                post_q_sd=st.stdev(r['q'] for r in post),post_signed_error_q=st.mean(r['q'] for r in post)-target,
                post_task_ids=[r['task_id'] for r in post],post_all_inside=all(lo<=r['q']<=hi for r in post))
            return result
        finally:
            result.update(programming_pulse_attempts=self.pulses-start[0],programming_decisions=self.decisions-start[1],
                programming_skipped_current=self.skipped-start[2])
            self.append('programming.jsonl',{**meta,**result})

    def mvm(self,cell,phase,active=True,**meta):
        wl,bl=cell;vol=[0.]*32
        if active:
            previous=self.last_mode7.get(tuple(cell))
            if previous is None:raise ValueError('Missing current mode7 readback')
            current=.1/previous[0]
            if not math.isfinite(current) or current>.005 or current*EXPECTED_BOARD['sum_gain']>4:
                self.append('rejected_commands.jsonl',dict(cell=cell,phase=phase,reason='provisional_single_input_model_guard',
                    predicted_current_a=current,predicted_output_v=current*EXPECTED_BOARD['sum_gain'],**meta))
                raise ValueError('Provisional single-input current/headroom model guard')
            vol[bl]=.1
        row=self.impact(wl,bl,10,vol,phase,**meta)
        return dict(task_id=row['task_id'],raw_adc=row['raw_adc'],voltage_v=row['output_voltage_v'])

    def anchors(self,active,phase,meta):
        records=[]
        for repeat in range(5):
            rows=[]
            for side in ('positive_cell','negative_cell'):
                zero=self.mvm(active[0][side],phase,False,repeat=repeat,side=side,role='zero',**meta)
                singles=[self.mvm(s[side],phase,True,repeat=repeat,side=side,role='single',tap=s['tap_index'],**meta) for s in active]
                rows.append(dict(zero=zero,singles=singles))
            records.append(rows)
        return records

    def block(self,active,chosen,phase,event):
        rows=[];values=[]
        for side in ('positive_cell','negative_cell'):
            before=self.mvm(active[0][side],phase,False,side=side,role='zero_before',**event)
            signals=[self.mvm(s[side],phase,True,side=side,role='signal',tap=s['tap_index'],**event) for s in chosen]
            if not chosen:signals=[self.mvm(active[0][side],phase,False,side=side,role='empty_zero_signal',**event)]
            after=self.mvm(active[0][side],phase,False,side=side,role='zero_after',**event)
            values.append(sum(s['voltage_v']-(before['voltage_v']+after['voltage_v'])/2 for s in signals))
            rows.append(dict(side=side,zero_before=before,signals=signals,zero_after=after))
        return values,rows

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
        correction=select_matrix(phases['train'],phases['selection'],mvm_output(1,EXPECTED_BOARD,self.converters))
        correction.update(reference_h_v=h,observations=phases,
            max_singleton_change_v=max(abs(apply_matrix(correction['matrix'],[h[0][i],h[1][i]])[j]-h[j][i])
                for i in range(len(active)) for j in range(2)))
        save_json(self.output/'frozen_matrix.json',correction)
        return correction

    def collect_tile(self,f,tile,fixture):
        active=slots(tile);fid=f['filter_id'];tid=tile['tile_id'];meta=dict(filter_id=fid,tile_id=tid)
        result=dict(**meta,status='failed',programming=[],partials=[])
        try:
            wmax=max(abs(s['trained_weight']) for t in f['tiles'] for s in slots(t))
            for s in active:
                w=s['trained_weight']
                for side,polarity in [('positive_cell',1),('negative_cell',-1)]:
                    target=.015+polarity*.005*w/wmax
                    result['programming'].append(self.program(s[side],target,dict(**meta,tap=s['tap_index'],side=side,trained_weight=w)))
            result['pre_readback']=[self.read(s[side],3,'tile_pre_readback',tap=s['tap_index'],side=side,**meta)
                for s in active for side in ('positive_cell','negative_cell')]
            anchors=self.anchors(active,'calibration',meta);save_json(self.output/'calibration_anchors.json',anchors)
            correction=self.matrix_calibration(active,anchors,meta);result['matrix']=correction
            differences=[]
            for row in anchors:
                corrected=[apply_matrix(correction['matrix'],[row[j]['singles'][i]['voltage_v']-row[j]['zero']['voltage_v'] for j in range(2)]) for i in range(len(active))]
                differences.append([pair[0]-pair[1] for pair in corrected])
            cal=scalar_calibration([s['trained_weight'] for s in active],differences,mvm_output(5,EXPECTED_BOARD,self.converters))
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


def verify_pilot(name,source,hashes,fixture,routing):
    directory=local_path(name);manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('stage')!='pilot' or manifest.get('status')!='completed' or manifest.get('source')!=source:
        raise ValueError('Pilot must be completed and from the same backend')
    if manifest.get('source_sha256')!=hashes:raise ValueError('Pilot source/input/protocol mismatch')
    recorded=manifest.get('result_sha256',{})
    required={'tiles/f0_t0/raw.jsonl','tiles/f0_t0/tile.json','tiles/f0_t0/partials.jsonl',
              'tiles/f0_t0/calibration_anchors.json','tiles/f0_t0/frozen_calibration.json',
              'tiles/f0_t0/frozen_matrix.json','tiles/f0_t0/matrix_probes.jsonl','summary.json','protocol.json','board.ini'}
    if not required<=set(recorded):raise ValueError('Pilot lacks required immutable artifacts')
    for relative,digest in recorded.items():
        path=directory/relative
        if path.resolve().is_relative_to(directory.resolve()) is False or sha256(path)!=digest:
            raise ValueError('Pilot artifact hash mismatch: '+relative)
    tile=json.loads((directory/'tiles/f0_t0/tile.json').read_text(encoding='utf-8'))
    if tile.get('status')!='completed' or not tile.get('decode_usable') or (tile.get('filter_id'),tile.get('tile_id'))!=(0,0):
        raise ValueError('Pilot is incomplete or scalar is unresolved; review before continuation')
    active=slots(routing['filters'][0]['tiles'][0])
    expected_cells={tuple(s[side]) for s in active for side in ('positive_cell','negative_cell')}
    if len(tile['programming'])!=28 or {tuple(p['cell']) for p in tile['programming']}!=expected_cells or any(p['status']!='completed' for p in tile['programming']):
        raise ValueError('Pilot programming coverage incomplete')
    expected={(c,r,t) for c in range(10) for r in range(5) for t in range(12)}
    actual={(p['case'],p['repeat'],p['t']) for p in tile['partials']}
    if len(tile['partials'])!=600 or actual!=expected:raise ValueError('Pilot partial ledger incomplete/duplicate')
    result_rows={};pending=None;freeze=None;result_count=0;first_cifar=None
    with (directory/'tiles/f0_t0/raw.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['event']=='request':
                if pending is not None or row['task_id']!=result_count+1:raise ValueError('Pilot request ledger')
                if row.get('phase')=='cifar' and first_cifar is None:first_cifar=row['task_id']
                pending=row
            elif row['event']=='result':
                if pending is None or row['task_id']!=pending['task_id'] or row['status']!='ok':raise ValueError('Pilot response ledger')
                if any(row.get(k)!=v for k,v in pending.items() if k!='event'):
                    raise ValueError('Pilot request/response immutable fields differ')
                if row['session_id']!=manifest['session_id'] or row['source']!=source:raise ValueError('Pilot source/session')
                result_rows[row['task_id']]=row;result_count+=1;pending=None
            elif row['event']=='calibration_frozen':
                if freeze is not None or first_cifar is not None or row['before_task_id']!=result_count+1 or pending is not None:
                    raise ValueError('Pilot calibration commitment order/count')
                freeze=row
    if pending is not None or result_count!=tile['impact_attempts'] or freeze is None or freeze['before_task_id']!=first_cifar:
        raise ValueError('Pilot incomplete raw')
    cal=json.loads((directory/'tiles/f0_t0/frozen_calibration.json').read_text(encoding='utf-8'))
    matrix=json.loads((directory/'tiles/f0_t0/frozen_matrix.json').read_text(encoding='utf-8'))
    if matrix!=tile['matrix'] or freeze.get('matrix_sha256')!=recorded['tiles/f0_t0/frozen_matrix.json']:
        raise ValueError('Pilot matrix commitment mismatch')
    if cal!=tile['calibration'] or freeze['sha256']!=recorded['tiles/f0_t0/frozen_calibration.json']:
        raise ValueError('Pilot scalar commitment mismatch')
    anchors=json.loads((directory/'tiles/f0_t0/calibration_anchors.json').read_text(encoding='utf-8'))
    if len(anchors)!=5 or any(len(row)!=2 or any(len(side['singles'])!=14 for side in row) for row in anchors):
        raise ValueError('Pilot calibration coverage incomplete')
    for repeat in anchors:
        for side in repeat:
            for entry in [side['zero']]+side['singles']:
                raw=result_rows.get(entry['task_id'])
                if raw is None or raw['phase']!='calibration' or raw['task_id']>=freeze['before_task_id'] or raw['raw_adc']!=entry['raw_adc'] or raw['output_voltage_v']!=entry['voltage_v']:
                    raise ValueError('Pilot calibration source mismatch')
    differences=[]
    for row in anchors:
        corrected=[apply_matrix(matrix['matrix'],[row[j]['singles'][i]['voltage_v']-row[j]['zero']['voltage_v'] for j in range(2)]) for i in range(14)]
        differences.append([pair[0]-pair[1] for pair in corrected])
    recomputed=scalar_calibration([s['trained_weight'] for s in active],differences,cal['five_adc_quality_reference_v'])
    if any(cal[k]!=v for k,v in recomputed.items()):raise ValueError('Pilot frozen scalar arithmetic mismatch')
    used=[]
    for p in tile['partials']:
        ci,r,t=p['case'],p['repeat'],p['t'];case=fixture['cases'][ci]
        chosen=[s for s in active if case['spikes_t12x75'][t][s['tap_index']]]
        if p['original_test_index']!=case['original_test_index'] or p['active_taps']!=[s['tap_index'] for s in chosen]:
            raise ValueError('Pilot inputs/taps changed')
        if len(p['measured_rows'])!=2:raise ValueError('Missing pilot WL')
        values=[]
        for row,side in zip(p['measured_rows'],('positive_cell','negative_cell')):
            entries=[row['zero_before']]+row['signals']+[row['zero_after']]
            if len(row['signals'])!=max(1,len(chosen)) or row['side']!=side:raise ValueError('Missing pilot partial')
            for i,e in enumerate(entries):
                raw=result_rows.get(e['task_id']);used.append(e['task_id'])
                if raw is None or raw.get('phase')!='cifar' or raw['task_id']<freeze['before_task_id']:
                    raise ValueError('Missing pilot measured response')
                if any(raw[k]!=p[k] for k in ('case','repeat','t','original_test_index')) or raw['side']!=side:
                    raise ValueError('Pilot response coordinate mismatch')
                if raw['raw_adc']!=e['raw_adc'] or raw['output_voltage_v']!=e['voltage_v']:raise ValueError('Pilot response changed')
                expected_vol=[0.]*32
                if 0<i<len(entries)-1 and chosen:expected_vol[chosen[i-1][side][1]]=.1
                if raw['command_si']['vol']!=expected_vol or raw['command_si']['wl']!=active[0][side][0]:
                    raise ValueError('Pilot physical routing mismatch')
            values.append(sum(e['voltage_v']-(row['zero_before']['voltage_v']+row['zero_after']['voltage_v'])/2 for e in row['signals']))
        corrected=apply_matrix(matrix['matrix'],values);difference=corrected[0]-corrected[1]
        if values!=p['raw_row_sums_v'] or corrected!=p['corrected_row_sums_v'] or values[0]-values[1]!=p['raw_signed_voltage_v']:
            raise ValueError('Pilot raw/matrix partial mismatch')
        if difference!=p['signed_voltage_v'] or difference/cal['alpha_v_per_weight']!=p['trained_unit_contribution']:
            raise ValueError('Pilot decoded partial mismatch')
    if len(used)!=len(set(used)) or len(used)!=4600 or {k for k,r in result_rows.items() if r['phase']=='cifar'}!=set(used):
        raise ValueError('Pilot measured coverage mismatch')
    return tile,dict(session=name,manifest_sha256=sha256(directory/'manifest.json'),result_sha256=recorded,
                     measured_data_reused=True,not_new_measurement=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    for flag in ('run-simulator','run-hardware','dry-run'):mode.add_argument('--'+flag,action='store_true')
    parser.add_argument('--stage',choices=('pilot','remaining'),default='pilot');parser.add_argument('--pilot-session')
    args=parser.parse_args(argv)
    if not(args.run_simulator or args.run_hardware or args.dry_run):parser.print_help();return 0
    if (args.stage=='remaining')!=bool(args.pilot_session):parser.error('Remaining requires --pilot-session; pilot forbids it')
    plan,routing,fixture=load_inputs();source='simulator' if args.run_simulator else 'hardware'
    config,snapshot,hashes=check_sources(source)
    for name in (*PINS,CONFIG,'cifar_memristor.py','cifar_memristor_analysis.py','experiment.py'):
        hashes[name]=sha256(local_path(name))
    selected=tiles(routing)[:1] if args.stage=='pilot' else tiles(routing)[1:]
    budget=counts(selected,fixture);imported=[];parent=None
    if args.pilot_session:
        pilot,parent=verify_pilot(args.pilot_session,source,hashes,fixture,routing);imported=[pilot]
    description=dict(protocol_id=plan['protocol_id'],stage=args.stage,source=source,source_sha256=hashes,
        hardware_access=source=='hardware' and not args.dry_run,**budget,aggregate_maximum_impacts=222118,
        aggregate_maximum_programming_pulses=66000,parent_pilot=parent,physical_pulse_width_s=None,
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
        sid=str(uuid.uuid4());output=local_path('results/cifar_memristor')/sid;output.mkdir(parents=True)
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
            run=TileRun(connector,converters,folder,sid,tile_plan,source)
            result=run.collect_tile(f,tile,fixture);results.append(result)
            print(f"filter {f['filter_id']} tile {tile['tile_id']}: alpha usable={result['decode_usable']}",flush=True)
            if sum(r['impact_attempts'] for r in results)>222118 or sum(r['programming_pulse_attempts'] for r in results)>66000:
                raise ValueError('Combined pilot/remaining budget exceeded')
        summary=aggregate(results,routing,fixture)
        summary.update(status='completed',stage=args.stage,technical_completed=True,
            decode_usable=all(r['decode_usable'] for r in results),filter_tiles_completed=len(results),
            acquisition_complete=args.stage=='remaining' and len(results)==28,
            decoded_conv_bn_lif_complete=args.stage=='remaining' and len(summary['trajectories'])==200 and all(r['decode_usable'] for r in results),
            pilot_only_partial_convolution=args.stage=='pilot',
            total_impacts_including_pilot=sum(r['impact_attempts'] for r in results),
            total_pulses_including_pilot=sum(r['programming_pulse_attempts'] for r in results),
            tile_status=[dict(filter_id=r['filter_id'],tile_id=r['tile_id'],decode_usable=r['decode_usable']) for r in results])
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
