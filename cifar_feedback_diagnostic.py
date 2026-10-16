"""Fresh full diagnostic CIFAR acquisition; fixed gain, no fitted-alpha decoding."""
import argparse
import copy
import hashlib
import json
import logging
import math
import random
import statistics as st
import uuid

import cifar_feedback as feedback
from cifar_feedback import FeedbackRun,candidate_pools,counts,fidelity_report
from cifar_memristor import slots,tiles,ROOT,local_path,sha256,save_json,now,EXPECTED_BOARD
from cifar_memristor_analysis import scalar_calibration,apply_matrix,aggregate
from native_conversions import mvm_output
from cell_calibration import check_sources,LOCK
from read_all import native_connector
from experiment_hardware import launch_provenance

CONFIG='configs/cifar_feedback_diagnostic.json'
PINS={**feedback.PINS,
    'cifar_feedback.py':'0cd7b6dbbb789afa6a9b3fe71e1f50a9dd9c66715bfd1675112b69f9eb8cb5b4',
    'configs/cifar_feedback.json':'c0978558dda13ebb28ca25e93c52237802907a7b6d2ce3c552374e2b71e5350c'}
SPEC={**feedback.SPEC,'protocol_id':'cifar-mode10-feedback-full-fixed-gain-diagnostic-v1',
    'collection_scope':'fresh_all_28_tiles_300_weights_no_pilot_import',
    'decoder':'signed_C_actual_two_channel_voltage / fixed_source_target_gain',
    'fitted_alpha_usage':'diagnostic_only_never_decoder_never_abs',
    'quality_failure_policy':'retain_all_weight_errors_and_collect_all_tiles; transport_safety_invalid_ADC_insufficient_route_abort',
    'cifar_collection_on_fidelity_failure':'all_28_fresh_tiles_collected; quality_failure_never_blocks_collection',
    'acquisition_completion_is_accuracy_claim':False}
SPEC['raw_part_maximum_bytes']=32*1024*1024


class ChunkedJournal:
    """Exact UTF-8 line stream; request/result pairs may span ordered parts."""
    def __init__(self,stream,folder,limit):
        self.stream=stream;self.folder=folder;self.limit=limit;self.parts=[]
        self.current=dict(path='raw.jsonl',bytes=0,records=0)
        self.total_hash=hashlib.sha256();self.part_hash=hashlib.sha256();self.closed=False

    def _finish_part(self):
        self.stream.close()
        self.parts.append({**self.current,'sha256':self.part_hash.hexdigest()})

    def write(self,line):
        if self.closed:raise ValueError('Closed journal')
        if not line.endswith('\n') or '\n' in line[:-1]:raise ValueError('Journal requires one serialized JSON line')
        data=line.encode('utf-8')
        if len(data)>self.limit:raise ValueError('Single journal line exceeds part budget')
        if self.current['bytes']+len(data)>self.limit:
            self._finish_part();name=f'raw.{len(self.parts):05d}.jsonl'
            self.stream=(self.folder/name).open('x',encoding='utf-8',newline='\n')
            self.current=dict(path=name,bytes=0,records=0);self.part_hash=hashlib.sha256()
        self.stream.write(line);self.current['bytes']+=len(data);self.current['records']+=1
        self.part_hash.update(data);self.total_hash.update(data)

    def flush(self):self.stream.flush()

    def close(self):
        if self.closed:return
        self._finish_part();self.closed=True
        save_json(self.folder/'raw_parts.json',dict(parts=self.parts,sha256=self.total_hash.hexdigest(),
            bytes=sum(p['bytes'] for p in self.parts),records=sum(p['records'] for p in self.parts),
            order='concatenate parts in listed order; never pair within individual parts only',
            maximum_part_bytes=self.limit))


def decode_measured(matrix,measured_rows,gain):
    if not math.isfinite(gain) or gain<=0:raise ValueError('Invalid fixed decoder gain')
    values=apply_matrix(matrix,measured_rows)
    return (values[0]-values[1])/gain


def fixed_calibration(weights,differences,gain,adc_quality_reference):
    if not math.isfinite(gain) or gain<=0:raise ValueError('Invalid fixed calibration gain')
    fitted=scalar_calibration(weights,differences,adc_quality_reference)
    means=fitted['measured_difference_mean_v'];effective=[d/gain for d in means]
    residuals=[w-t for w,t in zip(effective,weights)]
    return dict(decoder_gain_v_per_weight=gain,decoder_gain_is_fixed_before_acquisition=True,
        fitted_alpha_diagnostic=fitted,decode_usable=True,decode_usable_means_finite_arithmetic_not_weight_quality=True,
        effective_weights=effective,weight_residuals=residuals,
        weight_rmse=math.sqrt(st.mean(x*x for x in residuals)),
        measured_difference_mean_v=means,calibration_uses_cifar=False)


def address_coverage(results):
    categories={key:set() for key in ('reset_probed','reset_eligible','selected','set_attempted','represented')}
    for result in results:
        for row in result['reset_candidates']:
            categories['reset_probed'].add(tuple(row['cell']))
            if row['eligible']:categories['reset_eligible'].add(tuple(row['cell']))
        for slot in result.get('effective_route',[]):
            for side in ('positive_cell','negative_cell'):categories['selected'].add(tuple(slot[side]))
        for pair in result['pairs']:
            if pair['programming_pulse_attempts']>0:categories['set_attempted'].add(tuple(pair[pair['active_set_side']]))
        if result.get('technical_completed'):
            for slot in result.get('effective_route',[]):
                for side in ('positive_cell','negative_cell'):categories['represented'].add(tuple(slot[side]))
    out={key:dict(count=len(value),addresses=[list(cell) for cell in sorted(value)]) for key,value in categories.items()}
    out['probed_but_not_selected']=[list(cell) for cell in sorted(categories['reset_probed']-categories['selected'])]
    out['coverage_does_not_prove_programmability']=True
    return out


class DiagnosticRun(FeedbackRun):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.log=ChunkedJournal(self.log,self.output,self.plan['raw_part_maximum_bytes'])

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

def load_inputs():
    for name,pin in PINS.items():
        if sha256(local_path(name))!=pin:raise ValueError('Pinned diagnostic dependency changed: '+name)
    _,routing,fixture=feedback.load_inputs()
    plan=json.loads(local_path(CONFIG).read_text(encoding='utf-8'))
    if plan!=SPEC:raise ValueError('Diagnostic protocol differs from reviewed configuration')
    return plan,routing,fixture


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    for flag in ('run-simulator','run-hardware','dry-run'):mode.add_argument('--'+flag,action='store_true')
    args=parser.parse_args(argv)
    if not(args.run_simulator or args.run_hardware or args.dry_run):parser.print_help();return 0
    plan,routing,fixture=load_inputs();source='simulator' if args.run_simulator else 'hardware'
    config,snapshot,hashes=check_sources(source)
    for name in (*PINS,CONFIG,'cifar_feedback_diagnostic.py','experiment.py'):
        hashes[name]=sha256(local_path(name))
    selected=tiles(routing);budget=counts(selected,fixture)
    description=dict(protocol_id=plan['protocol_id'],source=source,source_sha256=hashes,
        hardware_access=source=='hardware' and not args.dry_run,**budget,
        fixed_decoder_gain_v_per_weight=plan['target_gain_v_per_weight'][source],
        historical_pilot_imported=False,all_tiles_new_measurements=True,physical_pulse_width_s=None,
        task_id_scope='unique within tile; identify by session_id/filter_id/tile_id/task_id')
    if args.dry_run:print(json.dumps(description));return 0
    lock=local_path(LOCK);locked=False
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as stream:stream.write(f'CIFAR diagnostic {now()}\n')
        locked=True
    connector=None;manifest=None;output=None;results=[]
    try:
        sid=str(uuid.uuid4());output=local_path('results/cifar_feedback_diagnostic')/sid;output.mkdir(parents=True)
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
            serial=output/'array';create_crossbar_array(str(serial),32,8)
            manifest['array_initial_sha256']=sha256(serial.with_suffix('.cb'))
            connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                board_type='rp5_fpga_python',task_format='SI',config=config,crossbar_serial=str(serial))
        from manager.service import converters
        if not connector.open_port():raise ConnectionError('Connector.open_port returned false')
        for f,tile in selected:
            folder=output/'tiles'/f"f{f['filter_id']}_t{tile['tile_id']}";folder.mkdir(parents=True)
            per=counts([(f,tile)],fixture);tile_plan={**plan,**per,'maximum_programming_decisions':per['maximum_programming_pulses']}
            run=DiagnosticRun(connector,converters,folder,sid,tile_plan,source)
            result=run.collect_tile(f,tile,fixture);results.append(result)
            print(f"filter {f['filter_id']} tile {tile['tile_id']}: physical fidelity={result['physical_weight_quality_pass']}",flush=True)
            if sum(r['impact_attempts'] for r in results)>580210 or sum(r['programming_pulse_attempts'] for r in results)>74940:
                raise ValueError('Full diagnostic budget exceeded')
        summary=aggregate(results,routing,fixture)
        summary.update(status='completed',technical_completed=True,filter_tiles_completed=len(results),
            acquisition_complete=len(results)==28,
            decoded_conv_bn_lif_complete=len(results)==28 and len(summary['trajectories'])==200,
            physical_weight_quality_pass=all(r['physical_weight_quality_pass'] for r in results),
            fixed_decoder_gain_v_per_weight=plan['target_gain_v_per_weight'][source],
            fitted_alpha_used_for_decoding=False,diagnostic_acquisition_not_accuracy_qualification=True,
            total_impacts=sum(r['impact_attempts'] for r in results),
            total_pulses=sum(r['programming_pulse_attempts'] for r in results),address_coverage=address_coverage(results),
            tile_status=[dict(filter_id=r['filter_id'],tile_id=r['tile_id'],
                physical_weight_quality_pass=r['physical_weight_quality_pass'],
                fitted_alpha_v_per_weight=r['calibration']['fitted_alpha_diagnostic']['alpha_v_per_weight']) for r in results])
        save_json(output/'summary.json',summary);manifest.update(status='completed',acquisition_complete=summary['acquisition_complete'])
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
