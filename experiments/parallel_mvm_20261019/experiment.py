"""Read-only dense MVM and synthetic digital LIF; explicit native Connector launch."""
import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import statistics
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cell_calibration import Discovery, CELLS, WRITE_GUARD, SIM_INI, LOCK, check_sources
from read_all import EXPECTED_BOARD, local_path, native_connector, now, sha256, save_json, json_value
from native_conversions import voltage, mvm_output
from experiment_hardware import launch_provenance
from experiments.survey256_20261019.experiment import execution_environment

DIRECTORY = 'experiments/parallel_mvm_20261019'
CONFIG = DIRECTORY+'/protocol.json'
PINS = {
    'cell_calibration.py':'eff0b2f72586f6a6659a13fba410c6ad0d564bd95d3b83e24b13cf739121c2e8',
    'read_all.py':'b30ae2f4dd3cd166dbc5cefb07b558127de2bbdba1f6674c40b1b45f4dd74d5f',
    'native_conversions.py':'2ad2e9c057bb14979b083a6e846e8c417c080a6c5273d848e31f893b7970253e',
    'experiment_hardware.py':'059dc4f827943a9df631631f52e776b57182afc9d36db0c1ad5246427f73548c',
    'experiments/survey256_20261019/experiment.py':'0732c2f10ccae7bb9c590c56bbcf61686621e530ff8176eec99c5a86a621825a',
}
SPEC = dict(protocol_id='readonly-parallel-mvm-v1-20261019', seed=20261019,
    basis_levels_v=[.025,.05,.1], basis_repeats=3, fit_repeats=2, test_repeats=3,
    fit_densities=[2,4,8,16], test_densities=[1,2,4,8,16,32],
    lif_densities=[0,1,2,4,8,16,32,16,8,4,2,0], lif_patterns=2, lif_repeats=2,
    initial_final_read_repeats=3, fresh_active_cell_reads_before_every_vector=1,
    maximum_impacts=18304, maximum_mode7_reads=13472, maximum_mode10=4832,
    maximum_programming_decisions=0, maximum_programming_pulses=0, programming_guard=WRITE_GUARD,
    simulator_pulse_width_s=0., baseline_scope='fresh active-cell native R before each vector; all read-only',
    current_model_limit_a=.005, gui_output_limit_v=3.3, adc_headroom_fraction=.8,
    guard_voltage='max(requested_v,native_decoded_v)', source_states_matched=False,
    basis_headroom_factor=2., basis_sd_multiplier=3.,
    common_amplitude='largest safe basis level for all32 inputs/all8WL before fit, else .025 with explicit skips; frozen thereafter',
    calibration=dict(ridge=.0001, minimum_design_rank=9, maximum_condition=1000000.,
                     maximum_original_adc_operator_gain=10., minimum_complete_fit_masks=9,
                     targets='same-chip singleton superposition, not trained weights',
                     invalid_fit='calibration unavailable; raw baseline retained'),
    lif=dict(leak=.5, threshold=1., reset=0., diagnostic_gains_per_volt=[10.,100.,1000.],
             pairs=[[0,1],[2,3],[4,5],[6,7]], all_silent='uninformative, not passed'),
    physical_pulse_width_s=None, trained_weights_programmed=False, cifar_replayed=False,
    physical_lif=False, atomic_eight_output_vector=False)


def validate(plan):
    if plan != SPEC:
        raise ValueError('Protocol differs from frozen parallel MVM specification')


def schedule(plan):
    validate(plan)
    rng = random.Random(plan['seed'])
    traces = []
    reserved = set()
    for pattern in range(2):
        columns = list(range(32)); rng.shuffle(columns)
        masks = [sorted(columns[:d]) for d in plan['lif_densities']]
        reserved.update(tuple(m) for m in masks if len(m) in (2,4,8,16))
        traces.append(masks)
    def partitions(used):
        masks = []
        for density in (2,4,8,16):
            while True:
                columns = list(range(32)); rng.shuffle(columns)
                candidate = [tuple(sorted(columns[i:i+density])) for i in range(0,32,density)]
                if all(m not in used for m in candidate):
                    break
            used.update(candidate); masks.extend(map(list,candidate))
        return masks
    used = set(reserved)
    fit = partitions(used)
    test = [[bl] for bl in range(32)] + partitions(used) + [list(range(32))]
    entries = []
    def add(phase, mask, **extra):
        entries.append(dict(vector_id=len(entries)+1, phase=phase, mask=mask, **extra))
    for level in plan['basis_levels_v']:
        for repeat in range(3):
            for mask in [[]]+[[bl] for bl in range(32)]:
                add('basis',mask,nominal_v=level,repeat=repeat)
    for phase, masks, repeats in (('fit',fit,2),('test',test,3)):
        for repeat in range(repeats):
            add(phase,[],repeat=repeat,bracket='before')
            for mask in masks:
                add(phase,mask,repeat=repeat)
            add(phase,[],repeat=repeat,bracket='after')
    for pattern, masks in enumerate(traces):
        for repeat in range(2):
            for timestep, mask in enumerate(masks):
                add('lif',mask,pattern=pattern,repeat=repeat,t=timestep)
    mode10 = len(entries)*8
    mode7 = 1536 + 8*sum(len(e['mask']) for e in entries)
    if (mode7,mode10,mode7+mode10) != (13472,4832,18304):
        raise ValueError('Frozen schedule count mismatch')
    return entries


def summarize_basis(vectors):
    result = {}
    for level in SPEC['basis_levels_v']:
        points = []
        for mask in [[]]+[[bl] for bl in range(32)]:
            samples = [v['adc'] for v in vectors if v['phase']=='basis' and v['mask']==mask and v['nominal_v']==level]
            points.append([dict(mean=statistics.mean([s[wl] for s in samples]),
                                sd=statistics.stdev([s[wl] for s in samples]))
                           if len(samples)==3 and all(s[wl] is not None for s in samples) else None
                           for wl in range(8)])
        result[str(level)] = points
    return result


def reference(basis, level, mask):
    points = basis[str(level)]
    result = []
    for wl in range(8):
        if points[0][wl] is None or any(points[bl+1][wl] is None for bl in mask):
            result.append(None)
        else:
            zero = points[0][wl]['mean']
            result.append(zero+sum(points[bl+1][wl]['mean']-zero for bl in mask))
    return result


def freeze_calibration(vectors, basis, level):
    import numpy as np
    rows = [v for v in vectors if v['phase']=='fit' and all(x is not None for x in v['adc'])]
    paired = [(v,reference(basis,level,v['mask'])) for v in rows]
    paired = [(v,y) for v,y in paired if all(z is not None for z in y)]
    fit_masks = {tuple(v['mask']) for v,y in paired if v['mask']}
    result = dict(status='unavailable', fitting_vector_ids=[v['vector_id'] for v,y in paired],
                  complete_distinct_nonzero_masks=len(fit_masks), nominal_v=level,
                  target='same-chip singleton-superposition ADC, not trained weight accuracy',
                  frozen_before_test=True, calibration_spec=SPEC['calibration'])
    if len(fit_masks)<9:
        result['reason']='insufficient_complete_fit_masks'; return result
    x = np.asarray([v['adc'] for v,y in paired],dtype=float)
    y = np.asarray([y for v,y in paired],dtype=float)
    xm, ym = x.mean(0), y.mean(0)
    xs, ys = x.std(0), y.std(0)
    if np.any(xs<=1e-12) or np.any(ys<=1e-12):
        result['reason']='constant_fit_channel'; return result
    xn, yn = (x-xm)/xs, (y-ym)/ys
    design = np.column_stack([xn,np.ones(len(x))])
    rank, condition = int(np.linalg.matrix_rank(design)), float(np.linalg.cond(design))
    result.update(design_rank=rank, normalized_design_condition=condition if math.isfinite(condition) else None)
    if rank!=9 or not math.isfinite(condition) or condition>1e6:
        result['reason']='rank_or_condition_limit'; return result
    # Centered ridge leaves intercept unpenalized; no pseudoinverse or heldout selection.
    b = np.linalg.solve(xn.T@xn+SPEC['calibration']['ridge']*np.eye(8),xn.T@yn)
    c = (b*ys[None,:]/xs[:,None]).T
    offset = ym-c@xm
    gain = float(np.linalg.norm(c,2))
    result['original_adc_operator_gain'] = gain
    if not np.isfinite(c).all() or not np.isfinite(offset).all() or gain>10.:
        result['reason']='nonfinite_or_gain_limit'; return result
    result.update(status='available', C=c.tolist(), offset=offset.tolist(),
                  training_rmse_adc=float(np.sqrt(np.mean((x@c.T+offset-y)**2))))
    return result


def apply_calibration(adc, calibration):
    if calibration.get('status')!='available' or any(v is None for v in adc):
        return None
    return [sum(a*b for a,b in zip(row,adc))+d for row,d in zip(calibration['C'],calibration['offset'])]


def analyze(vectors,basis,level,calibration):
    from manager.service import converters
    by_density=[]
    for density in (1,2,4,8,16,32):
        selected=[v for v in vectors if v['phase']=='test' and len(v['mask'])==density]
        for wl in range(8):
            raw_errors=[]; corrected_errors=[]
            for v in selected:
                expected=reference(basis,level,v['mask'])[wl]
                if v['adc'][wl] is not None and expected is not None:
                    raw_errors.append(v['adc'][wl]-expected)
                corrected=apply_calibration(v['adc'],calibration)
                if corrected is not None and expected is not None:
                    corrected_errors.append(corrected[wl]-expected)
            by_density.append(dict(density=density,wl=wl,scheduled=len(selected),raw_valid=len(raw_errors),
                corrected_valid=len(corrected_errors),
                raw_rmse_adc=math.sqrt(statistics.mean(e*e for e in raw_errors)) if raw_errors else None,
                corrected_rmse_adc=math.sqrt(statistics.mean(e*e for e in corrected_errors)) if corrected_errors else None))
    traces=[]
    zero=reference(basis,level,[])
    def to_volts(adc):
        # Derived affine ADC values may be fractional; retain native converter rounding.
        return converters.convert_adc_to_volt(1,EXPECTED_BOARD['adc_bit'],EXPECTED_BOARD['vol_ref_adc'],adc)
    for gain,pattern in [(g,p) for g in SPEC['lif']['diagnostic_gains_per_volt'] for p in range(2)]:
        for repeat in range(2):
            rows=sorted([v for v in vectors if v['phase']=='lif' and v['pattern']==pattern and v['repeat']==repeat],key=lambda v:v['t'])
            trajectory={}
            for branch in ('reference','raw','calibrated'):
                states=[0.]*4; history=[]; complete=len(rows)==12
                for v in rows:
                    values=reference(basis,level,v['mask']) if branch=='reference' else v['adc'] if branch=='raw' else apply_calibration(v['adc'],calibration)
                    if values is None or any(x is None for x in values) or any(x is None for x in zero):
                        complete=False; break
                    currents=[gain*((to_volts(values[a])-to_volts(zero[a]))-(to_volts(values[b])-to_volts(zero[b]))) for a,b in SPEC['lif']['pairs']]
                    pre=[.5*s+i for s,i in zip(states,currents)]
                    spikes=[int(s>=1.) for s in pre]
                    states=[0. if spike else value for value,spike in zip(pre,spikes)]
                    history.append(dict(t=v['t'],vector_id=v['vector_id'],current=currents,pre_reset=pre,membrane=states[:],spikes=spikes))
                trajectory[branch]=dict(status='completed' if complete else 'incomplete',steps=history,
                    informative_spiking=bool(complete and any(any(s['spikes']) for s in history)),
                    interpretation='all-silent is uninformative, not passed')
            comparisons={}
            ref=trajectory['reference']
            for name in ('raw','calibrated'):
                item=trajectory[name]
                if ref['status']==item['status']=='completed':
                    errors=[b-a for rr,hh in zip(ref['steps'],item['steps']) for a,b in zip(rr['pre_reset'],hh['pre_reset'])]
                    mismatches=sum(a!=b for rr,hh in zip(ref['steps'],item['steps']) for a,b in zip(rr['spikes'],hh['spikes']))
                    comparisons[name]=dict(membrane_rmse=math.sqrt(statistics.mean(e*e for e in errors)),spike_mismatches=mismatches,total_spike_decisions=48)
            traces.append(dict(gain_per_volt=gain,pattern=pattern,repeat=repeat,trajectories=trajectory,comparisons=comparisons))
    return dict(per_wl_density=by_density,lif=traces,calibration_status=calibration.get('status'),
                reference_is_same_chip_superposition=True,trained_weight_accuracy_measured=False,
                classification_accuracy_measured=False)


class Acquisition(Discovery):
    def __init__(self,connector,converters,output,sid,plan,source):
        validate(plan)
        super().__init__(connector,converters,output,sid,plan,source)
        self.vectors=[]; self.reads=[]; self.basis={}; self.g={}; self.common_v=None
        self.mode7_count=self.mode10_count=0
        self.calibration=dict(status='unavailable',reason='fit_not_completed')

    def impact(self,wl,bl,mode,vol,phase,sign=0,**metadata):
        if mode==7:
            if vol!=0 or sign!=0:
                raise ValueError('Read-only protocol forbids programming')
            if self.mode7_count>=13472:
                raise ValueError('Read budget exhausted')
            self.mode7_count+=1
            result=super().impact(wl,bl,7,0.,phase,sign=0,**metadata)
            self.g[wl,bl]=result['conductance_siemens']
            self.reads.append({k:result[k] for k in ('task_id','wl','bl','phase','raw_adc','native_r','conductance_siemens')})
            return result
        if mode!=10 or wl not in range(8) or not isinstance(vol,list) or len(vol)!=32 or any(type(v) not in (int,float) or not math.isfinite(v) or v not in (0.,.025,.05,.1) for v in vol):
            raise ValueError('Unreviewed MVM command')
        if self.sequence>=18304 or self.mode10_count>=4832:
            raise ValueError('MVM budget exhausted')
        self.sequence+=1; self.mode10_count+=1
        command=dict(mode_flag=10,vol=list(vol),wl=wl,id=0)
        row=dict(session_id=self.session_id,source=self.source,task_id=self.sequence,wl=wl,phase=phase,
                 timestamp_utc=now(),monotonic_time_s=time.monotonic(),command_si=command,
                 native_voltage_conversion=[voltage(v,EXPECTED_BOARD,self.converters) for v in vol],**metadata)
        self.emit(dict(event='request',**row)); started=time.monotonic()
        try:
            result=self.connector.impact({**command,'vol':list(vol)})
            row['native_result']=json_value(result)
            if not isinstance(result,(tuple,list)) or len(result)!=3:
                raise ValueError('Expected native R,ID,ADC tuple')
            native,returned,adc=result
            row.update(native_r=json_value(native),returned_id=json_value(returned),raw_adc=json_value(adc))
            if type(returned) is not int or returned!=0:
                raise ValueError('Native MVM ID must be zero')
            row['output_voltage_v']=mvm_output(adc,EXPECTED_BOARD,self.converters)
            if row['output_voltage_v']>4.:
                raise ValueError('Observed MVM exceeds ADC headroom')
            row['status']='ok'; return row
        except (Exception,KeyboardInterrupt) as exc:
            row.update(status='error',error=f'{type(exc).__name__}: {exc}'); raise
        finally:
            self.emit(dict(event='result',latency_s=time.monotonic()-started,**row))

    def guard(self,wl,mask,level,use_basis=True):
        decoded=voltage(level,EXPECTED_BOARD,self.converters)['native_decoded_v']
        guard_voltage=max(level,decoded)
        current=guard_voltage*sum(self.g[wl,bl] for bl in mask)
        prediction=330*current
        reasons=[]
        if current>.005 or prediction>=3.3 or prediction>4.:
            reasons.append('fresh_native_R_total_model_limit')
        decision=dict(wl=wl,current_model_a=current,output_model_v=prediction,reasons=reasons,
                      requested_v=level,native_decoded_v=decoded,guard_voltage_v=guard_voltage,
                      physical_current_compliance=False,bound_certified=False)
        if use_basis:
            points=self.basis[str(level)]
            if points[0][wl] is None or any(points[bl+1][wl] is None for bl in mask):
                reasons.append('basis_missing')
            else:
                z=points[0][wl]
                upper=z['mean']+3*z['sd']+2*sum(max(0.,points[bl+1][wl]['mean']-z['mean']+3*math.hypot(points[bl+1][wl]['sd'],z['sd'])) for bl in mask)
                decision['basis_heuristic_upper_adc']=upper
                if upper>.8*(2**EXPECTED_BOARD['adc_bit']-1):
                    reasons.append('basis_headroom_heuristic')
        decision['allowed']=not reasons
        return decision

    def vector(self,entry):
        level=entry.get('nominal_v',self.common_v)
        meta={k:v for k,v in entry.items() if k!='nominal_v'}
        # Refresh every active address on every WL before the eight sequential outputs.
        for wl in range(8):
            for bl in entry['mask']:
                self.impact(wl,bl,7,0.,entry['phase']+'_guard',vector_id=entry['vector_id'])
        record={**entry,'nominal_v':level,'adc':[None]*8,'task_ids':[None]*8,'guards':[]}
        for wl in range(8):
            decision=self.guard(wl,entry['mask'],level,use_basis=entry['phase']!='basis')
            record['guards'].append(decision)
            self.emit(dict(event='vector_guard',source=self.source,nominal_v=level,**meta,**decision))
            if decision['allowed']:
                row=self.impact(wl,0,10,[level if b in entry['mask'] else 0. for b in range(32)],entry['phase'],
                                vector_id=entry['vector_id'])
                record['adc'][wl]=row['raw_adc'];record['task_ids'][wl]=row['task_id']
        self.vectors.append(record); self.append('vectors.jsonl',record)

    def read_array(self,phase):
        for wl,bl in CELLS:
            for repeat in range(3):
                self.impact(wl,bl,7,0.,phase,repeat=repeat)

    def freeze_amplitude(self):
        options=[]
        for level in (.1,.05,.025):
            decisions=[self.guard(wl,list(range(32)),level) for wl in range(8)]
            options.append(dict(nominal_v=level,guards=decisions))
        self.common_v=next((o['nominal_v'] for o in options if all(d['allowed'] for d in o['guards'])),.025)
        save_json(self.output/'frozen_amplitude.json',dict(nominal_v=self.common_v,options=options,
            selected_before_fit=True,changes_after_freeze=False,
            fallback_if_no_safe_dense_level='lowest level, retain per-WL guard skips'))

    def run(self):
        entries=schedule(self.plan)
        try:
            if not self.connector.open_port():
                raise ConnectionError('Connector.open_port returned false')
            self.read_array('initial_array')
            previous=None
            for entry in entries:
                if entry['phase']!=previous:
                    if previous=='basis':
                        self.basis=summarize_basis(self.vectors)
                        save_json(self.output/'basis.json',self.basis)
                        self.freeze_amplitude()
                    if previous=='fit':
                        self.calibration=freeze_calibration(self.vectors,self.basis,self.common_v)
                        save_json(self.output/'frozen_calibration.json',self.calibration)
                    previous=entry['phase'];print('parallel MVM phase '+previous,flush=True)
                self.vector(entry)
            self.read_array('final_array')
            self.summary['status']='completed'
        except (Exception,KeyboardInterrupt) as exc:
            self.summary['error']=f'{type(exc).__name__}: {exc}'
        finally:
            try:
                if not self.connector.close_port():
                    raise ConnectionError('Connector.close_port returned false')
            except Exception as exc:
                self.summary.update(status='failed',close_error=f'{type(exc).__name__}: {exc}')
            self.log.close()
            self.summary.update(impact_attempts=self.sequence,mode7_reads=self.mode7_count,mode10_attempts=self.mode10_count,
                programming_pulse_attempts=self.pulses,nominal_v=self.common_v,vectors_recorded=len(self.vectors),
                complete_vectors=sum(all(a is not None for a in v['adc']) for v in self.vectors),
                calibration_status=self.calibration['status'],finished_utc=now())
            save_json(self.output/'summary.json',self.summary)
            save_json(self.output/'observations.json',dict(vectors=self.vectors,readbacks=self.reads))
            if self.basis and self.common_v is not None:
                save_json(self.output/'analysis.json',analyze(self.vectors,self.basis,self.common_v,self.calibration))
        return self.summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    for flag in ('run-simulator','run-hardware','dry-run'):
        modes.add_argument('--'+flag,action='store_true')
    args=parser.parse_args(argv)
    if not(args.run_simulator or args.run_hardware or args.dry_run):
        parser.print_help();return 0
    source='simulator' if args.run_simulator else 'hardware'
    plan_bytes=local_path(CONFIG).read_bytes();plan=json.loads(plan_bytes);validate(plan)
    entries=schedule(plan)
    for name,pin in PINS.items():
        if sha256(local_path(name))!=pin:
            raise ValueError('Unreviewed helper: '+name)
    config,snapshot,hashes=check_sources(source)
    for name in [*PINS,CONFIG,DIRECTORY+'/experiment.py']:
        hashes[name]=sha256(local_path(name))
    description=dict(source=source,protocol=plan,source_sha256=hashes,hardware_access=args.run_hardware,
        schedule_sha256=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest())
    if args.dry_run:
        print(json.dumps(description));return 0
    lock=local_path(LOCK);locked=False
    if source=='hardware':
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as stream:stream.write('parallel MVM '+now()+'\n')
        locked=True
    try:
        sid=str(uuid.uuid4());output=local_path('results/parallel_mvm_20261019')/sid;output.mkdir(parents=True)
        (output/'protocol.json').write_bytes(plan_bytes);(output/'board.ini').write_bytes(snapshot)
        save_json(output/'schedule.json',entries)
        manifest=dict(session_id=sid,status='failed',started_utc=now(),invocation=launch_provenance(argv),**description)
        save_json(output/'start_manifest.json',manifest)
        try:
            if source=='hardware':connector=native_connector(config)
            else:
                from experiments.unn_rev3_20261014.paths import use_local_dependencies
                use_local_dependencies()
                import numpy as np
                from manager.board import Connector
                from simulator.src import create_crossbar_array
                random.seed(plan['seed']);np.random.seed(plan['seed'])
                serial=output/'array';create_crossbar_array(str(serial),32,8)
                (output/'simulator_settings.ini').write_bytes(local_path(SIM_INI).read_bytes())
                manifest['array_initial_sha256']=sha256(serial.with_suffix('.cb'))
                connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
                    board_type=config.get('board','board_type'),task_format='SI',config=config,crossbar_serial=str(serial))
            from manager.service import converters
            manifest['execution_environment']=execution_environment()
            result=Acquisition(connector,converters,output,sid,plan,source).run()
            manifest.update(status=result['status'],impact_attempts=result['impact_attempts'],error=result.get('error'))
        except (Exception,KeyboardInterrupt) as exc:
            manifest['error']=f'{type(exc).__name__}: {exc}'
        finally:
            manifest['finished_utc']=now()
            manifest['result_sha256']={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
            save_json(output/'manifest.json',manifest)
        print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'])))
        return 0 if manifest['status']=='completed' else 2
    finally:
        if locked:lock.unlink()


if __name__=='__main__':
    raise SystemExit(main())
