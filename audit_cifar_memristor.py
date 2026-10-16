"""Independent, stdlib-only audit of recorded CIFAR Conv1 measurements.

Never imports acquisition/analysis/native packages and never creates a Connector.
Native conversion arithmetic below is checked against recorded source fingerprints.
"""
import argparse
import ast
from collections import Counter
from configparser import ConfigParser
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess

PINS = {
    'weights/prepared_cifar100_conv1.json': '86113ff1d2fdbe155056d2e1377cf380e7395120fd93ccf2e1efc7a0a933167c',
    'weights/conv1_weights.npy': '8be312b9eafb9de23f8340b17f848e7e0e2e7884293c826a6d9f6cbb5125d440',
    'weights/best_fs.pt': 'ee4fc4f58a3d010218116e9ad4c7ba808787959f9ed7687cd58cbb32d90f62fb',
    'weights/candidate_cifar75_routing.json': 'bbe71222138fd28ef928dcca22b319abc1223ec87b6680edd56b03c63dac0245',
    'vendor/MemriBoard/manager/service/converters.py': '28c6495c56afa225501dd6b0dea999215f08b73f30e016ff7ea34d6c5412701c',
}
BOARD = dict(adc_bit=14, dac_bit=12, vol_ref_adc=5., vol_ref_dac=5., gain=11.,
             res_load=3000., res_switches=10., sum_gain=330., soft_cc=.002)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def local(root, value):
    root = Path(root).resolve(); path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root):
        raise ValueError('Expected repository-relative contained path: ' + str(value))
    return path


def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class Checks:
    def __init__(self): self.count = 0; self.errors = []
    def require(self, condition, label):
        self.count += 1
        if not condition: self.errors.append(label)
    def equal(self, actual, expected, label, tolerance=1e-11):
        if isinstance(expected, dict):
            self.require(isinstance(actual, dict), label + '/object')
            if isinstance(actual, dict):
                for key, value in expected.items():
                    self.require(key in actual, label + '/' + key + '/present')
                    if key in actual: self.equal(actual[key], value, label + '/' + key, tolerance)
        elif isinstance(expected, (list, tuple)):
            self.require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label + '/length')
            if isinstance(actual, (list, tuple)):
                for i, (a, b) in enumerate(zip(actual, expected)): self.equal(a, b, label + '/' + str(i), tolerance)
        elif type(expected) in (int, float):
            self.require(type(actual) in (int, float) and math.isfinite(actual) and
                         math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), label)
        else: self.require(actual == expected, label)


def output_voltage(adc):
    if type(adc) is not int or not 0 <= adc <= 16382: raise ValueError('Invalid MVM ADC')
    return round(adc * 5 / 16383, 5)


def resistance(adc):
    if type(adc) is not int or not 20 <= adc <= 16382: raise ValueError('Invalid mode7 ADC')
    r = round(11 * 3000 * .3 * 16384 / (adc * 5) - 3010, 2)
    if r <= 1e-8: raise ValueError('Nonpositive native resistance')
    return r


def dac(volts):
    value = abs(int(round(volts * 4095 / 5, 0)))
    return dict(requested_v=volts, dac_code=value, native_decoded_v=round(value * 5 / 4095, 3))


def read_weights(path):
    data = Path(path).read_bytes()
    if data[:8] != b'\x93NUMPY\x01\x00': raise ValueError('Expected NPY1')
    size = struct.unpack('<H', data[8:10])[0]
    header = ast.literal_eval(data[10:10+size].decode('ascii'))
    if header != dict(descr='<f4', fortran_order=False, shape=(64,3,5,5)):
        raise ValueError('Unexpected weight layout')
    return struct.unpack('<4800f', data[10+size:])


def neural_trace(currents, fixture, fid):
    if len(currents) != 12 or any(x is None for x in currents): return None
    parameters = fixture['bntt']['parameters_t12x64']; result = []; membrane = 0.
    for t, value in enumerate(currents):
        g = parameters['weight'][t][fid]; b = parameters['bias'][t][fid]
        mean = parameters['running_mean'][t][fid]; variance = parameters['running_var'][t][fid]
        current = g * ((value - mean) / math.sqrt(variance + 1e-5)) + b
        before = membrane * .5 + current
        spike = int(before >= 1.)
        membrane = 0. if spike else before
        result.append(dict(t=t, conv_sum=value, bntt_current=current, lif_before_reset=before,
                           lif_after_reset=membrane, lif_spike=spike))
    return result


def calibration(weights, differences, floor_adc=5):
    if len(differences) != 5 or any(len(row) != len(weights) for row in differences):
        raise ValueError('Expected five complete calibration repeats')
    den = math.fsum(w*w for w in weights)
    if den <= 0: raise ValueError('Undefined calibration')
    scales = [math.fsum(w*d for w,d in zip(weights,row))/den for row in differences]
    alpha = statistics.mean(scales); sd = statistics.stdev(scales)
    means = [statistics.mean(row[i] for row in differences) for i in range(len(weights))]
    rms = math.sqrt(statistics.mean(x*x for x in means)); floor = output_voltage(floor_adc)
    usable = math.isfinite(alpha) and alpha > 0 and alpha > 5*sd/math.sqrt(5) and rms > 0
    effective = [d/alpha for d in means] if usable else None
    residuals = [e-w for e,w in zip(effective,weights)] if usable else None
    return dict(alpha_v_per_weight=alpha, alpha_repeat_values=scales, alpha_sd=sd, alpha_se=sd/math.sqrt(5),
                heuristic_resolution_5se=alpha>5*sd/math.sqrt(5), difference_rms_v=rms,
                five_adc_quality_reference_v=floor, rms_at_least_five_adc=rms>=floor,
                quantization_uncertainty_not_in_sample_sd=True, decode_usable=usable, alpha_cv=sd/abs(alpha) if alpha else None,
                alpha_cv_within_5pct=alpha>0 and sd/alpha<=.05, measured_difference_mean_v=means,
                effective_weights=effective, weight_residuals=residuals,
                weight_rmse=math.sqrt(statistics.mean(x*x for x in residuals)) if residuals else None,
                sign_mismatch_count=sum(w*d<0 for w,d in zip(weights,means)),
                zero_difference_count=sum(d==0 for d in means), calibration_uses_cifar=False,
                alpha_is_supervised_common_units_fit_to_trained_weights=True,confidence_claim=False)


def verify_inputs(repo, ck):
    for name, digest in PINS.items(): ck.require(sha(local(repo,name)) == digest, 'pin/' + name)
    fixture = load(local(repo,'weights/prepared_cifar100_conv1.json'))
    routing = load(local(repo,'weights/candidate_cifar75_routing.json'))
    weights = read_weights(local(repo,'weights/conv1_weights.npy'))
    pixel_path=local(repo,'weights/cifar100_conv1_fixture.json')
    ck.equal(sha(pixel_path),'8cb9ec0fe30242328527073bb076f8019295d73a748343f58fc2a8671bf7c154','fixture/raw_pixel_sha')
    pixels=load(pixel_path)['cases']
    ck.equal(fixture['filters'], [0,1,2,3], 'fixture/filters')
    ck.equal(len(fixture['cases']), 10, 'fixture/cases')
    addresses = set(); taps = []
    for filt in routing['filters']:
        fid = filt['filter_id']; local_taps = []
        for tile in filt['tiles']:
            for slot in tile['slots']:
                if slot['padding']:
                    ck.equal(slot['tap_index'],None,'padding/tap'); ck.equal(slot['padding_input'],0,'padding/input')
                    continue
                i = slot['tap_index']; local_taps.append(i); taps.append((fid,i))
                ck.equal(slot['tap_chw'],[i//25,(i%25)//5,i%5],f'routing/{fid}/{i}/chw')
                ck.equal(slot['trained_weight'],weights[fid*75+i],f'routing/{fid}/{i}/weight',0.)
                for side in ('positive_cell','negative_cell'): addresses.add(tuple(slot[side]))
        ck.equal(local_taps,list(range(75)),f'routing/{fid}/coverage')
    ck.require(len(set(taps)) == len(taps) == 300,'routing/300_unique_weights')
    ck.equal(len(addresses),94,'routing/94_addresses')
    maximum = 0.; spike_mismatches = 0
    for original,case in zip(pixels,fixture['cases']):
        spikes = case['spikes_t12x75']
        ck.equal(case['original_test_index'],original['original_test_index'],'fixture/test_index')
        y,x=original['patch_top_left_yx'];image=original['image_chw_uint8']
        # Frozen source fixture stores flat CHW image bytes. For uint8 values the
        # TTFS integer bins are identical in float32 and real arithmetic.
        if isinstance(image[0],list):
            def flatten(value):
                return sum((flatten(v) if isinstance(v,list) else [v] for v in value),[])
            image=flatten(image)
        patch=[image[c*1024+(y+dy)*32+x+dx] for c in range(3) for dy in range(5) for dx in range(5)]
        expected_spikes=[[int(p>=26 and math.floor((1-p/255)*11)==t) for p in patch] for t in range(12)]
        ck.equal(spikes,expected_spikes,'fixture/real_pixel_TTFS')
        ck.require(len(spikes)==12 and all(len(s)==75 for s in spikes),'fixture/spike_shape')
        ck.require(all(x in (0,1) for row in spikes for x in row),'fixture/binary')
        for fid in range(4):
            currents = [math.fsum(weights[fid*75+i]*x for i,x in enumerate(row)) for row in spikes]
            ref = neural_trace(currents,fixture,fid)
            for t,row in enumerate(ref):
                for name in ('conv_sum','bntt_current','lif_before_reset','lif_after_reset'):
                    delta=abs(row[name]-case['reference_t12x64'][name][t][fid]); maximum=max(maximum,delta)
                    ck.require(delta <= 1e-5,'golden/float32_tolerance/'+name)
                spike_mismatches += row['lif_spike'] != case['reference_t12x64']['lif_spikes'][t][fid]
    ck.equal(spike_mismatches,0,'golden/spikes')
    return fixture,routing,dict(maximum_float64_vs_float32_difference=maximum,
                               golden_absolute_tolerance=1e-5,spike_mismatches=spike_mismatches)


def paired_difference(rows):
    """Both sides have their own fresh before/after zero bracket."""
    if len(rows)!=2: raise ValueError('Expected positive and negative measurement rows')
    totals=[]
    for row in rows:
        base=(output_voltage(row['zero_before']['raw_adc'])+output_voltage(row['zero_after']['raw_adc']))/2
        totals.append(math.fsum(output_voltage(s['raw_adc'])-base for s in row['signals']))
    return totals[0]-totals[1]


def pair_values(rows):
    return [math.fsum(output_voltage(s['raw_adc'])-(output_voltage(row['zero_before']['raw_adc'])+
                output_voltage(row['zero_after']['raw_adc']))/2 for s in row['signals']) for row in rows]


def multiply(matrix, vector):return [math.fsum(a*b for a,b in zip(row,vector)) for row in matrix]


def singular2(matrix):
    """Closed-form singular values; no NumPy or collector dependency."""
    (a,b),(c,d)=matrix
    largest=(math.hypot(a+d,c-b)+math.hypot(a-d,c+b))/2
    smallest=abs(a*d-b*c)/largest if largest else 0.
    return [largest,smallest]


def matrix_select(train, selection):
    x=[r['measured_mean_v'] for r in train];y=[r['reference_v'] for r in train]
    aa=math.fsum(v[0]*v[0] for v in x);bb=math.fsum(v[0]*v[1] for v in x);dd=math.fsum(v[1]*v[1] for v in x)
    lam=(aa+dd+math.hypot(aa-dd,2*bb))/2
    # Determinant via pairwise minors avoids cancellation for near-collinear X.
    determinant=math.fsum((a[0]*b[1]-a[1]*b[0])**2 for i,a in enumerate(x) for b in x[i+1:])
    top=math.sqrt(lam);small=math.sqrt(max(0.,determinant/lam)) if lam else 0.
    threshold=top*max(len(x),2)*2**-52
    rank=int(top>threshold)+int(small>threshold);condition=top/small if rank==2 else None
    candidates=[('identity',[[1.,0.],[0.,1.]],True)]
    diag=[math.fsum(a[i]*b[i] for a,b in zip(x,y))/den if den else 1. for i,den in enumerate((aa,dd))]
    candidates.append(('diagonal',[[diag[0],0.],[0.,diag[1]]],all(v>0 for v in diag)))
    if rank==2 and condition<=1000:
        ridge=.005*(aa+dd);a=aa+ridge;d=dd+ridge;det=a*d-bb*bb
        matrix=[]
        for j in range(2):
            u=math.fsum(v[0]*target[j] for v,target in zip(x,y));v=math.fsum(v[1]*target[j] for v,target in zip(x,y))
            matrix.append([(d*u-bb*v)/det,(a*v-bb*u)/det])
        candidates.append(('full_ridge_0.01',matrix,True))
    reports=[]
    for name,matrix,eligible in candidates:
        high,low=singular2(matrix);cond=high/low if low else None
        eligible=bool(eligible and math.isfinite(high) and high<=10 and low>=.1 and cond<=1000)
        checks=[];errors=[]
        for obs in selection:
            corrected=[multiply(matrix,row) for row in obs['measured_repeats_v']]
            for i in range(2):
                target=obs['reference_v'][i];samples=[v[i] for v in corrected]
                tolerance=max(2*output_voltage(1),.05*abs(target));error=statistics.mean(samples)-target;sd=statistics.stdev(samples)
                passes=abs(error)<=tolerance and sd<=tolerance
                checks.append(dict(channel=i,error_v=error,sd_v=sd,tolerance_v=tolerance,passes=passes))
                errors.append(statistics.mean((s-target)**2 for s in samples)/tolerance**2)
        reports.append(dict(name=name,matrix=matrix,operator_gain=high,eligible=eligible,singular_values=[high,low],
            matrix_condition=cond,selection_pass=eligible and all(c['passes'] for c in checks),
            selection_score=statistics.mean(errors),checks=checks))
    eligible=[r for r in reports if r['eligible']]
    best=min(eligible,key=lambda r:(r['selection_score'],r['name']!='identity',r['name'])) if eligible else reports[0]
    return dict(selected_model=best['name'],matrix=best['matrix'],offset_v=[0.,0.],selection_pass=best['selection_pass'],
        identity_fallback=not eligible,candidates=reports,
        selection_score_definition='mean_over_channels_and_probes(mean_over_repeats((corrected_sample-fixed_target)^2)/tolerance_squared)',
        training_design_rank=rank,training_design_condition=condition,
        inputs_only_measured_two_channel_voltage=True,uses_cifar_or_trained_targets=False)


def read_raw(path, source, ck, session_id=None):
    events=[]; latest={}; sequence=decisions=pulses=skips=0; previous_time=-math.inf
    with Path(path).open(encoding='utf-8') as stream:
        while line:=stream.readline():
            request=json.loads(line); kind=request['event']; label='raw/'+str(sequence+1)
            if session_id is not None:ck.equal(request.get('session_id'),session_id,label+'/session')
            if kind=='calibration_frozen':
                ck.equal(request['before_task_id'],sequence+1,label+'/freeze_sequence')
                events.append(request);continue
            ck.require(kind in ('request','skipped'),label+'/request_or_skip')
            ck.require(request['source']==source,label+'/source')
            ck.require(request['monotonic_time_s']>=previous_time,label+'/time')
            previous_time=request['monotonic_time_s']
            command=request['planned_command_si'] if kind=='skipped' else request['command_si']
            cell=(request['wl'],request['bl']);mode=command['mode_flag'];vol=command['vol']
            ck.require(0<=cell[0]<8 and 0<=cell[1]<32,label+'/address')
            ck.require(mode in (7,10),label+'/mode')
            if mode==7:
                ck.require(type(vol) in (int,float) and 0<=vol<=3,label+'/program_voltage')
                ck.equal(command['wl'],cell[0],label+'/wl');ck.equal(command['bl'],cell[1],label+'/bl')
                ck.require(command['sign'] in (0,1),label+'/polarity')
                ck.equal(request['native_voltage_conversion'],dac(vol),label+'/dac')
                ck.equal(command['pulse_width'],.000125 if source=='simulator' and vol else 0.,label+'/width')
                if vol:
                    decisions+=1;ck.require(cell in latest,label+'/previous_R_exists')
                    if cell in latest:
                        resistance_value,prior=latest[cell];limit=.04 if command['sign']==0 else .002
                        ck.equal(request,dict(programming_decision_id=decisions,previous_native_r_ohm=resistance_value,
                                 resistance_source_task_id=prior,predicted_current_a=vol/resistance_value,current_limit_a=limit),label+'/guard')
                        ck.require((vol/resistance_value>limit)==(kind=='skipped'),label+'/guard_decision')
                    if kind=='skipped':
                        skips+=1;ck.require(not any(k in request for k in ('task_id','raw_adc','q','native_result')),label+'/no_fabrication')
                        events.append(request);continue
                    pulses+=1
            else:
                ck.require(kind=='request',label+'/mvm_not_skipped')
                ck.require(isinstance(vol,list) and len(vol)==32 and all(v in (0,.1) for v in vol) and sum(v!=0 for v in vol)<=1,label+'/onehot')
                ck.equal(request['native_voltage_conversion'],[dac(v) for v in vol],label+'/dac')
                if any(vol):
                    ck.require(cell in latest,label+'/model_previous_R')
                    if cell in latest:ck.require(.1/latest[cell][0]<=.005 and .1/latest[cell][0]*330<=4,label+'/model_guard')
            sequence+=1;ck.equal(request['task_id'],sequence,label+'/sequence')
            ck.equal(command['id'],sequence if mode==7 else 0,label+'/command_id')
            next_line=stream.readline()
            if not next_line:raise ValueError('Unpaired request at '+label)
            result=json.loads(next_line);ck.equal(result['event'],'result',label+'/result')
            ck.equal(result,{k:v for k,v in request.items() if k!='event'},label+'/request_retained')
            ck.equal(result['status'],'ok',label+'/status')
            if result['status']!='ok':raise ValueError('Recorded hardware/transport error; partial audit required')
            ck.equal(result['returned_id'],command['id'],label+'/returned_id')
            ck.equal(result['native_result'],[result['native_r'],result['returned_id'],result['raw_adc']],label+'/native_tuple')
            ck.require(result['latency_s']>=0 and math.isfinite(result['latency_s']),label+'/latency')
            if mode==7:
                r=resistance(result['raw_adc']);ck.equal(result,dict(native_r=r,resistance_ohm=r,conductance_siemens=1/r,q=330/r,
                                      read_voltage_parameter_v=.3),label+'/readback')
                latest[cell]=(r,sequence)
            else:
                v=output_voltage(result['raw_adc']);ck.equal(result['output_voltage_v'],v,label+'/output_voltage')
                ck.require(v<=4,label+'/actual_headroom')
            ck.equal(result['physical_pulse_width_s'],None,label+'/unknown_physical_width')
            ck.equal(result['native_adc_samples'],10 if mode==7 and source=='hardware' else 1,label+'/native_samples')
            ck.equal(result['simulator_duration_steps'],125 if mode==7 and source=='simulator' and vol else None,label+'/simulator_steps')
            events.append(result)
    return events,dict(impact_attempts=sequence,programming_pulse_attempts=pulses,
                      programming_decisions=decisions,programming_skipped_current=skips)


class Replay:
    def __init__(self,events,ck):self.events=events;self.ck=ck;self.position=0
    def take(self,phase,cell,vol,**metadata):
        if self.position>=len(self.events):raise ValueError('Missing event '+phase)
        row=self.events[self.position];self.position+=1
        label=f'protocol/{self.position}/{phase}'
        self.ck.equal(row,dict(phase=phase,wl=cell[0],bl=cell[1],**metadata),label+'/metadata')
        mode=10 if isinstance(vol,list) else 7
        command=row['planned_command_si'] if row['event']=='skipped' else row['command_si']
        self.ck.equal(command,dict(mode_flag=mode,wl=cell[0],vol=vol),label+'/command')
        if mode==7:self.ck.equal(command['bl'],cell[1],label+'/bl')
        return row
    def reads(self,phase,cell,n,**meta):return [self.take(phase,cell,0.,repeat=i,**meta) for i in range(n)]
    def mvm(self,phase,cell,active,**meta):
        vector=[0.]*32
        if active:vector[cell[1]]=.1
        row=self.take(phase,cell,vector,**meta)
        return dict(task_id=row['task_id'],raw_adc=row['raw_adc'],voltage_v=output_voltage(row['raw_adc']))


def replay_block(replay,active,selected,phase,event):
    rows=[]
    for side in ('positive_cell','negative_cell'):
        before=replay.mvm(phase,active[0][side],False,side=side,role='zero_before',**event)
        signals=[replay.mvm(phase,s[side],True,side=side,role='signal',tap=s['tap_index'],**event) for s in selected]
        if not signals:signals=[replay.mvm(phase,active[0][side],False,side=side,role='empty_zero_signal',**event)]
        after=replay.mvm(phase,active[0][side],False,side=side,role='zero_after',**event)
        rows.append(dict(side=side,zero_before=before,signals=signals,zero_after=after))
    return pair_values(rows),rows


def audit_matrix(directory,replay,active,anchors,meta,ck):
    n=len(active);h=[[statistics.mean(a[side]['singles'][i]['voltage_v']-a[side]['zero']['voltage_v'] for a in anchors)
                      for i in range(n)] for side in range(2)]
    masks={'train':[[i] for i in range(n)]+[list(range(n)),list(range(0,n,2)),list(range(1,n,2))],
           'selection':[[]]+[[i] for i in range(n)]+[[i,i+1] for i in range(0,n-1,2)]}
    phases={};ledger=[]
    for phase,groups in masks.items():
        observations=[]
        for probe_id,mask in enumerate(groups):
            measurements=[]
            for repeat in range(3):
                values,rows=replay_block(replay,active,[active[i] for i in mask],'matrix_'+phase,
                    dict(**meta,probe_id=probe_id,repeat=repeat,mask_slots=mask))
                measurements.append(values)
                ledger.append(dict(phase=phase,probe_id=probe_id,repeat=repeat,mask_slots=mask,
                                   measured_rows=rows,measured_voltage_v=values))
            observations.append(dict(probe_id=probe_id,mask_slots=mask,measured_repeats_v=measurements,
                measured_mean_v=[statistics.mean(v[i] for v in measurements) for i in range(2)],
                reference_v=[math.fsum(h[i][j] for j in mask) for i in range(2)]))
        phases[phase]=observations
    ck.equal([json.loads(line) for line in (directory/'matrix_probes.jsonl').read_text(encoding='utf-8').splitlines()],ledger,'matrix/probe_ledger')
    correction=matrix_select(phases['train'],phases['selection'])
    correction.update(reference_h_v=h,observations=phases,
        max_singleton_change_v=max(abs(multiply(correction['matrix'],[h[0][i],h[1][i]])[j]-h[j][i]) for i in range(n) for j in range(2)))
    ck.equal(load(directory/'frozen_matrix.json'),correction,'matrix/independent_fit_selection',1e-9)
    return correction


def audit_tile(directory,filt,tile,fixture,plan,source,ck,session_id=None):
    saved=load(directory/'tile.json');fid=filt['filter_id'];tid=tile['tile_id']
    active=[s for s in tile['slots'] if not s['padding']];meta=dict(filter_id=fid,tile_id=tid)
    events,counters=read_raw(directory/'raw.jsonl',source,ck,session_id);replay=Replay(events,ck)
    wmax=max(abs(s['trained_weight']) for t in filt['tiles'] for s in t['slots'] if not s['padding'])
    assignments=[]
    for slot in active:
        for side,sign in [('positive_cell',1),('negative_cell',-1)]:
            cell=slot[side];target=.015+sign*.005*slot['trained_weight']/wmax
            md=dict(**meta,tap=slot['tap_index'],side=side,trained_weight=slot['trained_weight'])
            current=replay.reads('program_initial',cell,1,**md)[0]
            resetstop=None
            for step in range(55):
                if current['q']<=.005:break
                row=replay.take('program_reset',cell,round(.3+.05*step,2),step=step,**md)
                ck.equal((row.get('command_si') or row['planned_command_si'])['sign'],0,'reset/sign')
                if row['event']=='skipped':resetstop='current_guard';break
                current=row
            crossed=current['q']<=.005
            confirmation=replay.reads('reset_confirmation',cell,5,**md)
            confirmed=crossed and statistics.mean(r['q'] for r in confirmation)<=.005 and confirmation[-1]['q']<=.005
            current=confirmation[-1];lo=target*.99;hi=target*1.01;reason='reset_unconfirmed'
            if confirmed:
                reason='set_ramp_exhausted'
                for step in range(55):
                    if lo<=current['q']<=hi:reason='hit';break
                    if current['q']>hi:reason='overshoot';break
                    row=replay.take('program_set',cell,round(.3+.05*step,2),step=step,target_q=target,**md)
                    ck.equal((row.get('command_si') or row['planned_command_si'])['sign'],1,'set/sign')
                    if row['event']=='skipped':reason='current_guard';break
                    current=row
                if lo<=current['q']<=hi:reason='hit'
                elif current['q']>hi:reason='overshoot'
            post=replay.reads('program_postread',cell,5,**md)
            expected=dict(cell=cell,target_q=target,status='completed',reset_confirmed=confirmed,reset_crossed=crossed,
                reset_confirmation_task_ids=[r['task_id'] for r in confirmation],endpoint=reason,endpoint_q=current['q'],
                post_q_mean=statistics.mean(r['q'] for r in post),post_task_ids=[r['task_id'] for r in post],
                post_all_inside=all(lo<=r['q']<=hi for r in post))
            if resetstop:expected['reset_stop']=resetstop
            assignments.append(expected)
    ck.equal(saved['programming'],assignments,'tile/programming')
    journal=[json.loads(line) for line in (directory/'programming.jsonl').read_text(encoding='utf-8').splitlines()]
    ck.equal(journal,assignments,'tile/programming_ledger')
    prereads=[]
    for slot in active:
        for side in ('positive_cell','negative_cell'):
            prereads.append(replay.reads('tile_pre_readback',slot[side],3,tap=slot['tap_index'],side=side,**meta))
    ck.equal(saved['pre_readback'],[[{k:v for k,v in r.items() if k not in ('event','latency_s')} for r in group] for group in prereads],'tile/pre_readback')
    def anchors(phase):
        records=[]
        for repeat in range(5):
            pair=[]
            for side in ('positive_cell','negative_cell'):
                z=replay.mvm(phase,active[0][side],False,repeat=repeat,side=side,role='zero',**meta)
                singles=[replay.mvm(phase,s[side],True,repeat=repeat,side=side,role='single',tap=s['tap_index'],**meta) for s in active]
                pair.append(dict(zero=z,singles=singles))
            records.append(pair)
        return records
    calanchors=anchors('calibration');ck.equal(load(directory/'calibration_anchors.json'),calanchors,'tile/anchors')
    correction=audit_matrix(directory,replay,active,calanchors,meta,ck)
    ck.equal(saved['matrix'],correction,'tile/matrix',1e-9)
    differences=[]
    for plus,minus in calanchors:
        corrected=[multiply(correction['matrix'],[p['voltage_v']-plus['zero']['voltage_v'],m['voltage_v']-minus['zero']['voltage_v']])
                   for p,m in zip(plus['singles'],minus['singles'])]
        differences.append([pair[0]-pair[1] for pair in corrected])
    cal=calibration([s['trained_weight'] for s in active],differences,plan['alpha_absolute_floor_adc'])
    freeze=events[replay.position];replay.position+=1
    ck.equal(freeze,dict(event='calibration_frozen',**meta,sha256=sha(directory/'frozen_calibration.json'),
                        matrix_sha256=sha(directory/'frozen_matrix.json'),
                        before_task_id=events[replay.position]['task_id']),'tile/freeze_commitment')
    cal.update(frozen_before_task_id=events[replay.position]['task_id'],trained_weights=[s['trained_weight'] for s in active],wmax_per_filter=wmax)
    ck.require(set(saved['calibration'])==set(cal),'tile/calibration_schema_supported')
    ck.equal(saved['calibration'],cal,'tile/calibration',1e-9);ck.equal(load(directory/'frozen_calibration.json'),cal,'tile/frozen_calibration',1e-9)
    partials=[]
    for ci,case in enumerate(fixture['cases']):
        for repeat in range(5):
            for t,spikes in enumerate(case['spikes_t12x75']):
                selected=[s for s in active if spikes[s['tap_index']]]
                event=dict(**meta,case=ci,original_test_index=case['original_test_index'],repeat=repeat,t=t)
                values,rows=replay_block(replay,active,selected,'cifar',event)
                corrected=multiply(correction['matrix'],values);difference=corrected[0]-corrected[1]
                partials.append(dict(**event,active_taps=[s['tap_index'] for s in selected],measured_rows=rows,
                                     raw_row_sums_v=values,corrected_row_sums_v=corrected,raw_signed_voltage_v=values[0]-values[1],
                                     signed_voltage_v=difference,trained_unit_contribution=difference/cal['alpha_v_per_weight'] if cal['decode_usable'] else None))
    ck.equal(saved['partials'],partials,'tile/partials',1e-9)
    ck.equal([json.loads(line) for line in (directory/'partials.jsonl').read_text(encoding='utf-8').splitlines()],partials,'tile/partial_ledger',1e-9)
    ck.equal(saved['final_anchors'],anchors('final_anchors'),'tile/final_anchors')
    postreads=[]
    for slot in active:
        for side in ('positive_cell','negative_cell'):
            postreads.append(replay.reads('tile_post_readback',slot[side],3,tap=slot['tap_index'],side=side,**meta))
    ck.equal(saved['post_readback'],[[{k:v for k,v in r.items() if k not in ('event','latency_s')} for r in group] for group in postreads],'tile/post_readback')
    ck.equal(replay.position,len(events),'tile/no_unconsumed_events')
    ck.equal(saved,counters,'tile/counters');ck.equal(saved['status'],'completed','tile/completed')
    ck.require(counters['programming_decisions']<=220*len(active),'tile/decision_budget')
    ck.require(counters['programming_pulse_attempts']<=counters['programming_decisions'],'tile/pulse_budget')
    return dict(**meta,calibration=cal,matrix=correction,partials=partials,counters=counters,assignment_endpoint_counts=dict(Counter(a['endpoint'] for a in assignments)))


def aggregate_independent(tile_results,routing,fixture,ck):
    indexed={}
    for result in tile_results:
        key=(result['filter_id'],result['tile_id'])
        ck.require(key not in indexed,'aggregate/no_duplicate_tile');indexed[key]=result
    trajectories=[];quality=[]
    for filt in routing['filters']:
        fid=filt['filter_id'];keys=[(fid,t['tile_id']) for t in filt['tiles']]
        if not all(k in indexed for k in keys):continue
        ledgers={key:{(p['case'],p['repeat'],p['t']):p for p in indexed[key]['partials']} for key in keys}
        for key in keys:ck.equal(len(ledgers[key]),600,'aggregate/unique_partial_keys')
        for ci,case in enumerate(fixture['cases']):
            for repeat in range(5):
                actual=[];effective=[]
                for t in range(12):
                    contributions=[ledgers[key][ci,repeat,t]['trained_unit_contribution'] for key in keys]
                    actual.append(None if any(v is None for v in contributions) else math.fsum(contributions))
                    coefficients=[]
                    for tile,key in zip(filt['tiles'],keys):
                        ws=indexed[key]['calibration']['effective_weights']
                        if ws is None:coefficients=None;break
                        active=[s for s in tile['slots'] if not s['padding']]
                        coefficients.extend(w*case['spikes_t12x75'][t][s['tap_index']] for s,w in zip(active,ws))
                    effective.append(None if coefficients is None else math.fsum(coefficients))
                actual_trace=neural_trace(actual,fixture,fid);effective_trace=neural_trace(effective,fixture,fid)
                complete=actual_trace is not None and effective_trace is not None
                comparisons={}
                for reference_name,reference in [('trained',None),('effective',effective_trace)]:
                    if not complete:continue
                    reference_values=[case['reference_t12x64']['conv_sum'][t][fid] if reference is None else reference[t]['conv_sum'] for t in range(12)]
                    reference_spikes=[case['reference_t12x64']['lif_spikes'][t][fid] if reference is None else reference[t]['lif_spike'] for t in range(12)]
                    comparisons[reference_name]=dict(conv_rmse=math.sqrt(statistics.mean((a-b)**2 for a,b in zip(actual,reference_values))),
                        spike_mismatches=sum(row['lif_spike']!=s for row,s in zip(actual_trace,reference_spikes)))
                    q=dict(filter_id=fid,case=ci,repeat=repeat,reference=reference_name,**comparisons[reference_name])
                    for name in ('bntt_current','lif_before_reset','lif_after_reset'):
                        refs=[case['reference_t12x64'][name][t][fid] if reference is None else reference[t][name] for t in range(12)]
                        q[name+'_rmse']=math.sqrt(statistics.mean((row[name]-r)**2 for row,r in zip(actual_trace,refs)))
                        comparisons[reference_name][name+'_max_abs_error']=max(abs(row[name]-r) for row,r in zip(actual_trace,refs))
                    quality.append(q)
                trajectories.append(dict(filter_id=fid,case=ci,original_test_index=case['original_test_index'],repeat=repeat,
                    status='measured' if complete else 'inconclusive_unresolved_alpha',actual=actual_trace,
                    effective_weight_reference=effective_trace,comparison=comparisons))
    return trajectories,quality


PRIVATE = {
 'vendor/MemriBoard/MemriCORE/rp5_fpga_python/rpi_FPGAed.py':'bc717659fa5d4d36d2037eef978bf1cd8dcf0fe92f9bbfad51f1eb1d0d27df9c',
 'vendor/MemriBoard/MemriCORE/rp5_fpga_python/FPGA_SIM.py':'5bcb02c04bc185e332b2ec96acabc2d4303d21724c9c052de540ef92d603b634'}


def verify_manifest(repo,directory,ck):
    manifest=load(directory/'manifest.json');git=manifest['invocation']['git']
    ck.require(git['dirty_tracked'] is False,'manifest/clean_tracked')
    ck.equal(manifest['protocol_id'],'cifar-conv1-program-measure-c2-v1','manifest/protocol')
    ck.require(manifest['source'] in ('hardware','simulator'),'manifest/source')
    ck.equal(manifest['hardware_access'],manifest['source']=='hardware','manifest/hardware_access')
    for name,digest in manifest['result_sha256'].items():
        ck.equal(sha(local(directory,name)),digest,'result_sha/'+name)
    differences=[]
    for name,digest in manifest['source_sha256'].items():
        path=local(repo,name)
        if name in PRIVATE:
            ck.equal(digest,PRIVATE[name],'source/private_pin/'+name)
            if path.is_file():ck.equal(sha(path),digest,'source/private_local/'+name)
            continue
        process=subprocess.run(['git','-C',str(repo),'show',git['commit']+':'+name],capture_output=True)
        ck.equal(process.returncode,0,'source/git_exists/'+name)
        ck.equal(hashlib.sha256(process.stdout).hexdigest(),digest,'source/git_blob/'+name)
        current=sha(path) if path.is_file() else None
        if current!=digest:differences.append(dict(path=name,recorded_sha256=digest,current_sha256=current))
    # No repository code is imported by this auditor. Historical runtime is verified
    # against its recorded Git blob; the converter formula is independently pinned.
    for name,digest in PINS.items():ck.equal(manifest['source_sha256'].get(name),digest,'source/known_pin/'+name)
    board=ConfigParser();board.read(directory/'board.ini',encoding='utf-8')
    for key,value in BOARD.items():ck.equal(board.getfloat('board',key),value,'INI/'+key)
    ck.equal(board.getfloat('board','vol_read'),.3,'INI/vol_read')
    plan=load(directory/'protocol.json')
    config_blob=subprocess.run(['git','-C',str(repo),'show',git['commit']+':configs/cifar_memristor.json'],capture_output=True,check=True).stdout
    ck.equal(plan,json.loads(config_blob),'protocol/snapshot_matches_recorded_config')
    ck.equal(plan,dict(protocol_id='cifar-conv1-program-measure-c2-v1',filters=[0,1,2,3],repeats=5,timesteps=12,cases=10,
        ramp_v=[round(.3+.05*i,2) for i in range(55)],q_center=.015,q_delta=.005,q_relative_tolerance=.01,
        reset_q=.005,maximum_impacts=222118,maximum_programming_pulses=66000,maximum_programming_decisions=66000,
        pilot_maximum_impacts=9350,pilot_maximum_programming_pulses=3080,
        alpha_resolution_sd_multiplier=5,alpha_absolute_floor_adc=5,inference_voltage_v=.1),'protocol/reviewed_parameters')
    ck.equal(plan['matrix'],dict(shape=[2,2],offset_v=[0,0],repeats=3,train='singles_all_even_odd',
        selection='zero_singles_adjacent_pairs',ridge_relative=.01,minimum_singular=.1,maximum_singular=10,
        maximum_condition=1000,selection_relative_error=.05,selection_absolute_adc=2,
        reference='uncorrected_fresh_H_times_mask; no trained weights or CIFAR',
        selection_policy='minimum_score_among_numerically_eligible; quality_flags_do_not_gate',
        fallback='identity_if_no_numerically_eligible'),'protocol/matrix')
    return manifest,plan,differences


def audit_session(repo,directory,fixture,routing,ck,allow_parent=True):
    manifest,plan,differences=verify_manifest(repo,directory,ck)
    ck.equal(manifest['status'],'completed','session/completed')
    full=[(f,t) for f in routing['filters'] for t in f['tiles']]
    if manifest['stage']=='pilot':
        selected=full[:1];results=[];ck.equal(manifest['parent_pilot'],None,'pilot/no_parent')
    elif manifest['stage']=='remaining' and allow_parent:
        selected=full[1:];parent=manifest['parent_pilot'];parent_dir=local(repo,parent['session'])
        ck.equal(sha(parent_dir/'manifest.json'),parent['manifest_sha256'],'parent/manifest_sha')
        pm=load(parent_dir/'manifest.json')
        ck.equal(pm['source_sha256'],manifest['source_sha256'],'parent/same_sources')
        ck.equal(pm['source'],manifest['source'],'parent/same_backend')
        ck.equal(pm['result_sha256'],parent['result_sha256'],'parent/result_sha')
        ck.equal(parent['measured_data_reused'],True,'parent/reused')
        parent_report,results=audit_session(repo,parent_dir,fixture,routing,ck,False)
        if manifest['source']=='simulator':
            ck.equal(manifest['array_initial_sha256'],parent['result_sha256'].get('array.cb'),'parent/native_persisted_array_reused')
    else:raise ValueError('Unsupported stage / recursive parent')
    expected_folders={f"f{f['filter_id']}_t{t['tile_id']}" for f,t in selected}
    ck.equal(sorted(p.name for p in (directory/'tiles').iterdir() if p.is_dir()),sorted(expected_folders),'session/exact_tile_directories')
    expected_raw={f'tiles/{folder}/raw.jsonl' for folder in expected_folders}
    ck.equal(sorted(name for name in manifest['result_sha256'] if name.endswith('/raw.jsonl')),sorted(expected_raw),'session/exact_raw_ledgers')
    for filt,tile in selected:
        relative=f"tiles/f{filt['filter_id']}_t{tile['tile_id']}"
        for name in ('raw.jsonl','programming.jsonl','calibration_anchors.json','frozen_calibration.json','frozen_matrix.json','matrix_probes.jsonl','partials.jsonl','tile.json'):
            ck.require(relative+'/'+name in manifest['result_sha256'],'manifest/required_result/'+relative+'/'+name)
        results.append(audit_tile(directory/relative,filt,tile,fixture,plan,manifest['source'],ck,manifest['session_id']))
    saved=load(directory/'summary.json');trajectories,quality=aggregate_independent(results,routing,fixture,ck)
    ck.equal(saved['trajectories'],trajectories,'summary/trajectories',1e-8)
    ck.equal(len(trajectories),0 if manifest['stage']=='pilot' else 200,'summary/trajectory_count')
    ck.equal(saved['filter_tiles_completed'],len(results),'summary/tile_count')
    ck.equal(saved['decode_usable'],all(r['calibration']['decode_usable'] for r in results),'summary/decode')
    ck.equal(saved['acquisition_complete'],manifest['stage']=='remaining' and len(results)==28,'summary/acquisition_complete')
    ck.equal(saved['decoded_conv_bn_lif_complete'],manifest['stage']=='remaining' and len(trajectories)==200 and all(r['calibration']['decode_usable'] for r in results),'summary/decoded_complete')
    impacts=sum(r['counters']['impact_attempts'] for r in results);pulses=sum(r['counters']['programming_pulse_attempts'] for r in results)
    ck.equal(saved['total_impacts_including_pilot'],impacts,'summary/impacts')
    ck.equal(saved['total_pulses_including_pilot'],pulses,'summary/pulses')
    ck.require(impacts<=(9350 if manifest['stage']=='pilot' else 222118),'session/impact_bound')
    ck.require(pulses<=(3080 if manifest['stage']=='pilot' else 66000),'session/pulse_bound')
    report=dict(session_id=manifest['session_id'],source=manifest['source'],stage=manifest['stage'],
                manifest_sha256=sha(directory/'manifest.json'),acquisition_status=manifest['status'],
                impacts_including_pilot=impacts,pulses_including_pilot=pulses,
                partial_points=sum(len(r['partials']) for r in results),full_time_outputs=len(trajectories)*12,
                decoded_time_outputs=sum(12 for t in trajectories if t['actual'] is not None),
                source_local_differences=differences,
                tiles=[{k:v for k,v in r.items() if k!='partials'} for r in results],quality=quality,
                limitations=['Audit consistency is separate from scientific accuracy and weight representation quality.',
                    'Five repeats do not identify quantization uncertainty or long-term stability.',
                    'Ten fixed CIFAR100 patches, four Conv1 filters; no full-network classification accuracy.',
                    'Actual outputs use fresh ADC differences. Effective weights are a separate calibration-based reference.'])
    return report,results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.');parser.add_argument('--session',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv);repo=Path(args.repo).resolve();directory=local(repo,args.session);output=local(repo,args.output)
    if output==directory or output.is_relative_to(directory):raise ValueError('Audit output must not alter the acquired session')
    ck=Checks();report={}
    try:
        fixture,routing,golden=verify_inputs(repo,ck)
        report,_=audit_session(repo,directory,fixture,routing,ck)
        report['golden_validation']=golden
    except Exception as exc:ck.require(False,'exception/'+type(exc).__name__+': '+str(exc))
    report.update(audit_status='passed' if not ck.errors else 'failed',checks=ck.count,errors=ck.errors,
                  auditor_sha256=sha(Path(__file__)),hardware_io=False,repository_code_imported=False)
    output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    text=f"# Независимый аудит CIFAR\n\nСтатус: **{report['audit_status']}**; проверок: {ck.count}; ошибок: {len(ck.errors)}.\n\n"
    text+='Аудит выполняет только чтение файлов и арифметику. Код коллектора и анализатора не импортируется.\n\n'
    text+=f"Полных Conv/BN/LIF точек: {report.get('full_time_outputs',0)}; partial-точек: {report.get('partial_points',0)}.\n\n"
    text+='Прохождение аудита подтверждает согласованность журнала; качество записи и ошибки относительно Wtrained/W_eff приведены отдельно в audit.json.\n'
    (output/'REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(status=report['audit_status'],checks=ck.count,errors=ck.errors[:20],output=args.output)))
    return int(bool(ck.errors))


if __name__=='__main__':raise SystemExit(main())

