"""Read-only offline v2 analysis after frozen runner postprocessing KeyError('array').
Does not import the simulator, write arrays, or perform any MVM.
"""
import hashlib,json,pathlib,numpy as np,time
D=pathlib.Path(__file__).resolve().parent
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=json.loads((D/'PROTOCOL.json').read_text())
assert H(D/'pilot.py')==P['runner_sha256']
def load(name):return [json.loads(line) for line in (D/name).read_text().splitlines()]
patches=load('patches.jsonl'); readback=load('readback.jsonl'); safety=load('mvm_safety.jsonl');predrive=load('predrive_safety.jsonl'); metrics=load('metrics.jsonl'); arrays=load('arrays.jsonl')
assert len(arrays)==len(readback)==len(metrics)==9 and len(safety)==len(predrive)==14022 and len(patches)==7011
assert {(x['replicate'],x['tile']) for x in arrays}=={(r,t) for r in range(3) for t in range(3)}
assert len({a['array_id'] for a in arrays})==9
assert all(x['decoded_max_v']<=.3 and x['current_a']<.015 and x['margin_a']>0 for x in safety)
assert all(x['decoded_max_v']<=.3 and x['requested_max_v']<=.3 and x['current_a']<.015 for x in predrive)
def stats(a,b):
 a=np.asarray(a);b=np.asarray(b);sd=float(np.std(a));return {'n':len(a),'rmse':float(np.sqrt(np.mean((a-b)**2))),'pearson_r':float(np.corrcoef(a,b)[0,1]) if sd>0 and np.std(b)>0 else None,'ideal_sd':sd}
# Stage 32 outcome is in frozen gates.jsonl; no reinterpretation of its 90% sign heuristic.
gate=load('gates.jsonl');assert len(gate)==1 and gate[0]['passed'] and gate[0]['stage']=='32'
indexed={}
for x in patches:
 if x['noise_flag']!=1 or x['split'] not in ('train','val'):continue
 key=(x['replicate'],x['split'],x['split_position'],x['top'],x['left'],x['t'],x['read'])
 d=indexed.setdefault(key,{})
 assert x['tile'] not in d
 d[x['tile']]=x
full=[]
for key,tiles in indexed.items():
 assert set(tiles)=={0,1,2}
 full.append(dict(replicate=key[0],split=key[1],split_position=key[2],top=key[3],left=key[4],t=key[5],read=key[6],**{f:sum(v[f] for v in tiles.values()) for f in ('ideal','quantized_ideal','programmed_readback','quantized_programmed','sim_mvm')}))
assert len(full)==2304
full.sort(key=lambda x:(x['replicate'],x['split'],x['split_position'],x['top'],x['left'],x['t'],x['read']))
(D/'full75_patches_offline.json').write_text(json.dumps(full,indent=1,allow_nan=False))
result={'label':P['label'],'protocol_sha256':H(D/'PROTOCOL.json'),'runner_sha256':H(D/'pilot.py'),'runner_exit':'1: offline KeyError(array) after all measurements; recovered from immutable raw records without another measurement','stage32_gate_passed':gate[0]['passed'],'mvm_calls':len(safety),'arrays':len(arrays),'elapsed_until_runner_abort_s':load('abort.jsonl')[0]['elapsed_s'],'requested_voltage_max_v':max(x['requested_max_v'] for x in predrive),'decoded_voltage_max_v':max(x['decoded_max_v'] for x in safety),'peak_predicted_current_a':max(x['current_a'] for x in safety),'minimum_predicted_current_margin_a':min(x['margin_a'] for x in safety),'min_predrive_margin_a':min(.015-x['current_a'] for x in predrive),'per_replicate':{},'tiles':[]}
for r in range(3):
 tr=[x for x in full if x['replicate']==r and x['split']=='train'];va=[x for x in full if x['replicate']==r and x['split']=='val']
 assert len(tr)==len(va)==384
 x=np.asarray([z['sim_mvm'] for z in tr]);y=np.asarray([z['ideal'] for z in tr]);slope,intercept=np.linalg.lstsq(np.stack([x,np.ones(len(x))],axis=1),y,rcond=None)[0]
 ideal=np.asarray([z['ideal'] for z in va]);q={'train_only_surrogate':{'intercept':float(intercept),'slope':float(slope),'val':stats(ideal,intercept+slope*np.asarray([z['sim_mvm'] for z in va]))}}
 for f in ['quantized_ideal','programmed_readback','quantized_programmed','sim_mvm']:
  q[f]=stats(ideal,[z[f] for z in va])
 q['val_raw_gate_passed']=q['sim_mvm']['pearson_r']>=.8 and q['sim_mvm']['rmse']<=.6*q['sim_mvm']['ideal_sd']
 q['val_surrogate_gate_passed']=q['train_only_surrogate']['val']['pearson_r']>=.8 and q['train_only_surrogate']['val']['rmse']<=.6*q['sim_mvm']['ideal_sd']
 result['per_replicate'][str(r)]=q
for rb in readback:
 a=np.asarray(rb['target']);b=np.asarray(rb['measured']);err=np.asarray(rb['cell_error']);lo,hi=rb['taps'];w=(a[:,0]-a[:,1]); repeats=np.asarray(rb['repeat_read_weights']);region={}
 for sign,mask in [('positive',w>0),('negative',w<0)]:
  region[sign]={'count':int(sum(mask)),'mean_signed_mapped_error':float(np.mean(err[mask])) if sum(mask) else None,'mean_abs_mapped_error':float(np.mean(abs(err[mask]))) if sum(mask) else None}
 result['tiles'].append({'replicate':rb['replicate'],'tile':rb['tile'],'sign_agreement':rb['sign_fraction'],'median_abs_cell_error_over_maxabs':rb['median_abs_cell_error_over_maxabs'],'zero_noise_max_error_per_active_tap_over_maxabs':next(z['zero_noise_max_error_per_active_tap_over_maxabs'] for z in metrics if z['replicate']==rb['replicate'] and z['tile']==rb['tile']),'by_sign':region,'by_column':[{'column':c,'mean_signed_conductance_residual':float(np.mean(b[:,c]-a[:,c])),'rmse_conductance_residual':float(np.sqrt(np.mean((b[:,c]-a[:,c])**2)))} for c in (0,1)],'row_tap_residuals':{str(i+lo):float(e) for i,e in enumerate(err)},'within_state_read_std_mean':float(np.mean(np.std(repeats,axis=2))),'write_seconds':next(z['seconds'] for z in metrics if z['replicate']==rb['replicate'] and z['tile']==rb['tile'])})
# Each tap has three independent array states. Compare this across-array variation to fixed-state read noise.
by_tap={}
for rb in readback:
 for i,e in enumerate(rb['cell_error']):by_tap.setdefault(rb['taps'][0]+i,[]).append(e)
assert len(by_tap)==75 and all(len(v)==3 for v in by_tap.values())
result['across_array_median_tap_sample_sd']=float(np.median([np.std(v,ddof=1) for v in by_tap.values()]))
result['mean_within_state_read_sd']=float(np.mean([t['within_state_read_std_mean'] for t in result['tiles']]))
result['all_full75_raw_gates_passed']=all(x['val_raw_gate_passed'] for x in result['per_replicate'].values())
result['verdict']='VALIDATED (post-v1 revised v2 only)' if result['all_full75_raw_gates_passed'] else 'FULL75 VALIDATION GATE FAILED (post-v1 revised v2 only)'
(D/'full75_summary_offline.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in result.items() if k not in ('tiles',)},indent=2,allow_nan=False))
