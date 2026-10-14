"""Read-only exploratory analysis of original + two diagnostic readbacks."""
import hashlib,json,pathlib,numpy as np
ROOT=pathlib.Path('/mnt/d/projects/snn-mnist/.tmp/openclaw-spikes')
O=ROOT/'memriboard-conv1-20261007';D=ROOT/'memriboard-conv1-diagnostic-20261007'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mf=json.loads((D/'MANIFEST.json').read_text())
for name,expected in mf['original_sha256'].items():assert h(O/name)==expected, f'original changed: {name}'
orig=json.loads((O/'readback.jsonl').read_text().splitlines()[0])
new=[json.loads(s) for s in (D/'readback.jsonl').read_text().splitlines()]
assert [r['seed'] for r in new]==[17002,17003]
for r in new:assert h(pathlib.Path(r['array_id']+'.cb'))==r['final_sha256']
pre=json.loads((O/'preflight.json').read_text());w=np.asarray(pre['filter_weights'][:32]);scale=pre['filter_maxabs']
rows=[orig,*new];target=np.asarray(orig['target']);measured=np.array([r['measured'] for r in rows]); gaps=measured[:,:,0]-measured[:,:,1];tg=target[:,0]-target[:,1];e=gaps-tg
assert np.allclose(target,np.array([r['target'] for r in new])) and np.all(np.abs(w)>.05*scale)
wrong=[np.flatnonzero(np.sign(g)!=np.sign(w)).tolist() for g in gaps]
for idx,r in enumerate(new):assert wrong[idx+1]==r['wrong_sign_taps'] and abs((1-len(wrong[idx+1])/32)-r['sign_agreement'])<1e-12
logs=[json.loads(s) for s in (D/'programming.jsonl').read_text().splitlines()];assert len(logs)==128
repeated=[np.array(orig['repeat_read_weights'])]+[np.array(r['repeated_weight_resistance'])[:,:,:,0] for r in new]
summary={
 'classification':'EXPLORATORY_DIAGNOSTIC_NOT_ORIGINAL_PILOT','original_verdict':'PARTIAL (unchanged)','original_sha256':mf['original_sha256'],'diagnostic_manifest_sha256':h(D/'MANIFEST.json'), 'script_sha256':mf['script_sha256'], 'checkpoint_sha256':pre['checkpoint_sha256'], 'seeds':[17001,17002,17003], 'diagnostic_only_seeds':[17002,17003], 'mvm_calls':0,'full_filter_maxabs':scale,'target_conductance_range':[float(target.min()),float(target.max())],'measured_conductance_range_all_arrays':[float(measured.min()),float(measured.max())],
 'arrays':[{'seed':r['seed'],'array_id':r['array_id'],'initial_sha256':r['initial_sha256'] if r is not orig else json.loads((O/'arrays.jsonl').read_text().splitlines()[0])['initial_sha256'],'final_sha256':r['final_sha256'],'sign_agreement':float(1-len(wrong[i])/32),'wrong_sign_taps':wrong[i], 'median_abs_cell_error':float(np.median(abs(measured[i]-target))),'gap_rmse':float(np.sqrt(np.mean(e[i]**2))), 'max_within_state_repeat_read_sd':float(np.max(np.std(repeated[i],axis=2,ddof=1)))} for i,r in enumerate(rows)],
 'target_gaps':tg.tolist(),'measured_gaps_by_seed':gaps.tolist(),'gap_error_by_seed':e.tolist(),'target_cells':target.tolist(),'measured_cells_by_seed':measured.tolist(),
 'tap_records':[{'tap':j,'weight':float(w[j]),'target_plus':float(target[j,0]),'target_minus':float(target[j,1]),'target_gap':float(tg[j]),'measured_gaps':gaps[:,j].tolist(),'gap_errors':e[:,j].tolist(),'wrong_sign_seeds':[rows[k]['seed'] for k in range(3) if j in wrong[k]],'across_program_gap_sample_sd':float(np.std(gaps[:,j],ddof=1))} for j in range(32)],
 'original_wrong_sign_tap_recurrence':{str(j):[r['seed'] for k,r in enumerate(rows) if j in wrong[k]] for j in wrong[0]},
 'additional_recurrent_taps':{str(j):[r['seed'] for k,r in enumerate(rows) if j in wrong[k]] for j in range(32) if sum(j in z for z in wrong)>=2 and j not in wrong[0]},
 'across_program_cell_sample_sd_median':float(np.median(np.std(measured,axis=0,ddof=1))), 'across_program_gap_sample_sd_median':float(np.median(np.std(gaps,axis=0,ddof=1))),
 'median_abs_target_gap':float(np.median(abs(tg))),'median_abs_gap_error':float(np.median(abs(e))),'sign_correct_by_gap_group':{'abs_gap_le_0p005':[int(np.sum((np.sign(gaps[i])==np.sign(w))&(abs(tg)<=.005))) for i in range(3)],'count_le_0p005':int(np.sum(abs(tg)<=.005)),'abs_gap_gt_0p01':[int(np.sum((np.sign(gaps[i])==np.sign(w))&(abs(tg)>.01))) for i in range(3)],'count_gt_0p01':int(np.sum(abs(tg)>.01))},
 'target_quartiles':[float(x) for x in np.quantile(target,[0,.25,.5,.75,1])], 'lower_target_quartile_mean_abs_cell_error':float(np.mean(abs(measured-target)[:,target<=np.quantile(target,.25)])), 'upper_target_quartile_mean_abs_cell_error':float(np.mean(abs(measured-target)[:,target>=np.quantile(target,.75)])), 'lower_target_quartile_mean_signed_cell_error':float(np.mean((measured-target)[:,target<=np.quantile(target,.25)])), 'upper_target_quartile_mean_signed_cell_error':float(np.mean((measured-target)[:,target>=np.quantile(target,.75)])),
 'diagnostic_cells_programmed':len(logs),'diagnostic_total_pulses':sum(len(x['pulses_v']) for x in logs),'diagnostic_total_write_seconds':float(sum(x['write_seconds'] for x in logs)), 'programming_pulse_min_max_v':[min(v for x in logs for v in x['pulses_v']),max(v for x in logs for v in x['pulses_v'])], 'new_array_initial_final_hashes':[{k:r[k] for k in ('seed','array_id','initial_sha256','final_sha256')} for r in new],
 'interpretation':'Weak differential target gaps and stochastic reprogramming variation, with larger low-conductance errors, plausibly explain sign reversals. Measurements span both sides of 0.01..0.05; no global hard writable floor/ceiling established. Not proof that a new calibration scale is feasible. No MVM safety/performance measured.'}
(D/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:summary[k] for k in ('arrays','original_wrong_sign_tap_recurrence','additional_recurrent_taps','across_program_cell_sample_sd_median','across_program_gap_sample_sd_median','lower_target_quartile_mean_abs_cell_error','upper_target_quartile_mean_abs_cell_error','diagnostic_total_pulses','diagnostic_total_write_seconds')},indent=2))
