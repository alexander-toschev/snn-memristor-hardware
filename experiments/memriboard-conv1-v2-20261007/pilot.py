"""Bounded preregistered MemriBoard local simulation; no physical connector imports."""
import configparser, hashlib, importlib.util, json, math, os, pathlib, pickle, random, signal, subprocess, sys, time, types
import numpy as np
import torch
ROOT=pathlib.Path('/mnt/d/projects/snn-mnist'); OUT=ROOT/'.tmp/openclaw-spikes/memriboard-conv1-v2-20261007'
UP=pathlib.Path('/mnt/d/projects/memristor-aware-twd-snn/memristor_hw/MemriBoard')
P=json.loads((OUT/'PROTOCOL.json').read_text()); START=time.monotonic(); MVM_CALLS=0
random.seed(P['seeds']['python']); np.random.seed(P['seeds']['numpy']); torch.manual_seed(P['seeds']['python'])
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def emit(file,row):
 with open(OUT/file,'a') as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');f.flush()
def check(c,msg):
 if not c:raise RuntimeError(msg)
def commit(p):return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
# Reuse Lab's verified headless method: direct standalone converter import, stub manager.app/service.
spec=importlib.util.spec_from_file_location('pilot_converters',UP/'manager/service/converters.py'); cv=importlib.util.module_from_spec(spec);spec.loader.exec_module(cv)
manager=types.ModuleType('manager'); manager.__path__=[]; app=types.ModuleType('manager.app'); service=types.ModuleType('manager.service')
for a,b in [('d2v','convert_dac_to_volt'),('v2d','convert_volt_to_dac'),('a2r','convert_adc_to_res'),('r2a','convert_res_to_adc'),('r2w','convert_res_to_weight'),('w2r','convert_weight_to_res'),('a2v','convert_adc_to_volt'),('a2c','convert_adc_to_current')]:setattr(service,a,getattr(cv,b))
CFG=configparser.ConfigParser();check(CFG.read(UP/'simulator/simulator_settings.ini'),'simulator config missing')
class FakeApplication:
 def __init__(self):
  for k in ('dac_bit','adc_bit','res_load','sum_gain'):setattr(self,k,int(CFG['board'][k]))
  for k in ('vol_ref_dac','vol_ref_adc','vol_read','res_switches','gain','soft_cc'):setattr(self,k,float(CFG['board'][k]))
app.Application=FakeApplication;sys.modules.update({'manager':manager,'manager.app':app,'manager.service':service});sys.path.insert(0,str(UP))
from simulator.src import BoardSimulator,create_crossbar_array
from MemriNeurons.cores import HardCore

class GuardedBoard(BoardSimulator):
 def __init__(self):super().__init__();self.config=CFG;self.core=None;self.meta={};self.calls=0
 def mode_mvm(self,vols,tms,tus,rtms,rums,wl,task_id):
  check(type(self) is GuardedBoard and isinstance(self,BoardSimulator) and wl in (0,1),'connector/column not simulator')
  check(len(vols)==32 and all(isinstance(v,(int,np.integer)) and 0<=v<=4095 for v in vols),'DAC vector invalid')
  voltage=np.array([cv.convert_dac_to_volt(self.core.dac_bit,self.core.vol_ref_dac,int(v)) for v in vols])
  check(np.all(np.isfinite(voltage)) and np.all(voltage>=0) and np.all(voltage<=.3),'decoded inference voltage exceeds 0.3V')
  observed=np.asarray(self.core.current_resistances[:,wl]);actual=np.asarray([self.crossbar[i][wl].last_resistance for i in range(32)])
  check(np.all(np.isfinite(observed)) and np.all(observed>0) and np.all(np.isfinite(actual)) and np.all(actual>0),'nonpositive/nonfinite resistance')
  c_obs=float(np.sum(voltage/observed));c_act=float(np.sum(voltage/actual));limit=max(c_obs,c_act)
  global MVM_CALLS
  check(MVM_CALLS<P['budget']['max_mvm_calls'] and time.monotonic()-START<P['budget']['max_total_seconds'],'pre-MVM budget exhausted')
  audit=dict(self.meta,call=MVM_CALLS,column=wl,decoded_max_v=float(max(voltage)),current_a=limit,observed_current_a=c_obs,actual_current_a=c_act,margin_a=.015-limit)
  emit('mvm_safety.jsonl',audit) # independent PRE-call audit, even for rejected current
  check(math.isfinite(limit) and limit<.015,f'ABORT BEFORE MVM: {limit} A >= 0.015 A')
  self.calls+=1; MVM_CALLS+=1; r=super().mode_mvm(vols,tms,tus,rtms,rums,wl,task_id)
  check((r[0]>0 or not any(vols)) and 0<=r[0]<2**self.core.adc_bit-1,f'ADC saturation/out of range: {r[0]}');return r

def drive(board,core,x,col,meta):
 check(len(x)==32 and np.all(np.isfinite(x)) and np.all(x>=0),'invalid input')
 v=.3*x;check(np.all(v<=.3),'requested inference voltage exceeds 0.3 V (never clip)')
 codes=[cv.convert_volt_to_dac(core.dac_bit,core.vol_ref_dac,float(a)) for a in v]
 dec=np.array([cv.convert_dac_to_volt(core.dac_bit,core.vol_ref_dac,n) for n in codes]);check(all(n<=core.max_dac_value for n in codes) and np.all(dec<=.3),'DAC quantization over cap')
 observed=np.asarray(core.current_resistances[:,col]); actual=np.asarray([board.crossbar[i][col].last_resistance for i in range(32)])
 check(np.all(np.isfinite(observed)) and np.all(observed>0) and np.all(np.isfinite(actual)) and np.all(actual>0),'invalid resistance before library dot_product')
 c_obs=float(np.sum(dec/observed));c_actual=float(np.sum(dec/actual));peak=max(c_obs,c_actual)
 emit('predrive_safety.jsonl',dict(meta,column=col,current_a=peak,observed_current_a=c_obs,actual_current_a=c_actual,margin_a=.015-peak,requested_max_v=float(max(v)),decoded_max_v=float(max(dec))))
 check(math.isfinite(peak) and peak<.015,f'ABORT BEFORE dot_product/mode_mvm: {peak} A >= 0.015 A')
 check(all(cv.convert_volt_to_dac(core.dac_bit,core.vol_ref_dac,float(a))==n for a,n in zip(v,codes)),'internal DAC mismatch')
 board.meta=meta
 # Library silently zeros over-current drive; independent predrive and connector checks precede any actual MVM.
 out,adc=core.dot_product(v.tolist(),col)
 return out,dec,adc

def patches():
 # Import ONLY official training pickle; replicate temporal_snn/dataset.py split exactly.
 with open(ROOT/'data/cifar-100-python/train','rb') as f:raw=pickle.load(f,encoding='bytes')
 labels=np.asarray(raw[b'fine_labels']);selected=np.flatnonzero(np.isin(labels,[0,1,2,3,5,8,13,14,17,19])); filtered=labels[selected]
 rng=np.random.default_rng(20261006); val=[]
 for cls in np.unique(filtered):
  pool=np.flatnonzero(filtered==cls);val.extend(rng.permutation(pool)[:round(len(pool)*.1)].tolist())
 val=sorted(val);mask=np.ones(len(selected),bool);mask[val]=False; train=np.flatnonzero(mask).tolist()
 digest=hashlib.sha256(json.dumps({'train_indices':train,'val_indices':val},sort_keys=True,separators=(',',':')).encode()).hexdigest()
 check(digest=='f3a5acaf7d61d1221b28a6ba66e468f905a44463630a8bdc5cf38cbb956f0492','split hash mismatch')
 rows=[]
 for split,indices,positions in [('train',train,P['train_split_positions']),('val',val,P['val_split_positions'])]:
  for pos in positions:
   fi=indices[pos];oi=int(selected[fi]);img=raw[b'data'][oi].reshape(3,32,32).astype(np.float32)/255
   for top,left in P['spatial_top_left']:
    px=img[:,top:top+5,left:left+5].reshape(75)
    for t in P['spike_times']:
     fire=np.floor((1-px)*(P['T']-1)).astype(int).clip(0,P['T']-1)
     x=((px>=P['x_min'])&(fire==t)).astype(float)
     rows.append((dict(split=split,split_position=pos,filtered_index=fi,original_train_index=oi,top=top,left=left,t=t),x))
 check(len(rows)==256 and len(train)==4500 and len(val)==500,'patch/split count mismatch');return rows,digest

def metrics(a,b,c):
 a=np.asarray(a);b=np.asarray(b);c=np.asarray(c);sd=float(np.std(a))
 return dict(count=len(a),rmse=float(np.sqrt(np.mean((c-a)**2))),mae=float(np.mean(abs(c-a))),readback_rmse=float(np.sqrt(np.mean((b-a)**2))),mvm_readback_rmse=float(np.sqrt(np.mean((c-b)**2))),corr=float(np.corrcoef(a,c)[0,1]) if sd>0 and np.std(c)>0 else None,ideal_sd=sd)
def expired(*args):raise TimeoutError('single cell write exceeded 20s')
def run_tile(rep,tile,seed,span,w,scale,rows):
 lo,hi=span;random.seed(seed+101*tile);np.random.seed(seed+101*tile)
 serial=str(OUT/f'array_r{rep}_tile{tile}');check(not pathlib.Path(serial+'.cb').exists(),'array already exists; do not overwrite/reuse')
 create_crossbar_array(serial,32,8); board=GuardedBoard();check(board.connect(serial)==1 and type(board) is GuardedBoard,'simulator connection failed')
 core=HardCore(board);board.core=core;core.set_mvm();check((core.ROW_NUM,core.COL_NUM)==(32,8),'crossbar orientation/shape invalid')
 array=dict(replicate=rep,tile=tile,seed=seed+101*tile,taps=[lo,hi],array_id=serial,initial_sha256=sha(serial+'.cb'))
 emit('arrays.jsonl',array);beg=time.monotonic(); g=np.stack((.03+.02*w/scale,.03-.02*w/scale),axis=1)
 for row,tap in enumerate(range(lo,hi)):
  for col in (0,1):
   signal.alarm(20)
   try:
    t=time.monotonic();final,hist,vol=core.write_weight(row,col,float(g[tap,col]));dt=time.monotonic()-t
   finally:signal.alarm(0)
   emit('programming.jsonl',dict(array,row=row,column=col,tap=tap,target=float(g[tap,col]),final=float(final),write_seconds=dt,pulses_v=vol,weight_history=hist,array_sha256=sha(serial+'.cb')))
   if rep==0 and tile==0 and row==0 and col==0:
    print('FIRST_WRITE',dt,'PROJECTED_64',64*dt,flush=True);check(64*dt<=900,'first-cell projection >900s')
   check(time.monotonic()-beg<=900*(hi-lo)/32,'stage programming budget exceeded')
 measured=[];res=[];repeat=[]
 for row in range(hi-lo):
  measured.append([]);res.append([]);repeat.append([])
  for col in (0,1):
   readings=[core.read_one_weight(row,col) for _ in range(3)];measured[-1].append(float(np.mean([z[0] for z in readings])));res[-1].append(float(np.mean([z[1] for z in readings])));repeat[-1].append([float(z[0]) for z in readings])
 measured=np.asarray(measured);error=(measured[:,0]-measured[:,1]-(g[lo:hi,0]-g[lo:hi,1]))*scale/.04
 significant=np.abs(w[lo:hi])>.05*scale; sign=float(np.mean(np.sign(measured[significant,0]-measured[significant,1])==np.sign(w[lo:hi][significant]))) if any(significant) else 1.
 median=float(np.median(abs(error))/scale)
 emit('readback.jsonl',dict(array,final_sha256=sha(serial+'.cb'),target=g[lo:hi].tolist(),measured=measured.tolist(),resistance_ohm=res,repeat_read_weights=repeat,sign_fraction=sign,median_abs_cell_error_over_maxabs=median,cell_error=error.tolist()))
 check(median<=.35,f'cell readback gate failed median={median}; sign={sign} is descriptive')
 # deterministic control, then noise-enabled fixed-state measurements (3 repeated paired reads)
 out={'train':[[],[],[]],'val':[[],[],[]]};zero_max=0
 for noise in (0,1):
  for line in board.crossbar:
   for mem in line:mem.noise_flag=noise
  directed=[(dict(split='control',split_position=-1,filtered_index=-1,original_train_index=-1,top=-1,left=-1,t=-1,probe=j),np.pad(z,(lo,75-hi))) for j,z in enumerate([np.eye(hi-lo)[0],np.eye(hi-lo)[min(hi-lo-1,15)],np.ones(hi-lo)])]
  sequence=(rows[:8]+directed) if noise==0 else rows
  for meta,x in sequence:
   z=x[lo:hi];pad=np.pad(z,(0,32-len(z)));ideal=float(z@w[lo:hi]);quan=float(np.asarray([cv.convert_dac_to_volt(core.dac_bit,core.vol_ref_dac,cv.convert_volt_to_dac(core.dac_bit,core.vol_ref_dac,float(.3*u))) for u in z])@(g[lo:hi,0]-g[lo:hi,1])*scale/(.04*.3))
   for read in range(1 if noise==0 else 3):
    outcol=[];digcol=[];adcs=[]
    for col in (0,1):
     mv,decoded,adc=drive(board,core,pad,col,dict(array,**meta,noise_flag=noise,read=read));outcol.append(float(mv));adcs.append(int(adc));digcol.append(float(decoded[:hi-lo]@measured[:,col]))
    board_out=(outcol[0]-outcol[1])*scale/(.04*.3); quant_read=(digcol[0]-digcol[1])*scale/(.04*.3);program_read=float(z@(measured[:,0]-measured[:,1])*scale/.04)
    emit('patches.jsonl',dict(array,**meta,noise_flag=noise,read=read,ideal=ideal,quantized_ideal=quan,quantized_programmed=quant_read,programmed_readback=program_read,sim_mvm=board_out,raw_columns=outcol,adc_columns=adcs))
    if noise==0:zero_max=max(zero_max,abs(board_out-quant_read)/(scale*max(1,int(sum(z)))))
    else:
     key=meta['split'];out[key][0].append(ideal);out[key][1].append(program_read);out[key][2].append(board_out)
 stats={key:metrics(*v) for key,v in out.items()}; stats.update(array);stats.update(sign_fraction=sign,median_abs_cell_error_over_maxabs=median,zero_noise_max_error_per_active_tap_over_maxabs=zero_max,seconds=time.monotonic()-beg)
 emit('metrics.jsonl',stats);print('TILE_DONE',json.dumps({k:stats[k] for k in ('replicate','tile','sign_fraction','median_abs_cell_error_over_maxabs','zero_noise_max_error_per_active_tap_over_maxabs','seconds','val')}),flush=True)
 check(zero_max<=.05,'zero-noise MVM/readback scaling failure')
 return stats

def aggregate_full():
 raw=[json.loads(line) for line in (OUT/'patches.jsonl').read_text().splitlines()]
 grouped={}
 for r in raw:
  if r['noise_flag']!=1 or r['split'] not in ('train','val'):continue
  key=(r['array']['replicate'],r['split'],r['split_position'],r['top'],r['left'],r['t'],r['read'])
  grouped.setdefault(key,{})[r['array']['tile']]=r
 out=[]
 for key,tiles in grouped.items():
  check(set(tiles)=={0,1,2},'incomplete three-tile patch')
  out.append(dict(replicate=key[0],split=key[1],position=key[2],top=key[3],left=key[4],time=key[5],read=key[6],**{field:sum(v[field] for v in tiles.values()) for field in ('ideal','quantized_ideal','programmed_readback','quantized_programmed','sim_mvm')}))
 (OUT/'full75_patches.json').write_text(json.dumps(out,allow_nan=False))
 summary={}
 for rep in range(3):
  train=[x for x in out if x['replicate']==rep and x['split']=='train']; val=[x for x in out if x['replicate']==rep and x['split']=='val']
  check(len(train)==384 and len(val)==384,'frozen patch count mismatch')
  x=np.asarray([r['sim_mvm'] for r in train]);y=np.asarray([r['ideal'] for r in train]);slope,intercept=np.linalg.lstsq(np.stack((x,np.ones(len(x))),axis=1),y,rcond=None)[0]
  ideal=np.asarray([r['ideal'] for r in val]);sd=float(np.std(ideal));pred=intercept+slope*np.asarray([r['sim_mvm'] for r in val])
  report={'train_affine_intercept':float(intercept),'train_affine_slope':float(slope),'val_count':len(val),'ideal_sd':sd,'surrogate_val_rmse':float(np.sqrt(np.mean((pred-ideal)**2))),'surrogate_val_corr':float(np.corrcoef(ideal,pred)[0,1]) if sd>0 and np.std(pred)>0 else None}
  for field in ('programmed_readback','quantized_ideal','quantized_programmed','sim_mvm'):
   z=np.asarray([r[field] for r in val]);report[field]={'rmse':float(np.sqrt(np.mean((z-ideal)**2))),'corr':float(np.corrcoef(ideal,z)[0,1]) if sd>0 and np.std(z)>0 else None}
  report['validation_gate_passed']=bool(report['sim_mvm']['corr'] is not None and report['sim_mvm']['corr']>=.8 and report['sim_mvm']['rmse']<=.6*sd)
  summary[str(rep)]=report
 return {'replicates':summary,'all_full75_gates_passed':all(r['validation_gate_passed'] for r in summary.values()),'mvm_calls':MVM_CALLS,'elapsed_s':time.monotonic()-START}

def main():
 check(sha(OUT/'pilot.py')==P['runner_sha256'],'v2 runner differs from frozen protocol')
 for name,path in [('v1_protocol',ROOT/'.tmp/openclaw-spikes/memriboard-conv1-20261007/PROTOCOL.json'),('v1_runner',ROOT/'.tmp/openclaw-spikes/memriboard-conv1-20261007/pilot.py'),('v1_summary',ROOT/'.tmp/openclaw-spikes/memriboard-conv1-20261007/summary.json'),('diagnostic_summary',ROOT/'.tmp/openclaw-spikes/memriboard-conv1-diagnostic-20261007/summary.json'),('cifar_train',ROOT/'data/cifar-100-python/train'),('core',UP/'MemriNeurons/cores.py'),('simulator',UP/'simulator/src.py'),('converter',UP/'manager/service/converters.py'),('simulator_config',UP/'simulator/simulator_settings.ini')]:
  check(sha(path)==P['input_hashes'][name],f'input changed: {name}')
 cp=ROOT/'runs_csnn/confirmatory_wide_bntt_20261006/twd1_sigma0p0_seed0/best_fs.pt';check(sha(cp)==P['checkpoint_sha256'],'checkpoint hash mismatch')
 weights=torch.load(cp,map_location='cpu',weights_only=True)['conv_layers.0.weight'];w=weights[0].detach().numpy().ravel().astype(float)
 check(w.shape==(75,) and np.all(np.isfinite(w)),'Conv1 geometry invalid');scale=float(max(abs(w)));check(scale>0,'all zero weights')
 a=.03+.02*w/scale;b=.03-.02*w/scale;check(min(a.min(),b.min())>=.01-1e-12 and max(a.max(),b.max())<=.05+1e-12 and np.allclose((a-b)*scale/.04,w),'ideal mapping infeasible')
 rows,digest=patches();provenance=dict(start_epoch=time.time(),checkpoint_sha256=sha(cp),sources={'snn':commit(ROOT),'MemriBoard':commit(UP),'MemriNeurons':commit(UP/'MemriNeurons')},simulator_config=dict(CFG['board']),simulator_config_sha256=sha(UP/'simulator/simulator_settings.ini'),split_sha256=digest,protocol_sha256=sha(OUT/'PROTOCOL.json'),python=sys.executable,filter=0,filter_weights=w.tolist(),filter_maxabs=scale,connector='BoardSimulator')
 (OUT/'preflight.json').write_text(json.dumps(provenance,indent=2));print('PREFLIGHT_OK',provenance['sources'],flush=True)
 signal.signal(signal.SIGALRM,expired);stage=[]
 for rep,seed in enumerate(P['seeds']['array_reprogram']):stage.append(run_tile(rep,0,seed,(0,32),w,scale,rows))
 gate=all(s['val']['corr'] is not None and s['val']['corr']>=.8 and s['val']['rmse']<=.6*s['val']['ideal_sd'] and s['val']['mvm_readback_rmse']<=.05*scale*32 for s in stage)
 emit('gates.jsonl',dict(stage='32',passed=gate,metrics=stage));check(gate,'32-tap validation gate failed; no 75-tap extension')
 print('GATE32_PASSED',flush=True)
 for rep,seed in enumerate(P['seeds']['array_reprogram']):
  for tile in (1,2):run_tile(rep,tile,seed,P['tile_ranges'][tile],w,scale,rows)
 full=aggregate_full()
 (OUT/'full75_summary.json').write_text(json.dumps(full,indent=2,allow_nan=False))
 print('ALL_TILES_COMPLETED',time.monotonic()-START,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  emit('abort.jsonl',dict(error=repr(e),elapsed_s=time.monotonic()-START,epoch=time.time()));print('ABORT',repr(e),flush=True);raise
