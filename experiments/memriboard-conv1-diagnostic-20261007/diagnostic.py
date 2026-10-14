"""Exploratory readback-only diagnostic: two fresh simulator arrays, no MVM."""
import importlib.util, json, hashlib, pathlib, random, signal, sys, time
import numpy as np
import torch

ROOT=pathlib.Path('/mnt/d/projects/snn-mnist')
ORIG=ROOT/'.tmp/openclaw-spikes/memriboard-conv1-20261007'
OUT=ROOT/'.tmp/openclaw-spikes/memriboard-conv1-diagnostic-20261007'
EXPECTED={'PROTOCOL.json':'9eb55bf0bec21674c4c4c6132d02dd226d468706216328b0d5ba86c2f82a12c8','pilot.py':'f1172ccafbf5f65ae3da42a5b3a69c615984f96bb67fb89a0fae82f5f212a41f','summary.json':'3a0995b8e0b2ea4e73e877f6759a9756458bc28b6c40fd5c31af1af5188c992b','array_r0_tile0.cb':'51d4650f0d0185e96bec8037f2900019c8dedf4bda48d7c3fb0c10b6e89403cf'}
START=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def check(b,s):
 if not b:raise RuntimeError(s)
def emit(n,v):
 with (OUT/n).open('a') as f:f.write(json.dumps(v,sort_keys=True,allow_nan=False)+'\n');f.flush()
# Import exact already-inspected local pilot headless FakeApplication/converter plumbing.
# The __main__ guard prevents execution; redirects no original output and imports no physical connector.
spec=importlib.util.spec_from_file_location('readonly_pilot_setup',ORIG/'pilot.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
class NoMVMBoard(p.BoardSimulator):
 def __init__(self):super().__init__();self.config=p.CFG
 def mode_mvm(self,*args,**kwargs):raise RuntimeError('MVM forbidden in exploratory readback-only diagnostic')
def expired(*args):raise TimeoutError('cell write >20s')
def program(seed,w,scale):
 random.seed(seed);np.random.seed(seed)
 serial=str(OUT/f'array_seed{seed}')
 check(not pathlib.Path(serial+'.cb').exists(),f'array already exists: {serial}')
 p.create_crossbar_array(serial,32,8)
 initial_sha=sha(serial+'.cb')
 board=NoMVMBoard();check(board.connect(serial)==1 and type(board) is NoMVMBoard,'simulator connection failed')
 core=p.HardCore(board);core.set_mvm();check((core.ROW_NUM,core.COL_NUM)==(32,8),'shape/orientation')
 metadata={'seed':seed,'array_id':serial,'initial_sha256':initial_sha,'shape':[32,8],'programmed_columns':[0,1]}
 emit('arrays.jsonl',metadata)
 g=np.stack((.03+.02*w/scale,.03-.02*w/scale),axis=1)
 start=time.monotonic()
 for tap in range(32):
  for col in (0,1):
   signal.alarm(20)
   try:
    t=time.monotonic();final,hist,pulses=core.write_weight(tap,col,float(g[tap,col]));dt=time.monotonic()-t
   finally:signal.alarm(0)
   emit('programming.jsonl',{'seed':seed,'array_id':serial,'tap':tap,'row':tap,'column':col,'target':float(g[tap,col]),'final_write_return':float(final),'write_seconds':dt,'weight_history':hist,'pulses_v':pulses,'array_sha256_after_write':sha(serial+'.cb')})
   check(dt<=20 and time.monotonic()-start<900,'write time bound exceeded')
 measured=[];resistances=[];repeated=[]
 for tap in range(32):
  measured.append([]);resistances.append([]);repeated.append([])
  for col in (0,1):
   readings=[core.read_one_weight(tap,col) for _ in range(3)]
   measured[-1].append(float(np.mean([v[0] for v in readings])))
   resistances[-1].append(float(np.mean([v[1] for v in readings])))
   repeated[-1].append([[float(v[0]),float(v[1])] for v in readings])
 m=np.asarray(measured);sig=np.abs(w[:32])>.05*scale
 check(np.all(np.isfinite(m)) and np.all(np.asarray(resistances)>0),'readback invalid')
 result={'seed':seed,'array_id':serial,'initial_sha256':initial_sha,'final_sha256':sha(serial+'.cb'),'target':g[:32].tolist(),'measured':measured,'resistance_ohm':resistances,'repeated_weight_resistance':repeated,'target_gap':(g[:32,0]-g[:32,1]).tolist(),'measured_gap':(m[:,0]-m[:,1]).tolist(),'gap_error':((m[:,0]-m[:,1])-(g[:32,0]-g[:32,1])).tolist(),'sign_agreement':float(np.mean(np.sign(m[sig,0]-m[sig,1])==np.sign(w[:32][sig]))),'wrong_sign_taps':np.flatnonzero(sig&(np.sign(m[:,0]-m[:,1])!=np.sign(w[:32]))).tolist(),'elapsed_s':time.monotonic()-start}
 emit('readback.jsonl',result)
 print('ARRAY_DONE',seed,result['sign_agreement'],result['wrong_sign_taps'],result['elapsed_s'],flush=True)
 return result

def main():
 for name,digest in EXPECTED.items():check(sha(ORIG/name)==digest,f'original changed: {name}')
 manifest=json.loads((OUT/'MANIFEST.json').read_text());check(manifest['original_sha256']==EXPECTED and manifest['script_sha256']==sha(__file__),'manifest/script mismatch')
 cp=ROOT/'runs_csnn/confirmatory_wide_bntt_20261006/twd1_sigma0p0_seed0/best_fs.pt'
 check(sha(cp)=='ee4fc4f58a3d010218116e9ad4c7ba808787959f9ed7687cd58cbb32d90f62fb','checkpoint mismatch')
 w=torch.load(cp,map_location='cpu',weights_only=True)['conv_layers.0.weight'][0].detach().numpy().ravel().astype(float)
 check(w.shape==(75,) and np.all(np.isfinite(w)),'filter invalid');scale=float(np.max(np.abs(w)))
 check(scale>0 and not any(OUT.glob('array_seed*.cb')),'diagnostic arrays already exist')
 emit('preflight.jsonl',{'original_sha256':EXPECTED,'checkpoint_sha256':sha(cp),'filter':0,'full_filter_maxabs':scale,'seeds':[17002,17003],'connector':'NoMVMBoard(BoardSimulator)','python':sys.executable})
 signal.signal(signal.SIGALRM,expired)
 for seed in (17002,17003):program(seed,w,scale)
 for name,digest in EXPECTED.items():check(sha(ORIG/name)==digest,f'original changed after diagnostic: {name}')
 print('COMPLETE',time.monotonic()-START,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:emit('abort.jsonl',{'error':repr(e),'elapsed_s':time.monotonic()-START});raise
