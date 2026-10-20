"""Exact official CIFAR bytes -> pooled integer TTFS. No pickle callbacks."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickletools
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CLASSES=[0,1,2,3,5,8,13,14,17,19]
PINS={
    'train':'735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b',
    'test':'4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864',
    'meta':'a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9',
}
SPLIT_SHA='0967934d6f3fe1c6a8dbbaf42e32d3eb1416428ba47579a64734d37d25cc0690'
EXCLUDED={9,132,54,396,334,27,94,7,49,122}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def parse_official(raw):
    """Symbolically interpret the fixed protocol-2 container; never call GLOBAL."""
    stack=[];memo={};mark=object()
    def marked():
        index=len(stack)-1
        while stack[index] is not mark:index-=1
        values=stack[index+1:];del stack[index:];return values
    for op,arg,pos in pickletools.genops(raw):
        name=op.name
        if name=='PROTO':
            if arg!=2:raise ValueError('Expected protocol2')
        elif name=='MARK':stack.append(mark)
        elif name=='EMPTY_DICT':stack.append({})
        elif name=='EMPTY_LIST':stack.append([])
        elif name in ('BINPUT','LONG_BINPUT'):memo[arg]=stack[-1]
        elif name in ('BINGET','LONG_BINGET'):stack.append(memo[arg])
        elif name in ('SHORT_BINSTRING','BINSTRING','BINUNICODE','BININT','BININT1','BININT2'):
            stack.append(arg)
        elif name=='NONE':stack.append(None)
        elif name in ('NEWTRUE','NEWFALSE'):stack.append(name=='NEWTRUE')
        elif name=='GLOBAL':
            if arg not in ('numpy.core.multiarray _reconstruct','numpy ndarray','numpy dtype'):
                raise ValueError('Unexpected symbolic global')
            stack.append(('global',arg))
        elif name.startswith('TUPLE'):
            count={'TUPLE1':1,'TUPLE2':2,'TUPLE3':3}.get(name)
            if count is None:stack.append(tuple(marked()))
            else:
                values=tuple(stack[-count:]);del stack[-count:];stack.append(values)
        elif name=='REDUCE':
            arguments=stack.pop();function=stack.pop()
            if function not in (('global','numpy.core.multiarray _reconstruct'),('global','numpy dtype')):
                raise ValueError('Unrecognized symbolic reduce')
            stack.append(dict(symbolic_function=function[1],arguments=arguments))
        elif name=='BUILD':
            state=stack.pop()
            if not isinstance(stack[-1],dict) or 'symbolic_function' not in stack[-1]:
                raise ValueError('Unexpected symbolic BUILD')
            stack[-1]['state']=state
        elif name=='APPENDS':
            values=marked();stack[-1].extend(values)
        elif name=='SETITEMS':
            values=marked()
            if len(values)%2:raise ValueError('Odd dict items')
            stack[-1].update(zip(values[::2],values[1::2]))
        elif name=='STOP':
            if len(stack)!=1 or pos!=len(raw)-1:raise ValueError('Trailing/invalid pickle container')
            return stack[0]
        else:raise ValueError('Unsupported opcode: '+name)
    raise ValueError('No STOP')


def load_dataset(name):
    path=ROOT/'data/cifar-100-python'/name
    if sha(path)!=PINS[name]:raise ValueError('Official dataset checksum mismatch: '+name)
    value=parse_official(path.read_bytes())
    if name=='meta':return value
    state=value['data']['state']
    n=50000 if name=='train' else 10000
    if state[0]!=1 or state[1]!=(n,3072) or state[3] is not False:
        raise ValueError('Unexpected official array shape/order')
    dtype=state[2]
    if dtype['symbolic_function']!='numpy dtype' or dtype['arguments'][0]!='u1':
        raise ValueError('Unexpected official dtype')
    pixels=state[4].encode('latin1') if isinstance(state[4],str) else state[4]
    if len(pixels)!=n*3072:raise ValueError('Wrong pixel byte count')
    images=np.frombuffer(pixels,dtype=np.uint8).reshape(n,3,32,32)
    labels=np.asarray(value['fine_labels'],dtype=np.int64)
    if labels.shape!=(n,):raise ValueError('Wrong labels shape')
    return images,labels


def encode(images):
    if images.dtype!=np.uint8 or images.ndim!=4 or images.shape[1:]!=(3,32,32):
        raise ValueError('Expected uint8 NCHW32')
    sums=images.reshape(-1,3,4,8,4,8).sum(axis=(3,5),dtype=np.int32).reshape(-1,48)
    times=(11*(16320-sums)//16320).astype(np.int8)
    times[sums<1632]=-1
    spikes=(times[:,None,:]==np.arange(12)[None,:,None]).astype(np.float32)
    return sums,times,spikes


def prepare():
    if sha(ROOT/'weights/split_manifest.json')!=SPLIT_SHA:raise ValueError('Split checksum mismatch')
    split=json.loads((ROOT/'weights/split_manifest.json').read_text(encoding='utf-8'))
    train_images,train_labels=load_dataset('train')
    ids=np.flatnonzero(np.isin(train_labels,CLASSES))
    train_positions=np.asarray(split['train_indices'],dtype=np.int64)
    val_positions=np.asarray(split['val_indices'],dtype=np.int64)
    if (len(ids)!=5000 or len(train_positions)!=4500 or len(val_positions)!=500
        or sorted(np.concatenate([train_positions,val_positions]).tolist())!=list(range(5000))):
        raise ValueError('Train/validation split incomplete or overlaps')
    remap={v:i for i,v in enumerate(CLASSES)}
    images=train_images[ids];labels=np.asarray([remap[int(v)] for v in train_labels[ids]])
    test_images,test_labels=load_dataset('test')
    historical=json.loads((ROOT/'weights/cifar100_conv1_fixture.json').read_text(encoding='utf-8'))
    if {c['original_test_index'] for c in historical['cases']}!=EXCLUDED:
        raise ValueError('Historical fixture exclusion mismatch')
    chosen=sorted(i for label in CLASSES
                  for i in [int(x) for x in np.flatnonzero(test_labels==label) if int(x) not in EXCLUDED][:10])
    if len(chosen)!=100 or Counter(int(test_labels[i]) for i in chosen)!=Counter({v:10 for v in CLASSES}):
        raise ValueError('Pilot100 must have exactly10/class')
    return dict(images=images,labels=labels,official_train_ids=ids,
                train_positions=train_positions,val_positions=val_positions,
                test_images=test_images[chosen],test_labels=np.asarray([remap[int(test_labels[i])] for i in chosen]),
                official_test_ids=np.asarray(chosen))
