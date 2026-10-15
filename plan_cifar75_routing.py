"""Offline candidate routing of trained Conv1 coefficients; no acquisition or writes."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parent
INPUTS={
    'weights/prepared_cifar100_conv1.json':'86113ff1d2fdbe155056d2e1377cf380e7395120fd93ccf2e1efc7a0a933167c',
    'weights/conv1_weights.npy':'8be312b9eafb9de23f8340b17f848e7e0e2e7884293c826a6d9f6cbb5125d440',
    'configs/mapped_sequential_mvm.json':'80d1195f8288002e4013686468ce8ff04ac991c75b3210ccba5f78bc46a5b440'}


def local(value):
    path=(ROOT/value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(ROOT):raise ValueError('Repository-relative path required')
    return path


def read_weights(data):
    """Read only the fingerprinted, non-object little-endian float32 NPY array."""
    if data[:8]!=b'\x93NUMPY\x01\x00':raise ValueError('Expected NPY v1')
    size=struct.unpack('<H',data[8:10])[0];header=ast.literal_eval(data[10:10+size].decode('ascii'))
    if header!={'descr':'<f4','fortran_order':False,'shape':(64,3,5,5)}:raise ValueError('Unexpected NPY contract')
    payload=data[10+size:]
    if len(payload)!=64*75*4:raise ValueError('Unexpected weight payload size')
    values=struct.unpack('<4800f',payload)
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite trained weight')
    return [list(values[i*75:(i+1)*75]) for i in range(64)]


def build(prepared,mapping,weights):
    if prepared['schema']!='prepared-cifar100-conv1-v1' or prepared['flatten_order']!='channel,row,column':raise ValueError('Prepared fixture contract')
    if prepared['filters']!=[0,1,2,3] or prepared['weights_shape']!=[64,3,5,5] or prepared['timesteps']!=12:raise ValueError('Filter/time contract')
    if prepared['hardware_weights_programmed'] or prepared['hardware_representation_qualified']:raise ValueError('Unexpected hardware qualification claim')
    bn=prepared['bntt']
    if bn['training'] is not False or bn['eps']!=1e-5:raise ValueError('Expected eval BNTT eps=1e-5')
    pairs=mapping['pairs']
    if [len(p['positive_bl']) for p in pairs]!=[14,9,11,13] or [p['wl'] for p in pairs]!=[[0,1],[2,3],[4,5],[6,7]]:raise ValueError('Candidate routing capacities differ')
    filters=[]
    for fid,pair in zip(prepared['filters'],pairs):
        capacity=len(pair['positive_bl']);tiles=[]
        if len(pair['negative_bl'])!=capacity or any(len(set(pair[k]))!=capacity for k in ('positive_bl','negative_bl')):raise ValueError('Repeated/unequal BL routing')
        for tile_id,start in enumerate(range(0,75,capacity)):
            slots=[]
            for j,(plus,minus) in enumerate(zip(pair['positive_bl'],pair['negative_bl'])):
                tap=start+j;active=tap<75;weight=weights[fid][tap] if active else None
                slots.append(dict(slot=j,tap_index=tap if active else None,
                    tap_chw=[tap//25,(tap%25)//5,tap%5] if active else None,
                    trained_weight=weight,float32_le_hex=struct.pack('<f',weight).hex() if active else None,
                    positive_cell=[pair['wl'][0],plus],negative_cell=[pair['wl'][1],minus],
                    padding=not active,padding_input=0 if not active else None,
                    physical_target=None,representation_qualified=False))
            tiles.append(dict(tile_id=tile_id,tap_start=start,tap_stop=min(start+capacity,75),slots=slots,
                active_taps=min(capacity,75-start),padding_slots=max(0,start+capacity-75)))
        filters.append(dict(filter_id=fid,wl_pair=pair['wl'],capacity=capacity,tiles=tiles,
            bntt_t12={name:[rows[t][fid] for t in range(12)] for name,rows in bn['parameters_t12x64'].items()}))
    stages=[dict(stage=i,filter_tiles=[dict(filter_id=f['filter_id'],tile_id=i if i<len(f['tiles']) else None,
        action='candidate_tile' if i<len(f['tiles']) else 'idle_no_contribution') for f in filters]) for i in range(max(len(f['tiles']) for f in filters))]
    return dict(schema='candidate-cifar75-routing-v1',status='offline_candidate_not_execution_ready',
        dataset=prepared['dataset'],filters=filters,stages=stages,
        tile_counts=[len(f['tiles']) for f in filters],total_filter_tiles=sum(len(f['tiles']) for f in filters),
        shared_stages=len(stages),unique_physical_cells=94,trained_coefficients=300,
        source_flatten_order=prepared['flatten_order'],timesteps=12,lif=prepared['lif'],
        bntt_eps=bn['eps'],bntt_training=False,
        accumulation='For each logical t, sum ALL measured tiles per filter in one trained-weight unit system BEFORE one BN[t] and one LIF update.',
        scheduling_caveat='Potential future tile-major acquisition may buffer all T contributions; never reset LIF or apply BN separately per tile.',
        routing='A logical tap is routed to the listed physical BL separately for its positive and negative WL.',
        physical_encoding='No q/G/DAC write targets or trained-weight-to-measured-current scale are specified.',
        unsupported_policy='Do not replace unsupported coefficients with zero, basis lookup, or software dot products in a claimed hardware inference.',
        final_silent_input_steps='T10/T11 still require trained BN beta and LIF evaluation.',
        source_existing_tiles='Prepared fixture uses32+32+11; this planner replaces geometry only, not coefficients or golden outputs.',
        mapping_provisional=True,mapped_states_qualified=False,weights_programmed=False,cifar_executed=False,
        checkpoint_scale_verified_on_hardware=False,training_source_caveat=prepared['training_source_caveat'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='weights/candidate_cifar75_routing.json');args=parser.parse_args(argv)
    blobs={}
    for name,pin in INPUTS.items():
        blobs[name]=local(name).read_bytes()
        if hashlib.sha256(blobs[name]).hexdigest()!=pin:raise ValueError('Pinned input differs: '+name)
    prepared=json.loads(blobs['weights/prepared_cifar100_conv1.json'])
    result=build(prepared,json.loads(blobs['configs/mapped_sequential_mvm.json']),read_weights(blobs['weights/conv1_weights.npy']))
    result['input_sha256']=INPUTS;result['generator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['checkpoint_sha256']=prepared['checkpoint_sha256'];result['source_fixture_sha256']=prepared['input_fixture_sha256']
    output=local(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=args.output,tile_counts=result['tile_counts'],total_filter_tiles=result['total_filter_tiles'],shared_stages=result['shared_stages'],status=result['status'])))
    return 0


if __name__=='__main__':raise SystemExit(main())
