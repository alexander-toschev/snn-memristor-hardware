"""Offline checkpoint -> Conv1/BNTT/LIF fixture. No MemriBoard or hardware imports.

Uses embedded real CIFAR pixels from a fingerprinted, independently extracted
fixture. Recomputes spikes and every reference from pixels and checkpoint;
the fixture's stored neural outputs are comparison evidence, not inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

FIXTURE_SHA='8cb9ec0fe30242328527073bb076f8019295d73a748343f58fc2a8671bf7c154'
ROOT=Path(__file__).resolve().parent


def local(value):
    from experiments.unn_rev3_20261014.paths import repo_path
    return repo_path(value)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(pixels,torch,T=12,x_min=.1):
    image=torch.as_tensor(pixels,dtype=torch.uint8).to(torch.float32)/255.
    times=torch.floor((1-image)*(T-1)).to(torch.int64)
    return torch.stack([((times==t)&(image>=x_min)).float() for t in range(T)])


def export(checkpoint,fixture,output):
    import numpy as np
    import torch
    if sha(fixture)!=FIXTURE_SHA:raise ValueError('Golden raw-image fixture fingerprint differs')
    data=json.loads(fixture.read_bytes())
    for name,pin in data['source_sha256'].items():
        if name.startswith('weights/') and sha(local(name))!=pin:raise ValueError('Weight/config fingerprint: '+name)
    if sha(checkpoint)!=data['source_sha256']['weights/best_fs.pt']:raise ValueError('Checkpoint fingerprint differs')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    weights=state['conv_layers.0.weight'].detach().float().cpu()
    if weights.shape!=(64,3,5,5):raise ValueError('Unexpected Conv1 shape')
    if not np.array_equal(weights.numpy(),np.load(local('weights/conv1_weights.npy'),allow_pickle=False)):
        raise ValueError('NPY differs from actual checkpoint Conv1')
    bn={name:torch.stack([state[f'bn_layers.0.{t}.{name}'].float() for t in range(12)]) for name in ('weight','bias','running_mean','running_var')}
    cases=[];maximum={name:0. for name in ('conv_sum','bntt_current','lif_before_reset','lif_after_reset')};mismatches=0
    for original in data['cases']:
        image=torch.as_tensor(original['image_chw_uint8'],dtype=torch.uint8).reshape(3,32,32)
        y,x=original['patch_top_left_yx'];patch=image[:,y:y+5,x:x+5]
        spikes=encode(patch,torch);v=torch.zeros(64);trace={name:[] for name in (*maximum,'lif_spikes')}
        tile_values=[]
        for t in range(12):
            # Native float32 Conv2d reduction, followed by ONE BN after all75 taps.
            conv=torch.nn.functional.conv2d(spikes[t].unsqueeze(0),weights).reshape(64)
            current=torch.nn.functional.batch_norm(conv.reshape(1,64,1,1),bn['running_mean'][t],bn['running_var'][t],
                bn['weight'][t],bn['bias'][t],training=False,momentum=.1,eps=1e-5).reshape(64)
            before=.5*v+current;s=(before>=1.).float();v=before-s*before
            values=dict(conv_sum=conv,bntt_current=current,lif_before_reset=before,lif_after_reset=v,lif_spikes=s)
            for name,value in values.items():
                trace[name].append(value.tolist())
                expected=torch.tensor(original['reference_t12x64'][name][t],dtype=value.dtype)
                if name=='lif_spikes':mismatches+=int((value!=expected).sum())
                else:maximum[name]=max(maximum[name],float((value-expected).abs().max()))
            flat=spikes[t].reshape(75)
            tile_values.append([flat[a:b].tolist()+[0.]*(32-(b-a)) for a,b in ((0,32),(32,64),(64,75))])
        if spikes.reshape(12,75).tolist()!=original['spikes_t12x75']:raise ValueError('Recomputed TTFS differs from independent fixture')
        cases.append(dict(original_test_index=original['original_test_index'],fine_label=original['fine_label'],
            remapped_label=original['remapped_label'],patch_top_left_yx=[y,x],
            spikes_t12x75=spikes.reshape(12,75).tolist(),tiles_t12x3x32=tile_values,
            tile_sha256=hashlib.sha256(json.dumps(tile_values,separators=(',',':')).encode()).hexdigest(),
            reference_t12x64=trace))
    if mismatches or any(v>1e-5 for v in maximum.values()):raise ValueError('Independent golden neural reference mismatch')
    result=dict(schema='prepared-cifar100-conv1-v1',scope='Offline trained Conv1/BNTT/LIF; not hardware inference or full network accuracy',
        source_sha256=data['source_sha256'],input_fixture_sha256=sha(fixture),checkpoint_sha256=sha(checkpoint),
        torch_version=torch.__version__,training_source_commit=None,
        training_source_caveat='Present source hashes and checkpoint numerical compatibility, not cryptographic proof of historical training code',
        dataset=data['dataset'],filters=[0,1,2,3],all_filter_count=64,weights_shape=[64,3,5,5],
        flatten_order='channel,row,column',tiles=[[0,32],[32,64],[64,75]],last_tile_zero_padding=21,
        tile_accumulation='Sum three measured contributions BEFORE trained BN[t] and LIF; unsupported mapped weights may not be fabricated',
        bntt=dict(eps=1e-5,training=False,parameters_t12x64={name:value.tolist() for name,value in bn.items()}),
        lif=data['lif'],timesteps=12,x_min=.1,input_encoding='floor((1-pixel/255)*(T-1)), one spike if pixel/255 >= .1',
        final_two_timesteps='Conv input silent, BN beta and LIF still evaluated',cases=cases,
        validation=dict(maximum_absolute_difference=maximum,spike_mismatches=mismatches),
        hardware_weights_programmed=False,hardware_representation_qualified=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='weights/best_fs.pt')
    parser.add_argument('--fixture',default='weights/cifar100_conv1_fixture.json')
    parser.add_argument('--output',default='weights/prepared_cifar100_conv1.json')
    args=parser.parse_args(argv);result=export(local(args.checkpoint),local(args.fixture),local(args.output))
    print(json.dumps(result['validation']));return 0


if __name__=='__main__':raise SystemExit(main())
