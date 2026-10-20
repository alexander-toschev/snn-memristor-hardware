"""Same image IDs as pooled32 pilots; new integer8x8 pooling gives64 inputs."""
import gzip
import json
from pathlib import Path
from experiments.mnist_accuracy_20261020 import inputs as official
from tools.prepare_mnist_conductance_fixture import selected_images,sha_bytes,digest

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
EDGES=(0,3,7,10,14,17,21,24,28)
TRAIN_GROUPS={'fit':list(range(11000,11032)),'preprobe':list(range(12000,12016)),
    'train':list(range(128)),'postprobe':list(range(12000,12016)),'retentionprobe':list(range(12000,12016))}
ACCURACY_GROUPS={'readout_train':list(range(128)),'official_test':list(range(100000,100100))}


def encode(pixels):
    if len(pixels)!=784 or any(type(v)is not int or not 0<=v<=255 for v in pixels):raise ValueError('Expected uint8 image')
    schedule=[[] for _ in range(32)];blocks=[]
    for row in range(8):
        for col in range(8):
            feature=8*row+col;area=(EDGES[row+1]-EDGES[row])*(EDGES[col+1]-EDGES[col])
            total=sum(pixels[y*28+x] for y in range(EDGES[row],EDGES[row+1]) for x in range(EDGES[col],EDGES[col+1]))
            tick=min(31,32*(255*area-total)//(255*area)) if total else None
            blocks.append(dict(feature=feature,sum_uint8=total,area=area,spike_t=tick))
            if tick is not None:schedule[tick].append(feature)
    return dict(blocks=blocks,schedule=schedule,schedule_sha256=digest(schedule))


def groups(scope):return TRAIN_GROUPS if scope=='train' else ACCURACY_GROUPS


def image_calls(row):
    calls=16
    for mask in row['schedule']:
        for tile in (0,1):
            n=sum(tile*32<=b<(tile+1)*32 for b in mask)
            calls+=4*(n+(n+3)//4)
    return calls


def verify(fixture):
    scope=fixture['scope']
    if scope not in ('train','accuracy') or fixture['groups']!=groups(scope) or fixture['labels_included'] is not False:
        raise ValueError('Invalid fixture scope')
    order=list(dict.fromkeys(i for group in groups(scope).values() for i in group))
    if [r['train_index'] for r in fixture['images']]!=order:raise ValueError('Fixture order differs')
    for row in fixture['images']:
        i=row['train_index'];split='test' if i>=100000 else 'train'
        if row['dataset_split']!=split or row['dataset_index']!=i%100000:raise ValueError('Wrong dataset identity')
        if set(row)!={'train_index','dataset_split','dataset_index','image_sha256','pixels_uint8','blocks','schedule','schedule_sha256'}:
            raise ValueError('Unexpected fixture fields/labels')
        if sha_bytes(bytes(row['pixels_uint8']))!=row['image_sha256']:raise ValueError('Image SHA differs')
        for k,v in encode(row['pixels_uint8']).items():
            if row[k]!=v:raise ValueError('Pooled64 encoding differs')
    return {r['train_index']:r for r in fixture['images']}


def budget(fixture):
    images=verify(fixture)
    reads=sum(image_calls(images[i]) for group in groups(fixture['scope']).values() for i in group)
    if fixture['scope']=='accuracy':return dict(image_impacts=reads,maximum_impacts=reads+7680,maximum_pulses=0)
    return dict(image_impacts=reads,maximum_impacts=reads+11520+6144+1536,maximum_pulses=2560,
        noeligible_maximum_impacts=sum(image_calls(images[i]) for i in TRAIN_GROUPS['fit'])+7680+6144)


def prepare(scope):
    order=list(dict.fromkeys(i for group in groups(scope).values() for i in group))
    images={}
    for split,prefix,count,offset in [('train','train',60000,0),('test','t10k',10000,100000)]:
        wanted=[i-offset for i in order if (i>=100000)==(split=='test')]
        if not wanted:continue
        name=prefix+'-images-idx3-ubyte.gz';official.checked(name)
        with gzip.open(official.DATA/name,'rb') as stream:
            found=selected_images(stream,wanted,count)
        for i,pixels in found.items():
            images[i+offset]=dict(train_index=i+offset,dataset_split=split,dataset_index=i,
                image_sha256=sha_bytes(pixels),pixels_uint8=list(pixels),**encode(list(pixels)))
    fixture=dict(schema='mnist64-label-free-v1',scope=scope,groups=groups(scope),labels_included=False,
                 source_images={k:v for k,v in official.SOURCES.items() if 'images' in k},
                 images=[images[i] for i in order])
    verify(fixture)
    path=HERE/('fixture_'+scope+'.json');text=json.dumps(fixture,separators=(',',':'))+'\n'
    if path.exists() and path.read_text()!=text:raise ValueError('Existing immutable fixture differs')
    path.write_text(text,encoding='utf-8',newline='\n')
    return dict(scope=scope,sha256=sha_bytes(path.read_bytes()),**budget(fixture))


def prepare_labels():
    for split in ('train','test'):
        labels=official.labels(split)
        value=dict(schema='official-mnist-label-prefix-v1',split=split,indices=list(range(len(labels))),
                   labels=labels,source_sha256=official.SOURCES[('train' if split=='train' else 't10k')+'-labels-idx1-ubyte.gz'])
        path=HERE/('labels_'+split+'.json');text=json.dumps(value,separators=(',',':'))+'\n'
        if path.exists() and path.read_text()!=text:raise ValueError('Frozen labels changed')
        path.write_text(text,encoding='utf-8',newline='\n')


def labels(split):
    path=HERE/('labels_'+split+'.json')
    value=json.loads(path.read_text())
    n=128 if split=='train' else 100
    if value['split']!=split or value['indices']!=list(range(n)) or len(value['labels'])!=n:
        raise ValueError('Label split/index differs')
    if any(type(y)is not int or not 0<=y<10 for y in value['labels']):raise ValueError('Invalid labels')
    return value['labels']


if __name__=='__main__':
    result=[prepare(s) for s in ('train','accuracy')];prepare_labels();print(json.dumps(result))
