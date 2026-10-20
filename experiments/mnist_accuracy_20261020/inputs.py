"""Fixed official MNIST128 train / MNIST100 test, with labels kept off acquisition."""
import gzip
import json
import struct
from pathlib import Path
from tools.prepare_mnist_conductance_fixture import pooled_schedule, selected_images, sha_bytes, digest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT/'experiments/mnist_baseline_20261019/data/MNIST/raw'
SOURCES = {
    'train-images-idx3-ubyte.gz': '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609',
    'train-labels-idx1-ubyte.gz': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c',
    't10k-images-idx3-ubyte.gz': '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6',
    't10k-labels-idx1-ubyte.gz': 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6',
}
GROUPS = {'readout_train': list(range(128)), 'official_test': list(range(100000, 100100))}


def checked(name):
    data = (DATA/name).read_bytes()
    if sha_bytes(data) != SOURCES[name]:
        raise ValueError('Official IDX SHA differs: '+name)
    return data


def prepare():
    rows = []
    for split, prefix, count, wanted, offset in (
            ('train', 'train', 60000, 128, 0), ('test', 't10k', 10000, 100, 100000)):
        name = prefix+'-images-idx3-ubyte.gz'
        checked(name)
        with gzip.open(DATA/name, 'rb') as stream:
            images = selected_images(stream, list(range(wanted)), count)
        for i in range(wanted):
            pixels = list(images[i])
            rows.append(dict(train_index=i+offset, dataset_split=split, dataset_index=i,
                image_sha256=sha_bytes(images[i]), pixels_uint8=pixels, **pooled_schedule(pixels)))
    fixture = dict(schema='mnist-official-128train-100test-label-free-v1', groups=GROUPS,
        source_images={k:v for k,v in SOURCES.items() if 'images' in k},
        labels_included=False, images=rows)
    verify(fixture)
    target = HERE/'fixture.json'
    text = json.dumps(fixture, separators=(',', ':'), allow_nan=False)+'\n'
    if target.exists() and target.read_text(encoding='utf-8') != text:
        raise ValueError('Refusing to replace frozen fixture')
    target.write_text(text, encoding='utf-8', newline='\n')
    return dict(fixture=target.relative_to(ROOT).as_posix(), sha256=sha_bytes(target.read_bytes()),
                exact_impacts=impact_count(fixture), maximum_impacts=128064, pulses=0)


def verify(fixture):
    if fixture['schema'] != 'mnist-official-128train-100test-label-free-v1' or fixture['groups'] != GROUPS or fixture['labels_included'] is not False:
        raise ValueError('Wrong fixture scope')
    if fixture['source_images'] != {k:v for k,v in SOURCES.items() if 'images' in k}:
        raise ValueError('Wrong source image pins')
    expected = GROUPS['readout_train']+GROUPS['official_test']
    if [r['train_index'] for r in fixture['images']] != expected:
        raise ValueError('Wrong ordered official split coverage')
    for row in fixture['images']:
        i = row['train_index']; split = 'train' if i < 100000 else 'test'
        if row['dataset_split'] != split or row['dataset_index'] != i % 100000:
            raise ValueError('Dataset identity differs')
        if set(row) != {'train_index','dataset_split','dataset_index','image_sha256','pixels_uint8','blocks','schedule','schedule_sha256'}:
            raise ValueError('Unexpected fixture fields / labels')
        if sha_bytes(bytes(row['pixels_uint8'])) != row['image_sha256']:
            raise ValueError('Image bytes differ')
        for key,value in pooled_schedule(row['pixels_uint8']).items():
            if row[key] != value: raise ValueError('Input encoding differs')
    return {r['train_index']:r['schedule'] for r in fixture['images']}


def impact_count(fixture):
    return 7680 + sum(16+8*sum(len(m)+(len(m)+3)//4 for m in r['schedule']) for r in fixture['images'])


def labels(split):
    """Only the local readout calls this; acquisition never opens label files."""
    prefix,count,wanted = ('train',60000,128) if split=='train' else ('t10k',10000,100)
    data = gzip.decompress(checked(prefix+'-labels-idx1-ubyte.gz'))
    if data[:8] != struct.pack('>II',2049,count) or len(data) != count+8 or any(v>9 for v in data[8:]):
        raise ValueError('Invalid official label IDX')
    return list(data[8:8+wanted])


if __name__=='__main__':
    print(json.dumps(prepare()))
