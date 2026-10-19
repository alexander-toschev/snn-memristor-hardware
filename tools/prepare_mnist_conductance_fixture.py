"""Extract the fixed label-free MNIST176 input fixture; no network, model or board."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PureWindowsPath
import struct

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'experiments/mnist_baseline_20261019/data/MNIST/raw/train-images-idx3-ubyte.gz'
SOURCE_SHA = '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609'
GROUPS = {'train': list(range(128)), 'fit': list(range(11000, 11032)), 'probe': list(range(12000, 12016))}
DEFAULT_OUTPUT = 'experiments/mnist_conductance_20261019/fixture.json'
ROWS = (0, 7, 14, 21, 28)
COLS = (0, 3, 7, 10, 14, 17, 21, 24, 28)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def pooled_schedule(image):
    if len(image) != 784 or any(type(v) is not int or not 0 <= v <= 255 for v in image):
        raise ValueError('Expected784 uint8 image pixels')
    blocks = []; schedule = [[] for _ in range(32)]
    for row in range(4):
        for column in range(8):
            bl = row * 8 + column
            area = (ROWS[row + 1] - ROWS[row]) * (COLS[column + 1] - COLS[column])
            total = sum(image[y * 28 + x] for y in range(ROWS[row], ROWS[row + 1])
                        for x in range(COLS[column], COLS[column + 1]))
            tick = min(31, 32 * (255 * area - total) // (255 * area)) if total else None
            blocks.append(dict(bl=bl, sum_uint8=total, area=area, spike_t=tick))
            if tick is not None:
                schedule[tick].append(bl)
    return dict(blocks=blocks, schedule=schedule, schedule_sha256=digest(schedule))


def local(name):
    path, windows = Path(name), PureWindowsPath(name)
    if not name or path.is_absolute() or windows.drive or windows.root or '..' in path.parts or '..' in windows.parts:
        raise ValueError('Use a repository-relative path')
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError('Path escapes repository')
    return result


def selected_images(stream, selected, expected_count=60000):
    """Validate the complete IDX shape and length, retaining only requested IDs."""
    if len(selected) != len(set(selected)) or any(type(i) is not int or not 0 <= i < expected_count for i in selected):
        raise ValueError('Invalid or duplicate image indices')
    if stream.read(16) != struct.pack('>IIII', 2051, expected_count, 28, 28):
        raise ValueError('Invalid MNIST image IDX header')
    wanted = set(selected); found = {}
    for index in range(expected_count):
        image = stream.read(784)
        if len(image) != 784:
            raise ValueError('Truncated MNIST image IDX')
        if index in wanted:
            found[index] = image
    if stream.read(1):
        raise ValueError('Trailing MNIST IDX bytes')
    return found


def build_fixture(images):
    order = [i for group in GROUPS.values() for i in group]
    if set(images) != set(order) or len(order) != 176 or any(len(image) != 784 for image in images.values()):
        raise ValueError('Fixed176 image coverage differs')
    rows = [dict(train_index=i, shape=[28, 28], image_sha256=sha_bytes(images[i]),
                 pixels_uint8=list(images[i]), **pooled_schedule(images[i])) for i in order]
    return dict(schema='mnist-conductance-label-free176-v1', source_dataset='MNIST official train IDX',
        source_path=SOURCE, source_gzip_sha256=SOURCE_SHA, split='train', groups=GROUPS,
        image_count=176, images_sha256=sha_bytes(b''.join(images[i] for i in order)), images=rows,
        labels_included=False, labels_read=False, test_split_read=False, model_executed=False,
        board_access=False, extraction='Exact uint8 pixels, ordered train128 then fit32 then probe16; integer pooling and TTFS32 metadata')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = local(args.output)
    if output.exists() or output.suffix != '.json':
        raise ValueError('Choose a new JSON fixture path; existing files are immutable')
    source = local(SOURCE)
    if sha_bytes(source.read_bytes()) != SOURCE_SHA:
        raise ValueError('Pinned training-image IDX changed or is a Git LFS pointer')
    order = [i for group in GROUPS.values() for i in group]
    with gzip.open(source, 'rb') as stream:
        images = selected_images(stream, order)
    fixture = build_fixture(images)
    fixture['extractor_sha256'] = sha_bytes(Path(__file__).read_bytes())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(fixture, stream, separators=(',', ':'), allow_nan=False); stream.write('\n')
    print(json.dumps(dict(output=args.output, sha256=sha_bytes(output.read_bytes()), images=176,
                          labels_read=False, model_executed=False, board_access=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
