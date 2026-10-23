"""Restore a published terminal archive inside its repository result directory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_common.durable import sha
from experiment_common.runtime import repo_path
from experiment_common.selective_metrology import object_dict, number


def restore(base: Path, sid: str) -> int:
    sid = str(uuid.UUID(sid))
    base = base.resolve()
    if not base.is_relative_to(ROOT/'results'):
        raise ValueError('Expected repository results directory')
    receipt = object_dict(json.loads((base/'archives'/f'{sid}.json').read_text(encoding='utf-8')))
    archive = base/'archives'/f'{sid}.tar.gz'
    if receipt['byte_identical_archive_verified'] is not True or sha(archive) != receipt['archive_sha256']:
        raise ValueError('Archive receipt mismatch; fetch the matching LFS object')
    expected = object_dict(receipt['files'])
    folder = base/sid
    folder.mkdir(exist_ok=True)
    seen: set[str] = set()
    with tarfile.open(archive, 'r:gz') as tf:
        for member in tf:
            if not member.isfile() or not member.name.startswith(sid+'/'):
                raise ValueError('Only regular session files are allowed')
            relative = member.name[len(sid)+1:]
            posix = PurePosixPath(relative)
            target = (folder/relative).resolve()
            if posix.is_absolute() or '..' in posix.parts or not target.is_relative_to(folder.resolve()):
                raise ValueError('Archive path escapes its session')
            if relative in seen or relative not in expected:
                raise ValueError('Unexpected or duplicate archive member')
            evidence = object_dict(expected[relative])
            if member.size != int(number(evidence['bytes'])):
                raise ValueError('File size mismatch')
            seen.add(relative)
            if target.exists():
                if sha(target) != evidence['sha256']:
                    raise ValueError('Existing file differs; refusing overwrite')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = tf.extractfile(member)
            if stream is None:
                raise ValueError('Unreadable archive member')
            with stream, target.open('xb') as output:
                shutil.copyfileobj(stream, output, 1024**2)
            if sha(target) != evidence['sha256']:
                raise ValueError('Restored file checksum mismatch')
    if seen != set(expected):
        raise ValueError('Archive inventory is incomplete')
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='results/mnist_tiled_transfer_20261022')
    parser.add_argument('--session', required=True)
    args = parser.parse_args()
    count = restore(repo_path(args.base), args.session)
    print(json.dumps(dict(session=args.session, verified_files=count)))


if __name__ == '__main__':
    main()
