"""Pin a smaller map using programming retention and ALL pilot readout attempts.

No labels, accuracy or simulator outcome enter selection. Subsets of old
multi-input blocks still require a new post-write pilot before the full run.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_common.durable import atomic_json, sha
from experiment_common.selective_metrology import object_dict, number

BASE = ROOT/'results/mnist_tiled_transfer_20261022/archives'
SESSIONS = ('e39b8972-8199-4bbb-9a4c-7b95b21bd24d', 'aa6c49fc-42b3-4f38-bb7c-8609c73452d5')


def main() -> None:
    bad_cells: set[tuple[int, int]] = set()
    bad_blocks: set[str] = set()
    evidence: list[dict[str, object]] = []
    for sid in SESSIONS:
        receipt = object_dict(json.loads((BASE/f'{sid}.json').read_text(encoding='utf-8')))
        archive = BASE/f'{sid}.tar.gz'
        if sha(archive) != receipt['archive_sha256']:
            raise ValueError('Source archive checksum mismatch')
        evidence.append(dict(session=sid, archive_sha256=sha(archive)))
        with tarfile.open(archive) as tf:
            for member in tf:
                if not member.name.endswith(('_programming.json', '_operators.json')):
                    continue
                stream = tf.extractfile(member)
                if stream is None:
                    raise ValueError('Missing archived JSON')
                value: object = json.load(stream)
                if member.name.endswith('_programming.json'):
                    if not isinstance(value, list):
                        raise ValueError('Expected programming records')
                    for v in value:
                        row = object_dict(v)
                        if row['within_tolerance'] is not True:
                            bad_cells.add((int(number(row['wl'])), int(number(row['bl']))))
                else:
                    operators = object_dict(value)['operators']
                    if not isinstance(operators, list):
                        raise ValueError('Expected operator records')
                    for v in operators:
                        row = object_dict(v)
                        checks = row['checks']
                        if not isinstance(checks, list):
                            raise ValueError('Expected checks')
                        if any(object_dict(c)['passed'] is not True for c in checks):
                            bad_blocks.add(str(row['block']))
    source = ROOT/'experiments/mnist_tiled_transfer_20261022/map44.json'
    plan = object_dict(json.loads(source.read_text(encoding='utf-8')))
    blocks = plan['selected_blocks']
    if not isinstance(blocks, list):
        raise ValueError('Expected source map')
    selected: list[dict[str, object]] = []
    for value in blocks:
        row = dict(object_dict(value))
        if row['block'] in bad_blocks:
            continue
        wls, bls = row['wl_pair'], row['bls']
        if not isinstance(wls, list) or not isinstance(bls, list):
            raise ValueError('Expected coordinates')
        kept = [b for b in bls if all((int(number(w)), int(number(b))) not in bad_cells for w in wls)]
        if kept:
            row.update(bls=kept, coefficient_slots=len(kept), qualification='candidate selected after written pilot; new subset-mask validation required')
            selected.append(row)
    cells = 2*sum(int(number(b['coefficient_slots'])) for b in selected)
    result = dict(physical_cells=cells, selected_blocks=selected, source_map_sha256=sha(source), evidence=evidence,
        excluded_cells=sorted(bad_cells), excluded_blocks=sorted(bad_blocks),
        rule='Both legs within target tolerance in both retained readbacks; all complete MVM attempts pass; no test labels; subset masks revalidated before full run')
    atomic_json(source.with_name('map_qualified.json'), result)
    print(json.dumps(dict(physical_cells=cells, blocks=len(selected), excluded_blocks=sorted(bad_blocks))))


if __name__ == '__main__':
    main()
