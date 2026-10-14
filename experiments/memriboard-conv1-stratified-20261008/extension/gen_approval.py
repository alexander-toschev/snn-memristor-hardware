"""Generate pilot/remaining approval JSONs with exact source hashes (parent-owned guard)."""
import hashlib, json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
FROZEN = '2eae1ff82a1cc2bf55013beedfd0d23030ad6ff3659dd57cac77605cf8395467'
CHECKPOINT = 'ee4fc4f58a3d010218116e9ad4c7ba808787959f9ed7687cd58cbb32d90f62fb'

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

src = {n: sha(HERE / n) for n in ('runner.py', 'extension_core.py', 'PROTOCOL.json')}
stages = {'pilot': [63], 'remaining': [41, 32, 30, 38, 43, 60, 47, 2, 9, 46, 12, 0, 1]}
for stage, filters in stages.items():
    rec = dict(decision='APPROVED_BY_PARENT', stage=stage, frozen_protocol_sha256=FROZEN,
               source_sha256=src, checkpoint_sha256=CHECKPOINT, filters=filters,
               no_concurrent_training_confirmed=True)
    out = HERE / f'approval_{stage}.json'
    out.write_text(json.dumps(rec, indent=2, sort_keys=True) + '\n')
    print(out.name, 'written; runner.py', src['runner.py'][:12])
