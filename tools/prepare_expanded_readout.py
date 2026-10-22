"""Pin nonoverlapping extra pairs from completed bidirectional calibration."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import tarfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment_common.durable import sha, atomic_json
from experiment_common.selective_metrology import object_dict, number


def evidence(sid: str) -> dict[str, object]:
    base = ROOT/'results/adaptive_cells37_20261022/archives'
    receipt = object_dict(json.loads((base/(sid+'.json')).read_text(encoding='utf-8')))
    path = base/(sid+'.tar.gz')
    if sha(path)!=receipt['archive_sha256']:
        raise ValueError('Archive checksum mismatch')
    with tarfile.open(path) as tar:
        names = [n for n in tar.getnames() if n.endswith('/observations.json')]
        if len(names)!=1:
            raise ValueError('Ambiguous observations')
        stream = tar.extractfile(names[0])
        if stream is None:
            raise ValueError('Missing observations')
        data = stream.read()
    if hashlib.sha256(data).hexdigest()!=object_dict(object_dict(receipt['files'])['observations.json'])['sha256']:
        raise ValueError('Observation checksum mismatch')
    return object_dict(json.loads(data))


def main() -> int:
    first = evidence('71a00bab-a33f-4fb2-b459-bfce24ff0bee')
    second = evidence('40485ae9-bfc4-4a9a-b6bc-da568de8206b')
    rechecks = second['prior_rechecks']
    if not isinstance(rechecks,list):
        raise ValueError('Prior rechecks required')
    retained = {(int(number(r['wl'])),int(number(r['bl']))) for v in rechecks for r in [object_dict(v)] if r['passed'] is True}
    cells: dict[tuple[int,int],tuple[float,float]] = {}
    for data, prior in ((first,True),(second,False)):
        values = data['cells']
        if not isinstance(values,list):
            raise ValueError('Cells required')
        for value in values:
            row = object_dict(value)
            key = int(number(row['wl'])),int(number(row['bl']))
            if row['qualified_programming'] is not True or prior and key not in retained:
                continue
            targets = row['targets']
            if not isinstance(targets,list):
                raise ValueError('Targets required')
            goals = [number(object_dict(t)['target_g_s']) for t in targets]
            cells[key] = min(goals),max(goals)
    # Maximum number of non-overlapping pairs per BL; among equal-size
    # matchings prefer wider common, already tested G intervals.
    def matching(wls: tuple[int,...], bl: int) -> tuple[list[tuple[int,int]],float]:
        if len(wls)<2:
            return [],0.
        best, score = matching(wls[1:],bl)
        for j in range(1,len(wls)):
            a,b = wls[0],wls[j]
            lo,hi = max(cells[a,bl][0],cells[b,bl][0]),min(cells[a,bl][1],cells[b,bl][1])
            if hi-lo <= 4*2.5603554203687646e-6:
                continue
            rest,value = matching(wls[1:j]+wls[j+1:],bl)
            proposal = [(a,b),*rest]
            if (len(proposal),value+hi-lo)>(len(best),score):
                best,score = proposal,value+hi-lo
        return best,score
    old = object_dict(json.loads((ROOT/'reports/2026-10-22/block-readout-validation/STABLE_BLOCKS.json').read_text(encoding='utf-8')))['selected_blocks']
    if not isinstance(old,list):
        raise ValueError('Old blocks required')
    profiles: list[dict[str,object]] = []
    for value in old:
        row = object_dict(value)
        profiles.append(dict(name=row['block'],group=row['block'],wl_pair=row['wl_pair'],bls=row['bls'],mode=row['mode'],delta_v=row['delta_v'],new=False))
    new_pairs: list[dict[str,object]] = []
    for bl in sorted({b for _,b in cells}):
        pairs,_ = matching(tuple(sorted(w for w,b in cells if b==bl)),bl)
        for a,b in pairs:
            group = f'N{len(new_pairs)+1:02d}'
            span = [max(cells[a,bl][0],cells[b,bl][0]),min(cells[a,bl][1],cells[b,bl][1])]
            row = dict(group=group,wl_pair=[a,b],bls=[bl],calibrated_range=span,new=True)
            new_pairs.append(row)
            for i,(mode,delta) in enumerate((('ordinary',.05),('fixed_positive',.025),('fixed_positive',.05))):
                profiles.append(dict(**row,name=f'{group}_p{i}',mode=mode,delta_v=delta))
    out = ROOT/'experiments/expanded_blocks_20261022'
    out.mkdir(exist_ok=True)
    plan: dict[str,object] = dict(profiles=profiles,new_pairs=new_pairs,qualified_new_cells=len(cells),
        common_range_minimum_s=4*2.5603554203687646e-6,
        source_sessions=['71a00bab-a33f-4fb2-b459-bfce24ff0bee','40485ae9-bfc4-4a9a-b6bc-da568de8206b'],
        selection='maximum disjoint pairs within new cells with common calibrated range; retain old12 blocks for revalidation')
    atomic_json(out/'plan.json',plan)
    print(json.dumps(dict(qualified_new_cells=len(cells),new_pairs=new_pairs)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
