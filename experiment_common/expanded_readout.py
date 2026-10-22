"""Read-only validation of a pinned expansion plan, using native binary MVM."""
from __future__ import annotations
import json
from pathlib import Path
import statistics
from experiment_common.block_readout_validation import Block, Collector
from experiment_common.tiled_transfer import Profile, acquire_load
from experiment_common.selective_metrology import object_dict, number
from experiment_common.durable import atomic_json


def validate(reader: Collector, path: Path, out: Path) -> dict[str, object]:
    plan = object_dict(json.loads(path.read_text(encoding='utf-8')))
    values = plan['profiles']
    if not isinstance(values, list):
        raise ValueError('Pinned profiles required')
    profiles: list[Profile] = []
    metadata: dict[str, dict[str, object]] = {}
    for value in values:
        row = object_dict(value)
        wls, bls = row['wl_pair'], row['bls']
        if not isinstance(wls, list) or len(wls)!=2 or not isinstance(bls, list):
            raise ValueError('Invalid map')
        name = str(row['name'])
        b = Block(name, int(number(wls[0])), int(number(wls[1])), tuple(int(number(v)) for v in bls))
        if not 0 <= b.plus_wl < 8 or not 0 <= b.minus_wl < 8 or b.plus_wl == b.minus_wl or not 1<=len(b.bls)<=3 or any(not 0<=v<32 for v in b.bls):
            raise ValueError('Invalid bounded profile')
        profiles.append(Profile(b, row['mode']=='fixed_positive', number(row['delta_v']), row.get('global_bias') is True))
        metadata[name] = row
    operators, _ = acquire_load(reader, tuple(profiles), out, 0)
    selected: dict[str, dict[str, object]] = {}
    assessments: list[dict[str, object]] = []
    for value in operators:
        name = str(value['block'])
        row = metadata[name]
        checks = value['checks']
        if not isinstance(checks, list):
            raise ValueError('Missing heldout checks')
        passed = all(object_dict(c)['passed'] is True for c in checks)
        resolved = True
        if row['new'] is True:
            # A new physical leg must have a resolved response; cancellation
            # of a signed pair remains legitimate and is not itself a failure.
            pairs = [p for p in reader.pairs if p.block==name and p.phase=='basis']
            for leg in ('plus', 'minus'):
                values_adc = [(p.plus if leg=='plus' else p.minus).centered('mean4') for p in pairs]
                resolved &= abs(statistics.mean(values_adc)) > max(5.,5*statistics.stdev(values_adc))
        admitted = passed and resolved
        assessments.append(dict(name=name, group=row['group'], passed=admitted, repeatability=passed, resolved_legs=resolved, checks=checks))
        group = str(row['group'])
        if admitted and group not in selected:
            selected[group] = dict(block=group, bls=row['bls'], wl_pair=row['wl_pair'], mode=row['mode'],
                delta_v=row['delta_v'], coefficient_slots=len(row['bls']) if isinstance(row['bls'],list) else 0,
                new=row['new'], global_bias=row.get('global_bias', False), calibrated_range=row.get('calibrated_range'),
                qualification='current_state_binary_readout; post-weight-programming revalidation still required')
    all_cells: set[tuple[int, int]] = set()
    for row in selected.values():
        wls, bls = row['wl_pair'], row['bls']
        if not isinstance(wls,list) or not isinstance(bls,list):
            raise ValueError('Invalid selected coordinates')
        for w in wls:
            for b in bls:
                address = int(number(w)),int(number(b))
                if address in all_cells:
                    raise ValueError('Overlapping selected blocks')
                all_cells.add(address)
    result = dict(selected_blocks=list(selected.values()), assessments=assessments,
        physical_cells=len(all_cells), coefficient_slots=len(all_cells)//2,
        new_blocks=sum(r['new'] is True for r in selected.values()),
        qualified_scope='current-state readout; fine programming previously tested only on new cells')
    atomic_json(out/'selected_map.json', result)
    return result
