"""Offline, deterministic routing from historical endpoints; no device access."""
import copy
import json
from pathlib import Path

SIDES = ('positive_cell', 'negative_cell')


def build_route(original, evidence, width=7):
    if width not in range(1, 8):
        raise ValueError('Microtile width must be 1..7')
    cells = {tuple(c['cell']): c for c in evidence['cells']}
    used_ever = set()
    result = dict(schema='fresh-conv1-capacity-route-v1', qualification=False,
                  maximum_width=width, hardware_target_gain=.25,
                  simulator_target_gain=.03, filters=[])
    for old_filter in original['filters']:
        fid = old_filter['filter_id']
        weights = sorted((copy.deepcopy(s) for t in old_filter['tiles']
                          for s in t['slots'] if not s.get('padding')), key=lambda s: s['tap_index'])
        if [s['tap_index'] for s in weights] != list(range(75)):
            raise ValueError('Expected all 75 trained taps')
        pools = {side: sorted(c for c in cells if c[0] == 2*fid+j and c[1] in list(range(16))+[31])
                 for j, side in enumerate(SIDES)}
        out = dict(filter_id=fid, wl_pair=[2*fid, 2*fid+1], tiles=[])
        offset = 0
        while offset < 75:
            for n in range(min(width, 75-offset), 0, -1):
                subset = weights[offset:offset+n]
                chosen = {}; taken = set(); possible = True
                # Largest target first protects scarce high-response active legs.
                for s in sorted(subset, key=lambda s: (-abs(s['trained_weight']), s['tap_index'])):
                    side = SIDES[0] if s['trained_weight'] >= 0 else SIDES[1]
                    target = .25*abs(s['trained_weight'])
                    adequate = [c for c in pools[side] if c not in taken and
                                cells[c]['active_candidate_allowed'] and cells[c]['screening_amplitude_v'] >= target]
                    if not adequate:
                        possible = False; break
                    cell = min(adequate, key=lambda c: (c in used_ever, cells[c]['screening_amplitude_v'], c))
                    chosen[s['tap_index']] = {side: cell}; taken.add(cell)
                if not possible:
                    continue
                # Allocate quiet/inactive legs after all active requirements are reserved.
                for s in subset:
                    route = chosen[s['tap_index']]
                    side = next(side for side in SIDES if side not in route)
                    candidates = [c for c in pools[side] if c not in taken and
                                  cells[c]['inactive_candidate_allowed_only_after_fresh_joint_quiet']]
                    if not candidates:
                        possible = False; break
                    cell = min(candidates, key=lambda c: (c in used_ever,
                               cells[c]['active_candidate_allowed'], cells[c]['screening_amplitude_v'], c))
                    route[side] = cell; taken.add(cell)
                if possible:
                    break
            else:
                raise ValueError(f'No historically adequate active route for filter {fid}, tap {offset}')
            tile = dict(tile_id=len(out['tiles']), tap_start=offset, tap_stop=offset+n, slots=[])
            reserved = set(taken)
            # Every primary is protected from every fallback; a fallback is optional.
            for s in subset:
                slot = copy.deepcopy(s); slot['slot'] = len(tile['slots'])
                primary = chosen[s['tap_index']]
                slot.update({side: list(primary[side]) for side in SIDES})
                attempts = [{side: list(primary[side]) for side in SIDES}]
                active_side = SIDES[0] if s['trained_weight'] >= 0 else SIDES[1]
                inactive_side = SIDES[1] if active_side == SIDES[0] else SIDES[0]
                act = [c for c in pools[active_side] if c not in reserved and cells[c]['active_candidate_allowed']
                       and cells[c]['screening_amplitude_v'] >= .25*abs(s['trained_weight'])]
                ina = [c for c in pools[inactive_side] if c not in reserved and
                       cells[c]['inactive_candidate_allowed_only_after_fresh_joint_quiet']]
                if act and ina:
                    a = min(act, key=lambda c: (c in used_ever, cells[c]['screening_amplitude_v'], c))
                    b = min(ina, key=lambda c: (c in used_ever, cells[c]['active_candidate_allowed'], c))
                    attempts.append({active_side: list(a), inactive_side: list(b)})
                    reserved.update((a, b))
                slot['candidate_pairs'] = attempts
                slot['historical_capacity_is_not_fresh_reachability'] = True
                tile['slots'].append(slot)
            used_ever.update(taken)
            out['tiles'].append(tile); offset += n
        result['filters'].append(out)
    result['primary_distinct_addresses'] = [list(c) for c in sorted(used_ever)]
    result['primary_distinct_count'] = len(used_ever)
    result['trained_coefficients'] = 300
    result['candidate_pair_attempts'] = sum(len(s['candidate_pairs']) for f in result['filters']
                                            for t in f['tiles'] for s in t['slots'])
    return result


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    root = here
    original = json.loads((root/'weights/candidate_cifar75_routing.json').read_text(encoding='utf-8'))
    evidence = json.loads((root/'weights/fresh_conv1_capacity.json').read_text(encoding='utf-8'))
    route = build_route(original, evidence)
    destination=root/'weights/fresh_conv1_route.json'
    with destination.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(route,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k: route[k] for k in ('primary_distinct_count', 'candidate_pair_attempts')}))
    print('microtiles', [len(f['tiles']) for f in route['filters']])
