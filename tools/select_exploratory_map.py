"""Deterministic maximum-cardinality same-BL differential map; no test labels."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_common.durable import sha
from experiment_common.selective_metrology import object_dict, number


def select(source: Path, assets: Path, output: Path) -> None:
    evidence = object_dict(json.loads(source.read_text(encoding='utf-8')))
    values = evidence['cells']
    if not isinstance(values, list):
        raise ValueError('Cells required')
    cells: dict[tuple[int, int], dict[str, object]] = {}
    for value in values:
        row = object_dict(value)
        hw = object_dict(row['HW'])
        if hw.get('observed_docx_switching') is True:
            cells[int(number(row['wl'])), int(number(row['bl']))] = hw
    payload = object_dict(json.loads(assets.read_text(encoding='utf-8')))
    import numpy as np
    g = np.asarray(payload['conductances_s'], dtype=np.float64)
    low, high = float(g.min()), float(g.max())
    pairs: dict[tuple[int, int], list[int]] = defaultdict(list)
    unused: list[tuple[int, int]] = []
    for bl in range(32):
        wls = sorted(w for w, b in cells if b == bl)
        for i in range(0, len(wls)-1, 2):
            pairs[wls[i], wls[i+1]].append(bl)
        if len(wls) % 2:
            unused.append((wls[-1], bl))
    count = sum(len(v) for v in pairs.values())
    merge_budget = 74518//(8*2*44)-count
    selected: list[dict[str, object]] = []
    def common(wls: tuple[int, int], bls: list[int]) -> tuple[float, float]:
        rows = [cells[w,b] for w in wls for b in bls]
        return max(1/number(r['max_r_ohm']) for r in rows), min(1/number(r['min_r_ohm']) for r in rows)
    for pair_wls, columns in sorted(pairs.items()):
        remaining = list(columns)
        while remaining:
            size = 1
            if len(remaining) >= 2 and merge_budget > 0:
                a,b = common(pair_wls, remaining[:2])
                if b > a:
                    size = 2
                    merge_budget -= 1
            bls, remaining = remaining[:size], remaining[size:]
            lo, hi = common(pair_wls, bls)
            if hi <= lo:
                raise ValueError('No shared conductance interval; no union fallback')
            scale = .6*(hi-lo)/(high-low)
            offset = lo+.2*(hi-lo)-scale*low
            selected.append(dict(block=f'E{len(selected)+1:03d}', wl_pair=pair_wls, bls=bls,
                mode='fixed_positive', delta_v=.05, global_bias=True, require_resolved_legs=False,
                conductance_scale=scale, conductance_offset=offset,
                measured_common_g_s=[lo,hi], encoded_target_g_s=[lo+.2*(hi-lo),lo+.8*(hi-lo)],
                qualification='exploratory; strict MVM flags retained, not prequalified for MNIST'))
    masks = sum((1<<len(v['bls']))-1 for v in selected if isinstance(v['bls'],list))
    result = dict(selected_blocks=selected, physical_cells=2*count, coefficient_slots=count,
        source_session=evidence['session_hardware'], source_comparison_sha256=sha(source),
        source_observations_sha256=evidence['hardware_observations_sha256'], source_assets_sha256=sha(assets),
        source_g_min_s=low, source_g_max_s=high, unpaired_reserve=unused,
        theoretical_maximum_cells=2*count, grouping='lexicographic WL pairing within each BL; at most2BL perblock',
        nonempty_patterns=masks, read_calls_per_load=masks*8*2*44,
        loads=(1600+count-1)//count, test_labels_used=False, mnist_qualified=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='selected_blocks'}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    select(a.source,a.assets,a.out)
