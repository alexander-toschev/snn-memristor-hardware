"""Read-only compact trajectory diagnosis; no acquisition imports or Git."""
import json
from pathlib import Path
import math
import statistics as st
from collections import Counter

ROOT=Path(__file__).resolve().parents[3]
SESSION='results/cifar_feedback_remaining_reserves/58ca618b-c37f-454e-be25-db21d1d25f60'
wrappers=json.loads((ROOT/SESSION/'completed_microtiles.json').read_text())

def channels(rows):
    return [sum(v['voltage_v']-(row['zero_before']['voltage_v']+row['zero_after']['voltage_v'])/2 for v in row['signals']) for row in rows]

def leaf(w):
    o=w['origin']
    while 'ancestry_original_origin' in o:o=o['ancestry_original_origin']
    return o

records=[]
for wrapper in wrappers:
    r=wrapper['result'];reset={tuple(c['cell']):c for c in r['reset_candidates']};h=r['matrix']['reference_h_v'];o=leaf(wrapper)
    assert len(r['pairs'])==len(r['effective_route'])
    for i,(p,s) in enumerate(zip(r['pairs'],r['effective_route'])):
        assert p['tap']==s['tap_index']
        initial=channels(p['initial_mvm']);postrows=[channels(f['measured_rows']) for f in p['post_feedback']]
        own=[st.mean(row[k] for row in postrows) for k in range(2)];allwrite=[h[k][i] for k in range(2)]
        target=p['trained_weight'];post=p['post_mean_signed_v'];after=r['physical_fidelity']['raw_measured_weight'][i]
        assert math.isclose(post,own[0]-own[1],abs_tol=1e-9)
        assert math.isclose(after,allwrite[0]-allwrite[1],abs_tol=1e-9)
        cellp=p['positive_cell'];celln=p['negative_cell'];base=[reset[tuple(cell)]['baseline_mvm_mean_v'] for cell in (cellp,celln)]
        active=0 if p['active_set_side']=='positive_cell' else 1
        records.append(dict(filter=wrapper['filter_id'],parent=wrapper['parent_tile_id'],micro=wrapper['microtile_id'],slot=i,
            tap=p['tap'],target=target,cells=[cellp,celln],reset_means=base,initial_channels=initial,own_write_channels=own,
            all_write_channels=allwrite,initial_D=p['initial_signed_v'],post_own_D=post,post_all_D=after,
            post_cifar_D=r['final_physical_fidelity']['raw_measured_weight'][i],effective=r['calibration']['effective_weights'][i],
            stop=p['stop'],pulses=p['programming_pulse_attempts'],active_side=p['active_set_side'],active_index=active,
            own_error=post-target,all_error=after-target,later_change=after-post,read_change=r['final_physical_fidelity']['raw_measured_weight'][i]-after,
            origin=o,post_own_sd=p['post_sd_signed_v']))
assert len(records)==300 and len({(r['filter'],r['tap']) for r in records})==300
den=sum(r['target']**2 for r in records);sse=sum(r['all_error']**2 for r in records)
def stats(rows):
    denom=sum(r['target']**2 for r in rows)
    return dict(n=len(rows),target_rms=math.sqrt(denom/len(rows)),
        nrmse_own=math.sqrt(sum(r['own_error']**2 for r in rows)/denom),
        nrmse_all=math.sqrt(sum(r['all_error']**2 for r in rows)/denom),
        nrmse_cifar=math.sqrt(sum((r['post_cifar_D']-r['target'])**2 for r in rows)/denom),
        rms_later_change=math.sqrt(st.mean(r['later_change']**2 for r in rows)),
        rms_read_change=math.sqrt(st.mean(r['read_change']**2 for r in rows)),
        SSE_fraction=sum(r['all_error']**2 for r in rows)/sse)
top=sorted(records,key=lambda r:r['all_error']**2,reverse=True)
for r in top:r['SSE_fraction']=r['all_error']**2/sse
decomposition=dict(own_sse=sum(r['own_error']**2 for r in records),
    later_sse=sum(r['later_change']**2 for r in records),cross_term=2*sum(r['own_error']*r['later_change'] for r in records),total_sse=sse)
report=dict(source=SESSION,scope='compact results plus exact measured row channels, no raw replay',all=stats(records),
    by_stop={stop:stats([r for r in records if r['stop']==stop]) for stop in sorted({r['stop'] for r in records})},
    by_filter={str(f):stats([r for r in records if r['filter']==f]) for f in range(4)},
    decomposition=decomposition,top20=top[:20],records=records,
    counters=dict(stops=dict(Counter(r['stop'] for r in records)),
        later_change_gt_1mV=sum(abs(r['later_change'])>.001 for r in records),
        read_change_gt_1mV=sum(abs(r['read_change'])>.001 for r in records),
        initial_nonactive_gt_5mV=sum(abs(r['initial_channels'][1-r['active_index']])>.005 for r in records),
        inactive_ownchange_gt_5mV=sum(abs(r['own_write_channels'][1-r['active_index']]-r['initial_channels'][1-r['active_index']])>.005 for r in records)))
(Path(__file__).parent/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('records','top20')},ensure_ascii=False,indent=2))
print('TOP',json.dumps([{k:r[k] for k in ['filter','tap','target','cells','reset_means','initial_channels','own_write_channels','all_write_channels','stop','pulses','active_side','own_error','all_error','SSE_fraction']} for r in top[:10]],ensure_ascii=False,indent=2))
