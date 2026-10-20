"""Descriptive saved MNIST conductance main/reference comparison; no classifier or board."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def load(audit_path):
    audit=json.loads(audit_path.read_text());require(audit['status']=='PASS' and audit['outcome']=='completed','Independent completed audit required')
    session=ROOT/audit['session'];manifest=json.loads((session/'manifest.json').read_text())
    require(manifest['source']=='simulator' and manifest['status']=='completed','This comparison is native SIM only')
    require(sha(session/'manifest.json')==audit['manifest_sha256'],'Audited manifest changed')
    require(sha(session/'raw/index.json')==audit['raw_index_sha256'],'Audited raw index changed')
    actual={p.relative_to(session).as_posix() for p in session.rglob('*') if p.is_file()}-{'manifest.json'}
    require(actual==set(manifest['result_sha256']),'Result allowlist differs')
    inventory={}
    for name,pin in manifest['result_sha256'].items():
        path=(session/name).resolve();require(path.is_relative_to(session.resolve()),'Result path escapes session')
        require(sha(path)==pin,'Result file SHA differs')
        inventory[path.relative_to(ROOT).as_posix()]=dict(sha256=pin,bytes=path.stat().st_size)
    inventory[(session/'manifest.json').relative_to(ROOT).as_posix()]=dict(sha256=sha(session/'manifest.json'),bytes=(session/'manifest.json').stat().st_size)
    return audit,manifest,json.loads((session/'observations.json').read_text()),inventory


def phase_summary(observations):
    answer={}
    for phase,diag in observations['diagnostics']['phases'].items():
        outcomes=[r for r in observations['outcomes'] if r['phase']==phase]
        by_wl=[[tuple(r['neurons'][wl]['posts']) for r in outcomes] for wl in range(8)]
        require(len(outcomes)==diag['images'],'Phase images disagree')
        pairs=[]
        for p in diag['pairwise']:
            pairs.append(dict(p,active_conditioned_match_fraction=p['matching_active_images']/p['active_images'] if p['active_images'] else None))
        answer[phase]=dict(images=len(outcomes),posts_by_wl=[sum(sum(v) for v in values) for values in by_wl],silent_images_by_wl=[sum(not any(v) for v in values) for values in by_wl],over8posts_images_by_wl=[sum(sum(v)>8 for v in values) for values in by_wl],unique_eight_neuron_patterns=len({tuple(tuple(n['posts']) for n in r['neurons']) for r in outcomes}),unique_learning_patterns=len({tuple(tuple(n['posts']) for n in r['neurons'][:4]) for r in outcomes}),unique_control_patterns=len({tuple(tuple(n['posts']) for n in r['neurons'][4:]) for r in outcomes}),pairwise=pairs)
    return answer


def checkpoint_delta(observations,start='after_preprobe',end='after_train_128'):
    checkpoints={r['phase']:r['mean_inverse_r'] for r in observations['checkpoints']}
    rows=[]
    for wl in range(8):
        before,after=checkpoints[start][wl],checkpoints[end][wl]
        delta=[b-a for a,b in zip(before,after)]
        rows.append(dict(wl=wl,bank='learning' if wl<4 else 'paired_control',mean_g_before_s=statistics.mean(before),mean_g_after_s=statistics.mean(after),mean_delta_g_s=statistics.mean(delta),mean_absolute_delta_g_s=statistics.mean(map(abs,delta)),changed_cells_exact=sum(v!=0 for v in delta),positive_cells=sum(v>0 for v in delta),negative_cells=sum(v<0 for v in delta),delta_g_s=delta))
    return dict(start_checkpoint=start,end_checkpoint=end,rows=rows,zero_threshold='exact equality, no posthoc noise cutoff')


def updates(observations):
    rows=[r for r in observations['updates'] if r['phase']=='training_update']
    admitted=[r for r in rows if r['admitted']]
    occupancy=Counter((r['wl'],r['bl'],r['sign']) for r in admitted)
    signs={}
    for sign in (0,1):
        group=[r for r in admitted if r['sign']==sign]
        signs['SET' if sign else 'RESET']=dict(pulses=len(group),positive=sum(r['delta_inverse_r']>0 for r in group),negative=sum(r['delta_inverse_r']<0 for r in group),unchanged=sum(r['delta_inverse_r']==0 for r in group),expected_sign=sum((r['delta_inverse_r']>0 if sign else r['delta_inverse_r']<0) for r in group))
    return dict(training_attempts=len(rows),admitted=len(admitted),reference_suppressed=sum(r['reference_suppressed'] for r in rows),guard_skipped=sum(not r['guard_admissible'] for r in rows),unique_updated_cells=len({(r['wl'],r['bl']) for r in admitted}),updated_cells_by_wl=[len({r['bl'] for r in admitted if r['wl']==wl}) for wl in range(8)],pulses_by_wl=[sum(r['wl']==wl for r in admitted) for wl in range(8)],signs=signs,occupancy=[dict(wl=w,bl=b,sign=s,count=n) for (w,b,s),n in sorted(occupancy.items())],paired_control_immediate_delta_g=[1/r['control_post']['r']-1/r['control_guard']['r'] for r in admitted],nonzero_learning_eligibility=sum(e!=0 for outcome in observations['outcomes'] if outcome['phase']=='train' for neuron in outcome['neurons'][:4] for e in neuron['eligibility']))


def trajectory_comparison(left,right,left_phase,right_phase):
    a={r['train_index']:r for r in left['outcomes'] if r['phase']==left_phase}
    b={r['train_index']:r for r in right['outcomes'] if r['phase']==right_phase}
    require(set(a)==set(b),'Probe/train identities differ')
    answer=[]
    for wl in range(8):
        pair=[(a[k]['neurons'][wl]['posts'],b[k]['neurons'][wl]['posts']) for k in sorted(a)]
        active=[(x,y) for x,y in pair if any(x) or any(y)]
        union=sum(bool(i or j) for x,y in pair for i,j in zip(x,y));intersection=sum(bool(i and j) for x,y in pair for i,j in zip(x,y))
        answer.append(dict(wl=wl,images=len(pair),identical_images=sum(x==y for x,y in pair),active_union_images=len(active),identical_active_images=sum(x==y for x,y in active),hamming_bits=sum(i!=j for x,y in pair for i,j in zip(x,y)),jaccard=intersection/union if union else None))
    return answer


def compare(main_audit,reference_audit):
    ma,mm,mo,mi=load(main_audit);ra,rm,ro,ri=load(reference_audit)
    require(mm['reference'] is None and rm['reference']['session']==ma['session'],'Wrong main/reference binding')
    require(rm['reference']['manifest_sha256']==ma['manifest_sha256'],'Reference main manifest mismatch')
    require(mm['array_initial_sha256']==rm['array_initial_sha256']==sha(ROOT/ma['session']/'array_initial.cb')==sha(ROOT/ra['session']/'array_initial.cb'),'Initial states differ')
    require(mm['source_sha256']==rm['source_sha256'] and mm['protocol_sha256']==rm['protocol_sha256'],'Sources/protocol differ')
    require(mo['gain']==ro['gain']==rm['reference']['gain'],'Common frozen gain differs')
    require(ra['native_no_update_reference'] and ra['required_comparison_status']=='reference_and_main_replayed','Independent reference replay required')
    require(ra['main_independent_audit']['manifest_sha256']==ma['manifest_sha256'],'Nested main audit mismatch')
    summaries={}
    for label,a,m,o in [('main',ma,mm,mo),('reference',ra,rm,ro)]:
        summaries[label]=dict(session=a['session'],manifest_sha256=a['manifest_sha256'],gain=o['gain'],impacts=o['impacts'],pulses=o['pulses'],train_images=128,measured_presentations=o['measured_images'],checkpoint_cells=256,image_active_bl=[bl for bl in range(32) if bl not in a['no_image_event_bl']],no_image_event_bl=a['no_image_event_bl'],image_contributing_cells=sum(v>0 for row in a['image_cell_participation'] for v in row),phase=phase_summary(o),updates=updates(o),training_g_change=checkpoint_delta(o),retention_g_change=checkpoint_delta(o,'after_postprobe','after_retentionprobe'),conductance_diversity=o['diagnostics']['conductance'])
    inventory={**mi,**ri}
    report=dict(schema='mnist-conductance-native-main-reference-description-v1',source='native MemriBoard simulator',audits={label:dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for label,p in [('main',main_audit),('reference',reference_audit)]},initial_array_sha256=mm['array_initial_sha256'],gain=mo['gain'],main=summaries['main'],reference=summaries['reference'],main_pre_post=trajectory_comparison(mo,mo,'preprobe','postprobe'),main_post_retention=trajectory_comparison(mo,mo,'postprobe','retentionprobe'),reference_pre_post=trajectory_comparison(ro,ro,'preprobe','postprobe'),between_sessions={p:trajectory_comparison(mo,ro,p,p) for p in ('preprobe','train','postprobe','retentionprobe')},result_allowlist=inventory,hardware_measured=False,classification_accuracy_measured=False,full10k_qualified=False,hardware_qualified=False,limitations=['One native SIM main/reference pair; no physical-board measurement or classification.', 'Shared initial bytes and gain do not guarantee identical stochastic draws after update branches diverge.', 'Paired rows begin with different conductances; between-row differences are not a matched effect estimator.', 'Exact nonzero conductance differences include model/readout variability; no posthoc noise threshold.', 'Probe retention in simulator is not demonstrated physical retention.', 'Training scope is128images/128eligible cells; all256 checkpoints do not imply all256 image-active or updated.'])
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--main-audit',required=True);p.add_argument('--reference-audit',required=True);p.add_argument('--output',required=True);args=p.parse_args(argv)
    paths=[(ROOT/v).resolve() for v in (args.main_audit,args.reference_audit,args.output)]
    require(all(v.is_relative_to(ROOT/'reports') for v in paths),'Reports paths required')
    require(not paths[2].exists(),'Preserve earlier analysis')
    report=compare(paths[0],paths[1]);paths[2].parent.mkdir(parents=True,exist_ok=True)
    with paths[2].open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='descriptive_comparison_saved',output=args.output,classification_accuracy_measured=False)))


if __name__=='__main__':main()
