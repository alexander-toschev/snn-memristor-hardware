"""New diagnostic interpretation of saved A observations; original verdict remains."""
import copy
import json
import math
from pathlib import Path
from experiments.stable_cell_calibration_20261020 import metrics,plan
from experiments.cifar_resumable_20261019.durable import sha,atomic_json

ROOT=Path(__file__).resolve().parents[2]
RULE=dict(schema='resolved-span-diagnostic-v1',
    lower_bound='min_cycle(H_mean-R_mean-3*hypot(sdH,sdR)/sqrt(10))-2ADC > 0',
    quantization_margin_adc=2.,se_multiplier=3.,
    rationale='Difference of two separately zero-centered endpoints: conservative2ADC quantization uncertainty',
    selection='oneWLpair; upto8bestcommonBL; maxK,minworstqualityscore,lex',
    quality_qualified=False,labels_used=False,rerank_after_B=False)


def diagnostic_cell(original):
    row=copy.deepcopy(original)
    row['original_hard_reasons']=list(original['hard_reasons'])
    row['original_eligible']=original['eligible']
    row['diagnostic_lower_span_adc']=original.get('span_adc',-2)-2.
    other=[r for r in original['hard_reasons'] if r!='insufficient_reproducible_span']
    row['hard_reasons']=other+(['nonpositive_noise_and_quantization_lower_bound']
        if not math.isfinite(row['diagnostic_lower_span_adc']) or row['diagnostic_lower_span_adc']<=0 else [])
    row['hard_eligible']=not row['hard_reasons'];row['eligible']=False
    return row


def select(observations):
    checkpoints=observations['checkpoints']
    if len(checkpoints)!=6656 or [r['id'] for r in checkpoints]!=list(range(1,6657)):
        raise ValueError('Complete original A checkpoint sequence required')
    summaries=[{**r,'summary':metrics.checkpoint(r['brackets'])} for r in checkpoints]
    cells=[]
    for wl,bl in plan.CELLS:
        original=metrics.cell_summary(wl,bl,summaries,observations['updates'])
        row=diagnostic_cell(original)
        cells.append(row)
    selected=metrics.select_rectangle(cells)
    return dict(rule=RULE,cells=cells,selection=selected,qualified=False,
                scientific_scope='diagnostic resolution only; old10ADC verdict unchanged',
                full_raw_semantic_replay_performed=False)


def make_candidate(session,output):
    manifest=json.loads((session/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] not in ('completed','no_stable_rectangle','pair_qualification_failed'):
        raise ValueError('Original acquisition failed or incomplete')
    path=session/'observations.json'
    if sha(path)!=manifest['result_sha256']['observations.json']:
        raise ValueError('Saved observations bytes changed')
    observations=json.loads(path.read_text(encoding='utf-8'))
    result=select(observations)
    result.update(source=manifest['source'],original_session=session.relative_to(ROOT).as_posix(),
                  original_manifest_sha256=sha(session/'manifest.json'),observations_sha256=sha(path),
                  original_verdict=manifest['status'])
    if output.exists():raise FileExistsError('Preserve existing diagnostic analysis')
    output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,result)
    return dict(candidate=output.relative_to(ROOT).as_posix(),
                selected=result['selection']['selected'],
                diagnostic_resolvable=sum(c['hard_eligible'] for c in result['cells']),qualified=False)


def pair_diagnostic(selection,cells,observations):
    try:
        if any(r.get('status')=='measured' and (len(r.get('differences_adc',[]))!=10 or not all(math.isfinite(v) for v in r['differences_adc'])) for r in observations):
            raise ValueError('Invalid pair repeats')
        original=metrics.qualify_pairs(selection,cells,observations)
    except (ValueError,KeyError,TypeError):original=dict(qualified=False,mandatory_pass=False,error='invalid_pair_observations')
    rows=[];mandatory=[]
    for pair in selection['pairs']:
        for bl in selection['bls']:
            for configuration in ('plus','minus','zero','checkerboard'):
                for role in ('immediate','delayed'):
                    found=[r for r in observations if r['pair']==pair and r['bl']==bl
                           and r['configuration']==configuration and r['role']==role]
                    reason=None;lower=None
                    if len(found)!=1 or found[0]['status']!='measured':reason='missing_or_duplicate'
                    elif len(found[0].get('differences_adc',[]))!=10 or not all(math.isfinite(v) for v in found[0]['differences_adc']):reason='invalid_repeat_count_or_values'
                    elif configuration in ('plus','minus'):
                        s=metrics.stats(found[0]['differences_adc']);direction=1 if configuration=='plus' else -1
                        lower=direction*s['mean']-3*s['sd']/s['count']**.5-2
                        if lower<=0:reason='nonpositive_signed_noise_and_quantization_lower_bound'
                    record=dict(pair=pair,bl=bl,configuration=configuration,role=role,
                                lower_bound_adc=lower,reason=reason)
                    rows.append(record)
                    if reason:mandatory.append(record)
    return dict(mandatory_pass=not mandatory,qualified=False,diagnostic=True,
                original_quality_evaluation=original,measurements=rows,failures=mandatory,rerank=False)
