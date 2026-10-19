"""Separate full STDP+new-weight evaluation gate; frozen smoke source stays unchanged."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
ENTRY=ROOT/'experiments/mnist_stdp_retraining_20261019/experiment.py'
ENTRY_SHA='7f2a35d9fe5b832686d73b57d940acbc6fae99a3406de41258377d4d2fd9ee3f'
SMOKE=ROOT/'results/mnist_stdp_retraining_20261019/440fc54b-a3bc-48f9-ae3c-43196809b96d'
SMOKE_MANIFEST_SHA='3761f285f1a659c0a03aa6d235b9fadb655840f1d90e53ef3f5b9565918e5f4a'
QUALIFICATION_REVIEW=ROOT/'reports/2026-10-19/mnist-stdp-training-review/SMOKE128_INDEPENDENT_AUDIT.json'
QUALIFICATION_REVIEW_SHA='6c200ce9f7db12e82afa68d1eb181bf4666a15ddfd18dc36066b1833516058a5'
FULL_SPEC=dict(schema='mnist-full-stdp-qualified-launch-v1',training_helper_sha256=ENTRY_SHA,
               training_spec_sha256='db6a90ce72a25a37144971f405c6a23097de83c9583dd0c9e713848c50e205bd',
               smoke_manifest_sha256=SMOKE_MANIFEST_SHA,fresh_training_images=30000,training_batch=1,timesteps=100,
               qualification_review_sha256=QUALIFICATION_REVIEW_SHA,
               feature_train_samples=30000,feature_test_samples=10000,head_epochs=25,test_evaluations=1,
               seed=42,maximum_wall_seconds=28800,maximum_output_bytes=8*1024**3,
               reconstruction=True,physical_memristor=False,resume_smoke_weights=False)
APPROVED_FULL_SPEC_SHA256=frozenset({'5e0e64cf86142de7c55ac0e505014c56e619c2c2e26f93349deac89720843ebe'})


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def import_reviewed_helper():
    if sha(ENTRY)!=ENTRY_SHA:raise ValueError('Frozen smoke acquisition source changed')
    sys.path.insert(0,str(ROOT))
    module=importlib.import_module('experiments.mnist_stdp_retraining_20261019.experiment')
    if Path(module.__file__).resolve()!=ENTRY or digest(module.SPEC)!=FULL_SPEC['training_spec_sha256']:
        raise ValueError('Wrong acquisition helper/specification')
    return module


def qualify_smoke():
    if sha(SMOKE/'manifest.json')!=SMOKE_MANIFEST_SHA:raise ValueError('Qualifying smoke manifest changed')
    manifest=json.loads((SMOKE/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed' or manifest['scope']!='train_only128_smoke' or manifest['entrypoint_sha256']!=ENTRY_SHA:
        raise ValueError('Wrong qualifying smoke session')
    if manifest['training']['training_samples']!=128 or not manifest['training']['learning_beyond_initial_clipping_observed']:
        raise ValueError('Smoke did not qualify STDP learning')
    if {p.name for p in SMOKE.iterdir() if p.is_file()}!=set(manifest['result_sha256'])|{'manifest.json'}:
        raise ValueError('Smoke inventory changed')
    for name,pin in manifest['result_sha256'].items():
        path=SMOKE/name
        if not path.resolve().is_relative_to(SMOKE.resolve()) or sha(path)!=pin:raise ValueError('Smoke artifact changed')
    reviewed=QUALIFICATION_REVIEW_SHA is not None
    if reviewed and sha(QUALIFICATION_REVIEW)!=QUALIFICATION_REVIEW_SHA:raise ValueError('Independent qualification changed')
    return dict(smoke_manifest_sha256=SMOKE_MANIFEST_SHA,independently_reviewed=reviewed,
                independent_review_path=QUALIFICATION_REVIEW.relative_to(ROOT).as_posix(),
                independent_review_sha256=QUALIFICATION_REVIEW_SHA)


def confirmed_events(output):
    """Completed event coverage, never inferred from preflight/planned counts."""
    result=dict(training_images=0,feature_batches={'train':0,'test':0},feature_rows={'train':0,'test':0},
                head_updates_at_last_completed_epoch=0,completed_head_epochs=0,test_evaluations=0)
    path=output/'events.jsonl'
    if not path.exists():return result
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);kind=row.get('kind')
            if kind=='stdp_image':result['training_images']+=1
            elif kind=='new_weight_features':
                result['feature_batches'][row['split']]+=1
                result['feature_rows'][row['split']]+=row['end']-row['start']
            elif kind=='head_epoch':
                result['completed_head_epochs']=row['epoch'];result['head_updates_at_last_completed_epoch']=row['updates']
            elif kind=='test_complete':result['test_evaluations']+=1
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--run',action='store_true');args=parser.parse_args(argv)
    full_sha=digest(FULL_SPEC)
    if args.run and (full_sha not in APPROVED_FULL_SPEC_SHA256 or QUALIFICATION_REVIEW_SHA is None):
        raise ValueError('Full source/scientific qualification and registration required')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG',':4096:8')!=':4096:8':raise ValueError('Unreviewed CUDA determinism setting')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    helper=import_reviewed_helper();qualification=qualify_smoke()
    p=helper.recovery.helpers();checked=helper.recovery.preflight(p);ref,cfg=helper.inputs(p)
    planned={name:checked.pop(name) for name in ['head_updates','feature_batches','counts_bytes']}
    checked['source_sha256'][ENTRY.relative_to(ROOT).as_posix()]=ENTRY_SHA
    checked['source_sha256'][helper.SUMMARY.relative_to(ROOT).as_posix()]=helper.SUMMARY_SHA
    checked['source_sha256'][Path(helper.recovery.__file__).relative_to(ROOT).as_posix()]=helper.RECOVERY_SHA
    if args.preflight:
        print(json.dumps(dict(full_spec=FULL_SPEC,full_spec_sha256=full_sha,qualification=qualification,planned=planned,
                             effective_cfg=vars(cfg),**checked)));return 0
    output=ROOT/'results/mnist_stdp_retraining_20261019'/str(uuid.uuid4());output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='failed',started_utc=p.now(),spec=helper.SPEC,spec_sha256=digest(helper.SPEC),full_run_spec=FULL_SPEC,
        full_run_spec_sha256=full_sha,effective_cfg=vars(cfg),entrypoint_sha256=sha(__file__),acquisition_helper_sha256=ENTRY_SHA,
        qualification=qualification,planned=planned,physical_memristor=False,**checked)
    p.write_json(output/'start_manifest.json',manifest)
    try:helper.execute(output,p,ref,cfg,False,manifest)
    except (Exception,KeyboardInterrupt) as exc:manifest['error']=type(exc).__name__+': '+str(exc)
    finally:
        manifest['finished_utc']=p.now()
        try:manifest['confirmed_execution']=confirmed_events(output)
        except Exception as exc:manifest['coverage_audit_error']=str(exc);manifest['status']='failed'
        manifest['result_sha256']={f.name:sha(f) for f in output.iterdir() if f.is_file()}
        p.write_json(output/'manifest.json',manifest)
    print(json.dumps(dict(status=manifest['status'],output=output.relative_to(ROOT).as_posix(),accuracy=manifest.get('accuracy'),error=manifest.get('error'))))
    return 0 if manifest['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
