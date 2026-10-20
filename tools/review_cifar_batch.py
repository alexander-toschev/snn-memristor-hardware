"""Local completed-batch evidence review; frozen numerical raw audit runs separately."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import compare_cifar_continuation as compare
from tools import classify_cifar_continuation as composer
from tools import plan_cifar_continuation as planner
from tools import runtime_provenance as provenance_policy
from experiments.cifar_resumable_20261019 import contract

PINS={
 'tools/compare_cifar_continuation.py':'061eeead6bc2b6d4d8ce9a92e960aed92c954503ffcaaa1f138ce24f13fd502e',
 'tools/plan_cifar_continuation.py':'57edfc2f68c5d3de9ef4aad85ad8ab097cdabe080206722a23e9076df5e61908',
 'experiments/cifar_resumable_20261019/contract.py':'d7902489769b808f5a29a9e7e8e43d1e2662b915802cd6851302eb16024fe9be',
 'experiments/cifar_resumable_20261019/collector.py':'794eaced2fc551a7b26898cd57409dfe5276f3d4d7351787b1bc875f2fe6da1f',
}
require=compare.require
sha=compare.sha
TERMINAL_PART_LIMIT=8*1024**2


def local(name):return contract.local(name,root=ROOT)
def read(path):return json.loads(path.read_text(encoding='utf-8'))


def terminal_record(path):
    with gzip.open(path,'rb') as stream:data=stream.read(TERMINAL_PART_LIMIT+1)
    require(len(data)<=TERMINAL_PART_LIMIT,'Terminal raw part exceeds reviewed8MiB limit')
    require(data.endswith(b'\n') and bool(data.strip()),'Missing complete terminal line')
    return json.loads(data.splitlines()[-1])


def verify_tools():
    current=provenance_policy.versions(ROOT)['current']
    for name,pin in PINS.items():require(sha(local(name))==current.get(name,pin),'Reviewer dependency changed: '+name)
    compare.verify_tools();contract.verify_pins()


def source_pins():
    return {name:sha(local(name)) for name in {*PINS,*compare.PINNED_TOOLS,*contract.PINS,'tools/runtime_provenance.py',provenance_policy.REGISTRATION}}


def check_coefficients(rows,stages):
    expected={(s,(s//3)*4+p,(s%3)*32+b,b,2*p,2*p+1)
              for s in stages for p in range(4) for b in range((32,32,11)[s%3])}
    actual={(r['stage'],r['filter'],r['tap'],r['bl'],r['positive_cell'][0],r['negative_cell'][0]) for r in rows}
    require(actual==expected and len(rows)==len(expected),'Missing/duplicate/misplaced coefficient observation')
    require(all(r['positive_cell'][1]==r['negative_cell'][1]==r['bl']
                and r['per_coefficient_fitting_used'] is False for r in rows),'Invalid cell mapping or per-weight fitting')


def review_report(name,expected_plan_sha=None):
    path=local(name);report=read(path);report_sha=sha(path);session=local(report['session'])
    original=report.get('status')=='verified_complete_stage_prefix'
    require(expected_plan_sha is None or not original,'New batch requires completed literal stages')
    start=read(session/'start_manifest.json')
    reference=dict(acquisition_session=report['session'],
        accepted_stages=[r['stage'] for r in report['accepted_stages']],
        native_impacts=report['native_impacts'],programming_pulses=report['programming_pulses'],
        execution_grant=None if original else composer.verify_stage_grant(session,start),
        unknown_prior_tail=original)
    stages=compare.validate_stage_report(report,reference)
    binding=compare.bind_stage_sidecars(report,reference)
    require(report.get('frozen_full_auditor_pass_claimed') is False
            and report.get('physical_state_now_inferred') is False,'Unsupported audit/state claim')
    inventory_count=0
    if original:require(report.get('unknown_actual_tail_operations') is True,'Original discarded tail must remain explicit')
    if not original:
        require(len(stages)==1 and report.get('terminal_session_event_verified') is True
                and report.get('unknown_actual_tail_operations') is False,'Incomplete or ambiguous new stage')
        manifest=read(session/'manifest.json')
        require(manifest['stage_range']==[stages[0],stages[0]+1],'Wrong literal stage range')
        if expected_plan_sha is not None:
            require(start['continuation_plan_sha256']==manifest['continuation_plan_sha256']==expected_plan_sha
                    and sha(session/'continuation_plan.json')==expected_plan_sha,'Stage belongs to another finite plan')
        actual={p.relative_to(session).as_posix() for p in session.rglob('*') if p.is_file()}-{'manifest.json'}
        require(actual==set(manifest['result_sha256']),'Incomplete/extra result inventory')
        for relative,pin in manifest['result_sha256'].items():
            target=(session/relative).resolve()
            require(target.is_relative_to(session.resolve()) and sha(target)==pin,'Result bytes changed')
            inventory_count+=1
    raw=report['raw_inventory']
    require(bool(raw['parts']),'Missing raw evidence')
    require(len({p['path'] for p in raw['parts']})==len(raw['parts']),'Duplicate raw inventory part')
    for part in raw['parts']:
        target=(session/'raw'/part['path']).resolve()
        require(target.is_relative_to((session/'raw').resolve()) and target.stat().st_size==part['bytes']
                and sha(target)==part['sha256'],'Raw bytes changed')
    if not original:
        require(all(p['admitted'] for p in raw['parts'])
                and raw['excluded_crc_valid_tail_records']==1,'New completed stage has unexplained raw tail')
        require(terminal_record(session/'raw'/raw['parts'][-1]['path'])==dict(event='session_complete',
            full_classifier_accuracy_measured=False,impact_attempts=report['native_impacts'],
            programming_pulses=report['programming_pulses'],scope='full',source='hardware',
            stage=stages[0],status='completed'),'Missing/mismatched successful close footer')
    for cert in report['accepted_stages']:
        require(set(cert['payloads'])=={'raw','calibrated'},'Missing measured branch')
        for branch,item in cert['payloads'].items():
            data=local(item['path']).read_bytes()
            require(item['shape']==[12,4,28,28] and item['dtype']=='<f4'
                    and len(data)==compare.STAGE_BYTES
                    and hashlib.sha256(data).hexdigest()==item['sha256']==cert['payload_sha256'][branch]
                    and all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',data)),'Invalid measured stage payload')
    check_coefficients(report['weight_observations'],stages)
    require(sha(path)==report_sha,'Stage report changed during review')
    return dict(path=name,sha256=report_sha,stages=stages,session=report['session'],
        binding=binding,native_impacts=report['native_impacts'],programming_pulses=report['programming_pulses'],
        observed_coefficients=report['observed_coefficients'],
        strict_numeric_replay=report['strict_numeric_replay'],
        condition_only_stages=[r['stage'] for r in report['numeric_diagnostics'] if r['strict_numeric_replay']=='FAIL_condition_only'],
        raw_parts_rehashed=len(raw['parts']),result_inventory_files_rehashed=inventory_count,
        payloads_rehashed=2*len(stages),unknown_prior_tail=original)


def check_coverage(old,new,first,stop):
    require([s for r in old for s in r['stages']]==list(range(first)),'Prior prefix missing/duplicate/reordered')
    require(len(new)==stop-first and [r['stages'] for r in new]==[[s] for s in range(first,stop)],
            'Batch must contain every planned completed stage exactly once in order')
    sessions=[r['session'] for r in old+new]
    require(len(sessions)==len(set(sessions)),'Duplicate acquisition session')


def review(plan_name,stage_reports,output_name,next_stop=None):
    verify_tools()
    plan_path=local(plan_name);plan_sha=sha(plan_path);plan=read(plan_path)
    first,stop=contract.validate_plan(plan,'hardware')
    require(plan['purpose']=='full','Only full-spatial hardware batches')
    require(len(set(stage_reports))==len(stage_reports),'Duplicate stage report path')
    output=local(output_name)
    relative=output.relative_to(ROOT)
    require(relative.parts and relative.parts[0] in ('reports','.tmp'),'Output must be under reports/ or .tmp/')
    require(not output.exists(),'Output exists; preserve prior reviews')
    old=[review_report(ref['path']) for ref in plan['prefix_reports']]
    new=[review_report(name,plan_sha) for name in stage_reports]
    check_coverage(old,new,first,stop)
    refs=[dict(path=r['path'],sha256=r['sha256']) for r in old+new]
    require(refs[:len(old)]==plan['prefix_reports'],'Old prefix report changed')
    next_plan=None;bounds=[]
    if next_stop is not None:
        require(type(next_stop) is int and stop<next_stop<=48,'Next finite stop must advance within48')
        next_plan=planner.build('hardware',[r['path'] for r in refs],'full',next_stop)
        require(contract.validate_plan(next_plan,'hardware')==(stop,next_stop),'Wrong next plan range')
        from experiments.cifar_resumable_20261019 import collector
        from experiments.cifar_full_case0_20261019 import plan as physical
        bounds=[collector.stage_bounds(physical.fixture()['image_chw_uint8'],s) for s in range(stop,next_stop)]
    grant=contract.execution_grant('hardware') if next_plan is not None else None
    if next_plan is not None:require(grant is not None,'Missing reviewed hardware grant')
    require(sha(plan_path)==plan_sha and all(sha(local(r['path']))==r['sha256'] for r in refs),'Input changed during review')
    result=dict(schema='cifar-completed-batch-review-v1',status='prefix_reuse_qualified',
        reviewer_sha256=sha(Path(__file__)),source_sha256=source_pins(),
        input_plan=dict(path=plan_name,sha256=plan_sha),
        accepted_stages=list(range(stop)),newly_accepted_stages=list(range(first,stop)),
        first_unaccepted_stage=stop,remaining_stages=list(range(stop,48)),reports=old+new,
        verified_accepted_stage_impacts=sum(r['native_impacts'] for r in old+new),
        verified_accepted_stage_pulses=sum(r['programming_pulses'] for r in old+new),
        unknown_prior_lost_tail_operations=any(r['unknown_prior_tail'] for r in old+new),
        numerical_raw_replay_repeated=False,model_or_board_access=False,
        full_classifier_executed=False,full_dataset_accuracy_measured=False,autorun_qualified=False,
        publication_status_checked=False,next_finite_plan=None,
        limitations=['Review checks saved audit/public-source/payload/raw bytes, not another numerical replay. Private runtime is recorded, not independently reverified; legacy receipts remain optional evidence.',
          'FAIL_condition_only remains explicit; it is not strict frozen-auditor PASS.',
          'Accepted-stage impacts are a lower bound: earlier discarded/lost tail is not counted as zero.',
          'Complete stage coverage is not classification success or dataset accuracy.',
          'Publication/commit/push and fresh Pi preflight are separate checks.'])
    output.mkdir(parents=True)
    if next_plan is not None:
        write_new(output/'NEXT_PLAN.json',next_plan)
        result['next_finite_plan']=dict(path=(output/'NEXT_PLAN.json').relative_to(ROOT).as_posix(),
            sha256=sha(output/'NEXT_PLAN.json'),stage_range=[stop,next_stop],stage_bounds=bounds,
            maximum_impacts=sum(r['maximum_impacts'] for r in bounds),
            maximum_pulses=sum(r['maximum_pulses'] for r in bounds),
            execution_grant_sha256=grant['sha256'],launch_requires_fresh_pi_preflight=True)
    write_new(output/'PREFIX_INDEX.json',dict(schema='cifar-reviewed-prefix-index-v1',
        accepted_stages=list(range(stop)),prefix_reports=refs,complete_full_tensor=False))
    write_new(output/'BATCH_REVIEW.json',result)
    return result


def write_new(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--stage-report',action='append',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--next-stop-stage-exclusive',type=int,help='Optional plan only; never launches or publishes')
    args=p.parse_args(argv)
    result=review(args.plan,args.stage_report,args.output_dir,args.next_stop_stage_exclusive)
    print(json.dumps(dict(status=result['status'],accepted_stages=result['accepted_stages'],
        output_dir=args.output_dir,next_plan_created=result['next_finite_plan'] is not None)))


if __name__=='__main__':main()
