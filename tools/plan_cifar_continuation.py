"""Create a finite acquisition plan from independently replayed stage certificates."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.cifar_resumable_20261019 import contract as c


def build(source,reports,purpose,stop_stage=None):
    refs=[];stages=[]
    for name in reports:
        path=c.local(name);report=json.loads(path.read_text())
        refs.append(dict(path=name,sha256=c.sha(path)))
        stages.extend(row['stage'] for row in report['accepted_stages'])
    target=3 if purpose=='qualification' else 48
    result=dict(schema='cifar-continuation-plan-v1',source=source,purpose=purpose,
        spec_sha256=c.digest(c.SPEC),physical_protocol_sha256=c.PHYSICAL_PROTOCOL_SHA,
        first_stage=len(stages),stop_stage_exclusive=target if stop_stage is None else stop_stage,
        target_stop_stage_exclusive=target,
        accepted_stage_ids=sorted(stages),prefix_reports=refs,
        incomplete_stage_policy='Discard no evidence; reacquire entire first unaccepted stage with fresh programming/R/C',
        retry_policy='No automatic retry; a later invocation requires a newly verified plan',
        prior_actual_impacts='Preserved report lower bound, buffered lost tail unknown; each original attempt upper bound15000000')
    c.validate_plan(result,source)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=('simulator','hardware'),required=True)
    p.add_argument('--purpose',choices=('qualification','full'),required=True)
    p.add_argument('--prefix-report',action='append',default=[])
    p.add_argument('--output',required=True)
    p.add_argument('--stop-stage-exclusive',type=int,help='Finite full-purpose batch stop≤48; final classifier still requires every stage0..47')
    args=p.parse_args(argv);output=c.local(args.output)
    if output.exists():raise ValueError('Preserve previous plan')
    result=build(args.source,args.prefix_report,args.purpose,args.stop_stage_exclusive)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(plan=args.output,plan_sha256=c.sha(output),stage_range=[result['first_stage'],result['stop_stage_exclusive']])))


if __name__=='__main__':main()
