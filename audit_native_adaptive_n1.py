"""Offline component evidence replay; no full-run qualification or device import."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--session',required=True);p.add_argument('--output',required=True);args=p.parse_args()
repo=Path(args.repo).resolve()
expected='09d8ac17e57e8a2df12c26d12fe187c52ceeb25f007ed93baeebe67262b4c92f'
if hashlib.sha256((repo/'audit_cifar_feedback_adaptive.py').read_bytes()).hexdigest()!=expected:raise ValueError('Independent helper changed')
sys.path.insert(0,str(repo))
import audit_cifar_feedback_adaptive as a
ck=a.CompactChecks();directory=a.local(repo,args.session);output=a.local(repo,args.output)
a.require(not output.is_relative_to(directory),'Output must not modify data')
m=a.load(directory/'manifest.json');plan=a.load(directory/'protocol.json')
ck.equal(m,dict(source='simulator',component_only=True,hardware_io=False,production_cifar_execution=False,full300_coverage_claim=False,original_candidate_count=34,pending_taps=[13]),'component/explicit_scope')
for name,digest in m['result_sha256'].items():ck.equal(a.sha(a.local(directory,name)),digest,'component/result/'+name)
for name,digest in m['source_sha256'].items():ck.equal(a.sha(a.local(repo,name)),digest,'component/local_recorded_source/'+name)
component_pins={k:v for k,v in a.ADAPTIVE_PINS.items() if k!='experiment.py'}
component_pins['run_native_adaptive_n1.py']='9c944a68b039e007c6bfeaa9c9e7872c67e349eeae844f44d022e7c06738781c'
for name,digest in component_pins.items():ck.equal(m['source_sha256'].get(name),digest,'component/frozen_runtime/'+name)
fixture,routing,golden=a.verify_inputs(repo,ck)
filt=routing['filters'][0];parent=filt['tiles'][0];pending=[s for s in parent['slots'] if not s['padding']][-1:]
r=a.audit_microtile(directory/'microtile',filt,parent,pending,1,fixture,plan,'simulator',ck,m['session_id'])
ck.equal(r['matrix']['reference_design_rank'],1,'component/reference_rank_one')
ck.require(not any(c['eligible'] for c in r['matrix']['candidates'] if c['name'].startswith('full_')),'component/full_matrix_ineligible')
ck.equal(len(r['partials']),600,'component/partial_points')
ck.equal(m['impact_attempts'],r['counters']['impact_attempts'],'component/impacts')
ck.equal(m['programming_pulse_attempts'],r['counters']['programming_pulse_attempts'],'component/pulses')
report=dict(component_consistency='passed' if not ck.errors else 'failed',checks=ck.count,errors=ck.errors,source='simulator',hardware_io=False,
    session=args.session,manifest_sha256=a.sha(directory/'manifest.json'),auditor_helper_sha256=expected,replay_script_sha256=a.sha(Path(__file__)),
    counts=r['counters'],partial_points=len(r['partials']),active_taps=[s['tap_index'] for s in r['effective_route']],reset_candidates=len(r['reset_candidates']),
    reference_design_rank=r['matrix']['reference_design_rank'],selected_model=r['matrix']['selected_model'],
    physical_weight_quality_pass=r['physical_weight_quality_pass'],full300_coverage_claim=False,
    provenance_limitations='Standalone component launched with staged/untracked non-runtime work. Recorded source bytes were checked locally against the manifest; this report does not assert a fully clean committed acquisition tree or successful adaptive 13+1 history. The pending singleton was explicitly supplied; native measurements are unchanged.')
output.mkdir(parents=True,exist_ok=False);(output/'audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
(output/'REPORT.md').write_text('# Native SIM singleton component\n\n'+json.dumps({k:report[k] for k in ('component_consistency','checks','counts','partial_points','reference_design_rank','selected_model')},indent=2)+'\n\n'+report['provenance_limitations']+'\n',encoding='utf-8')
print(json.dumps(report));raise SystemExit(bool(ck.errors))
