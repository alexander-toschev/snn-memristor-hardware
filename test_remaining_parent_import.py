"""Offline real-parent import exercise; no acquisition and no saved config edits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SESSION='results/cifar_feedback_reserve/55134577-a813-4d1b-a3de-67c81ab285ca'
MANIFEST='8d728c0b8a94c1282f5f4120ffb7abf90b755c7bdcd652fd48eec334343debbc'
AUDIT='results/cifar_feedback_reviews/reserve_simulator_20261016/audit.json'
AUDIT_SHA='6691aa35788b1d06f281b44ec0c5c70d84c42bfb9a1265e30f76d9d5f2e30fe8'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(repo,output):
    repo=repo.resolve();output=output.resolve()
    if output.is_relative_to(repo/SESSION):raise ValueError('Report must not modify the source session')
    sys.path.insert(0,str(repo))
    import cifar_feedback_remaining_reserves as runtime
    report=dict(offline_only=True,native_backend_opened=False,public_configuration_modified=False,
        parent_session=SESSION,parent_manifest_sha256=MANIFEST,parent_audit_sha256=AUDIT_SHA,
        runner_sha256=sha(Path(__file__)),runtime_sha256=sha(repo/'cifar_feedback_remaining_reserves.py'),
        checks=[],errors=[],limitations=['Import consistency exercise, not a new raw numerical audit or acquisition.',
            'Only the actual completed simulator A parent is bound in memory; no hardware parent is invented.',
            'The baseline-plus-failed-A branch is not exercised with real data here.'])
    config=repo/'configs/cifar_feedback_remaining_reserves.json';before=sha(config)
    def check(ok,name):
        report['checks'].append(name)
        if not ok:raise AssertionError(name)
    def reject(call,name):
        try:call()
        except (ValueError,KeyError) as exc:
            report['checks'].append(name);report.setdefault('rejections',{})[name]=str(exc)
        else:raise AssertionError('Accepted invalid input: '+name)
    try:
        binding=dict(source='simulator',choice='completed_A',session=SESSION,
            manifest_sha256=MANIFEST,audit_path=AUDIT,audit_sha256=AUDIT_SHA)
        plan=copy.deepcopy(runtime.SPEC)
        plan.update(draft=False,native_execution_enabled=True,parent_bindings={'simulator':binding})
        _,routing,fixture=runtime.adaptive.load_inputs()
        with patch.object(runtime,'native_connector',side_effect=AssertionError('No native backend permitted')):
            chunks,meta=runtime.verify_parent('simulator',routing,fixture,plan)
        parent=json.loads((repo/SESSION/'manifest.json').read_text(encoding='utf-8'))
        old=json.loads((repo/SESSION/'completed_microtiles.json').read_text(encoding='utf-8'))
        actual={(c['filter_id'],s['tap_index']) for c in chunks for s in c['result']['effective_route']}
        excluded={(2,t) for t in range(11)}|{(1,t) for t in range(9)}
        expected={(f,t) for f in range(4) for t in range(75)}-excluded
        check(actual==expected and sum(len(c['result']['effective_route']) for c in chunks)==280,'exact_retained280_unique_taps')
        check(len(excluded)==20 and not actual&excluded,'exact_B11_C9_excluded')
        for chunk in chunks:
            origin=chunk['origin'];previous=old[origin['wrapper_index']]
            check(chunk['result']==previous['result'],'unchanged_result_'+str((chunk['filter_id'],chunk['parent_tile_id'],chunk['microtile_id'])))
            check(origin['ancestry_original_origin']==previous['origin'] and origin['session']==SESSION
                and origin['sha256']==parent['result_sha256']['completed_microtiles.json']
                and origin['reused'] is True and origin['new_measurement'] is False,'immutable_wrapper_origin_'+str(origin['wrapper_index']))
        first42=[c for c in chunks if c['filter_id']==0 and c['parent_tile_id'] in (0,1,2)]
        check(sum(len(c['result']['effective_route']) for c in first42)==42,'ancestral42_retained')
        lineage=[]
        for chunk in first42:
            origin=chunk['origin'];sessions=[]
            while origin:
                sessions.append(origin['session']);origin=origin.get('ancestry_original_origin')
            lineage.append(sessions)
            check(sessions[-1]=='results/cifar_feedback_diagnostic/e1e1d0a7-00e2-403b-ac4e-defbf1fe80b8','original42_diagnostic_origin')
        check(meta['parent_cumulative_impacts']==572402 and meta['parent_cumulative_pulses']==36893,'cumulative572402_36893_once')
        check(len(meta['historical_attempt_ledger'])==1,'no_double_count_baseline_or_A')
        check(meta['verified_ancestry']==parent['parent_import'],'recursive_ancestry_proof_equal_recorded_parent')
        bad=copy.deepcopy(plan);bad['parent_bindings']['simulator']['manifest_sha256']=MANIFEST[:-1]+('0' if MANIFEST[-1]!='0' else '1')
        reject(lambda:runtime.verify_parent('simulator',routing,fixture,bad),'bad_manifest_SHA_rejected')
        bad=copy.deepcopy(plan);bad['parent_bindings']['simulator']['audit_sha256']=AUDIT_SHA[:-1]+('0' if AUDIT_SHA[-1]!='0' else '1')
        reject(lambda:runtime.verify_parent('simulator',routing,fixture,bad),'bad_audit_SHA_rejected')
        bad=copy.deepcopy(plan);bad['parent_bindings']={'hardware':{**binding,'source':'hardware'}}
        reject(lambda:runtime.verify_parent('hardware',routing,fixture,bad),'simulator_parent_as_hardware_rejected')
        # Simulate corrupted bytes at the hashing boundary; immutable originals are untouched.
        victim=next(name for name in parent['result_sha256'] if name.endswith('raw.jsonl'))
        victim_path=(repo/SESSION/victim).resolve();original_hash=runtime.sha256
        with patch.object(runtime,'sha256',side_effect=lambda path:'0'*64 if Path(path).resolve()==victim_path else original_hash(path)):
            reject(lambda:runtime.verify_parent('simulator',routing,fixture,plan),'changed_parent_raw_bytes_rejected')
        report.update(retained_coefficients=280,excluded_coefficients=20,ancestral_coefficients=42,
            retained_A_reserve_coefficients=sum(len(c['result']['effective_route']) for c in chunks if (c['filter_id'],c['parent_tile_id'])==(3,4)),
            retained_wrapper_count=len(chunks),cumulative_impacts=572402,cumulative_writes=36893,first42_lineage=lineage)
    except Exception as exc:report['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        report['public_config_sha256_before']=before;report['public_config_sha256_after']=sha(config)
        if before!=sha(config):report['errors'].append('Public config changed')
        report['status']='passed' if not report['errors'] else 'failed'
        output.mkdir(parents=True,exist_ok=False)
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (output/'REPORT.md').write_text('# Offline A to B/C import verification\n\n'+
            f"{report['status']}: {len(report['checks'])} checks, {len(report['errors'])} errors.\n\n"+
            'Actual simulator A parent, temporary in-memory binding only. Public configuration unchanged by this offline check.\n'+
            'Retained280, excluded20 and ancestral42 checks, exact origins, cumulative counters and rejection cases are in report.json.\n'+
            'No acquisition, no new raw numerical audit, no hardware parent binding.\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','errors','runtime_sha256')}))
    return report


class RealParentImportTest(unittest.TestCase):
    def test_actual_simulator_A_parent_import(self):
        repo=Path(__file__).resolve().parent
        required=[repo/SESSION/'manifest.json',repo/AUDIT,
            repo/'results/cifar_feedback_adaptive/64036fb0-2d33-4350-b672-96ef8380acec/manifest.json',
            repo/'results/cifar_feedback_diagnostic/e1e1d0a7-00e2-403b-ac4e-defbf1fe80b8/manifest.json']
        missing=[str(p.relative_to(repo)) for p in required if not p.exists()]
        if missing:self.skipTest('Actual simulator ancestry unavailable; no verification performed: '+', '.join(missing))
        scratch=repo/'.tmp';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            report=verify(repo,Path(tmp)/'proof')
        self.assertEqual(report['status'],'passed',report['errors'])
        self.assertEqual((report['retained_coefficients'],report['excluded_coefficients'],report['ancestral_coefficients']),(280,20,42))
        self.assertEqual(len(report['checks']),65)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,help='New repo-relative report directory; existing output is never overwritten')
    args=parser.parse_args();repo=Path(__file__).resolve().parent
    path=Path(args.output)
    if path.is_absolute() or not (repo/path).resolve().is_relative_to(repo):
        parser.error('Output must remain repo-relative')
    report=verify(repo,repo/path)
    return int(bool(report['errors']))


if __name__=='__main__':
    if '--output' in sys.argv:raise SystemExit(main())
    unittest.main()
