"""Second review of saved stages0..18: hashes/sidecars/certificates, no raw replay or board."""
import hashlib
import gzip
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import compare_cifar_continuation as compare
from tools import classify_cifar_continuation as composer


def sha(path):
    return compare.sha(path)


def main():
    assert sha(ROOT/'tools/compare_cifar_continuation.py') == '061eeead6bc2b6d4d8ce9a92e960aed92c954503ffcaaa1f138ce24f13fd502e'
    compare.verify_tools()
    original = ROOT/'reports/2026-10-19/cifar-full-case0-review/hardware-prefix-diagnostic/PREFIX_AUDIT.json'
    assert sha(original) == '49cdfe3f5b5d76854431c8a3e58e855a43e4b8fe76ea887ad1316e6e045011b6'
    paths = [original] + [ROOT/f'reports/2026-10-20/cifar-hardware-batch-13-18-review/stage-{stage}/PREFIX_AUDIT.json' for stage in range(13,19)]
    reports = []; accepted = []; parts_checked = payloads_checked = inventory_checked = 0
    for number,path in enumerate(paths):
        report = json.loads(path.read_text(encoding='utf-8'))
        session = ROOT/report['session']; old = number == 0
        start = json.loads((session/'start_manifest.json').read_text(encoding='utf-8'))
        reference = dict(acquisition_session=report['session'],
            accepted_stages=[row['stage'] for row in report['accepted_stages']],
            native_impacts=report['native_impacts'], programming_pulses=report['programming_pulses'],
            execution_grant=None if old else composer.verify_stage_grant(session,start),
            unknown_prior_tail=old)
        stages = compare.validate_stage_report(report,reference)
        assert stages == (list(range(13)) if old else [number+12])
        binding = compare.bind_stage_sidecars(report,reference)
        assert not report['frozen_full_auditor_pass_claimed'] and not report['physical_state_now_inferred']
        if not old:
            assert report['terminal_session_event_verified'] and not report['unknown_actual_tail_operations']
            manifest = json.loads((session/'manifest.json').read_text())
            assert manifest['stage_range']==[stages[0],stages[0]+1]
            actual = {p.relative_to(session).as_posix() for p in session.rglob('*') if p.is_file()}-{'manifest.json'}
            assert actual==set(manifest['result_sha256'])
            for relative,pin in manifest['result_sha256'].items():
                target=(session/relative).resolve()
                assert target.is_relative_to(session.resolve()) and sha(target)==pin
                inventory_checked+=1
        inv=report['raw_inventory']
        for part in inv['parts']:
            target=session/'raw'/part['path']
            assert target.stat().st_size==part['bytes'] and sha(target)==part['sha256']
            parts_checked+=1
        if old:
            assert len(inv['parts'])==56 and len(inv['admitted_parts'])==55
            assert inv['excluded_crc_valid_tail_records']==828468 and not inv['parts'][-1]['admitted']
        else:
            assert all(part['admitted'] for part in inv['parts'])
            assert inv['excluded_crc_valid_tail_records']==1
            with gzip.open(session/'raw'/inv['parts'][-1]['path'],'rt',encoding='utf-8') as stream:
                terminal=None
                for terminal in stream:pass
            assert json.loads(terminal)==dict(event='session_complete',full_classifier_accuracy_measured=False,
                impact_attempts=report['native_impacts'],programming_pulses=report['programming_pulses'],
                scope='full',source='hardware',stage=stages[0],status='completed')
        for cert in report['accepted_stages']:
            for branch,payload in cert['payloads'].items():
                data=(ROOT/payload['path']).read_bytes()
                assert len(data)==150528 and hashlib.sha256(data).hexdigest()==payload['sha256']==cert['payload_sha256'][branch]
                assert all(math.isfinite(value[0]) for value in struct.iter_unpack('<f',data))
                payloads_checked+=1
        rows=report['weight_observations']; expected=set()
        for stage in stages:
            group,tile=divmod(stage,3); width=(32,32,11)[tile]
            expected.update((stage,group*4+pair,tile*32+bl,bl,2*pair,2*pair+1) for pair in range(4) for bl in range(width))
        actual={(row['stage'],row['filter'],row['tap'],row['bl'],row['positive_cell'][0],row['negative_cell'][0]) for row in rows}
        assert actual==expected and len(rows)==len(expected)
        assert all(row['positive_cell'][1]==row['negative_cell'][1]==row['bl'] and row['per_coefficient_fitting_used'] is False for row in rows)
        accepted.extend(stages)
        denominator=sum(row['trained_weight']**2 for row in rows)
        errors={branch:math.sqrt(sum((row[f'post_all_{branch}_weight']-row['trained_weight'])**2 for row in rows)/denominator) for branch in ('raw','calibrated')}
        reports.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),stages=stages,
            session=report['session'],manifest_binding=binding['manifest_identity'],
            strict_numeric_replay=report['strict_numeric_replay'],
            condition_only_stages=[r['stage'] for r in report['numeric_diagnostics'] if r['strict_numeric_replay']!='PASS'],
            native_impacts=report['native_impacts'],programming_pulses=report['programming_pulses'],
            observed_coefficients=len(rows),immediate_within_tolerance=sum(row['immediate_within_tolerance'] for row in rows),
            post_all_weight_relative_l2=errors,heldout=report['heldout'],retention=report['retention'],
            maximum_shared_zero_shift_v=report['maximum_shared_zero_shift_v'],
            maximum_position_bracket_duration_s=report['maximum_position_bracket_duration_s'],
            private_source_verification=binding['source_verification']))
    assert accepted==list(range(19))
    from experiments.cifar_resumable_20261019 import contract,collector
    from experiments.cifar_full_case0_20261019 import plan as physical_plan
    plan_path=ROOT/'reports/2026-10-20/cifar-hardware-batch-13-18-review/HW_BATCH_19_25_PLAN.json'
    assert sha(plan_path)=='c3838b01b7855b57a5b38d26965b0ea9a5e4ef35043334007dea328bde383a15'
    next_plan=json.loads(plan_path.read_text())
    assert contract.validate_plan(next_plan,'hardware')==(19,25)
    assert next_plan['prefix_reports']==[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in paths]
    grant=contract.execution_grant('hardware');assert grant is not None
    image=physical_plan.fixture()['image_chw_uint8']
    bounds=[collector.stage_bounds(image,stage) for stage in range(19,25)]
    assert sum(row['maximum_impacts'] for row in bounds)==1807920
    assert sum(row['maximum_pulses'] for row in bounds)==14400
    output=ROOT/'reports/2026-10-20/cifar-hardware-batch-13-18-review/SECOND_REVIEW.json'
    result=dict(schema='cifar-batch-second-review-v1',status='reuse_qualified',
        reviewer_tool_sha256=sha(Path(__file__)),accepted_stages=accepted,first_unaccepted_stage=19,
        newly_accepted_stages=list(range(13,19)),remaining_stages=list(range(19,48)),
        reports=reports,raw_compressed_files_rehashed=parts_checked,
        result_inventory_files_rehashed=inventory_checked,payloads_rehashed=payloads_checked,
        native_raw_replay_repeated=False,model_or_board_access=False,
        full_classifier_executed=False,full_dataset_accuracy_measured=False,
        prior_interrupted_tail_reused=False,hardware_launch_performed=False,
        next_finite_plan=dict(path=plan_path.relative_to(ROOT).as_posix(),sha256=sha(plan_path),
            stage_range=[19,25],target_stop_stage_exclusive=48,hardware_grant_sha256=grant['sha256'],
            maximum_impacts=1807920,maximum_pulses=14400,stage_bounds=bounds,
            launch_requires_fresh_pi_preflight=True,autorun_qualified=False),
        limitations=['Certificates are grounded in the separately completed frozen raw replay; this second review checks bytes and bindings, not a second numerical replay.',
            'Private source receipt is remote operator evidence; local private source bytes remain unavailable.',
            'Condition-only exclusions remain explicit, not a claim that frozen strict numerical replay passed.',
            'Original interrupted stage13 tail is excluded; new whole-stage13 has its own source/grant/calibration.',
            'The one record after each completed-stage boundary is the separately verified terminal session_complete footer, not unused physical operations.',
            'Reuse qualification does not infer current board state or replace fresh launch preflight.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],accepted_stages=accepted,report=output.relative_to(ROOT).as_posix())))


if __name__=='__main__':main()
