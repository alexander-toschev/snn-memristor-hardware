"""Local strict replay/composition of stage sessions; no physical I/O or missing-data fill."""
import argparse
import hashlib
import gzip
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for folder in (ROOT,ROOT/'.deps-classifier',ROOT/'.deps'):
    if folder.is_dir():sys.path.insert(0,str(folder))
from experiments.cifar_resumable_20261019 import contract as c
from tools import runtime_provenance as provenance_policy

APPROVED_AUDIT_SOURCES={
    'tools/audit_cifar_interrupted_prefix.py':'1c19c6e4b0db4af3892a681035b27123633d2ca7c840f1537928b13e35ec3b00',
    'tools/cifar_prefix_replay.py':'0f6bb5743dd8b4218b68701c65749db00874d8e444d6df18feda08f127a39c69',
}
APPROVED_CLASSIFIER_SHA='64ba6e85cda19367f93a0c8294f77b6c9b6f5f26cbf40d188685bc47430053e5'


def combine(stages):
    """Exactly one measured contribution for each T/filter/y/x/tile; no fill."""
    import numpy as np
    if set(stages)!=set(range(48)):raise ValueError('Full classifier requires all48 distinct full stages')
    output={}
    for branch in ('raw','calibrated'):
        arrays=[]
        for group in range(16):
            tiles=[]
            for tile in range(3):
                payload=stages[group*3+tile][branch]
                if not isinstance(payload,bytes) or len(payload)!=12*4*28*28*4:raise ValueError('Incomplete full stage payload')
                array=np.frombuffer(payload,dtype='<f4').reshape(12,4,28,28)
                if not np.isfinite(array).all():raise ValueError('Nonfinite physical contribution')
                tiles.append(array)
            arrays.append(np.stack(tiles,axis=-1))
        output[branch]=np.concatenate(arrays,axis=1).astype('<f4',copy=False).tobytes()
    return output


def verify_stage_grant(session,start,require_final=True):
    name=start.get('execution_grant_path')
    grant=c.execution_grant(start['source'],historical_path=name if name!=c.GRANTS[start['source']] else None)
    if grant is None or start.get('execution_grant_path')!=grant['path'] or start.get('execution_grant_sha256')!=grant['sha256']:
        raise ValueError('Stage execution grant not explicitly reviewed')
    if c.sha(session/'execution_grant.json')!=grant['sha256']:raise ValueError('Stage grant snapshot differs')
    if require_final:
        final=json.loads((session/'manifest.json').read_text())
        for name in ('execution_grant_path','execution_grant_sha256'):
            if final.get(name)!=start.get(name):raise ValueError('Start/final execution grant differs')
    return dict(path=grant['path'],sha256=grant['sha256'],source=start['source'])


def replay_sessions(session_names,output,qualification=False,portable_condition=False,private_receipts=None):
    from tools import audit_cifar_interrupted_prefix as audit
    reviewed=provenance_policy.versions(ROOT)['current']
    for name in ('tools/audit_cifar_interrupted_prefix.py','tools/cifar_prefix_replay.py','tools/runtime_provenance.py'):
        if c.sha(ROOT/name)!=reviewed[name]:raise ValueError('Independent audit source changed')
    accepted={};reports=[];source=None
    if len(set(session_names))!=len(session_names):raise ValueError('Duplicate acquisition session')
    for index,name in enumerate(session_names):
        session=c.local(name);start=json.loads((session/'start_manifest.json').read_text())
        is_stage=start.get('acquisition_kind')=='stage_segment'
        grant=verify_stage_grant(session,start) if is_stage else None
        folder=output/('audit-%02d'%index)
        report=audit.audit_session(session,folder,portable_condition=portable_condition,
            private_receipt=(private_receipts or {}).get(name),stage_session=is_stage)
        if report['scope']!='full' or report['protocol_sha256']!=c.PHYSICAL_PROTOCOL_SHA:raise ValueError('Only same-protocol full spatial stages admitted')
        if source is not None and source!=report['source']:raise ValueError('Cannot mix hardware and nativeSIM contributions')
        source=report['source']
        for cert in report['accepted_stages']:
            stage=cert['stage']
            if stage in accepted:raise ValueError('Duplicate physical stage; no choose-best replacement')
            if cert['shape']!=[12,4,28,28] or cert['dtype']!='<f4':raise ValueError('Wrong stage shape')
            branch={}
            for label in ('raw','calibrated'):
                entry=cert['payloads'][label];path=c.local(entry['path'])
                if c.sha(path)!=entry['sha256']:raise ValueError('Audited stage payload changed')
                branch[label]=path.read_bytes()
            accepted[stage]=branch
        reports.append(dict(path=(folder/'PREFIX_AUDIT.json').relative_to(ROOT).as_posix(),sha256=c.sha(folder/'PREFIX_AUDIT.json'),
            acquisition_session=name,accepted_stages=[v['stage'] for v in report['accepted_stages']],
            native_impacts=report['native_impacts'],programming_pulses=report['programming_pulses'],
            execution_grant=grant,
            unknown_prior_tail=report.get('unknown_actual_tail_operations',False)))
    expected=set(range(3 if qualification else 48))
    if set(accepted)!=expected:raise ValueError('Missing physical stages; no CPU/zero fill or partial classifier')
    if qualification and source!='simulator':raise ValueError('Qualification only accepts nativeSIM')
    return accepted,dict(source=source,stage_reports=reports,accepted_stage_ids=sorted(accepted),
        complete_full_tensor=not qualification,unknown_prior_lost_tail_operations=any(r['unknown_prior_tail'] for r in reports),
        validated_stage_impacts=sum(r['native_impacts'] for r in reports),
        validated_stage_pulses=sum(r['programming_pulses'] for r in reports),
        interpretation='Every stage replayed from actual raw. Prior incomplete tail preserved and excluded, not assumed zero cost.')


def classify(payloads):
    import torch
    torch.set_num_threads(2)
    path=ROOT/'experiments/cifar_full_case0_20261019/classifier.py'
    if c.sha(path)!=APPROVED_CLASSIFIER_SHA:raise ValueError('Frozen digital bridge changed')
    from experiments.cifar_full_case0_20261019 import classifier as bridge,collector
    ref=bridge.digital();state,model,case,image=ref.assets(ROOT,0)
    flat,weights=collector.load_assets()
    if flat!=image.flatten().tolist() or not torch.equal(torch.from_numpy(weights.copy()),state['conv_layers.0.weight'].reshape(64,75)):
        raise ValueError('Image/programming checkpoint differs from classifier')
    with torch.no_grad():
        cpu_conv,cpu_result,validation=bridge.cpu_control(state,model,image)
        predictions={'cpu_reference':ref.describe(cpu_result)};errors={}
        for label,payload in payloads.items():
            conv=bridge.decode_payload(payload,hashlib.sha256(payload).hexdigest())
            predictions[label]=ref.describe(ref.downstream(conv,state))
            residual=conv-cpu_conv
            errors[label]=dict(conv1_rmse=float(residual.square().mean().sqrt()),conv1_max_abs=float(residual.abs().max()),relative_l2=float(residual.norm()/cpu_conv.norm()))
    return dict(predictions=predictions,conv1_errors=errors,cpu_validation=validation,
        case={k:case[k] for k in ('original_test_index','fine_label','remapped_label','image_chw_uint8_sha256')},
        artifact_sha256=ref.PINS,output_class_fine_labels=ref.LABELS,
        frozen_digital_bridge_sha256=APPROVED_CLASSIFIER_SHA,digital_downstream_sha256=bridge.DIGITAL_SHA)


def interruption_evidence(session):
    """Preserve failed SIM evidence; it contributes no tensor and gets no full audit PASS."""
    from tools import audit_cifar_interrupted_prefix as audit
    from experiments.cifar_full_case0_20261019.trace import records
    start,provenance=audit.verify_sources(session,stage_session=True)
    grant=verify_stage_grant(session,start,require_final=False)
    if (session/'manifest.json').exists():raise ValueError('Abrupt-process qualification must not have a fabricated final manifest')
    marker=json.loads((session/'fault_injection.json').read_text())
    if start.get('source')!='simulator' or start.get('scope')!='full' or start.get('stage_range')!=[1,2] or start.get('controlled_interruption_after_positions')!=17 or start.get('controlled_process_loss') is not True:
        raise ValueError('Qualification requires the controlled failed nativeSIM stage1')
    if marker.get('schema')!='cifar-sim-process-loss-v1' or marker.get('exit_code')!=86 or marker.get('source')!='simulator' or marker.get('stage')!=1 or marker.get('software_observed_positions')!=17:
        raise ValueError('Invalid explicit fault-injection marker')
    if c.sha(session/'continuation_plan.json')!=start['continuation_plan_sha256'] or start['launcher_sha256']!=start['source_sha256'].get('experiments/cifar_resumable_20261019/experiment.py'):
        raise ValueError('Interrupted plan/launcher binding differs')
    inventory={p.relative_to(session).as_posix():c.sha(p) for p in session.rglob('*') if p.is_file()}
    index=json.loads((session/'raw/index.json').read_text())
    if index.get('stream_finalized') is not False or marker['raw_index_sha256']!=c.sha(session/'raw/index.json'):
        raise ValueError('Interrupted index must be the unfinalized durable checkpoint')
    positions=[];impacts=0;pending=None;last=None
    for row in records(session/'raw'):
        last=row
        if row['event']=='position_complete':positions.append([row['y'],row['x']])
        if row['event'] in ('stage_complete','session_complete','abort'):raise ValueError('Unexpected terminal/abort in abrupt durable prefix')
        if row['event']=='request':
            impacts+=1
            if pending is not None or row['task_id']!=impacts:raise ValueError('Durable request sequence mismatch')
            pending=row
        if row['event']=='result':
            if pending is None or any(row.get(k)!=v for k,v in pending.items() if k!='event'):raise ValueError('Durable result/request mismatch')
            native=row.get('native_result');task=pending['command_si']
            if not isinstance(native,list) or len(native)!=3:raise ValueError('Malformed durable native tuple')
            r,echo,adc=native
            if type(echo) is not int or echo!=task['id'] or type(adc) is not int or not 0<=adc<=16382:raise ValueError('Invalid durable native ID/ADC')
            if task['mode_flag']==7 and (adc<20 or type(r) not in (int,float) or not math.isfinite(r) or r<=1e-8):raise ValueError('Invalid durable nativeR')
            pending=None
    if positions!=[[0,x] for x in range(16)] or pending is not None or last.get('event')!='position_complete':
        raise ValueError('Durable index does not end exactly at closed position16')
    indexed={part['path'] for part in index['parts']}
    unindexed=sorted(p.name for p in (session/'raw').glob('part-*.jsonl.gz') if p.name not in indexed)
    if unindexed!=['part-%06d.jsonl.gz'%len(indexed)]:raise ValueError('Missing or unexpected unindexed position17 tail')
    tail=session/'raw'/unindexed[0]
    tail_incomplete=tail.stat().st_size<20
    if not tail_incomplete:
        try:
            with gzip.open(tail,'rb') as stream:
                for _ in stream:pass
        except (EOFError,OSError):tail_incomplete=True
        else:raise ValueError('Fault tail unexpectedly has a complete gzip footer')
    if inventory!={p.relative_to(session).as_posix():c.sha(p) for p in session.rglob('*') if p.is_file()}:raise ValueError('Interrupted evidence changed during audit')
    return dict(session=session.relative_to(ROOT).as_posix(),start_manifest_sha256=c.sha(session/'start_manifest.json'),
        source_provenance=provenance,execution_grant=grant,inventory_sha256=inventory,durable_completed_positions=16,
        durable_impact_attempts=impacts,software_marker_observed_positions=17,unindexed_tail=unindexed,
        incomplete_tail_verified=tail_incomplete,actual_unrecorded_operations_unknown=True,
        no_close_or_final_manifest_claimed=True,durable_position16_verified=True,
        failed_stage_payload_used=False,failed_stage_full_semantic_audit_claimed=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',action='append',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--qualification-only',action='store_true')
    p.add_argument('--interrupted-session',help='Required only for qualification: actual controlled failed stage1 session')
    p.add_argument('--portable-condition',action='store_true')
    p.add_argument('--private-receipts',help='Repository-relative JSON object mapping each exact acquisition session to its own reviewed receipt path')
    args=p.parse_args(argv);output=c.local(args.output_dir)
    if bool(args.interrupted_session)!=args.qualification_only:raise ValueError('Qualification requires its explicit failed-stage evidence; full classification does not consume it')
    if output.exists():raise ValueError('Preserve previous analysis')
    output.mkdir(parents=True)
    receipts={}
    if args.private_receipts:
        mapping=json.loads(c.local(args.private_receipts).read_text())
        if not isinstance(mapping,dict) or not set(mapping).issubset(args.session):raise ValueError('Receipt map contains unrelated sessions')
        receipts={session:c.local(path) for session,path in mapping.items()}
    stages,report=replay_sessions(args.session,output,args.qualification_only,args.portable_condition,receipts)
    report.update(schema='cifar-continuation-classification-v1',analyzer_sha256=c.sha(__file__),
        physical_protocol_sha256=c.PHYSICAL_PROTOCOL_SHA,spec_sha256=c.digest(c.SPEC),
        hardware_acquisition_performed=False,full_dataset_accuracy_measured=False,
        source_gain_v_per_weight={'simulator':.015,'hardware':.125}[report['source']],
        gain_confounds_source_comparison=True,downstream='Checkpoint BN/LIF and remaining network executed digitally once per logical timestep after sum of3tiles',
        status_semantics='Complete measured Conv1 one-case inference, not dataset accuracy; silent/tied argmax is not valid recognition')
    if not args.qualification_only:
        payloads=combine(stages)
        for label,payload in payloads.items():(output/(label+'.f32')).write_bytes(payload)
        report['payload_sha256']={k:hashlib.sha256(v).hexdigest() for k,v in payloads.items()}
        report.update(classify(payloads))
    else:report.update(status='qualification_pass',classifier_executed=False,qualification='3fulltiles/4filters; no fullnetwork prediction',
                       controlled_interruption=interruption_evidence(c.local(args.interrupted_session)))
    with (output/'COMPOSITE_REPORT.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output_dir=args.output_dir,full_tensor=report['complete_full_tensor'],source=report['source'])))


if __name__=='__main__':main()
