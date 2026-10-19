"""Independent report/source review; does not replay raw stream or execute a model."""
import hashlib
import json
import math
from pathlib import Path
from collections import Counter
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/2026-10-19/cifar-full-case0-review'
SESSION=ROOT/'results/cifar_full_case0_20261019/5ee64f20-d499-40df-a742-95a76d2dc386'
EXP=ROOT/'experiments/cifar_full_case0_20261019'


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review():
    r=json.loads((REPORT/'NATIVE_SIM_FULL_CLASSIFIER.json').read_text())
    a=r['native_audit'];m=json.loads((SESSION/'manifest.json').read_text());start=json.loads((SESSION/'start_manifest.json').read_text())
    freeze=json.loads((REPORT/'SCIENTIFIC_FREEZE.json').read_text())
    require(sha(SESSION/'manifest.json')==r['acquisition_manifest_sha256'],'Report manifest binding differs')
    require(m['status']=='completed' and m['scope']=='full' and m['source']=='simulator','Unexpected acquisition scope/outcome')
    for key in ('session_id','source','scope','protocol_sha256','source_sha256','invocation','execution_environment'):
        require(m[key]==start[key],'Start/final provenance differs: '+key)
    require(not m['invocation']['git']['dirty_tracked'],'Runtime tracked checkout was dirty')
    for name,pin in freeze['source_sha256'].items():require(sha(EXP/name)==pin==m['source_sha256']['experiments/cifar_full_case0_20261019/'+name],'Frozen source differs: '+name)
    require(sha(SESSION/'protocol.json')==freeze['protocol_file_sha256'] and m['protocol_sha256']==freeze['canonical_protocol_sha256'],'Protocol differs')
    for name,pin in m['source_sha256'].items():require(sha(ROOT/name)==pin,'Source bytes differ: '+name)
    for name,pin in r['artifact_sha256'].items():require(sha(ROOT/name)==pin,'Classifier artifact differs: '+name)
    require(r['classifier_sha256']==sha(EXP/'classifier.py') and r['auditor_sha256']==sha(EXP/'audit_native.py') and r['digital_downstream_sha256']==sha(ROOT/'full_cifar_classifier.py'),'Classifier/auditor bytes differ')
    actual={p.relative_to(SESSION).as_posix() for p in SESSION.rglob('*') if p.is_file()}-{'manifest.json'}
    require(actual==set(m['result_sha256']),'Evidence file inventory differs')
    for name in m['result_sha256']:
        if not name.endswith('.jsonl.gz'):require(sha(SESSION/name)==m['result_sha256'][name],'Sidecar evidence differs: '+name)
    index=json.loads((SESSION/'raw/index.json').read_text())
    require(r['registered_classifier_import_verified'] is True and a['status']=='native_trace_replayed' and a['complete_full_tensor'] is True,'Registered native replay incomplete')
    require(a['native_impacts']==m['impacts']<=15000000 and a['programming_pulses']==m['pulses']<=115200,'Impact/pulse bounds differ')
    require(sum(a['phase_counts'].values())==m['impacts'],'Phase call sum differs')
    require(a['observed_coefficients']==4800 and a['covered_tensor_elements']==12*64*28*28*3,'Full coverage differs')
    observations=a['weight_observations']
    require(len(observations)==4800 and {(x['filter'],x['tap']) for x in observations}=={(f,t) for f in range(64) for t in range(75)},'Coefficient identity coverage differs')
    cells={tuple(x[k]) for x in observations for k in ('positive_cell','negative_cell')}
    require(cells=={(w,b) for w in range(8) for b in range(32)},'Physical cell union differs')
    require(not any(x['per_coefficient_fitting_used'] for x in observations),'Unexpected per-weight fitting')
    require([x['stage'] for x in a['heldout']]==list(range(48)) and [x['stage'] for x in a['retention']]==list(range(48)),'Stage validation coverage differs')
    require(r['cpu_validation']['direct_forward_max_abs']<=1e-6 and max(r['cpu_validation']['tiled_forward_max_abs'].values())<=1e-5 and r['cpu_validation']['tiled_forward_max_abs']['spikes']==0,'CPU bridge disagreement')
    preds={}
    for branch,b in r['predictions'].items():
        spikes=b['output_spikes'];mem=b['output_membranes_post_reset'];scores=b['twd_scores']
        require(len(spikes)==len(mem)==12 and all(len(row)==10 for row in spikes+mem) and len(scores)==10,'Output trace shape differs')
        require(all(v in (0.,1.) for row in spikes for v in row) and all(math.isfinite(v) for row in mem for v in row),'Invalid output trace')
        first=[next((t for t in range(12) if spikes[t][j]>.5),12) for j in range(10)]
        membrane=[sum(row[j] for row in mem) for j in range(10)]
        prediction=dict(first_spike=min(range(10),key=first.__getitem__),membrane=max(range(10),key=membrane.__getitem__),twd=max(range(10),key=scores.__getitem__))
        silent=not any(any(row) for row in spikes);ties=sum(t==min(first) for t in first)
        require(prediction==b['predicted_subset_indices'] and silent==b['all_output_classes_silent'] and ties==b['first_spike_winner_count'],'Decoder/ambiguity flags differ')
        require({k:r['output_class_fine_labels'][v] for k,v in prediction.items()}==b['predicted_cifar100_fine_labels'],'Class remapping differs')
        preds[branch]=dict(predicted_subset_indices=prediction,predicted_fine_labels=b['predicted_cifar100_fine_labels'],total_output_spikes=sum(sum(row) for row in spikes),all_silent=silent,first_spike_ties=ties,twd_score_ties=sum(s==max(scores) for s in scores),spike_based_recognition_informative=not silent and ties==1)
    require(r['case']['original_test_index']==9 and r['case']['fine_label']==r['case']['remapped_label']==0,'Case differs')
    norm=sum(x['trained_weight']**2 for x in observations)
    weight_error={k:math.sqrt(sum((x[k]-x['trained_weight'])**2 for x in observations)/norm) for k in ('immediate_weight','post_all_raw_weight','post_all_calibrated_weight','final_raw_weight','final_calibrated_weight')}
    heldout=a['heldout'];improved=sum(x['calibrated_rmse_v']<x['raw_rmse_v'] for x in heldout);worse=sum(x['calibrated_rmse_v']>x['raw_rmse_v'] for x in heldout)
    return dict(status='pass_report_and_source_review',reviewer_tool_sha256=sha(__file__),classifier_report_sha256=sha(REPORT/'NATIVE_SIM_FULL_CLASSIFIER.json'),manifest_sha256=sha(SESSION/'manifest.json'),
        session=SESSION.relative_to(ROOT).as_posix(),runtime=m['invocation'],source_count=len(m['source_sha256']),result_inventory_count=len(actual),raw_index_sha256=sha(SESSION/'raw/index.json'),raw_records=index['records'],raw_parts=len(index['parts']),
        audit_scope='Independently verified report/source/sidecars/coverage/decoder arithmetic. Full compressed/expanded stream replay was performed by the registered classifier run; reviewer did not duplicate the large stream or neural forward.',
        acquisition_seconds=(datetime.fromisoformat(m['finished_utc'])-datetime.fromisoformat(m['started_utc'])).total_seconds(),native_impacts=m['impacts'],programming_pulses=m['pulses'],coefficients=4800,tensor_elements=1806336,physical_cells_union=len(cells),
        cpu_validation=r['cpu_validation'],conv1_errors=r['conv1_errors'],predictions=preds,case=r['case'],dataset_accuracy=None,
        weight_relative_l2=weight_error,immediate_within_tolerance=sum(x['immediate_within_tolerance'] for x in observations),programming_stop_counts=dict(Counter(x['programming_stop'] for x in observations)),
        heldout=dict(stages=48,improved=improved,worsened=worse,unchanged=48-improved-worse,mean_stage_raw_rmse_v=sum(x['raw_rmse_v'] for x in heldout)/48,mean_stage_calibrated_rmse_v=sum(x['calibrated_rmse_v'] for x in heldout)/48),
        retention_max_abs_v=max(x['max_abs_v'] for x in a['retention']),maximum_shared_zero_shift_v=a['maximum_shared_zero_shift_v'],maximum_position_bracket_duration_s=a['maximum_position_bracket_duration_s'],
        hardware_full_go=True,hardware_conditions=['Same frozen7py/protocol and same preregistered case0','Fresh Raspberry strict native sources/INI/free space preflight, public Connector only','Exclusive hardware lock; no simultaneous STDP/calibration','Approximate weights retained as designed; invalid native R/ADC/transport/current guards remain binding','Hardware classifier/audit on Raspberry with native private files and qualified numeric environment','Report CPU/raw/calibrated branches and all-silent/ties without claiming successful recognition','Preserve source-specific gainHW.125 versusSIM.015; comparison not identical physical targets'],
        conclusion='Full measurement-to-digital-SNN path is qualified for a bounded approximate hardware case. Calibrated first-spike/TWD class0 is an all-silent tie, not successful recognition; CPU/raw predict wrong class1. No dataset accuracy measured.')


if __name__=='__main__':
    output=REPORT/'NATIVE_SIM_FULL_INDEPENDENT_REVIEW.json'
    require(not output.exists(),'Preserve prior report')
    result=review();output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
