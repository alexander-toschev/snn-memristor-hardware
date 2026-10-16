"""Offline comparison of two independently audited full adaptive diagnostic CIFAR sessions.

Reads recorded summaries only; never imports acquisition, analyzer or driver code.
Audit PASS is evidence of consistency, not a requirement that weight quality passed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st

PROTOCOL='cifar-feedback-adaptive-two-microtiles-v1'
AUDITOR_SHA='09d8ac17e57e8a2df12c26d12fe187c52ceeb25f007ed93baeebe67262b4c92f'
FIXTURE='weights/prepared_cifar100_conv1.json'
FIXTURE_SHA='86113ff1d2fdbe155056d2e1377cf380e7395120fd93ccf2e1efc7a0a933167c'
FIELDS=('conv_sum','bntt_current','lif_before_reset','lif_after_reset')
EXPECTED_KEYS={(f,c,r,t) for f in range(4) for c in range(10) for r in range(5) for t in range(12)}


def require(condition,message):
    if not condition:raise ValueError(message)


def local(repo,name):
    require(not Path(name).is_absolute(),'Paths must be repository-relative')
    path=(repo/name).resolve()
    require(path.is_relative_to(repo),'Path escapes repository')
    return path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def indexed_summary(summary,case_indices):
    """Strict Cartesian coverage: no duplicate, omitted or zero-filled trajectory."""
    require(summary.get('status')=='completed' and summary.get('technical_completed') is True,'Incomplete summary')
    for key in ('acquisition_complete','decoded_conv_bn_lif_complete'):
        require(summary.get(key) is True,'Incomplete '+key)
    require(type(summary.get('microtiles_completed')) is int and 28<=summary['microtiles_completed']<=56,'Invalid completed microtile count')
    require(summary.get('fitted_alpha_used_for_decoding') is False,'Fitted alpha used for decoding')
    require(summary.get('actual_branch_uses_software_dot_product') is False,'CPU replacement is not a measured comparison')
    require(summary.get('diagnostic_acquisition_not_accuracy_qualification') is True,'Missing diagnostic interpretation')
    require(summary.get('full_network_accuracy_claim') is False,'Unexpected full-network accuracy claim')
    actual={};effective={};seen=set()
    require(len(summary['trajectories'])==200,'Expected 200 trajectories')
    for tr in summary['trajectories']:
        f,c,r=(tr[k] for k in ('filter_id','case','repeat'))
        require(all(type(v) is int for v in (f,c,r)),'Noninteger trajectory key')
        require(0<=f<4 and 0<=c<10 and 0<=r<5,'Out-of-range trajectory key')
        require((f,c,r) not in seen,'Duplicate trajectory');seen.add((f,c,r))
        require(tr['original_test_index']==case_indices[c],'Different CIFAR test sample')
        require(tr['status']=='measured','Unmeasured trajectory')
        for field,target in (('actual',actual),('effective_weight_reference',effective)):
            require(isinstance(tr[field],list) and len(tr[field])==12,'Missing timestep in '+field)
            for t,row in enumerate(tr[field]):
                require(type(row['t']) is int and row['t']==t,'Reordered/invalid timestep')
                require(all(finite(row[k]) for k in FIELDS),'Nonfinite or nonnumeric neural output')
                require(type(row['lif_spike']) is int and row['lif_spike'] in (0,1),'Invalid spike')
                target[f,c,r,t]={k:row[k] for k in (*FIELDS,'lif_spike')}
    require(set(actual)==EXPECTED_KEYS and set(effective)==EXPECTED_KEYS,'Incomplete 2400 output keys')
    return actual,effective


def trained_reference(fixture):
    out={}
    for f,c,r,t in sorted(EXPECTED_KEYS):
        gold=fixture['cases'][c]['reference_t12x64']
        row={field:gold[field][t][f] for field in FIELDS}
        row['lif_spike']=gold['lif_spikes'][t][f]
        require(all(finite(row[k]) for k in FIELDS),'Invalid trained reference')
        require(row['lif_spike'] in (0,1),'Invalid trained spike reference')
        out[f,c,r,t]=row
    return out


def error_stats(values,reference):
    require(len(values)==len(reference) and bool(values),'Unequal or empty compared arrays')
    errors=[a-b for a,b in zip(values,reference)]
    rmse=math.sqrt(st.mean(e*e for e in errors));rms=math.sqrt(st.mean(r*r for r in reference))
    return dict(n=len(errors),mean_error=st.mean(errors),mae=st.mean(abs(e) for e in errors),rmse=rmse,
        max_abs_error=max(abs(e) for e in errors),reference_rms=rms,nrmse=rmse/rms if rms else None)


def comparison(actual,reference):
    require(set(actual)==set(reference),'Comparison key sets differ')
    def group(keys):
        numeric={k:error_stats([actual[i][k] for i in keys],[reference[i][k] for i in keys]) for k in FIELDS}
        mismatch=sum(actual[i]['lif_spike']!=reference[i]['lif_spike'] for i in keys)
        return dict(points=len(keys),numeric=numeric,spike_mismatches=mismatch,spike_mismatch_fraction=mismatch/len(keys))
    keys=sorted(actual)
    return dict(overall=group(keys),by_filter={str(f):group([k for k in keys if k[0]==f]) for f in range(4)})


def repeat_variation(actual):
    """Five acquired repeats; report variability without claiming independence."""
    require(set(actual)==EXPECTED_KEYS,'Repeat coverage incomplete')
    groups={(f,c,t):[actual[f,c,r,t] for r in range(5)] for f in range(4) for c in range(10) for t in range(12)}
    def group(keys):
        numeric={}
        for field in FIELDS:
            vectors=[[row[field] for row in groups[key]] for key in keys]
            sds=[st.stdev(v) for v in vectors];ranges=[max(v)-min(v) for v in vectors]
            numeric[field]=dict(rms_sample_sd=math.sqrt(st.mean(s*s for s in sds)),max_sample_sd=max(sds),
                maximum_repeat_range=max(ranges),nonidentical_groups=sum(r>0 for r in ranges))
        disagree=sum(len({row['lif_spike'] for row in groups[key]})>1 for key in keys)
        return dict(groups=len(keys),repeats_per_group=5,numeric=numeric,spike_disagreement_groups=disagree)
    return dict(overall=group(sorted(groups)),by_filter={str(f):group([k for k in sorted(groups) if k[0]==f]) for f in range(4)})


def build_comparison(sim,hardware,trained):
    return dict(hardware_vs_simulator=comparison(hardware['actual'],sim['actual']),
        simulator_vs_trained=comparison(sim['actual'],trained),hardware_vs_trained=comparison(hardware['actual'],trained),
        simulator_vs_own_effective_weights=comparison(sim['actual'],sim['effective']),
        hardware_vs_own_effective_weights=comparison(hardware['actual'],hardware['effective']),
        repeat_variation={source:repeat_variation(data['actual']) for source,data in [('simulator',sim),('hardware',hardware)]})


COUNTERS=('impact_attempts','programming_pulse_attempts','programming_decisions','programming_skipped_current')


def counters(value):
    require(set(value)==set(COUNTERS),'Unexpected counter schema')
    require(all(type(value[k]) is int and value[k]>=0 for k in COUNTERS),'Invalid counters')
    require(value['programming_decisions']==value['programming_pulse_attempts']+value['programming_skipped_current'],'Decision accounting mismatch')
    return value


def verify_parent(repo,imported,historical,source):
    if imported is None:
        require(historical is None,'Audit reports an unexpected parent')
        return None
    require(isinstance(historical,dict),'Missing independently audited parent provenance')
    require(imported.get('imported_prefix')==['f0_t0','f0_t1','f0_t2'] and imported.get('imported_trained_coefficients')==42,'Incorrect reused prefix')
    require(imported.get('failed_parent_tile_contributes') is False and imported.get('other_parent_tiles_ignored') is True,'Unapproved parent contribution')
    directory=local(repo,imported['session']);manifest_path=directory/'manifest.json'
    require(sha(manifest_path)==imported['manifest_sha256'],'Parent manifest hash mismatch')
    parent=load(manifest_path);audit_path=local(repo,imported['audit_path'])
    require(sha(audit_path)==imported['audit_sha256'],'Parent audit hash mismatch')
    audit=load(audit_path)
    require(parent['source']==source and audit['source']==source and historical['source']==source,'Parent backend differs')
    require(parent['protocol_id']=='cifar-mode10-feedback-full-fixed-gain-diagnostic-v1','Incorrect parent protocol')
    require(audit['audit_status']=='passed' and audit['errors']==[],'Parent audit not PASS')
    require(audit['auditor_sha256']=='e7367990b6fc2a00f65e3bb0c4e4e5bf097b9f6b21a00700355edb84e0cbc70f','Unreviewed parent auditor')
    require(audit['manifest_sha256']==imported['manifest_sha256'] and audit['session_id']==parent['session_id'],'Parent audit binding mismatch')
    require(imported['result_sha256']==parent['result_sha256'],'Parent result commitments differ')
    require(historical['manifest_sha256']==imported['manifest_sha256'] and historical['audit_sha256']==imported['audit_sha256'],'Child audit parent binding mismatch')
    require(historical['session']==imported['session'] and historical['session_id']==parent['session_id'],'Historical identity mismatch')
    require(historical['acquisition_status']==parent['status'],'Historical failed/completed status changed')
    require((historical['imported_coefficients'],historical['imported_partial_points'],historical['imported_tiles'])==(42,1800,3),'Incorrect imported coverage')
    require(historical['unimported_attempts_not_erased'] is True,'Missing parent-attempt accounting')
    total=counters(historical['all_parent_attempt_counters']);selected=counters(historical['selected_tile_counters'])
    require(total==audit['counters'],'Parent all-attempt counters differ from original audit')
    require(all(selected[k]<=total[k] for k in COUNTERS),'Selected parent counters exceed all attempts')
    require(imported['parent_total_impacts']==total['impact_attempts'] and imported['parent_total_pulses']==total['programming_pulse_attempts'],'Parent totals mislabeled as imported-only')
    return dict(session=imported['session'],session_id=parent['session_id'],source=source,
        acquisition_status=parent['status'],manifest_sha256=imported['manifest_sha256'],audit_sha256=imported['audit_sha256'],
        imported_coefficients=42,imported_tiles=3,selected_tile_counters=selected,
        all_parent_attempt_counters=total,unimported_attempts_retained=True)


def verify_session(repo,session_name,audit_name,source,case_indices):
    directory=local(repo,session_name);audit_path=local(repo,audit_name)
    manifest_path=directory/'manifest.json';m=load(manifest_path);audit=load(audit_path)
    require(m.get('source')==source,'Wrong backend label')
    require(m.get('status')=='completed' and m.get('acquisition_complete') is True,'Session did not complete')
    require(m.get('protocol_id')==PROTOCOL,'Different adaptive protocol')
    require(m.get('hardware_access') is (source=='hardware'),'Hardware access label mismatch')
    gain=1. if source=='hardware' else .03
    require(m.get('fixed_decoder_gain_v_per_weight')==gain,'Incorrect fixed backend gain')
    require(m['source_sha256'].get(FIXTURE)==FIXTURE_SHA,'Different trained CIFAR fixture')
    require(audit.get('audit_status')=='passed' and audit.get('errors')==[] and audit.get('checks',0)>0,'Independent audit did not pass')
    require(audit.get('auditor_sha256')==AUDITOR_SHA,'Unreviewed adaptive auditor')
    require(audit.get('manifest_sha256')==sha(manifest_path),'Audit is not bound to this manifest')
    require(audit.get('session_id')==m['session_id'] and audit.get('source')==source and audit.get('protocol_id')==PROTOCOL,'Audit session identity mismatch')
    require(audit.get('acquisition_complete') is True and audit.get('acquisition_status')=='completed','Audit does not confirm complete acquisition')
    cov=audit['coefficient_coverage'];n=m['microtiles_completed']
    require(type(n) is int and 28<=n<=56,'Invalid total microtile count')
    require(cov==dict(coefficients=300,by_filter={str(f):75 for f in range(4)},measured_tiles=n,exact_300_coverage=True),'Incomplete coefficient geometry')
    require((audit.get('full_time_outputs'),audit.get('partial_points'))==(2400,600*n),'Audit output coverage mismatch')
    parent=verify_parent(repo,m.get('parent_import'),audit.get('parent_import_audit'),source)
    imported_tiles=3 if parent else 0
    require(m['trained_coefficients']==(258 if parent else 300),'New coefficient count incorrectly includes/excludes parent')
    require(m['new_attempted_microtiles']==n-imported_tiles==len(audit['new_microtiles']),'New microtile count mismatch')
    partition=[]
    if parent:partition=[dict(filter_id=0,parent_tile_id=i,microtile_id=0,taps=list(range(i*14,(i+1)*14)),historical=True) for i in range(3)]
    for tile in audit['new_microtiles']:
        require(tile['technical_completed'] is True,'Audit includes incomplete microtile')
        partition.append(dict(filter_id=tile['filter_id'],parent_tile_id=tile['parent_tile_id'],microtile_id=tile['microtile_id'],
            taps=[s['tap_index'] for s in tile['effective_route']],historical=False))
    ids=[(r['filter_id'],r['parent_tile_id'],r['microtile_id']) for r in partition]
    require(len(set(ids))==n,'Duplicate microtile identity')
    represented=[(r['filter_id'],t) for r in partition for t in r['taps']]
    require(len(represented)==300 and set(represented)=={(f,t) for f in range(4) for t in range(75)},'Partition does not represent all 300 coefficients exactly once')
    new=counters(m['new_counters']);require(new==audit['new_counters'],'New counters disagree with audit')
    total=parent['all_parent_attempt_counters'] if parent else dict(impact_attempts=0,programming_pulse_attempts=0)
    cumulative=dict(impact_attempts=new['impact_attempts']+total['impact_attempts'],programming_pulse_attempts=new['programming_pulse_attempts']+total['programming_pulse_attempts'])
    require(m['cumulative_impacts_including_all_parent_attempts']==cumulative['impact_attempts'],'Cumulative impacts omit/include wrong parent attempts')
    require(m['cumulative_pulses_including_all_parent_attempts']==cumulative['programming_pulse_attempts'],'Cumulative pulses omit/include wrong parent attempts')
    summary_path=directory/'summary.json';require(sha(summary_path)==m['result_sha256'].get('summary.json'),'Summary hash mismatch')
    summary=load(summary_path)
    require(summary['fixed_decoder_gain_v_per_weight']==gain,'Summary gain mismatch')
    require(type(summary['physical_weight_quality_pass']) is bool and summary['physical_weight_quality_pass']==audit['physical_weight_quality_pass'],'Quality flag disagrees with audit')
    require(summary['new_counters']==new and summary['parent_import']==m.get('parent_import'),'Summary provenance/counts differ')
    require(summary['microtiles_completed']==n,'Summary microtile count differs')
    actual,effective=indexed_summary(summary,case_indices)
    provenance=dict(session=session_name,session_id=m['session_id'],source=source,protocol_id=PROTOCOL,
        source_commit=m['invocation']['git']['commit'],manifest_sha256=sha(manifest_path),summary_sha256=sha(summary_path),
        independent_audit=audit_name,independent_audit_sha256=sha(audit_path),auditor_sha256=audit['auditor_sha256'],
        source_sha256=m['source_sha256'],fixed_decoder_gain_v_per_weight=gain,
        physical_weight_quality_pass=summary['physical_weight_quality_pass'],new_counters=new,
        parent_import=parent,cumulative_including_all_parent_attempts=cumulative,microtile_partition=partition,
        completed_microtiles=n,address_coverage=summary['address_coverage'])
    return dict(actual=actual,effective=effective,provenance=provenance)


def markdown(report):
    lines=['# Adaptive CIFAR: hardware and native simulator','',
        'Exactly 300 trained coefficients and 2400 aligned outputs per session: 4 filters, 10 real patches, 12 steps, 5 repeats. '
        'Both sessions have a bound independent consistency PASS. Physical quality failure does not hide measured outputs.','',
        '| Comparison | Conv RMSE | BN RMSE | Membrane before reset RMSE | Membrane after reset RMSE | Spike mismatches /2400 |',
        '|---|---:|---:|---:|---:|---:|']
    labels={'hardware_vs_simulator':'HW - SIM','simulator_vs_trained':'SIM - Wtrained','hardware_vs_trained':'HW - Wtrained',
        'simulator_vs_own_effective_weights':'SIM - own W_eff','hardware_vs_own_effective_weights':'HW - own W_eff'}
    for key,label in labels.items():
        row=report['comparisons'][key]['overall'];metrics=row['numeric']
        lines.append('| '+label+' | '+' | '.join(f'{metrics[k]["rmse"]:.8g}' for k in FIELDS)+f' | {row["spike_mismatches"]} |')
    lines+=['','Per-filter metrics, bias/MAE/max error, reference RMS and repeat sample SD are in comparison.json. '
        'Zero reference RMS yields NRMSE=null. Five repeats are not claimed independent.','',
        'The independently acquired microtile partitions may differ. Identical inputs do not imply matched conductance states, '
        'time histories or noise. Hardware gain=1 and simulator gain=0.03 are fixed provisional scales. '
        'Fitted alpha is never used for decoding. Agreement with own W_eff does not establish trained-weight accuracy.','',
        '| Source | Completed microtiles | Imported coefficients | New impacts | All historical parent impacts | Cumulative impacts | Conv repeat RMS sample SD |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for source in ('simulator','hardware'):
        p=report['sessions'][source];parent=p['parent_import'];sd=report['comparisons']['repeat_variation'][source]['overall']['numeric']['conv_sum']['rms_sample_sd']
        lines.append(f"| {source} | {p['completed_microtiles']} | {42 if parent else 0} | {p['new_counters']['impact_attempts']} | {parent['all_parent_attempt_counters']['impact_attempts'] if parent else 0} | {p['cumulative_including_all_parent_attempts']['impact_attempts']} | {sd:.8g} |")
    lines+=['','Imported prefixes remain historical. All parent attempts, including its unimported failed tile, remain in cumulative counts. '
        'They are not mislabeled as new or as represented measurements. No raw ledger is rewritten or CPU value inserted.','',
        'This comparison checks bound audit/manifest/summary hashes and exact output/partition coverage; it does not replay full raw a second time. '
        'These Conv1/BN/LIF outputs are not CIFAR100 classification accuracy.','']
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',default='.')
    for name in ('sim-session','hardware-session','sim-audit','hardware-audit','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv);repo=Path(args.repo).resolve();output=local(repo,args.output)
    for source in (args.sim_session,args.hardware_session):
        folder=local(repo,source);require(output!=folder and not output.is_relative_to(folder),'Output would modify acquired session')
    require(not output.exists(),'Output must be a new directory')
    fixture_path=local(repo,FIXTURE);require(sha(fixture_path)==FIXTURE_SHA,'Reviewed fixture hash mismatch');fixture=load(fixture_path)
    indices=[c['original_test_index'] for c in fixture['cases']]
    sim=verify_session(repo,args.sim_session,args.sim_audit,'simulator',indices)
    hardware=verify_session(repo,args.hardware_session,args.hardware_audit,'hardware',indices)
    require(sim['provenance']['session_id']!=hardware['provenance']['session_id'],'Same session used for both sources')
    for data in (sim,hardware):
        parent=data['provenance']['parent_import']
        if parent:
            folder=local(repo,parent['session'])
            require(output!=folder and not output.is_relative_to(folder),'Output would modify imported parent')
    report=dict(schema='audited-adaptive-cifar-hardware-simulator-comparison-v1',hardware_io=False,
        comparison_script_sha256=sha(Path(__file__)),fixture_sha256=FIXTURE_SHA,
        sessions={'simulator':sim['provenance'],'hardware':hardware['provenance']},
        comparisons=build_comparison(sim,hardware,trained_reference(fixture)),
        microtile_partition_may_differ=True,interpretation='Descriptive aligned-input comparison, not matched states, physical model validation or network classification accuracy.',
        invocation=dict(sim_session=args.sim_session,hardware_session=args.hardware_session,sim_audit=args.sim_audit,hardware_audit=args.hardware_audit,output=args.output))
    output.mkdir(parents=True,exist_ok=False)
    (output/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (output/'REPORT.md').write_text(markdown(report),encoding='utf-8')
    print(json.dumps(dict(status='completed',points_per_session=2400,output=args.output)))
    return 0


if __name__=='__main__':raise SystemExit(main())
