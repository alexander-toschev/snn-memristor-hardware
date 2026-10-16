"""Offline comparison of two independently audited full diagnostic CIFAR sessions.

Reads recorded summaries only; never imports acquisition, analyzer or driver code.
Audit PASS is evidence of consistency, not a requirement that weight quality passed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st

PROTOCOL='cifar-mode10-feedback-full-fixed-gain-diagnostic-v1'
AUDITOR_SHA='e7367990b6fc2a00f65e3bb0c4e4e5bf097b9f6b21a00700355edb84e0cbc70f'
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
    require(summary.get('filter_tiles_completed')==28,'Expected all 28 tiles')
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


def verify_session(repo,session_name,audit_name,source,case_indices):
    directory=local(repo,session_name);audit_path=local(repo,audit_name)
    manifest_path=directory/'manifest.json';m=load(manifest_path);audit=load(audit_path)
    require(m.get('source')==source,'Wrong backend label: '+session_name)
    require(m.get('status')=='completed' and m.get('acquisition_complete') is True,'Session did not complete')
    require(m.get('protocol_id')==PROTOCOL,'Different diagnostic protocol')
    require(m.get('hardware_access') is (source=='hardware'),'Hardware access label mismatch')
    require(m.get('historical_pilot_imported') is False and m.get('all_tiles_new_measurements') is True,'Historical pilot import is not allowed')
    require('parent_pilot' not in m,'Unexpected parent session')
    require((m.get('trained_coefficients'),m.get('tiles'),m.get('physical_assignments'))==(300,28,600),'Incomplete coefficient geometry')
    gain=1. if source=='hardware' else .03
    require(m.get('fixed_decoder_gain_v_per_weight')==gain,'Incorrect fixed backend gain')
    require(m['source_sha256'].get(FIXTURE)==FIXTURE_SHA,'Different trained CIFAR fixture')
    require(audit.get('audit_status')=='passed' and audit.get('errors')==[] and audit.get('checks',0)>0,'Independent audit did not pass')
    require(audit.get('auditor_sha256')==AUDITOR_SHA,'Unreviewed independent auditor')
    require(audit.get('manifest_sha256')==sha(manifest_path),'Audit is not bound to this manifest')
    require(audit.get('session_id')==m['session_id'] and audit.get('source')==source,'Audit session identity mismatch')
    require(audit.get('acquisition_complete') is True and audit.get('acquisition_status')=='completed','Audit does not confirm complete acquisition')
    require((audit.get('full_time_outputs'),audit.get('partial_points'))==(2400,16800),'Audit output coverage mismatch')
    summary_path=directory/'summary.json'
    require(sha(summary_path)==m['result_sha256'].get('summary.json'),'Summary hash mismatch')
    summary=load(summary_path)
    require(summary['fixed_decoder_gain_v_per_weight']==gain,'Summary gain mismatch')
    require(type(summary['physical_weight_quality_pass']) is bool,'Missing physical quality flag')
    require(summary['physical_weight_quality_pass']==audit['physical_weight_quality_pass'],'Quality flag disagrees with audit')
    require(summary['total_impacts']==audit['counters']['impact_attempts'],'Impact count disagrees with audit')
    require(summary['total_pulses']==audit['counters']['programming_pulse_attempts'],'Pulse count disagrees with audit')
    actual,effective=indexed_summary(summary,case_indices)
    provenance=dict(session=session_name,session_id=m['session_id'],source=source,protocol_id=m['protocol_id'],
        source_commit=m['invocation']['git']['commit'],manifest_sha256=sha(manifest_path),summary_sha256=sha(summary_path),
        independent_audit=audit_name,independent_audit_sha256=sha(audit_path),auditor_sha256=audit['auditor_sha256'],
        source_sha256=m['source_sha256'],fixed_decoder_gain_v_per_weight=gain,
        physical_weight_quality_pass=summary['physical_weight_quality_pass'],impact_attempts=summary['total_impacts'],
        programming_pulses=summary['total_pulses'],address_coverage=summary['address_coverage'])
    return dict(actual=actual,effective=effective,provenance=provenance)


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


def markdown(report):
    lines=['# Аппаратный и native SIM диагностический CIFAR','',
        'Сравниваются одинаковые 10 патчей, 4 фильтра, 12 шагов и 5 повторов: 2400 точек каждого прогона. '
        'Оба прогона имеют собственный PASS независимого аудита. Это не совпадающие физические начальные состояния и не доказательство точности модели платы.','',
        '| Сравнение | Conv RMSE | BN RMSE | Membrane до reset RMSE | Membrane после reset RMSE | Различий спайков /2400 |',
        '|---|---:|---:|---:|---:|---:|']
    labels={'hardware_vs_simulator':'HW − SIM','simulator_vs_trained':'SIM − Wtrained','hardware_vs_trained':'HW − Wtrained',
        'simulator_vs_own_effective_weights':'SIM − собственный W_eff','hardware_vs_own_effective_weights':'HW − собственный W_eff'}
    for key,label in labels.items():
        row=report['comparisons'][key]['overall'];metrics=row['numeric']
        lines.append('| '+label+' | '+' | '.join(f'{metrics[k]["rmse"]:.8g}' for k in FIELDS)+f' | {row["spike_mismatches"]} |')
    lines+=['','В comparison.json приведены те же показатели по каждому фильтру, signed bias/MAE/max error, '
        'NRMSE с явным знаменателем и повторяемость по 480 группам патч/фильтр/время. '
        'При нулевом RMS эталона NRMSE=null, а не нулевой или бесконечный результат.','',
        'Масштабы заранее фиксированы: HW=1 V/единицу веса, SIM=0.03. Fitted alpha не применяется. '
        'W_eff каждого прогона — отдельный эталон его измеренных состояний; совпадение с ним не подтверждает точность обученных весов. '
        'Повторы не объявляются независимыми, ошибка спайков не называется accuracy CIFAR100.','',
        'Этот инструмент повторно не читает полный raw. Он проверяет manifest SHA, соответствующий PASS независимого аудита, '
        'SHA summary и точное покрытие выходов, затем пересчитывает сравнение из actual/эталонных траекторий. '
        'Протокол, raw и ранее опубликованные отчёты не изменяются.','',
        '| Источник | RMS sample SD Conv между 5 повторами | Максимальный диапазон Conv | Групп с несовпадающими спайками /480 |',
        '|---|---:|---:|---:|']
    for source in ('simulator','hardware'):
        variation=report['comparisons']['repeat_variation'][source]['overall'];conv=variation['numeric']['conv_sum']
        lines.append(f'| {source} | {conv["rms_sample_sd"]:.8g} | {conv["maximum_repeat_range"]:.8g} | {variation["spike_disagreement_groups"]} |')
    lines+=['','## Источники','']
    for source in ('simulator','hardware'):
        p=report['sessions'][source]
        lines.append(f'- {source}: `{p["session"]}`, source commit `{p["source_commit"]}`, physical_weight_quality_pass=`{p["physical_weight_quality_pass"]}`.')
    return '\n'.join(lines)+'\n'


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
    report=dict(schema='audited-diagnostic-cifar-hardware-simulator-comparison-v1',hardware_io=False,
        comparison_script_sha256=sha(Path(__file__)),fixture_sha256=FIXTURE_SHA,
        sessions={'simulator':sim['provenance'],'hardware':hardware['provenance']},
        comparisons=build_comparison(sim,hardware,trained_reference(fixture)),
        interpretation='Descriptive aligned-input comparison; not matched states, physical model validation or network classification accuracy.',
        invocation=dict(sim_session=args.sim_session,hardware_session=args.hardware_session,sim_audit=args.sim_audit,hardware_audit=args.hardware_audit,output=args.output))
    output.mkdir(parents=True,exist_ok=False)
    (output/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (output/'REPORT.md').write_text(markdown(report),encoding='utf-8')
    print(json.dumps(dict(status='completed',points_per_session=2400,output=args.output)))
    return 0


if __name__=='__main__':raise SystemExit(main())
