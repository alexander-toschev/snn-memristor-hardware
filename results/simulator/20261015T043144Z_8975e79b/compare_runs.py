"""Compare completed old/new simulator executions; no simulator imports or writes."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def scientific_summary(run):
    phase1=read(run/'characterization.json')
    result={'phase1':{'attempts':phase1['attempt_count'],
                      'target_crossings':phase1['target_reached_count'],
                      'reset_failures':sum(g['attempt_count']-g['reset_success_count'] for g in phase1['groups']),
                      'bias_uS_range':[min(g['bias_s'] for g in phase1['groups'])*1e6,max(g['bias_s'] for g in phase1['groups'])*1e6],
                      'sample_sd_uS_range':[min(g['error_std_s_ddof1'] for g in phase1['groups'])*1e6,max(g['error_std_s_ddof1'] for g in phase1['groups'])*1e6],
                      'rmse_uS_range':[min(g['rmse_s'] for g in phase1['groups'])*1e6,max(g['rmse_s'] for g in phase1['groups'])*1e6]}}
    for phase in ('phase2','phase3','phase4'):
        data=read(run/(phase+'.json'))
        rows=data['results']
        result[phase]={
            'filters':data['filter_count'],'repetitions':data['repetitions'],
            'programmed_cells':data['programmed_cell_count'],
            'reset_failures':sum(150-r['reset_success_count'] for r in rows),
            'target_misses':sum(150-r['target_reached_count'] for r in rows),
            'comparisons':{metric:{'passed':sum(r[metric]['historical_gate_pass'] for r in rows),
                                   'total':len(rows),
                                   'failed':[{'filter':r['filter_id'],'repetition':r['repetition'],
                                              'rmse':r[metric]['rmse'],'pearson':r[metric]['pearson'],
                                              'rmse_over_expected_sd':r[metric]['rmse']/r[metric]['expected_std_ddof0']}
                                             for r in rows if not r[metric]['historical_gate_pass']]}
                           for metric in ('weights','mvm_vs_ideal','mvm_vs_programmed')}}
        if phase=='phase4':
            prefix=data['prefix_means']
            result[phase]['prefix_means']=[{
                'filter_id':p['filter_id'],'K':p['K'],
                'weight_rmse':p['weights']['rmse'],'weight_pearson':p['weights']['pearson'],
                'mvm_rmse':p['mvm_vs_ideal']['rmse'],'mvm_pearson':p['mvm_vs_ideal']['pearson'],
                'mvm_rmse_change_from_K1_percent':100*(p['mvm_vs_ideal']['rmse']/prefix[0]['mvm_vs_ideal']['rmse']-1),
                'weight_rmse_change_from_K1_percent':100*(p['weights']['rmse']/prefix[0]['weights']['rmse']-1),
                'historical_mvm_gate':p['mvm_vs_ideal']['historical_gate_pass']}
                for p in prefix]
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--old',required=True)
    parser.add_argument('--new',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    old,new=args.repo/args.old,args.repo/args.new
    manifests=[read(run/'manifest.json') for run in (old,new)]
    if any(m['status']!='completed' or m['source']!='simulator' for m in manifests):
        raise ValueError('Both executions must be completed simulator runs')
    if manifests[0]['session_id']==manifests[1]['session_id']:
        raise ValueError('Separate run IDs required')
    files=[]
    for path in sorted(new.iterdir()):
        baseline_files={'baseline.json','baseline.csv','characterization.json',
                        'characterization_attempts.csv','characterization_stats.csv'}
        phase_pattern=r'phase[234](?:\.json|_filter_\d{2}_rep_\d+\.json|_mvm_inputs\.json|_mvm_outputs\.csv|_programming\.csv)'
        if path.name not in baseline_files and re.fullmatch(phase_pattern,path.name) is None:
            continue
        before=old/path.name
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        old_hash,new_hash=digest(before),digest(path)
        files.append({'file':path.name,'old_sha256':old_hash,'new_sha256':new_hash,'identical':old_hash==new_hash})
    old_summary,new_summary=scientific_summary(old),scientific_summary(new)
    report={
        'source':'simulator','hardware_qualification':False,
        'old_run':args.old,'new_run':args.new,
        'old_identity':{k:manifests[0][k] for k in ('session_id','started_utc','finished_utc','seed','raw_task_count')},
        'new_identity':{k:manifests[1][k] for k in ('session_id','started_utc','finished_utc','seed','raw_task_count')},
        'separate_executions':True,'scientific_files_compared':len(files),
        'scientific_files_identical':all(f['identical'] for f in files),
        'scientific_summaries_identical':old_summary==new_summary,
        'files':files,'new_scientific_outcomes':new_summary,
        'interpretation':'A new completed execution reproduced historical scientific outputs at the same seed. Execution and arithmetic consistency do not imply that all scientific accuracy comparisons passed.',
        'limitations':['Synthetic simulator probes only; no hardware or classifier accuracy.',
                       'The same seed tests behavior preservation, not independent scientific replication.',
                       'Target crossing is not tolerance-based precise programming.',
                       'K prefix means use one filter and three sequential programming attempts; this does not establish a population-level averaging law.',
                       'Raspberry Pi board INI files concern a future, separately reviewed hardware stage.']}
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    outcomes={'source':'simulator','run':args.new,'session_id':manifests[1]['session_id'],
              'execution_complete':True,'hardware_qualification':False,
              'all_historical_mvm_comparisons_passed':all(new_summary[p]['comparisons']['mvm_vs_ideal']['passed']==new_summary[p]['comparisons']['mvm_vs_ideal']['total'] for p in ('phase2','phase3','phase4')),
              **new_summary,'limitations':report['limitations']}
    (args.output.parent/'scientific_outcomes.json').write_text(json.dumps(outcomes,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('scientific_files_compared','scientific_files_identical','scientific_summaries_identical')}))


if __name__=='__main__':
    main()
