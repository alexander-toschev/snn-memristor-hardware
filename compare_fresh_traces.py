"""Compare two verified actual trace exports without fitting or backend access."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

EXPORTER_SHA = 'ecc2874fae1509f5e4d942024c11c29e831fa5b79e63159bcc345a993b17826e'
ANALYZER_SHA = '0ced0d8d3b3f5b126374571659c38b5399be4af8cdfc0ebe39a99a0c9331d8c0'
FIELDS = ['filter','case','original_test_index','repeat','t','conv','bn','v_before','v_after','spike']
NUMBERS = ('conv','bn','v_before','v_after')
UNITS = dict(conv='checkpoint weight units after measured tile sums / fixed source gain',bn='trained BNTT current',
    v_before='digital LIF potential before reset',v_after='digital LIF potential after reset',spike='0 or 1')
AUDITORS = {'bfab9b2044388c11a6362e94471b7872fab134451e8c09187eda59c9e8caa5bb':'fresh300-jit-capacity-fixed-gain-v1',
    '07d2dd0faecff90360df9c9c1771746b8d01045fb10119fa626263a3f201e9ca':'fresh300-jit-capacity-no-bl8-fixed-gain-v1'}


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def local(root,name):
    result=(root/name).resolve()
    require(not Path(name).is_absolute() and result.is_relative_to(root),'Repository-relative path required')
    return result


def read_trace(root,name):
    directory=local(root,name);metadata_path=local(root,str(Path(name)/'TRACE_PROVENANCE.json'))
    m=json.loads(metadata_path.read_text(encoding='utf-8'));csv_info=m['csv']
    require(m['schema']=='audited-fresh-actual-trace-v2' and m['generator_sha256']==EXPORTER_SHA,'Unknown export contract (v2 re-export required)')
    require(m['route_canonicalization']=='parsed JSON; sort_keys=True; separators=(comma,colon); ensure_ascii=False; allow_nan=False; UTF-8; no trailing newline','Unknown route canonicalization')
    for key in ('planned_route_sha256','planned_route_canonical_sha256'):
        require(isinstance(m[key],str) and len(m[key])==64 and all(x in '0123456789abcdef' for x in m[key]),'Invalid route SHA')
    require(m['analyzer_sha256']==ANALYZER_SHA and AUDITORS.get(m['auditor_sha256'])==m['protocol_id'],'Unknown analyzer/auditor contract')
    require(m['acquisition_complete'] and m['inventory_verified'] and m['actual_only'],'Incomplete/unverified actual export')
    require(m['source'] in ('hardware','simulator'),'Unknown backend')
    require(m['decoder_gain_v_per_weight']==(.25 if m['source']=='hardware' else .03),'Unexpected source gain')
    require(m['units']==UNITS,'Unit contract mismatch')
    require(csv_info['path']=='ACTUAL_TRACE.csv' and csv_info['rows']==2400 and csv_info['columns']==FIELDS,'Invalid CSV contract')
    path=local(root,str(Path(name)/csv_info['path']))
    require(path.stat().st_size==csv_info['bytes'] and sha(path)==csv_info['sha256'],'CSV size/SHA mismatch')
    with path.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream);require(reader.fieldnames==FIELDS,'CSV columns mismatch');raw=list(reader)
    expected=[(f,c,r,t) for f in range(4) for c in range(10) for r in range(5) for t in range(12)]
    require(len(raw)==len(expected),'Missing/excess rows');rows=[];cases={}
    for item,key in zip(raw,expected):
        require(None not in item and all(value is not None for value in item.values()),'Malformed CSV row')
        row={k:int(item[k]) for k in FIELDS[:5]}
        require(tuple(row[k] for k in ('filter','case','repeat','t'))==key,'Missing/duplicate/wrong-order key')
        case,index=row['case'],row['original_test_index'];require(index>=0,'Invalid original index')
        require(cases.setdefault(case,index)==index,'Inconsistent original case index')
        row.update({k:float(item[k]) for k in NUMBERS});require(all(math.isfinite(row[k]) for k in NUMBERS),'Nonfinite output')
        require(item['spike'] in ('0','1'),'Invalid spike');row['spike']=int(item['spike'])
        require(row['spike']==int(row['v_before']>=1.),'Spike/threshold inconsistency')
        require(row['v_after']==(0. if row['spike'] else row['v_before']),'Hard-reset inconsistency')
        rows.append(row)
    require(len(set(cases.values()))==10,'Duplicate original cases')
    return dict(metadata=m,rows=rows,path=name,metadata_sha256=sha(metadata_path),cases=cases)


def metrics(first,second):
    require(len(first)==len(second) and bool(first),'Unaligned rows')
    result={}
    for field in NUMBERS:
        differences=[a[field]-b[field] for a,b in zip(first,second)]
        require(all(math.isfinite(x) for x in differences),'Nonfinite comparison')
        result[field]=dict(rmse=math.sqrt(math.fsum(x*x for x in differences)/len(differences)),
            max_abs=max(abs(x) for x in differences),mean_difference=math.fsum(differences)/len(differences))
    mismatches=[a for a,b in zip(first,second) if a['spike']!=b['spike']]
    result.update(timepoints=len(first),spike_mismatches=len(mismatches),
        unique_logical_spike_mismatches=len({(r['filter'],r['case'],r['t']) for r in mismatches}),
        logical_points=len({(r['filter'],r['case'],r['t']) for r in first}))
    return result


def compare(first,second):
    a,b=first['metadata'],second['metadata']
    require({a['source'],b['source']}=={'hardware','simulator'},'Requires one HW and one native SIM export')
    for key in ('fixture_sha256','trained_weights_sha256','planned_route_canonical_sha256','protocol_id','units'):
        require(a[key]==b[key],'Different shared contract: '+key)
    require(first['cases']==second['cases'],'Original case alignment mismatch')
    hardware,simulator=(first,second) if a['source']=='hardware' else (second,first)
    h,s=hardware['rows'],simulator['rows']
    require([(x['filter'],x['case'],x['original_test_index'],x['repeat'],x['t']) for x in h]==
            [(x['filter'],x['case'],x['original_test_index'],x['repeat'],x['t']) for x in s],'Output keys differ')
    def provenance(value):
        m=value['metadata']
        return dict(export_directory=value['path'],trace_provenance_sha256=value['metadata_sha256'],
            **{k:m[k] for k in ('session_id','source','source_commit','manifest_sha256','audit_sha256','auditor_sha256',
                'fixture_sha256','trained_weights_sha256','planned_route_sha256','planned_route_canonical_sha256','summary_sha256','decoder_gain_v_per_weight')},
            csv_sha256=m['csv']['sha256'])
    return dict(schema='direct-fresh-hardware-simulator-comparison-v1',generator_sha256=sha(Path(__file__)),
        hardware=provenance(hardware),simulator=provenance(simulator),difference_direction='hardware minus simulator',
        total=metrics(h,s),per_filter=[dict(filter=f,**metrics([x for x in h if x['filter']==f],[x for x in s if x['filter']==f])) for f in range(4)],
        comparison_is_calibrated_actual_pipeline_not_raw_voltage=True,fit_or_rescale_performed=False,
        route_raw_bytes_equal=a['planned_route_sha256']==b['planned_route_sha256'],
        route_semantically_equal=True,
        trained_and_effective_reference_metrics_recomputed=False,
        limits=['Direct comparison of independent measured acquisitions; no shared physical state/noise assumption.',
            'Fixed source gains differ (.25 HW/.03 SIM); no further scaling is applied here. C matrices may differ.',
            'Shared route, fixture and trained weights are required; actual programmed weights may differ.',
            'This does not replace separate comparisons against trained weights and W_eff.',
            'Repeated 2400 points and unique 480 logical points are counted separately.',
            'Bound export provenance and CSV are verified; full raw inventory/audit is not repeated here.',
            'Four filters and ten prepared patches do not establish full-classifier accuracy.'])


def run(repo,first,second,output):
    root=Path(repo).resolve();destination=local(root,output)
    require(not destination.exists(),'Output directory must be new')
    for name in (first,second):
        source=local(root,name)
        require(not destination.is_relative_to(source) and not source.is_relative_to(destination),'Input/output overlap')
    result=compare(read_trace(root,first),read_trace(root,second));m=result['total']
    lines=['# Прямое сравнение измеренных HW и native SIM выходов','',
        f"HW: `{result['hardware']['session_id']}`; SIM: `{result['simulator']['session_id']}`.",'',
        'Разности ниже: HW минус SIM. Это выходы калиброванного измеренного Conv→BN→цифровой LIF, а не сырые напряжения. '+
        'Общие fixture, обученные веса, маршрут и все 2400 ключей проверены. Дополнительной подгонки масштаба нет.','',
        '| Величина | RMSE | Максимальная абсолютная разность | Средняя разность |',
        '|---|---:|---:|---:|']
    for key in NUMBERS: lines.append(f"| {key} | {m[key]['rmse']:.9g} | {m[key]['max_abs']:.9g} | {m[key]['mean_difference']:.9g} |")
    lines += ['',f"Несовпадения спайков: {m['spike_mismatches']}/{m['timepoints']} с повторами; "+
        f"{m['unique_logical_spike_mismatches']}/{m['logical_points']} уникальных логических точек.",'',
        'Gain HW=.25 и SIM=.03, C и физически записанные состояния могут различаться. Это не доказательство одинаковой физики или шума. '+
        'Сравнения каждого прогона с обученным эталоном и W_eff остаются отдельными результатами и здесь не пересчитываются. '+
        'Полный raw повторно не проверяется: используются SHA-связанные экспорты из независимо проверенных сессий. '+
        'Результат не является точностью полного классификатора.','']
    payload=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    require(len(payload)<50000,'Unexpectedly large comparison')
    destination.mkdir(parents=True)
    (destination/'COMPARISON.json').write_bytes(payload);(destination/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return dict(output=output,**m,files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(destination.iterdir())})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.')
    for name in ('first','second','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args(argv);print(json.dumps(run(a.repo,a.first,a.second,a.output),indent=2))

if __name__=='__main__':main()
