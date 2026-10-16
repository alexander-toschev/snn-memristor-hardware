"""Package bounded derived fresh300 reports. Standard library; no backend/network."""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import struct
import sys

ANALYZER_SHA = '0ced0d8d3b3f5b126374571659c38b5399be4af8cdfc0ebe39a99a0c9331d8c0'
AUDITORS = {
    'bfab9b2044388c11a6362e94471b7872fab134451e8c09187eda59c9e8caa5bb':'fresh300-jit-capacity-fixed-gain-v1',
    '07d2dd0faecff90360df9c9c1771746b8d01045fb10119fa626263a3f201e9ca':'fresh300-jit-capacity-no-bl8-fixed-gain-v1'}
MAX_FILE_BYTES = 500000
MAX_PACKAGE_BYTES = 950000


def require(value, message):
    if not value: raise ValueError(message)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text(encoding='utf-8'))


def local(root, name):
    relative = Path(name)
    require(not relative.is_absolute(), 'Repository-relative path required')
    path = (root/relative).resolve()
    require(path.is_relative_to(root), 'Path escapes repository')
    return path


def bind_session(root, row):
    directory=local(root,row['session']);manifest=directory/'manifest.json';audit=local(root,row['audit'])
    require(sha(manifest)==row['manifest_sha256'], 'Manifest SHA mismatch')
    require(sha(audit)==row['audit_sha256'], 'Audit SHA mismatch')
    m=load(manifest);a=load(audit)
    require(a['auditor_sha256'] in AUDITORS and a['auditor_sha256']==row['auditor_sha256'], 'Unknown auditor')
    require(m['protocol_id']==AUDITORS[a['auditor_sha256']], 'Auditor/protocol mismatch')
    require(a['audit_status']=='passed' and not a['errors'], 'Independent audit must pass')
    require(a['manifest_sha256']==row['manifest_sha256'], 'Audit does not bind manifest')
    require(a['session_id']==m['session_id']==row['session_id'], 'Session identity mismatch')
    require(a['source']==m['source']==row['source'], 'Backend identity mismatch')
    require(m['invocation']['git']['commit']==row['source_commit'], 'Source commit mismatch')
    require(bool(row['complete']) == bool(a['acquisition_complete']) == (m['status']=='completed'), 'Acquisition status mismatch')
    require(len(row['weights'])<=300 and len(row['cells'])==256 and 0<=row['full_time_outputs']<=2400, 'Invalid compact coverage')
    if row['complete']:
        require(len(row['weights'])==300 and row['full_time_outputs']==2400, 'Complete status lacks full coverage')
    return dict(session_id=row['session_id'],source=row['source'],source_commit=row['source_commit'],
        manifest_sha256=row['manifest_sha256'],audit_sha256=row['audit_sha256'],auditor_sha256=row['auditor_sha256'],
        audit_checks=a['checks'],audit_errors=0,acquisition_complete=bool(row['complete']),
        acquisition_status=m['status'],observed_weight_records=len(row['weights']),full_time_outputs=row['full_time_outputs'],
        full_raw_local_repo_relative_path=row['session'],
        full_audit_local_repo_relative_path=row['audit'],acquisition_source_sha256={k:v for k,v in m['source_sha256'].items()
            if k in ('fresh_conv1.py','fresh_conv1_no_bl8.py','experiment.py','configs/fresh_conv1.json','configs/fresh_conv1_no_bl8.json')})


def csv_rows(path):
    with path.open(encoding='utf-8',newline='') as stream: return list(csv.DictReader(stream))


def expected_weight_rows(rows):
    return [{k:str(json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v) if v is not None else ''
             for k,v in row.items() if k!='leg_evidence'} for row in rows]


def expected_cell_rows(rows):
    result=[]
    for row in rows:
        values=dict(wl=row['cell'][0],bl=row['cell'][1],status=row['observation_status'],probes=len(row['probes']),
            selected=len(row['selected']),SET_events=len(row['SET']),quiet_failures=len(row['failures']),
            selected_observations=row['selected_observations'],max_abs_post_all_minus_own_v=row['max_abs_post_all_minus_own_v'],
            signed_mean_post_all_minus_own_v=row['signed_mean_post_all_minus_own_v'],max_abs_final_minus_post_all_v=row['max_abs_final_minus_post_all_v'])
        result.append({k:'' if v is None else str(v) for k,v in values.items()})
    return result


def package(root, source_name, science_digest, output_name, include_png=False):
    root=Path(root).resolve();source=local(root,source_name);output=local(root,output_name)
    require(source.is_dir() and not output.exists(), 'Source directory and new output directory required')
    require(not output.is_relative_to(source) and not source.is_relative_to(output), 'Source/output must be separate')
    def artifact(name):return local(root,(source.relative_to(root)/name).as_posix())
    science_path=artifact('SCIENCE.json')
    require(sha(science_path)==science_digest.lower(), 'Explicit SCIENCE SHA mismatch')
    science=load(science_path)
    require(science['schema']=='fresh300-stability-observations-v1' and science['analyzer_sha256']==ANALYZER_SHA, 'Unknown scientific analysis')
    analyzer=root/'analyze_fresh_stability.py'
    require(sha(analyzer)==ANALYZER_SHA, 'Pinned offline analyzer changed')
    sys.path.insert(0,str(root))
    analysis=importlib.import_module('analyze_fresh_stability')
    require(Path(analysis.__file__).resolve()==analyzer.resolve(), 'Wrong offline analyzer location')
    sessions=science['sessions'];require(1<=len(sessions)<=2, 'One or two sessions required')
    require(len({r['session_id'] for r in sessions})==len(sessions),'Duplicate session')
    bindings=[bind_session(root,row) for row in sessions]
    report=analysis.markdown_report(science)+analysis.extra_report(sessions)
    require(artifact('REPORT.md').read_text(encoding='utf-8')==report, 'REPORT differs from pinned analyzer and SCIENCE')
    payload={};origins={}
    def add(name,data,origin=None):
        require(len(data)<=MAX_FILE_BYTES, 'Compact file size limit: '+name)
        require(name not in payload and Path(name).name==name, 'Duplicate/unsafe compact filename')
        payload[name]=data
        if origin:origins[name]=dict(repo_relative_source=origin.relative_to(root).as_posix(),source_sha256=sha(origin))
    for index,row in enumerate(sessions):
        prefix=row['source']+'_'+row['session_id']
        require(row['artifact_prefix']==prefix,'Unexpected artifact prefix')
        short='' if len(sessions)==1 else f'SESSION{index+1}_'
        svg=artifact(prefix+'_MAP.svg')
        require(svg.read_text(encoding='utf-8')==analysis.evidence_map(row), 'SVG differs from scientific observations')
        add(short+'MAP.svg',svg.read_bytes(),svg)
        for suffix,expected in (('CELLS',expected_cell_rows(row['cells'])),('WEIGHTS',expected_weight_rows(row['weights']))):
            path=artifact(prefix+'_'+suffix+'.csv')
            require(csv_rows(path)==expected, suffix+' CSV differs from SCIENCE')
            add(short+suffix+'.csv',path.read_bytes(),path)
        if include_png:
            png=artifact(prefix+'_MAP.png');data=png.read_bytes()
            require(data[:8]==b'\x89PNG\r\n\x1a\n' and len(data)>=24,'Invalid optional PNG')
            width,height=struct.unpack('>II',data[16:24]);require(0<width<=5000 and 0<height<=5000,'PNG dimension limit')
            add(short+'MAP.png',data,png)
        report=report.replace(prefix+'_MAP.svg',short+'MAP.svg')
    report=report.replace('История каждой ячейки сохранена в SCIENCE.json.',
        'Полная история каждой ячейки сохранена в локальном SCIENCE.json, который не включён в компактный пакет.')
    add('REPORT.md',report.encode('utf-8'),artifact('REPORT.md'))
    lines=['# Компактная сводка fresh300','',
        'Это производный пакет наблюдений, а не полный набор воспроизводимости raw-аудита. '+
        'Полные raw, audit.json, SCIENCE.json и история остаются в локальном checkout/Raspberry; '+
        'их относительные пути и SHA указаны в PROVENANCE.json. Они сюда не копируются.',
        'Цвет карты не квалифицирует точность веса или исправность ячейки. Исключённый BL8 не означает deadcell '+
        'или электрическую изоляцию линии при 0 V. Даже полное покрытие не равно качеству весов, параллельного MVM или классификатора.',
        'Own использует bracket zero; anchors — общий zero. Ошибка пары не приписывается отдельной ноге. '+
        'C, обученные веса и пороги упаковщик не изменяет.','', '[Отчёт](REPORT.md)', '']
    for index,row in enumerate(sessions):
        short='' if len(sessions)==1 else f'SESSION{index+1}_'
        nrmse=row.get('weight_metrics',{}).get('post_all_raw',{}).get('nrmse')
        metric='недоступно' if nrmse is None else f'{nrmse:.4%}'
        state='завершено' if row['complete'] else 'неполное измерение'
        lines += [f"- {row['source']} `{row['session_id']}`: {state}; audit PASS, {bindings[index]['audit_checks']} проверок; "+
            f"записей весов {len(row['weights'])}/300, выходов {row['full_time_outputs']}/2400; "+
            f"качество тайлов {row['quality_tiles']}/{len(row['tiles'])}, NRMSE {metric}. "+
            f"[Карта]({short}MAP.svg), [адреса]({short}CELLS.csv), [веса]({short}WEIGHTS.csv)."]
    lines += ['', 'Полный raw replay из этого пакета невозможен. Проверены явный SHA SCIENCE, связи manifest/audit/source, '+
        'а также точное соответствие REPORT/SVG/CSV закреплённому offline-анализатору. Optional PNG только проверен как '+
        'ограниченный по размеру PNG и связан SHA: его визуальная эквивалентность SVG здесь автоматически не доказывается.',
        'Private-файлы, входные веса/checkpoint и credentials не включены. Новых аппаратных действий нет.','']
    add('README.md','\n'.join(lines).encode('utf-8'))
    provenance=dict(schema='bounded-fresh-report-package-v1',packager_sha256=sha(Path(__file__)),
        full_science_local_repo_relative_path=science_path.relative_to(root).as_posix(),full_science_sha256=science_digest.lower(),
        analyzer_sha256=ANALYZER_SHA,sessions=bindings,full_data_included=False,raw_reaudit_performed=False,
        map_is_not_qualification=True,private_or_input_files_included=False,
        compact_files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),**origins.get(name,{})) for name,data in payload.items()})
    add('PROVENANCE.json',(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    require(sum(map(len,payload.values()))<=MAX_PACKAGE_BYTES,'Compact package total size limit')
    output.mkdir(parents=True)
    for name,data in payload.items():
        with (output/name).open('xb') as stream:stream.write(data)
    return dict(output=output.relative_to(root).as_posix(),files={name:len(data) for name,data in payload.items()},total_bytes=sum(map(len,payload.values())))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.');parser.add_argument('--science-dir',required=True)
    parser.add_argument('--science-sha256',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--include-png',action='store_true');args=parser.parse_args(argv)
    print(json.dumps(package(args.repo,args.science_dir,args.science_sha256,args.output,args.include_png),indent=2))


if __name__=='__main__':main()
