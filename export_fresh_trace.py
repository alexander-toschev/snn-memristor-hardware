"""Export audited measured Conv/BN/LIF traces only; stdlib, no backend/network."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ANALYZER_SHA = '0ced0d8d3b3f5b126374571659c38b5399be4af8cdfc0ebe39a99a0c9331d8c0'
MAX_BYTES = 400000
FIELDS = ('filter','case','original_test_index','repeat','t','conv','bn','v_before','v_after','spike')


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_route_sha(path):
    value=json.loads(path.read_text(encoding='utf-8'))
    data=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def trace_rows(trajectories, fixture):
    require(len(trajectories)==200 and len(fixture['cases'])==10,'Expected 200 complete trajectories and 10 cases')
    rows=[];seen=set()
    for trajectory in trajectories:
        f,c,r=trajectory['filter_id'],trajectory['case'],trajectory['repeat']
        require(f in range(4) and c in range(10) and r in range(5),'Invalid trajectory identity')
        require(trajectory['original_test_index']==fixture['cases'][c]['original_test_index'],'Original case identity mismatch')
        actual=trajectory['actual'];require(actual is not None and len(actual)==12,'Incomplete actual trace')
        for t,value in enumerate(actual):
            key=(f,c,r,t);require(key not in seen,'Duplicate timepoint');seen.add(key)
            require(value['t']==t,'Timestep ordering mismatch')
            numbers=[value[k] for k in ('conv_sum','bntt_current','lif_before_reset','lif_after_reset')]
            require(all(math.isfinite(x) for x in numbers),'Nonfinite actual output')
            require(value['lif_spike'] in (0,1),'Invalid spike')
            rows.append(dict(zip(FIELDS,(f,c,trajectory['original_test_index'],r,t,*numbers,value['lif_spike']))))
    require(seen=={(f,c,r,t) for f in range(4) for c in range(10) for r in range(5) for t in range(12)},'Incomplete timepoint coverage')
    return sorted(rows,key=lambda row:(row['filter'],row['case'],row['repeat'],row['t']))


def csv_bytes(rows):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def export(repo,session,audit,output):
    repo=Path(repo).resolve();analyzer=repo/'analyze_fresh_stability.py'
    require(sha(analyzer)==ANALYZER_SHA,'Pinned analyzer changed')
    import analyze_fresh_stability as analysis
    require(Path(analysis.__file__).resolve()==analyzer,'Unexpected analyzer location')
    directory=analysis.local(repo,session);destination=analysis.local(repo,output)
    require(not destination.exists(),'Output directory must be new')
    require(not destination.is_relative_to(directory) and not directory.is_relative_to(destination),'Source/output overlap')
    report,trajectories=analysis.analyze(repo,session,audit)
    require(report['complete'] and report['full_time_outputs']==2400,'Complete audited acquisition required')
    fixture=analysis.load(repo/'weights/prepared_cifar100_conv1.json')
    manifest=analysis.load(directory/'manifest.json')
    rows=trace_rows(trajectories,fixture);data=csv_bytes(rows)
    metadata=dict(schema='audited-fresh-actual-trace-v2',generator_sha256=sha(Path(__file__)),
        analyzer_sha256=ANALYZER_SHA,session=session,session_id=report['session_id'],source=report['source'],
        source_commit=report['source_commit'],protocol_id=manifest['protocol_id'],
        manifest_sha256=report['manifest_sha256'],audit=audit,audit_sha256=report['audit_sha256'],
        auditor_sha256=report['auditor_sha256'],audit_checks=report['audit_checks'],
        fixture_sha256=report['fixture_sha256'],trained_weights_sha256=report['trained_weights_sha256'],
        summary_sha256=manifest['result_sha256']['summary.json'],
        planned_route_sha256=manifest['result_sha256']['planned_route.json'],
        planned_route_canonical_sha256=canonical_route_sha(directory/'planned_route.json'),
        route_canonicalization='parsed JSON; sort_keys=True; separators=(comma,colon); ensure_ascii=False; allow_nan=False; UTF-8; no trailing newline',
        decoder_gain_v_per_weight=report['decoder_gain'],acquisition_complete=True,
        csv=dict(path='ACTUAL_TRACE.csv',sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),rows=len(rows),columns=list(FIELDS)),
        inventory_verified=True,raw_reaudit_performed=False,actual_only=True,
        units=dict(conv='checkpoint weight units after measured tile sums / fixed source gain',bn='trained BNTT current',
            v_before='digital LIF potential before reset',v_after='digital LIF potential after reset',spike='0 or 1'),
        limits=['Independent acquisitions; same case keys do not imply equal physical programmed weights or noise.',
            'HW and native SIM use distinct fixed gains; source scale is recorded, not claimed physically equivalent.',
            'Each timestep accumulates all 75 measured taps before trained BN and one digital LIF step.',
            'No fixture inputs, trained weights, raw journal or private files are exported.',
            'Completion and valid audit do not qualify weight fidelity or classifier accuracy.'])
    meta=(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    require(len(data)+len(meta)<=MAX_BYTES,'Export exceeds 400000 bytes; no silent truncation')
    require(sha(analyzer)==ANALYZER_SHA and sha(directory/'manifest.json')==report['manifest_sha256'],'Source changed during export')
    destination.mkdir(parents=True)
    for name,payload in [('ACTUAL_TRACE.csv',data),('TRACE_PROVENANCE.json',meta)]:
        with (destination/name).open('xb') as stream: stream.write(payload)
    return dict(output=output,rows=len(rows),total_bytes=len(data)+len(meta),
        files={name:dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
               for name,payload in [('ACTUAL_TRACE.csv',data),('TRACE_PROVENANCE.json',meta)]})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.')
    for name in ('session','audit','output'): p.add_argument('--'+name,required=True)
    args=p.parse_args(argv);print(json.dumps(export(args.repo,args.session,args.audit,args.output),indent=2))


if __name__=='__main__':main()
