"""Offline actual A parent evidence exercise; never imports acquisition code."""
import argparse,hashlib,json,sys
from pathlib import Path

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',default='.')
    parser.add_argument('--output',required=True);args=parser.parse_args(argv)
    root=Path(args.repo).resolve();output=(root/args.output).resolve()
    if not output.is_relative_to(root):raise ValueError('Output must remain repo-relative')
    if hashlib.sha256((root/'audit_cifar_feedback_remaining_reserves.py').read_bytes()).hexdigest()!='4f2f73eb96fd286aabde4e0513b7aac2ef96e57ef873ce36573a73a29a3b0525':raise ValueError('Reviewed independent auditor changed')
    sys.path.insert(0,str(root))
    import audit_cifar_feedback_remaining_reserves as a
    a.initialize(root)
    loaded_sha=a.sha(root/'audit_cifar_feedback_remaining_reserves.py')
    fixture=a.load(root/'weights/prepared_cifar100_conv1.json');routing=a.load(root/'weights/candidate_cifar75_routing.json')
    bindings={
    'hardware':dict(source='hardware',choice='completed_A',session='results/cifar_feedback_reserve/e17464e9-a601-4b7e-adf5-8a1d56de41d2',manifest_sha256='83afba83e474cbd453fb82c5785123f9751eca32605cb33cb9a725af8f8cf9fe',audit_path='results/cifar_feedback_reviews/reserve_hardware_20261016/audit.json',audit_sha256='eeeb47b7a4f5cd98fec4fc99fc749b7d9af13ffe529f09e0aa1213d921d095a9'),
    'simulator':dict(source='simulator',choice='completed_A',session='results/cifar_feedback_reserve/55134577-a813-4d1b-a3de-67c81ab285ca',manifest_sha256='8d728c0b8a94c1282f5f4120ffb7abf90b755c7bdcd652fd48eec334343debbc',audit_path='results/cifar_feedback_reviews/reserve_simulator_20261016/audit.json',audit_sha256='6691aa35788b1d06f281b44ec0c5c70d84c42bfb9a1265e30f76d9d5f2e30fe8')}
    results={}
    for source,b in bindings.items():
        parent=a.load(root/b['session']/'manifest.json');ck=a.CompactChecks()
        metadata={**b,'result_sha256':parent['result_sha256'],'imported_trained_coefficients':280,
          'excluded_blocks':[dict(filter_id=2,parent_tile_id=0,taps=list(range(11))),dict(filter_id=1,parent_tile_id=0,taps=list(range(9)))],
          'verified_ancestry':parent['parent_import'],**a.historical_totals(b['session'],parent)}
        chunks,totals=a.verify_parent(root,metadata,source,{'parent_bindings':bindings},fixture,routing,ck)
        coeff=[(w['filter_id'],s['tap_index']) for w in chunks for s in w['result']['effective_route']]
        ck.require(len(coeff)==280 and len(set(coeff))==280,'retained_exact280')
        ck.require(sum(f==3 and 52<=t<=64 for f,t in coeff)==13,'retained_A13')
        ck.require(sum(f==0 and t<42 for f,t in coeff)==42,'retained_ancestral42')
        report=a.load(root/b['audit_path'])
        results[source]=dict(status='passed' if not ck.errors else 'failed',checks=ck.count,errors=ck.errors,
          parent_binding=b,parent_audit_checks=report['checks'],retained_coefficients=len(coeff),retained_wrappers=len(chunks),
          retained_A13=13,ancestral42=42,excluded20=20,**totals)
    if any(output.is_relative_to(root/b['session']) for b in bindings.values()):raise ValueError('Output cannot modify source session')
    output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(dict(offline=True,new_backend_calls=0,collector_imported=False,
        loaded_auditor_snapshot_sha256=loaded_sha,results=results),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results))
    return int(any(x['errors'] for x in results.values()))


if __name__=='__main__':raise SystemExit(main())
