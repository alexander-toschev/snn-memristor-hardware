"""Manifest-exact MNIST result publication inventory and batched raw/decoded secret scan."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
scan_temp=ROOT/'.tmp/mnist-publication-secret-scan'
scan_temp.mkdir(parents=True,exist_ok=True)
os.environ['TMP']=str(scan_temp)
os.environ['TEMP']=str(scan_temp)
sys.path.insert(0,str(ROOT))
from tools import run_cifar_batch as batch
from tools.summarize_mnist_conductance_comparison import sha,require,load
BASE=ROOT/'reports/2026-10-20/mnist-native-sim'


def run():
    inventory={};sessions=[]
    for filename in ('MAIN_AUDIT.json','REFERENCE_AUDIT.json'):
        audit,manifest,observations,files=load(BASE/filename)
        require(audit['status']=='PASS','Raw audit must pass before preservation scan')
        sessions.append(audit['session']);inventory.update(files)
    require(len(sessions)==len(set(sessions))==2,'Exactly main and reference required')
    output=BASE/'PUBLICATION_MANIFEST_ALLOWLIST.json'
    require(not output.exists(),'Preserve previous publication inventory')
    names=sorted(inventory);scans={name:{'raw_scanned':False} for name in names}
    scanner=batch.security.find_scanner(ROOT)
    total={'raw_bytes':0,'decoded_bytes':0,'gzip_files':0}
    def raw_stream():
        for name in names:
            yield b'\n'
            for block in batch.raw_chunks(ROOT/name):
                total['raw_bytes']+=len(block);yield block
            scans[name]['raw_scanned']=True
        yield b'\n'
    batch.scan_chunks(scanner,raw_stream())
    def decoded_stream():
        for name in names:
            if not name.endswith('.gz'):continue
            yield b'\n';entry=scans[name]
            for block in batch.decoded_chunks(ROOT/name,entry):yield block
            require(entry.get('complete_gzip') and not entry.get('truncated_prefix'),'Decoded gzip incomplete')
            total['decoded_bytes']+=entry['decoded_bytes'];total['gzip_files']+=1
        yield b'\n'
    batch.scan_chunks(scanner,decoded_stream())
    for name,item in inventory.items():
        require(sha(ROOT/name)==item['sha256'] and (ROOT/name).stat().st_size==item['bytes'],'Evidence changed during scan')
        item['secret_scan']=scans[name]
    report=dict(schema='mnist-native-pair-publication-allowlist-v1',status='PASS',observed_utc=datetime.now(timezone.utc).isoformat(),sessions=sessions,file_count=len(names),file_sha256=inventory,secret_scan='PASS',raw_and_decoded_scan_policy='Two scanner stdin streams; newline separators; every original file raw bytes and every gzip fully decoded with CRC; no size exclusions or evidence repacking',scanner_sha256=sha(scanner),scanner_helper_sha256=sha(ROOT/'tools/run_cifar_batch.py'),security_helper_sha256=sha(ROOT/'tools/security/scan_staged_lfs.py'),analysis_helper_sha256=sha(ROOT/'tools/summarize_mnist_conductance_comparison.py'),publication_scanner_sha256=sha(__file__),totals=total,git_operations_performed=False,hardware_or_model_executed=False)
    with output.open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,indent=2,ensure_ascii=False);stream.write('\n')
    print(json.dumps(dict(status='PASS',files=len(names),totals=total,output=output.relative_to(ROOT).as_posix())))


if __name__=='__main__':run()
