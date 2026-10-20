"""Archive one terminal calibration session, with two streaming secret scans."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_cifar_batch as b


def inventory(folder):
    files=sorted(p for p in folder.rglob('*') if p.is_file())
    if any(p.is_symlink() or not p.resolve().is_relative_to(folder.resolve()) for p in folder.rglob('*')):
        raise ValueError('Session contains links or paths outside its directory')
    return files,{p.relative_to(folder).as_posix():dict(bytes=p.stat().st_size,sha256=b.sha(p)) for p in files}


def verify_archive(archive,sid,expected):
    seen=set()
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf:
            if not member.isfile() or not member.name.startswith(sid+'/'):
                raise ValueError('Unexpected archive member')
            rel=member.name[len(sid)+1:]
            if rel not in expected or rel in seen or member.size!=expected[rel]['bytes']:
                raise ValueError('Archive inventory mismatch')
            seen.add(rel);digest=hashlib.sha256();stream=tf.extractfile(member)
            for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
            if digest.hexdigest()!=expected[rel]['sha256']:raise ValueError('Archive byte mismatch')
    if seen!=set(expected):raise ValueError('Archive missing files')


def preserve(folder,base=None):
    base=(base or ROOT/'results/stable_cell_calibration_20261020').resolve()
    folder=folder.resolve()
    if folder.parent!=base or folder.name=='archives':raise ValueError('Expected direct session directory')
    manifest_path=folder/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') not in ('completed','failed','no_stable_rectangle','pair_qualification_failed','uninformative_calibration') or not manifest.get('finished_utc'):
        raise ValueError('Terminal manifest required; never archive a live acquisition')
    files,before=inventory(folder)
    for name,digest in manifest.get('result_sha256',{}).items():
        if name not in before or before[name]['sha256']!=digest:raise ValueError('Final result inventory mismatch')
    destination=base/'archives';destination.mkdir(exist_ok=True)
    archive=destination/(folder.name+'.tar.gz');report_path=destination/(folder.name+'.json')
    if archive.exists() or report_path.exists():raise FileExistsError('Existing preservation is immutable')
    temp=ROOT/'.tmp/calibration-archive-scan';temp.mkdir(parents=True,exist_ok=True)
    if not temp.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Invalid scan temp path')
    old_temp=tempfile.tempdir;tempfile.tempdir=str(temp)
    decoded={};scanner=b.security.find_scanner(ROOT)
    def chunks(decoded_only=False):
        for path in files:
            rel=path.relative_to(folder).as_posix()
            if decoded_only and path.suffix!='.gz':continue
            yield ('\nFILE '+rel+'\n').encode('utf-8')
            if decoded_only:
                summary={};yield from b.decoded_chunks(path,summary);decoded[rel]=summary
                if manifest['status']!='failed' and not summary['complete_gzip']:
                    raise ValueError('Terminal successful session has incomplete gzip')
            else:yield from b.raw_chunks(path)
    try:
        scans=dict(raw=b.scan_chunks(scanner,chunks()),decoded_gzip=b.scan_chunks(scanner,chunks(True)))
    finally:tempfile.tempdir=old_temp
    if inventory(folder)[1]!=before:raise ValueError('Session changed during secret scan')
    with tarfile.open(archive,'x:gz',compresslevel=1) as tf:
        for path in files:tf.add(path,arcname=folder.name+'/'+path.relative_to(folder).as_posix(),recursive=False)
    verify_archive(archive,folder.name,before)
    if inventory(folder)[1]!=before:raise ValueError('Session changed during preservation')
    report=dict(session=folder.name,source=manifest.get('source'),acquisition_status=manifest['status'],
        archive=archive.relative_to(ROOT).as_posix(),archive_sha256=b.sha(archive),archive_bytes=archive.stat().st_size,
        impacts=manifest.get('impacts'),pulses=manifest.get('pulses'),files=before,decoded_gzip=decoded,
        secret_scan=scans,scanner_sha256=b.sha(Path(scanner)),
        scan_helper_sha256=b.sha(ROOT/'tools/security/scan_staged_lfs.py'),
        raw_bytes=sum(v['bytes'] for v in before.values()),
        decoded_bytes=sum(v['decoded_bytes'] for v in decoded.values()),
        byte_identical_archive_verified=True,original_files_removed=False,
        notes='Preservation only; no semantic audit, hardware operation or accuracy claim.')
    with report_path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    return {k:report[k] for k in ('session','acquisition_status','archive','archive_bytes','raw_bytes','decoded_bytes')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',required=True)
    args=parser.parse_args();path=Path(args.session)
    if path.is_absolute() or '..' in path.parts:raise ValueError('Use repository-relative session path')
    print(json.dumps(preserve(ROOT/path),allow_nan=False))

if __name__=='__main__':main()
