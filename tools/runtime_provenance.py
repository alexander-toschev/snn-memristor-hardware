"""Explicit offline provenance: Git-versioned sources; private runtime is descriptive.

No hardware/network calls. Historical source verification uses only a reviewed,
fixed local Git commit, never HEAD or a revision supplied by an input report.
"""
from configparser import ConfigParser
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REGISTRATION='tools/runtime_provenance_versions.json'
OLD_COMMIT='b7232c4367096d1c29eeb4d4c0c7e40645a99142'
PRIVATE={
 'vendor/MemriBoard/MemriCORE/rp5_fpga_python/rpi_FPGAed.py':'bc717659fa5d4d36d2037eef978bf1cd8dcf0fe92f9bbfad51f1eb1d0d27df9c',
 'vendor/MemriBoard/MemriCORE/rp5_fpga_python/FPGA_SIM.py':'5bcb02c04bc185e332b2ec96acabc2d4303d21724c9c052de540ef92d603b634',
}
STATUS='private_runtime_not_independently_reverified'
OLD_REGISTRY='tools/cifar_stage_source_registry.json'
NEW_REGISTRY='tools/cifar_stage_source_registry_v2.json'


def require(ok,message):
    if not ok:raise ValueError(message)


def local(name,root=ROOT):
    p,w=Path(name),PureWindowsPath(name)
    require(isinstance(name,str) and name and not p.is_absolute() and not w.drive and not w.root and '..' not in p.parts,'Repository-relative source required')
    result=(root/p).resolve();require(result.is_relative_to(root.resolve()),'Source path escapes repository')
    return result


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions(root=ROOT):
    value=json.loads(local(REGISTRATION,root).read_text(encoding='utf-8'))
    require(value.get('schema')=='runtime-provenance-versions-v2' and value.get('registration')=='approved_after_review','Provenance v2 registration closed')
    require(value.get('historical_commit')==OLD_COMMIT,'Unreviewed historical Git revision')
    require(value['current']['tools/runtime_provenance.py']==sha(local('tools/runtime_provenance.py',root)),'Provenance implementation changed')
    return value


def verify_source(name,pin,root=ROOT):
    require(name not in PRIVATE,'Private bytes need the explicit descriptive policy')
    path=local(name,root)
    if path.is_file() and sha(path)==pin:return 'current_local_bytes'
    known=versions(root)['historical_sources']
    require(known.get(name)==pin,'Unknown/missing source version: '+name)
    # No checkout, filters, hooks, credentials or network. Do not print Git stderr.
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0')
    result=subprocess.run(['git','--no-pager','show',OLD_COMMIT+':'+name],cwd=root,env=env,capture_output=True,check=False)
    require(result.returncode==0 and hashlib.sha256(result.stdout).hexdigest()==pin,'Pinned historical Git object unavailable or changed: '+name)
    return 'historical_git_object_bytes'


def hardware_runtime(root=ROOT):
    """The same board INI/public required pins, without opening private files."""
    from read_all import INI, VENDOR, EXPECTED_BOARD, EXPECTED_SOURCES, FPGA_SOURCE, FPGA_SHA256
    snapshot=local(INI,root).read_bytes();config=ConfigParser();config.read_string(snapshot.decode('utf-8-sig'))
    require(config.get('board','board_type')=='rp5_fpga_python','Wrong hardware INI backend')
    require(all(config.getfloat('board',k)==v for k,v in EXPECTED_BOARD.items()),'Hardware INI settings changed')
    expected={VENDOR+'/'+name:pin for name,pin in {**EXPECTED_SOURCES,FPGA_SOURCE:FPGA_SHA256}.items()}
    for name,pin in expected.items():
        if name not in PRIVATE:verify_source(name,pin,root)
    expected[INI]=sha(local(INI,root))
    return snapshot,expected


def private_recorded(recorded):
    require(all(recorded.get(name)==pin for name,pin in PRIVATE.items()),'Recorded private runtime pins differ from reviewed acquisition')
    return dict(status=STATUS,recorded_sha256=dict(PRIVATE),fresh_remote_check_performed=False,
                private_bytes_git_verified=False,limitation='Acquisition recorded these private runtime hashes; local audit does not independently reverify private file contents.')


def audit_sources(report,root=ROOT,verify_bytes=True):
    """Exact v1/v2 producer inventories; preserve historical identities."""
    v=versions(root);sources=report.get('audit_source_sha256',{})
    auditor='tools/audit_cifar_interrupted_prefix.py';replay='tools/cifar_prefix_replay.py'
    generation=report.get('provenance',{}).get('provenance_generation')
    if sources.get(auditor)==v['historical_sources'][auditor]:
        require(generation is None,'Historical producer cannot claim v2 provenance')
        expected={n:v['historical_sources'][n] for n in (auditor,replay)}
    else:
        require(generation=='v2','New producer requires explicit v2 provenance')
        expected={n:v['current'][n] for n in (auditor,replay,'tools/runtime_provenance.py')}
        expected[REGISTRATION]=sha(local(REGISTRATION,root))
    prov=report.get('provenance',{})
    if prov.get('stage_source_registry') is not None:
        name=prov.get('stage_source_registry_path',OLD_REGISTRY)
        require(name in (OLD_REGISTRY,NEW_REGISTRY),'Unknown stage registry')
        expected[name]=prov['stage_source_registry_sha256']
        approved=v['historical_sources'].get(name) if name==OLD_REGISTRY else v.get('current_stage_registry_sha256')
        require(expected[name]==approved,'Unreviewed stage registry version')
        require(json.loads(local(name,root).read_text())==prov['stage_source_registry'],'Stage registry body changed')
    require(sources==expected,'Exact reviewed prefix audit dependency inventory required')
    if verify_bytes:
        for name,pin in sources.items():verify_source(name,pin,root)
    return sources


def registry_for(recorded,root=ROOT):
    matches=[]
    for name in (OLD_REGISTRY,NEW_REGISTRY):
        p=local(name,root)
        if not p.exists():continue
        value=json.loads(p.read_text())
        if not all(recorded.get(n)==s for n,s in value['source_sha256'].items()):continue
        approved=versions(root)['historical_sources'].get(name) if name==OLD_REGISTRY else versions(root).get('current_stage_registry_sha256')
        require(sha(p)==approved,'Stage source registry is not independently registered')
        matches.append((name,value))
    require(len(matches)==1,'Unknown or ambiguous acquisition source generation')
    return matches[0]
