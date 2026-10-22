"""Foreground publication of one terminal session; never invokes acquisition.

Uses the existing byte-preserving archive + secret scans, standard Git hooks
and LFS. No credentials are read from files, no force push or index clearing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Protocol, cast

from experiment_common.durable import atomic_json
from experiment_common.selective_metrology import object_dict

REMOTE_BRANCH = 'codex/full-cifar-diagnostic'
ALLOWED_REMOTES = (
    'https://github.com/alexander-toschev/snn-memristor-hardware',
    'https://github.com/alexander-toschev/snn-memristor-hardware.git',
    'git@github.com:alexander-toschev/snn-memristor-hardware.git',
)


class Preserver(Protocol):
    # Exact existing tools.archive_calibration.preserve(folder, base=None).
    def __call__(self, folder: Path, base: Path) -> object: ...


def git(root: Path, arguments: list[str], log: Path) -> str:
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith('GIT_CONFIG_') and key not in (
                       'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_ASKPASS', 'SSH_ASKPASS',
                       'GIT_LFS_SKIP_PUSH', 'GITLEAKS_CONFIG', 'GITLEAKS_CONFIG_TOML')}
    environment.update(GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never', GIT_ASKPASS='',
                       SSH_ASKPASS='', GIT_LFS_SKIP_SMUDGE='1', GIT_SSH_COMMAND='ssh -oBatchMode=yes')
    command = ['git', '-c', 'core.hooksPath=.githooks', '-c', 'gc.auto=0',
               '-c', 'lfs.allowincompletepush=false', *arguments]
    result = subprocess.run(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding='utf-8', errors='replace', timeout=3600, check=False)
    with log.open('a', encoding='utf-8') as stream:
        stream.write('\nCOMMAND ' + json.dumps(arguments) + '\n' + result.stdout)
    if result.returncode != 0:
        raise RuntimeError('Git ' + arguments[0] + ' failed; see local publication log')
    return result.stdout.strip()


def preflight(root: Path, log: Path) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    if Path(git(root, ['rev-parse', '--show-toplevel'], log)).resolve() != root.resolve():
        raise ValueError('Wrong Git checkout')
    if git(root, ['diff', '--cached', '--name-only'], log):
        raise ValueError('Pre-existing staged files; publication will not alter the index')
    if git(root, ['remote', 'get-url', 'origin'], log) not in ALLOWED_REMOTES:
        raise ValueError('Unexpected publication remote')
    if git(root, ['remote', 'get-url', '--push', 'origin'], log) not in ALLOWED_REMOTES:
        raise ValueError('Unexpected push remote')
    return git(root, ['rev-parse', 'HEAD'], log)


def publish(root: Path, session: Path, *, log: Path) -> dict[str, object]:
    session, root = session.resolve(), root.resolve()
    if not session.is_relative_to(root / 'results') or session.parent.name == 'archives':
        raise ValueError('Expected terminal results session')
    preflight(root, log)
    decoded: object = json.loads((session / 'manifest.json').read_text(encoding='utf-8'))
    manifest = object_dict(decoded)
    if manifest.get('status') not in ('completed', 'failed', 'no_stable_rectangle', 'pair_qualification_failed') or not manifest.get('finished_utc'):
        raise ValueError('Cannot publish live session')
    archive = session.parent / 'archives' / (session.name + '.tar.gz')
    receipt = archive.with_name(session.name + '.json')
    # Existing archive helper performs raw and decoded secret scans and exact
    # byte verification. Its untyped legacy boundary is described above only.
    if not archive.exists() and not receipt.exists():
        from tools.archive_calibration import preserve
        cast(Preserver, preserve)(session, session.parent)
    else:
        # A retry may use only a complete previously verified archive; no overwrite.
        if not archive.is_file() or not receipt.is_file():
            raise ValueError('Incomplete archive; preserve existing evidence for review')
        from experiment_common.durable import sha
        old = object_dict(json.loads(receipt.read_text(encoding='utf-8')))
        if old.get('archive_sha256') != sha(archive) or old.get('byte_identical_archive_verified') is not True:
            raise ValueError('Archive retry identity mismatch')
        files = object_dict(old['files'])
        for relative, evidence in files.items():
            item = object_dict(evidence)
            target = (session / relative).resolve()
            if not target.is_relative_to(session) or item['sha256'] != sha(target):
                raise ValueError('Session differs from existing archive')
    summary = session.parent / 'summaries' / (session.name + '.json')
    summary.parent.mkdir(exist_ok=True)
    data: object = json.loads((session / 'summary.json').read_text(encoding='utf-8'))
    if summary.exists():
        if json.loads(summary.read_text(encoding='utf-8')) != data:
            raise ValueError('Existing published summary differs')
    else:
        atomic_json(summary, data)
    html = summary.with_suffix('.html')
    source_html = (session / 'map.html').read_bytes()
    if html.exists() and html.read_bytes() != source_html:
        raise ValueError('Existing published map differs')
    html.write_bytes(source_html)
    allowlist = [path.relative_to(root).as_posix() for path in (archive, receipt, summary, html)]
    # Acquisition is closed before any checkout update. FF only, no merges/rebases.
    git(root, ['fetch', 'origin', REMOTE_BRANCH], log)
    git(root, ['merge', '--ff-only', 'FETCH_HEAD'], log)
    preflight(root, log)
    attribute = git(root, ['check-attr', 'filter', '--', allowlist[0]], log)
    if not attribute.endswith(': filter: lfs'):
        raise ValueError('Archive must use Git LFS')
    git(root, ['add', '--', *allowlist], log)
    staged = set(git(root, ['diff', '--cached', '--name-only'], log).splitlines())
    if not staged <= set(allowlist):
        raise ValueError('Unexpected staged file; not committing')
    if staged:
        git(root, ['commit', '-m', 'Preserve remaining-cell calibration ' + session.name], log)
    commit = git(root, ['rev-parse', 'HEAD'], log)
    git(root, ['push', 'origin', 'HEAD:' + REMOTE_BRANCH], log)
    return dict(status='published', commit=commit, allowlist=allowlist,
                acquisition_status=manifest['status'], source=manifest['source'])
