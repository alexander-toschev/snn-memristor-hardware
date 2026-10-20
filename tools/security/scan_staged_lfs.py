"""Scan exact staged LFS payloads locally. No checkout, fetch, or network calls."""
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


class Blocked(Exception):
    """A safe diagnostic: never includes subprocess output or payload content."""


def git(repo, *args, data=None):
    try:
        p = subprocess.run(['git', *args], cwd=repo, input=data,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        raise Blocked('Git is unavailable.') from None
    if p.returncode:
        raise Blocked('A required local Git query failed.')
    return p.stdout


def staged_entries(repo):
    # --no-renames represents a rename as add/delete; the added pointer is scanned.
    changed = set(git(repo, 'diff', '--cached', '--name-only', '-z',
                      '--diff-filter=AMT', '--no-renames', '--no-ext-diff').split(b'\0')) - {b''}
    if not changed:
        return []
    attrs = git(repo, 'check-attr', '--cached', '-z', '--stdin', 'filter',
                data=b'\0'.join(sorted(changed)) + b'\0').split(b'\0')[:-1]
    if len(attrs) % 3:
        raise Blocked('Malformed staged attributes response.')
    filters = {}
    for i in range(0, len(attrs), 3):
        path, key, value = attrs[i:i+3]
        if key != b'filter' or path in filters:
            raise Blocked('Ambiguous staged attributes response.')
        filters[path] = value
    found = {}
    for row in git(repo, 'ls-files', '--stage', '-z').split(b'\0'):
        if not row:
            continue
        metadata, path = row.split(b'\t', 1)
        if path not in changed:
            continue
        mode, oid, stage = metadata.split(b' ')
        if stage != b'0' or path in found:
            raise Blocked('Unmerged or ambiguous staged entry.')
        if not re.fullmatch(rb'[0-9a-f]{40}|[0-9a-f]{64}', oid):
            raise Blocked('Invalid staged Git object ID.')
        found[path] = (mode, oid, filters.get(path))
    if set(found) != changed or set(filters) != changed:
        raise Blocked('Staged entry coverage changed or is incomplete.')
    return sorted((path, *values) for path, values in found.items())


POINTER = re.compile(rb'version https://git-lfs.github.com/spec/v1\n'
                     rb'oid sha256:([0-9a-f]{64})\nsize (0|[1-9][0-9]*)\n?')


def parse_pointer(blob, lfs_attribute=False):
    match = POINTER.fullmatch(blob.replace(b'\r\n', b'\n'))
    if match:
        return match.group(1).decode('ascii'), int(match.group(2))
    if lfs_attribute or blob.startswith(b'version https://git-lfs.github.com/spec/'):
        raise Blocked('Invalid or unsupported staged LFS pointer (extensions fail closed).')
    return None


def local_media_dir(repo):
    # Git LFS 2.13.3 has no contentlocation command. env honors lfs.storage and
    # linked worktrees. Do not print its output: it can contain remote URLs.
    lines = git(repo, 'lfs', 'env').splitlines()
    paths = [os.fsdecode(line[len(b'LocalMediaDir='):]) for line in lines
             if line.startswith(b'LocalMediaDir=')]
    if len(paths) != 1 or not paths[0]:
        raise Blocked('Cannot locate the local LFS object store.')
    path = Path(paths[0])
    return path if path.is_absolute() else Path(repo) / path


def copy_verified_object(media, oid, size, destination):
    if not re.fullmatch('[0-9a-f]{64}', oid) or size < 0:
        raise Blocked('Invalid LFS content identity.')
    path = Path(media) / oid[:2] / oid[2:4] / oid
    digest = hashlib.sha256()
    count = 0
    try:
        with path.open('rb') as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > size:
                    raise Blocked('Local LFS payload size does not match staged pointer.')
                digest.update(chunk)
                destination.write(chunk)
    except OSError:
        raise Blocked('Required local LFS payload is missing or unreadable; fetch it separately.') from None
    if count != size or digest.hexdigest() != oid:
        raise Blocked('Local LFS payload SHA256/size does not match staged pointer.')
    destination.flush()
    destination.seek(0)


def find_scanner(repo):
    scanner = shutil.which('gitleaks')
    if scanner:
        return str(Path(scanner).resolve())
    for name in ('gitleaks.exe', 'gitleaks'):
        candidate = Path(repo) / '.tools' / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    raise Blocked('Install Gitleaks on PATH or in repository .tools.')


# Gitleaks 8.30.1 stdin still MIME-sniffs its first fragment and silently
# skips application/* (including gzip). Scan-only ASCII framing prevents that
# optimization; every original byte follows unchanged, including NULs.
SCAN_FRAME = b'gitleaks-full-byte-stream-v1\n' + b'\n' * 1024
REPOSITORY_CONFIG = Path(__file__).resolve().parents[2] / '.gitleaks.toml'


def scan_chunks(scanner, chunks, cwd, env=None):
    environment = dict(os.environ if env is None else env)
    environment.pop('GITLEAKS_CONFIG', None)
    environment.pop('GITLEAKS_CONFIG_TOML', None)
    process = None
    payload_bytes = 0
    try:
        if not REPOSITORY_CONFIG.is_file():
            raise Blocked('Repository scanner configuration is missing.')
        process = subprocess.Popen([scanner, 'stdin', '--config', str(REPOSITORY_CONFIG), '--redact=100', '--no-banner',
            '--ignore-gitleaks-allow', '--max-target-megabytes=0', '--exit-code=1'],
            stdin=subprocess.PIPE, cwd=cwd, env=environment,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if process.stdin.write(SCAN_FRAME) != len(SCAN_FRAME):
            raise Blocked('Scanner input write incomplete.')
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise Blocked('Scanner requires binary chunks.')
            if process.stdin.write(chunk) != len(chunk):
                raise Blocked('Scanner input write incomplete.')
            payload_bytes += len(chunk)
        process.stdin.close()
        code = process.wait(timeout=300)
        if code == 1:
            raise Blocked('Gitleaks detected a potential secret in a staged LFS payload.')
        if code:
            raise Blocked('Gitleaks failed to scan payload; publication blocked.')
        return dict(payload_bytes=payload_bytes, framing_bytes=len(SCAN_FRAME),
                    scanner_input_bytes=payload_bytes + len(SCAN_FRAME),
                    scan_framing='ascii-lf-v1')
    except (OSError, subprocess.SubprocessError):
        raise Blocked('Gitleaks stream failed; publication blocked.') from None
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except OSError:
                    pass


def scan_stream(scanner, stream, cwd):
    # The caller supplies an isolated directory and a verified immutable snapshot.
    return scan_chunks(scanner, iter(lambda: stream.read(1024**2), b''), cwd)


def scan(repo):
    initial = staged_entries(repo)
    pointers = set()
    for _, mode, oid, attribute in initial:
        if mode not in (b'100644', b'100755'):
            if attribute == b'lfs':
                raise Blocked('LFS attribute on unsupported staged object type.')
            continue
        length = int(git(repo, 'cat-file', '-s', oid.decode('ascii')).strip())
        if length > 1024:
            if attribute == b'lfs':
                raise Blocked('Staged LFS file is not a canonical pointer.')
            continue  # Ordinary non-LFS blob: scanned by existing git --staged hook.
        blob = git(repo, 'cat-file', 'blob', oid.decode('ascii'))
        pointer = parse_pointer(blob, attribute == b'lfs')
        if pointer is not None:
            pointers.add(pointer)
    if pointers:
        media, scanner = local_media_dir(repo), find_scanner(repo)
        with tempfile.TemporaryDirectory(prefix='staged-lfs-scan-') as scratch:
            for oid, size in sorted(pointers):
                # TemporaryFile avoids retaining a second named payload on failure.
                with tempfile.TemporaryFile(dir=scratch) as stream:
                    copy_verified_object(media, oid, size, stream)
                    scan_stream(scanner, stream, scratch)
    if staged_entries(repo) != initial:
        raise Blocked('Git index changed during LFS scanning; retry the commit.')
    return len(pointers)


def main():
    try:
        root = os.fsdecode(git('.', 'rev-parse', '--show-toplevel')).strip()
        count = scan(root)
        print('Staged LFS payload scan passed: %d unique object(s).' % count)
        return 0
    except (Blocked, OSError, ValueError):
        # Never print arbitrary OS/Git/scanner exceptions (paths can be secrets).
        print('Commit blocked: staged LFS payload verification/secret scan failed. '
              'Check local object availability, pointer integrity and scanner installation.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
