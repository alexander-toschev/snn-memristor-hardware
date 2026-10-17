# Credentials and secret scanning

Keep credentials outside this repository. Supply them through the environment
or a local credential manager. Never put tokens in Git remote URLs, tracked INI
files, source code, experiment logs, or reports. Board INI files may be committed
only after removing passwords and tokens; numeric board settings are not secrets.

The repository ignores common local credential files. This does not protect files
that Git already tracks: review staged changes and use the scanner as well.

Install [Gitleaks](https://github.com/gitleaks/gitleaks/releases) (version 8.30.1
was used for the publication audit) and put its executable on PATH. On Windows,
placing it in the ignored `.tools/gitleaks.exe` also works; this clone has that
local copy installed. Enable the
tracked local hook for each clone:

```sh
git config core.hooksPath .githooks
```

The pre-commit hook scans staged changes and blocks the commit when Gitleaks is
missing or finds a potential secret. It prints redacted diagnostics. GitHub Actions
also scans available Git history on pushes and pull requests, using pinned tool
and checkout versions. Neither check needs a personal access token in a file.

Run a manual history check before publishing a branch:

```sh
gitleaks git . --log-opts="--all --full-history" --redact=100 --ignore-gitleaks-allow
```

Automated detection cannot prove the absence of every kind of secret. If a real
credential is exposed, revoke or rotate it first, then remove it from affected
files and history. Deleting only the latest copy does not remove historical copies.
Do not paste the credential into a public issue or audit report.

Publication audit: [secret-audit.json](.github/secret-audit.json). No detected
credentials required history rewriting. This is a scan result, not a guarantee.

## Git LFS payload protection

## Contract

The hook first runs the existing `gitleaks git --staged --redact=100
--no-banner --ignore-gitleaks-allow`. If that succeeds, the Python stdlib helper
lists added/modified/type-changed index entries using NUL-delimited Git queries.
Renames are represented as add/delete. It reads staged blobs by Git object ID,
not from working-tree files. Staged `filter=lfs` attributes are queried with
`--cached`; canonical LFS pointers are recognized even without that attribute.
Malformed pointers and unsupported pointer extensions fail closed.

Git LFS **2.13.3** was inspected: `contentlocation` is not supported. The helper
uses the supported local `git lfs env` query and its `LocalMediaDir` (including
custom lfs.storage and linked-worktree storage), then the SHA256 aa/bb/OID
fanout. It does not print env output, invoke fetch/pull, use a remote, or fall
back to the unstaged worktree. Missing payload blocks commit.

Every unique staged LFS payload is copied in chunks to a private temporary
snapshot while checking its full byte count and SHA256 against the staged
pointer. Only the verified snapshot is scanned. No payload-size limit exists;
the 1024-byte limit in the script applies only to the Git pointer format,
not to LFS payloads. Temporary disk space approximately equal to the largest
staged LFS object is needed. Insufficient space or other I/O failures block.

Gitleaks scans stdin, not a directory walk: all payload bytes, including NULs,
are submitted without filename/binary-extension or file-size skipping.
`--max-target-megabytes=0`, `--ignore-gitleaks-allow`, `--redact=100` are explicit.
An isolated temporary cwd and removal of GITLEAKS_CONFIG/GITLEAKS_CONFIG_TOML
prevent checkout allowlists/config from suppressing this additional default-rule
scan. Scanner stdout and stderr are suppressed completely, including on errors.
The helper emits only counts or a generic blocking diagnostic, never file names,
payload data, Git stderr, scanner findings, or remote URLs. Scanner failure and
secret detection both block. The staged entry/attribute snapshot is checked
again after scanning; index changes block.

This is a scan of the exact stored bytes, not a claim that arbitrary compressed,
encrypted, or proprietary formats were fully decoded. Gitleaks default rules
and its built-in decoding cannot prove absence of all secrets. Archives needing
semantic inspection require a separate reviewed unpack-and-scan procedure.
This local hook is not retroactive history protection and can be bypassed by a
user disabling hooks; existing history scanning still matters before push.

## Dependencies and commands

Python3 stdlib; Git; Git LFS; Gitleaks on PATH or `.tools/gitleaks[.exe]`.
Gitleaks8.30.1 stdin help was checked locally, including all required flags.
Hook picks repository `.venv` before Python on PATH. No auto-install.

```sh
python -B tools/security/scan_staged_lfs.py
python -B -m unittest discover -s tests -p test_scan_staged_lfs.py -v
```

Optional real-scanner test: set `LFS_SCAN_TEST_GITLEAKS` to the installed
executable before running tests. This uses only a generated, unusable synthetic
token after a 12MiB NUL prefix, followed by a clean binary control. It does not
print the token or scanner output. Unit tests never stage/commit or contact a
remote. Temporary dirs may be redirected with standard TEMP/TMP if a sandbox
denies access to its default temp directory.

