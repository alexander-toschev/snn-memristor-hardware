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
