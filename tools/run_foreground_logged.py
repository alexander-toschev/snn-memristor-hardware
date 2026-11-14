"""Run one Python entrypoint in the foreground with stdout/stderr on disk.

Install in repository tools/. This is not a supervisor and never retries.
The entrypoint runs in this process, preserving PID, group, terminal and signals.
Redirection prevents progress prints from depending on an SSH output pipe;
it does not guarantee survival of a lost SSH connection or SIGHUP.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


def inside(root: Path, value: str) -> Path:
    """Require an explicit repository-relative path, including symlink checks."""
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Paths must be repository-relative without '..'")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Path escapes repository")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="New relative log under results/")
    parser.add_argument("entrypoint", help="Repository-relative experiment.py")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    entry = inside(root, str(args.entrypoint))
    log = inside(root, str(args.log))
    if not entry.is_file() or entry.suffix != ".py":
        parser.error("Entrypoint must be an existing Python file")
    if not log.is_relative_to(root / "results"):
        parser.error("Log must be inside results/")
    log.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects previous native logs. No stdout/stderr writes
    # before replacement: the caller's pipe may already have disappeared.
    with log.open("xb", buffering=0) as stream:
        os.dup2(stream.fileno(), 1, inheritable=True)
        os.dup2(stream.fileno(), 2, inheritable=True)
    arguments = [str(value) for value in args.arguments]
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    sys.argv = [str(entry), *arguments]
    sys.path[0] = str(entry.parent)
    # No child process, new session, background job or signal suppression.
    # SystemExit and KeyboardInterrupt propagate exactly as in a direct launch.
    runpy.run_path(str(entry), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
