"""All configurable paths are relative to this repository, never the shell cwd."""

from pathlib import Path, PureWindowsPath
import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]


def repo_path(value):
    """Resolve a portable repository path; reject escapes, drives and symlinks out."""
    raw = str(value)
    path = Path(raw)
    windows = PureWindowsPath(raw)
    if not raw or path.is_absolute() or windows.drive or windows.root:
        raise ValueError("Use a nonempty path relative to the repository")
    if ".." in path.parts or ".." in windows.parts:
        raise ValueError("Parent-directory traversal is not allowed")
    resolved = (PROJECT_DIR / path).resolve()
    if not resolved.is_relative_to(PROJECT_DIR):
        raise ValueError("Path must remain inside the repository")
    return resolved


def relative_path(path):
    return Path(path).resolve().relative_to(PROJECT_DIR).as_posix()


def use_local_dependencies():
    """Make the repository's packages available for ordinary Python imports."""
    for relative in (".deps", "vendor/MemriBoard"):
        directory = repo_path(relative)
        if directory.is_dir() and str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
