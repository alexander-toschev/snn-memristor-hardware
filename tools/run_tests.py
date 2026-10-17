"""Run offline tests after locating public dependencies; never launch a backend."""
import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PINNED_ROOT_TESTS = ('test_audit_cifar_feedback', 'test_audit_cifar_feedback_adaptive', 'test_audit_cifar_memristor')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('modules', nargs='*', help='Specific tests.test_* modules; default is the full offline suite')
    parser.add_argument('--pattern', default='test_*.py', help='Discovery pattern; used when modules are omitted')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    for folder in (ROOT, ROOT / '.deps', ROOT / 'vendor/MemriBoard'):
        if folder.is_dir(): sys.path.insert(0, str(folder))
    loader = unittest.TestLoader()
    if args.modules:
        suite = loader.loadTestsFromNames(args.modules)
    else:
        suite = loader.discover(str(ROOT / 'tests'), pattern=args.pattern, top_level_dir=str(ROOT))
        if args.pattern == 'test_*.py': suite.addTests(loader.loadTestsFromNames(PINNED_ROOT_TESTS))
    return 0 if unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite).wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
