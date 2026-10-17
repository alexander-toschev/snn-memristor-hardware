"""Convenient front door for unchanged, hash-pinned offline tools."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    'analyze-stability': 'analyze_fresh_stability.py',
    'diagnose-fidelity': 'diagnose_fresh_fidelity.py',
    'export-trace': 'export_fresh_trace.py',
    'compare-traces': 'compare_fresh_traces.py',
    'package-report': 'package_fresh_report.py',
    'analyze-capacity': 'analyze_cifar_adaptive_capacity.py',
    'audit-fresh': 'audit_fresh_conv1.py',
    'audit-no-bl8': 'audit_fresh_conv1_no_bl8.py',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Tool arguments and output paths are relative to the repository root.')
    parser.add_argument('tool', choices=COMMANDS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    return subprocess.call([sys.executable, '-B', str(ROOT / COMMANDS[args.tool]), *arguments], cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
