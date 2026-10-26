"""Build the public, static cell-problem map; no board or private runtime access."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/2026-10-26/cell-problems'
REPO_URL = 'https://github.com/alexander-toschev/snn-memristor-hardware'


def build(output: Path) -> None:
    """Export only the reviewed report and pin evidence links to its Git revision."""
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Expected a full Git revision')
    output.mkdir(parents=True, exist_ok=True)
    page = (SOURCE / 'MAP.html').read_text(encoding='utf-8')
    page = page.replace('href="../../../', f'href="{REPO_URL}/blob/{revision}/')
    if 'href="../../../' in page or '<script' in page.lower():
        raise ValueError('Unexpected relative evidence path or executable script')
    (output / 'index.html').write_text(page, encoding='utf-8')
    description = (SOURCE / 'REPORT.md').read_text(encoding='utf-8').replace('MAP.html', 'index.html')
    (output / 'REPORT.md').write_text(description, encoding='utf-8')
    for name in ['CELLS.csv', 'PROBLEMS.json']:
        (output / name).write_bytes((SOURCE / name).read_bytes())
    (output / '.nojekyll').write_bytes(b'')
    (output / 'README.md').write_text(
        '# MemriBoard: recorded cell problems\n\n'
        'Static report for all 256 physical addresses. Select an address on the map '
        'to see the recorded problems and supporting measurements.\n\n'
        f'Source: [{revision}]({REPO_URL}/tree/{revision}/reports/2026-10-26/cell-problems).\n\n'
        'The site contains only the public report. Evidence links point to the source repository.\n'
        'Rebuild with `python tools/build_cell_problems_pages.py --output <site-directory>` '
        'from the research checkout, then commit and push the generated files '
        'to `codex/cell-problems-site` with the standard secret-scanning hooks.\n',
        encoding='utf-8',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.tmp/cell-problems-pages')
    build(parser.parse_args().output.resolve())


if __name__ == '__main__':
    main()
