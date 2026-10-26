"""Show the recorded MNIST selection separately from accumulated cell problems."""
from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'reports/2026-10-26/cell-problems'
SELECTION = ROOT / 'reports/2026-10-25/mnist-experiment-report/CELLS.csv'
LABELS = {
    'used': ('✓', 'Взяли в MNIST'),
    'conductance': ('G', 'Не взяли: проводимость вне допуска'),
    'partner': ('P', 'Не взяли: исключён партнёр'),
    'mvm': ('M', 'Не взяли: дрейф MVM пары'),
    'unqualified': ('—', 'Не проверяли в этом отборе'),
}
CSS = '''<style id="mnist-selection-style">
.map td.selection-used{background:#c9efd5;box-shadow:inset 0 0 0 2px #238144}
.map td.selection-conductance{background:#ffe0b2}.map td.selection-partner{background:#fff4cd}
.map td.selection-mvm{background:#ffd5d9}.map td.selection-unqualified{background:#edf0f4}
.selection-marker{display:block;font-size:14px;font-weight:800;margin:2px 0 4px}.selection-used .selection-marker{color:#12692f}
.cell-tip{display:none;position:fixed;right:20px;bottom:20px;z-index:20;width:460px;max-width:calc(100vw - 70px);padding:18px;background:#fff;color:#172638;border:2px solid #4c627b;border-radius:10px;box-shadow:0 6px 30px #192c3b40;font:15px/1.5 system-ui,Arial;text-align:left;pointer-events:none}
.map a:hover .cell-tip,.map a:focus .cell-tip{display:block}.tip-title{display:block;font-size:19px;font-weight:750}.tip-status{display:block;font-weight:700;margin:6px 0}.tip-context{display:block;color:#526577;font-size:12px;margin-top:10px}
.selection-legend{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}.selection-legend span{padding:7px 12px;border:1px solid #c4cdd6;border-radius:5px}
.selected-swatch{background:#c9efd5}.conductance-swatch{background:#ffe0b2}.partner-swatch{background:#fff4cd}.mvm-swatch{background:#ffd5d9}.unqualified-swatch{background:#edf0f4}
.selection-card{padding:14px 18px;border:1px solid #b5c6d4;border-radius:6px;background:#f3f7fb}.selection-card.selection-used{background:#e3f6e8}
@media print{.cell-tip{display:none!important}}
</style>'''
INTRO = '''<h1>Какие ячейки взяли в MNIST и почему</h1>
<p><b>Цвет фона — результат отбора для последнего MNIST-прогона 23.10.2026.</b> Наведите указатель на адрес: появятся решение, причина и партнёр. Нажмите на ячейку, чтобы открыть подробности; на телефоне используйте нажатие.</p>
<div class="selection-legend"><span class="selected-swatch">✓ Взяли — 24</span><span class="conductance-swatch">G Не достигнута проводимость — 7</span><span class="partner-swatch">P Исключён партнёр — 5</span><span class="mvm-swatch">M Дрейф MVM пары — 8</span><span class="unqualified-swatch">— Не проверяли в этом отборе — 212</span></div>
<p class="notice"><b>Серые ячейки не объявлены неисправными:</b> они не входили в проверку 44 кандидатов после записи для этого переноса. Зелёные использовались в 134 загрузках MVM; это не означает точную запись каждой цели G. Мелкие цветные коды внутри ячейки — зафиксированные проблемы из всех опытов, а не причина последнего исключения.</p>
'''


def annotate() -> None:
    with SELECTION.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    selected = {(int(r['wl']), int(r['bl'])): r for r in rows}
    expected = {'used': 24, 'conductance': 7, 'partner': 5, 'mvm': 8, 'unqualified': 212}
    if len(selected) != 256 or dict(Counter(r['category'] for r in rows)) != expected:
        raise ValueError('Selection evidence does not match the reviewed map')
    path = FOLDER / 'MAP.html'
    page = path.read_text(encoding='utf-8')
    page = re.sub(r'<style id="mnist-selection-style">.*?</style>', '', page, flags=re.S)
    page = page.replace('</head>', CSS + '</head>')
    page = re.sub(r'<h1>.*?(?=<p><a href="REPORT.md">)', INTRO, page, count=1, flags=re.S)
    page = page.replace('Нажмите адрес. Цветные метки перечисляют разные зафиксированные проблемы и не ранжируют тяжесть.',
                        'Коды ниже описывают наблюдавшиеся проблемы. Решение об использовании показывает фон ячейки; точная причина — в подсказке.')

    def cell(match: re.Match[str]) -> str:
        wl, bl = int(match[1]), int(match[2])
        row = selected[(wl, bl)]
        category = row['category']
        marker, label = LABELS[category]
        chips = ''.join(re.findall(r'<span class="chip [A-Z]">[A-Z]</span>', match[0]))
        context = f'Блок {row["block"]}; партнёр {row["partner"]}.' if row['partner'] else 'В отбор после записи не включалась.'
        tooltip = (f'<span class="cell-tip" id="tip-{wl}-{bl}" role="tooltip">'
                   f'<span class="tip-title">WL{wl} / BL{bl}</span>'
                   f'<span class="tip-status">{html.escape(label)}</span>'
                   f'<span>{html.escape(row["reason"])}</span>'
                   f'<span class="tip-context">{html.escape(context)} Отбор MNIST: 23.10.2026.</span></span>')
        return (f'<td class="selection-{category}"><a href="#cell-{wl}-{bl}" '
                f'aria-label="WL{wl} BL{bl}: {html.escape(label, quote=True)}" aria-describedby="tip-{wl}-{bl}">'
                f'<b>{wl},{bl}</b><span class="selection-marker">{marker}</span>{chips}{tooltip}</a></td>')

    page, count = re.subn(r'<td(?: class="selection-[a-z]+")?><a href="#cell-(\d+)-(\d+)"[^>]*>.*?</a></td>', cell, page, flags=re.S)
    if count != 256:
        raise ValueError(f'Expected 256 map entries, found {count}')
    page = re.sub(r'<div class="selection-card selection-[a-z]+">.*?</div>', '', page, flags=re.S)
    for (wl, bl), row in selected.items():
        label = LABELS[row['category']][1]
        card = (f'<div class="selection-card selection-{row["category"]}"><b>{html.escape(label)}</b>'
                f'<p>{html.escape(row["reason"])}</p><small>Блок: {html.escape(row["block"] or "—")}; '
                f'партнёр: {html.escape(row["partner"] or "—")}. '
                '<a href="../../../reports/2026-10-25/mnist-experiment-report/CELLS.csv">Основание отбора</a></small></div>')
        needle = f'<section class="cell" id="cell-{wl}-{bl}">'
        page = page.replace(needle, needle + card)
    path.write_text(page, encoding='utf-8')


if __name__ == '__main__':
    annotate()
