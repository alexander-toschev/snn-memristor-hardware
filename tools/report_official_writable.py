"""Offline HW/SIM switching report; no Connector, writes only requested output."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_common.selective_metrology import number, object_dict


class Session(TypedDict):
    summary: dict[str, object]
    manifest: dict[str, object]
    cells: dict[str, dict[str, object]]
    observations_sha256: str


def read_json(path: Path) -> dict[str, object]:
    return object_dict(json.loads(path.read_text(encoding='utf-8-sig')))


def load_session(path: Path) -> Session:
    manifest = read_json(path / 'manifest.json')
    summary = read_json(path / 'summary.json')
    if manifest.get('status') != 'completed' or not manifest.get('port_closed'):
        raise ValueError('Only terminal completed sessions with closed port can be compared')
    observations = path / 'observations.json'
    raw = read_json(observations).get('cells')
    if not isinstance(raw, list):
        raise ValueError('Cell observations required')
    cells: dict[str, dict[str, object]] = {}
    for item in raw:
        row = object_dict(item)
        wl, bl = row.get('wl'), row.get('bl')
        if type(wl) is not int or type(bl) is not int or not 0 <= wl < 8 or not 0 <= bl < 32:
            raise ValueError('Invalid cell coordinate')
        key = f'{wl},{bl}'
        if key in cells:
            raise ValueError('Duplicate cell')
        cells[key] = row
    digest = hashlib.sha256(observations.read_bytes()).hexdigest()
    if object_dict(manifest['result_sha256']).get('observations.json') != digest:
        raise ValueError('Observations do not match terminal manifest hash')
    return dict(summary=summary, manifest=manifest, cells=cells, observations_sha256=digest)


def stage(row: dict[str, object], name: str) -> dict[str, object]:
    values = row.get('stages', [])
    if isinstance(values, list):
        for value in values:
            parsed = object_dict(value)
            if parsed.get('name') == name:
                return parsed
    return {}


def stage_condition(row: dict[str, object], name: str) -> bool:
    value = stage(row, name).get('last_r_ohm')
    if value is None:
        return False
    resistance = number(value)
    return resistance <= 296 if name == 'SET' else resistance >= 2998


def counts(session: Session, priority: set[str]) -> dict[str, object]:
    rows = session['cells']
    passed = {key for key, row in rows.items() if row.get('observed_docx_switching') is True}
    result: dict[str, object] = dict(measured=len(rows), switching=len(passed), priority44_switching=len(passed & priority),
                outside44_switching=len(passed - priority),
                measurement_errors=sum(bool(r.get('error')) for r in rows.values()),
                guard_modified=sum(r.get('guard_modified_protocol') is True for r in rows.values()),
                SET_target_reached=sum(stage_condition(r, 'SET') for r in rows.values()),
                RESET_target_reached=sum(stage_condition(r, 'RESET') for r in rows.values()),
                rmin_failed=sum('min_r_ohm' in r and number(r['min_r_ohm']) <= 100 for r in rows.values()),
                ratio_failed=sum('ratio' in r and number(r['ratio']) <= 2 for r in rows.values()))
    for name in ('SET', 'RESET'):
        result[name+'_already_satisfied'] = sum(stage(r, name).get('stopped_by_docx_resistance') is True
                and stage(r, name).get('submitted_tasks') == 0 for r in rows.values())
    return result


def render(hardware_path: Path, simulator_path: Path, output: Path, priority_path: Path) -> None:
    hw, sim = load_session(hardware_path), load_session(simulator_path)
    if hw['manifest'].get('source') != 'hardware' or sim['manifest'].get('source') != 'simulator':
        raise ValueError('Expected hardware and simulator session roles')
    if hw['summary'].get('expected_cells') != sim['summary'].get('expected_cells'):
        raise ValueError('Different intended cell scope')
    hw_protocol, sim_protocol = object_dict(hw['manifest']['protocol']), object_dict(sim['manifest']['protocol'])
    for key in ('schema', 'official_json_sha256', 'stop_resistance_ohm', 'requested_width_s',
                'current_limits_a', 'docx_classification', 'gui_classification', 'cells'):
        if hw_protocol.get(key) != sim_protocol.get(key):
            raise ValueError('Protocol mismatch: '+key)
    hw_sources, sim_sources = object_dict(hw['manifest']['source_sha256']), object_dict(sim['manifest']['source_sha256'])
    for path in ('experiments/official_writable_cells_20261114/experiment.py',
                 'experiment_common/official_writable.py', 'experiment_common/adaptive_programming.py',
                 'experiment_common/native_memriboard.py', 'configs/raspberry_board.ini'):
        if not isinstance(hw_sources.get(path), str) or hw_sources.get(path) != sim_sources.get(path):
            raise ValueError('Executed source/INI mismatch: '+path)
    raw_priority: object = json.loads(priority_path.read_text(encoding='utf-8-sig'))
    if not isinstance(raw_priority, list):
        raise ValueError('Priority array required')
    priority = {f'{v[0]},{v[1]}' for v in raw_priority if isinstance(v, list) and len(v) == 2}
    if len(priority) != 44:
        raise ValueError('Exactly44 previous cells required')
    records: list[dict[str, object]] = []
    for wl in range(8):
        for bl in range(32):
            key = f'{wl},{bl}'
            records.append(dict(wl=wl, bl=bl, priority44=key in priority,
                                HW=hw['cells'].get(key), SIM=sim['cells'].get(key)))
    count_hw, count_sim = counts(hw, priority), counts(sim, priority)
    payload: dict[str, object] = dict(HW=count_hw, SIM=count_sim, cells=records,
        session_hardware=hw['summary'].get('session_id'), session_simulator=sim['summary'].get('session_id'),
        hardware_observations_sha256=hw['observations_sha256'], simulator_observations_sha256=sim['observations_sha256'],
        protocol=hw_protocol, initial_states_equal=False, mnist_qualified=False)
    output.mkdir(parents=True, exist_ok=True)
    (output/'COMPARISON.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    columns = ['WL', 'BL', 'previous44']
    metrics = ('complete', 'observed_docx_switching', 'observed_gui_switching', 'min_r_ohm', 'max_r_ohm',
               'ratio', 'skipped_current_guard', 'error')
    for label in ('HW', 'SIM'):
        columns += [label+'_'+m for m in metrics] + [label+'_SET_reached', label+'_RESET_reached', label+'_SET_tasks', label+'_RESET_tasks', label+'_SET_stop_flag', label+'_RESET_stop_flag']
    with (output/'CELLS.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            flat: dict[str, object] = dict(WL=record['wl'], BL=record['bl'], previous44=record['priority44'])
            for label in ('HW', 'SIM'):
                value = record[label]
                cell = {} if value is None else object_dict(value)
                flat.update({label+'_'+m: cell.get(m) for m in metrics})
                for name in ('SET', 'RESET'):
                    flat[label+'_'+name+'_reached'] = stage_condition(cell, name)
                    flat[label+'_'+name+'_tasks'] = stage(cell, name).get('submitted_tasks')
                    flat[label+'_'+name+'_stop_flag'] = stage(cell, name).get('stopped_by_docx_resistance')
            writer.writerow(flat)
    md = ['# Резистивное переключение: плата и симулятор', '',
          '| Показатель | Плата | Симулятор |', '|---|---:|---:|']
    labels = {'measured':'Получены записи ячеек', 'switching':'Критерий переключения выполнен',
              'priority44_switching':'Из прежних44', 'outside44_switching':'За пределами прежних44',
              'measurement_errors':'Ошибки измерения', 'guard_modified':'Ячейки с пропусками по току',
              'SET_target_reached':'SET: условие296 Ом в конце этапа (включая исходное)', 'RESET_target_reached':'RESET: условие2998 Ом в конце этапа (включая исходное)',
              'SET_already_satisfied':'SET: условие уже выполнено, задач не было',
              'RESET_already_satisfied':'RESET: условие уже выполнено, задач не было',
              'rmin_failed':'Не выполнено Rmin>100 Ом', 'ratio_failed':'Не выполнено Rmax/Rmin>2'}
    md += [f'| {label} | {count_hw[key]} | {count_sim[key]} |' for key, label in labels.items()]
    md += ['', 'Одинаковые форма команд и критерий, но разные исходные состояния: симулятор создан заново, плата сохраняет физическое состояние.', '',
           'Зеленый цвет означает только Rmin>100 Ом и Rmax/Rmin>2. Это не допуск к MNIST и не проверка группового MVM. Пропуски по току изменяют протокол; физическая длительность импульсов неизвестна.', '',
           'Наведение или нажатие на ячейку открывает Rmin/Rmax, отношение, ошибки и итоги SET/RESET для платы и симулятора. Обводка отмечает прежние44 ячейки.']
    (output/'RESULTS.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    data_json = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    document = r'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Карта переключения256 ячеек</title>
<style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 16px;color:#17242b}table{border-collapse:separate;border-spacing:3px}td{min-width:27px;height:34px;text-align:center;border:1px solid #aaa;cursor:pointer;font-size:11px}td.old{outline:2px solid #334f75}th{font-size:12px}.green{background:#afe3b0}.orange{background:#ffd591}.red{background:#f6aaa7}.gray{background:#ddd}#details{white-space:pre-wrap;background:#f3f6f8;padding:16px;min-height:180px;font:13px monospace}.scroll{overflow:auto}.cards{display:flex;gap:30px;flex-wrap:wrap}.card{padding:15px;background:#f2f6fa;border-radius:8px}button{padding:7px}p{line-height:1.5}</style>
<h1>Плата и симулятор: переключение256 ячеек</h1><p><b>Зеленый: наблюдаемый критерий Rmin&gt;100 Ом и Rmax/Rmin&gt;2.</b> Оранжевый: критерий не выполнен. Красный: ошибка измерения. Серый: нет данных. Обводка: прежние44. Цвет не означает пригодность для MNIST.</p>
<p>Одинаковый протокол, разные исходные состояния. Физическая длительность импульсов неизвестна. Пропуски по току изменяют протокол; подробности видны по каждой ячейке.</p><div class="cards" id="counts"></div><p>Окраска по: <button id="hw">плате</button> <button id="sim">симулятору</button>. Наведите курсор или нажмите ячейку.</p><div class="scroll"><table id="map"></table></div><pre id="details">Выберите ячейку</pre>
<script>const data=DATA;let selected='HW';const names={HW:'Плата',SIM:'Симулятор'};
document.getElementById('counts').innerHTML=['HW','SIM'].map(k=>'<div class="card"><b>'+names[k]+'</b><br>Критерий: '+data[k].switching+'/256<br>Из прежних44: '+data[k].priority44_switching+'<br>Вне прежних44: '+data[k].outside44_switching+'<br>Ошибки: '+data[k].measurement_errors+'; пропуски по току: '+data[k].guard_modified+'</div>').join('');
function info(r){let s='WL'+r.wl+' / BL'+r.bl+'; прежние44: '+(r.priority44?'да':'нет')+'\n';for(const k of ['HW','SIM']){let c=r[k];s+='\n'+names[k]+':\n';if(!c){s+='Нет измерений\n';continue;}s+='Критерий: '+c.observed_docx_switching+'; GUI>=2: '+c.observed_gui_switching+'\nRmin='+c.min_r_ohm+' Ом, Rmax='+c.max_r_ohm+' Ом; отношение='+c.ratio+'\nПропуски по току: '+c.skipped_current_guard+'\nПричина: '+(c.error?'ошибка измерения':c.min_r_ohm<=100?'Rmin не выше100 Ом':c.ratio<=2?'отношение не выше2':'критерий выполнен')+'\nОшибка: '+(c.error||'нет')+'\n';for(const t of c.stages||[])s+=t.name+': задач '+t.submitted_tasks+'/'+t.maximum_tasks+', условие в конце этапа '+(t.name==='SET'?t.last_r_ohm<=296:t.name==='RESET'?t.last_r_ohm>=2998:'не применяется')+', raw stop flag='+t.stopped_by_docx_resistance+(t.stopped_by_docx_resistance&&t.submitted_tasks===0?' (уже выполнено до этапа)':'')+', Rпосл='+t.last_r_ohm+' Ом, пропусков '+t.skipped_current_guard+'\n';}return s;}
function draw(){const table=document.getElementById('map');table.textContent='';let head=document.createElement('tr');head.innerHTML='<th>WL / BL</th>'+Array.from({length:32},(_,i)=>'<th>'+i+'</th>').join('');table.append(head);for(let w=0;w<8;w++){let tr=document.createElement('tr');tr.innerHTML='<th>'+w+'</th>';for(let b=0;b<32;b++){const r=data.cells[w*32+b],c=r[selected],td=document.createElement('td');td.className=(!c?'gray':c.error?'red':c.observed_docx_switching?'green':'orange')+(r.priority44?' old':'');td.textContent=b;td.title=info(r);td.tabIndex=0;const show=()=>document.getElementById('details').textContent=info(r);td.onmouseenter=show;td.onclick=show;td.onfocus=show;tr.append(td);}table.append(tr);}}
document.getElementById('hw').onclick=()=>{selected='HW';draw()};document.getElementById('sim').onclick=()=>{selected='SIM';draw()};draw();</script></html>'''
    (output/'MAP.html').write_text(document.replace('DATA', data_json), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', type=Path, required=True)
    parser.add_argument('--simulator', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--priority', type=Path, default=ROOT/'experiments/official_writable_cells_20261114/priority44.json')
    args = parser.parse_args()
    render(args.hardware, args.simulator, args.out, args.priority)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
