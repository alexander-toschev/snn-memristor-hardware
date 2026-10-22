"""Compare completed, native-converter-corrected MNIST transfer results."""
from __future__ import annotations

import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiment_common.durable import atomic_json, sha
from experiment_common.selective_metrology import object_dict, number

BASE = ROOT / 'results/mnist_tiled_transfer_20261022'
HW_ACQUISITION = 'cb53be86-ae2b-485f-bf40-3b7d994e7a50'
SIM_ACQUISITION = '0c7d20fc-045f-4f27-a031-cdcf1889500c'
HW_INFERENCE = '0d2465d8-7095-43bd-a2b5-55a72b4c4042'
SIM_INFERENCE = '0b909723-b4b3-4e4c-b704-a03fea42404c'


def read(path: Path) -> dict[str, object]:
    return object_dict(json.loads(path.read_text(encoding='utf-8')))


def main() -> int:
    summaries = {name: read(BASE/sid/'summary.json') for name, sid in (
        ('hardware', HW_INFERENCE), ('simulator', SIM_INFERENCE))}
    for name, sid in (('hardware', HW_INFERENCE), ('simulator', SIM_INFERENCE)):
        manifest = read(BASE/sid/'manifest.json')
        accuracy = object_dict(summaries[name]['accuracy'])
        if manifest['status'] != 'completed' or accuracy['test_images'] != 10000 or not manifest.get('conversion'):
            raise ValueError('Completed corrected inference on all10000 required')
        for relative, digest in object_dict(manifest['result_sha256']).items():
            if sha(BASE/sid/relative) != digest:
                raise ValueError('Inference evidence checksum mismatch')
    cells: dict[tuple[int, int], list[dict[str, object]]] = {}
    for path in sorted((BASE/HW_ACQUISITION).glob('load_*_programming.json')):
        decoded: object = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(decoded, list):
            raise ValueError('Programming records must be a list')
        for value in decoded:
            row = object_dict(value)
            key = (int(number(row['wl'])), int(number(row['bl'])))
            cells.setdefault(key, []).append(row)
    cell_report: list[dict[str, object]] = []
    for (wl, bl), rows in sorted(cells.items()):
        errors = np.asarray([number(r['error_g_s']) for r in rows], dtype=np.float64)*1e6
        cell_report.append(dict(wl=wl, bl=bl, writes=len(rows), misses=sum(r['within_tolerance'] is not True for r in rows),
            mae_us=float(np.mean(abs(errors))), rmse_us=float(np.sqrt(np.mean(errors**2))), maximum_abs_error_us=float(max(abs(errors)))))
    sim = np.load(BASE/SIM_INFERENCE/'inference_lut.npy', allow_pickle=False)
    hw = np.load(BASE/HW_INFERENCE/'inference_lut.npy', allow_pickle=False)
    diagnostic: dict[str, object] = dict(unweighted_unique_binary_masks=sim.shape[0], filters=64,
        pearson_flat=float(np.corrcoef(sim.ravel(), hw.ravel())[0, 1]),
        note='Descriptive over unique masks, not weighted by image frequencies; no calibration fitted to test.')
    with np.load(BASE/SIM_INFERENCE/'predictions.npz', allow_pickle=False) as sp, np.load(BASE/HW_INFERENCE/'predictions.npz', allow_pickle=False) as hp:
        if not np.array_equal(sp['labels'], hp['labels']):
            raise ValueError('Test labels/order differ')
        sc, hc = sp['strict_correct'], hp['strict_correct']
        diagnostic['paired_predictions'] = dict(same_test_order=True,
            argmax_agreement=float(np.mean(sp['argmax'] == hp['argmax'])),
            both_strict_correct=int(np.sum(sc & hc)), hardware_only_correct=int(np.sum(hc & ~sc)),
            simulator_only_correct=int(np.sum(sc & ~hc)), neither_strict_correct=int(np.sum(~hc & ~sc)))
    for name, values in (('simulator', sim), ('hardware', hw)):
        diagnostic[name] = dict(absolute_current_quantiles_50_90_99_max=np.quantile(abs(values), [.5, .9, .99, 1]).tolist(),
            zero_fraction=float(np.mean(values == 0)), negative_fraction=float(np.mean(values < 0)))
    report: dict[str, object] = dict(schema='mnist12-transfer-comparison-v1',
        scope='Frozen measured partial MVM operators; CPU sum before LIF; CPU LIF and frozen source head.',
        live_hardware_per_image=False, new_stdp=False, new_readout=False,
        calibration='Fixed source physical scale330ohm and native DAC; no fitted per-cell calibration matrix or readout adaptation.',
        original_source_strict_accuracy=.6636, results=summaries,
        hardware_acquisition=read(BASE/HW_ACQUISITION/'summary.json'),
        simulator_acquisition=read(BASE/SIM_ACQUISITION/'summary.json'),
        acquisition_accuracy_superseded='Use corrected inference results above, not earlier acquisition accuracy.',
        operator_diagnostics=diagnostic, physical_cell_programming=cell_report,
        sources={sid: dict(manifest_sha256=sha(BASE/sid/'manifest.json'), summary_sha256=sha(BASE/sid/'summary.json'))
                 for sid in (HW_ACQUISITION, SIM_ACQUISITION, HW_INFERENCE, SIM_INFERENCE)})
    out = ROOT/'reports/2026-10-22/mnist-tiled-transfer'
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'RESULTS.json', report)
    rows_html = ''
    for name, label in (('simulator', 'Симулятор'), ('hardware', 'Плата')):
        acc = object_dict(summaries[name]['accuracy'])
        rows_html += f'<tr><td>{label}</td><td>{100*number(acc["strict_accuracy"]):.2f}%</td><td>{int(number(acc["strict_correct"]))}/10000</td><td>{100*number(acc["conventional_argmax_accuracy"]):.2f}%</td></tr>'
    cell_html = ''.join(f'<tr><td>WL{r["wl"]}/BL{r["bl"]}</td><td>{r["misses"]}/{r["writes"]}</td><td>{number(r["mae_us"]):.2f}</td></tr>' for r in cell_report)
    page = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>MNIST: 12 блоков</title>'
        '<style>body{font:16px system-ui;max-width:1050px;margin:2em auto;padding:1em}table{border-collapse:collapse}td,th{padding:.5em 1em;border:1px solid #bbb;text-align:left}pre{white-space:pre-wrap}</style>'
        '<h1>MNIST64: перенос обученных STDP-весов через 12 блоков</h1>'
        '<p>89 загрузок,18 пар,36 физических ячеек;1600 коэффициентов.10 000 test; head/scaler заморожены.</p>'
        '<table><tr><th>Источник</th><th>Strict accuracy</th><th>Верно</th><th>Argmax accuracy</th></tr>'+rows_html+'</table>'
        '<p>Исходное обучение в прежней раскладке:66,36%. Текущие числа относятся к переносу через12блоков.</p>'
        '<p>Это гибридный инференс по замороженным измеренным откликам: MemriBoard выполняет запись и частичные MVM; CPU складывает результаты до LIF и применяет прежний классификатор. Нет live-запроса платы для каждой картинки и нет аналогового LIF.</p>'
        '<p>Все89 аппаратных загрузок прошли заданную проверку повторяемости. При этом2295из3200 записей ячеек не попали в допуск G;905попали. Это количество записей при многократном использовании36ячеек. Ошибки не заменялись идеальными весами.</p>'
        '<p>Использована фиксированная шкала исходной модели; индивидуальная калибровочная матрица и переобучение head не применялись. Это результат прямого переноса, а не оценка предельной достижимой точности платы. Округление штатного конвертера исправлено пересчётом каждого raw-отсчёта; аппаратный сбор не повторялся.</p>'
        '<p>Близкая суммарная accuracy не доказывает совпадение MVM или верное воспроизведение каждого веса. Отклики различаются по масштабу и знакам; отдельные прогнозы и парное сравнение сохранены ниже.</p>'
        '<h2>Ошибки записи по физическим ячейкам</h2><table><tr><th>Ячейка</th><th>Промахов/записей</th><th>Средняя абсолютная ошибка,мкСм</th></tr>'+cell_html+'</table>'
        '<details><summary>Данные и происхождение</summary><pre>'+html.escape(json.dumps(report, ensure_ascii=False, indent=2))+'</pre></details></html>')
    (out/'RESULTS.html').write_text(page, encoding='utf-8')
    print(out.relative_to(ROOT).as_posix())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
