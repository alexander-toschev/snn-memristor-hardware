"""Read-only screening map, all 256 cells. Hardware access only via public Connector."""
import argparse
import csv
import html
import json
import math
import statistics
import time
import uuid

from read_all import ROOT, INI, EXPECTED_BOARD, local_path, preflight, native_connector, now, sha256, save_json, json_value
from native_conversions import voltage, readback, mvm_output
from experiment_hardware import launch_provenance, READ_ALL_SHA256

CONFIG = "configs/cell_map.json"
LOCK = "results/read_all/.hardware.lock"
CELLS = tuple((wl, bl) for wl in range(8) for bl in range(32))
CRITERIA_KEYS = {"noise_floor_adc", "minimum_snr", "maximum_cv_high", "linearity_fraction",
                 "linearity_se_multiplier", "linearity_floor_adc", "drift_fraction", "drift_se_multiplier"}


def load_criteria(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    criteria = data["criteria"]
    if data["schema_version"] != 1 or set(criteria) != CRITERIA_KEYS:
        raise ValueError("Unsupported cell-map criteria schema")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in criteria.values()):
        raise ValueError("Criteria must be finite positive numbers")
    return data


def stats(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sd": statistics.stdev(values) if len(values) > 1 else None}


def classify(cell, zero, criteria):
    """Signed ADC signals; no clipped baseline, fitted gain, or invented missing samples."""
    result = {"wl": cell["wl"], "bl": cell["bl"], "status": "unmeasured",
              "reasons": list(cell["reasons"]), "metrics": {}}
    groups = {name: stats(cell[name]) for name in ("pre_g", "post_g", "low_adc", "high_adc")}
    groups["zero_adc"] = stats(zero)
    result["metrics"].update(groups)
    if result["reasons"]:
        result["status"] = "reject"
    if any(groups[name]["n"] != count for name, count in
           (("pre_g",3),("post_g",3),("low_adc",5),("high_adc",5),("zero_adc",5))):
        result["reasons"].append("incomplete_measurements")
        return result
    pre, post, low, high, base = (groups[k] for k in ("pre_g","post_g","low_adc","high_adc","zero_adc"))
    signals = {}
    for label, group in (("low",low),("high",high)):
        signal = group["mean"] - base["mean"]
        signal_sd = math.sqrt(group["sd"]**2 + base["sd"]**2)
        noise = max(criteria["noise_floor_adc"], signal_sd)
        threshold = criteria["minimum_snr"]*noise
        signals[label] = signal
        result["metrics"].update({f"signal_{label}_adc": signal, f"signal_{label}_sd_adc": signal_sd,
                                  f"signal_{label}_threshold_adc": threshold, f"snr_{label}": signal/noise})
        if signal <= 0 or signal < threshold:
            result["reasons"].append(f"weak_or_nonpositive_signal_{label}")
    high_sd = result["metrics"]["signal_high_sd_adc"]
    cv = high_sd / signals["high"] if signals["high"] > 0 else None
    result["metrics"]["cv_high"] = cv
    if cv is None or cv > criteria["maximum_cv_high"]:
        result["reasons"].append("unstable_high_signal")
    epsilon = high["mean"] - 2*low["mean"] + base["mean"]
    se = math.sqrt(high["sd"]**2/5 + 4*low["sd"]**2/5 + base["sd"]**2/5)
    tolerance = max(criteria["linearity_fraction"]*abs(signals["high"]),
                    criteria["linearity_se_multiplier"]*se, criteria["linearity_floor_adc"])
    delta = post["mean"] - pre["mean"]
    drift_se = math.sqrt(pre["sd"]**2/3 + post["sd"]**2/3)
    fractional = delta/pre["mean"]
    drift_tolerance = max(criteria["drift_fraction"], criteria["drift_se_multiplier"]*drift_se/pre["mean"])
    result["metrics"].update(linearity_epsilon_adc=epsilon, linearity_se_adc=se, linearity_tolerance_adc=tolerance,
        g_delta_siemens=delta, g_delta_se_siemens=drift_se, g_fractional_delta=fractional, g_fractional_tolerance=drift_tolerance)
    if abs(epsilon) > tolerance:
        result["reasons"].append("nonlinear_zero_adjusted_response")
    if abs(fractional) > drift_tolerance:
        result["reasons"].append("pre_post_conductance_difference")
    result["status"] = "reject" if result["reasons"] else "candidate"
    return result


def export_map(output, cells, run_status):
    save_json(output / "cells.json", cells)
    flattened=[]
    for row in cells:
        values={k:row[k] for k in ('wl','bl','status')}
        values['reasons']='; '.join(row['reasons'])
        for name,value in row['metrics'].items():
            if isinstance(value,dict): values.update({f'{name}_{key}':item for key,item in value.items()})
            else: values[name]=value
        flattened.append(values)
    fields=['wl','bl','status','reasons']+sorted(set().union(*(row.keys() for row in flattened))-{'wl','bl','status','reasons'})
    with (output / "cell_map.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)
    with (output / "candidate_cells.csv").open("x", encoding="utf-8", newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(('wl','bl'))
        if run_status=='completed':
            writer.writerows((row['wl'],row['bl']) for row in cells if row['status']=='candidate')
    colors = {"candidate":"#a7dba1", "reject":"#efaaaa", "unmeasured":"#dddddd"}
    page = ['<!doctype html><meta charset="utf-8"><title>Cell screening map</title>',
            '<h1>Existing-conductance screening, not programming qualification</h1>',
            f'<h2>Run status: {html.escape(run_status)}. Mask usable: {run_status=="completed"}.</h2>',
            '<p>Green: candidate. Red: reject. Grey: unmeasured. Provisional criteria; see criteria.json and raw.jsonl.</p>',
            '<table border="1" cellpadding="5"><tr><th>WL / BL</th>'+''.join(f'<th>{bl}</th>' for bl in range(32))+'</tr>']
    for wl in range(8):
        page.append(f'<tr><th>{wl}</th>')
        for row in cells[wl*32:(wl+1)*32]:
            title=html.escape(('; '.join(row['reasons']) or 'candidate under provisional criteria')+' '+json.dumps(row['metrics']),quote=True)
            page.append(f'<td style="background:{colors[row["status"]]}" title="{title}">{row["status"][0].upper()}</td>')
        page.append('</tr>')
    page.append('</table><p>candidate_cells.csv is a GUI mask for this conductance state and 0.05–0.1 V only. Programming invalidates the map. Pair/selected-mask MVM must be checked separately. No proof of programmability or CIFAR inference accuracy.</p>')
    (output / "map.html").write_text('\n'.join(page),encoding="utf-8")


def collect(connector, converters, output, session_id, criteria):
    board = EXPECTED_BOARD
    state = {(wl,bl):dict(wl=wl,bl=bl,pre_g=[],post_g=[],low_adc=[],high_adc=[],reasons=[]) for wl,bl in CELLS}
    zeros = {wl:[] for wl in range(8)}
    summary = {"status":"failed", "attempted":0, "responded":0, "skipped_mvm":0,
               "maximum_impacts":4136,"weights_written":False,"screening_validated":False}
    try:
        with (output/'raw.jsonl').open('x',encoding='utf-8') as stream:
            def emit(record):
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                stream.flush()

            def impact(mode, wl, bl, level, **metadata):
                task_id=summary['attempted']+1
                values = [level if i==bl else 0.0 for i in range(32)] if mode==10 else level
                task = {'mode_flag':10,'vol':values,'id':0,'wl':wl} if mode==10 else {
                    'mode_flag':9,'vol':level,'id':task_id,'wl':wl,'bl':bl,'pulse_width':0.0,'sign':0}
                dac = [voltage(v,board,converters)['dac_code'] for v in values] if mode==10 else voltage(level,board,converters)['dac_code']
                row = dict(session_id=session_id,source='hardware',task_id=task_id,timestamp_utc=now(),
                           monotonic_time_s=time.monotonic(),command_si=task,vol_dac=dac,**metadata)
                emit({**row,'event':'request'})
                summary['attempted']+=1
                started=time.monotonic()
                try:
                    result=connector.impact({**task,'vol':list(values) if mode==10 else values})
                    row['native_result']=json_value(result)
                    if not isinstance(result,(list,tuple)) or len(result)!=3:
                        raise ValueError('Expected native (R,id,ADC)')
                    native_r, echoed, adc=result
                    row.update(native_r=json_value(native_r),returned_id=json_value(echoed),raw_adc=json_value(adc))
                    if type(echoed) is not int or echoed!=0:
                        raise ValueError('Native ID must be zero for mode9/GUI mode10')
                    if type(adc) is not int or not 0<=adc<2**board['adc_bit']-1:
                        raise ValueError('ADC protocol error or saturation')
                    summary['responded']+=1
                    row['status']='ok'
                    if mode==9:
                        try:
                            row.update(readback(adc,board,converters,read_voltage=.1))
                        except ValueError as exc:
                            row.update(status='invalid_readback_model',model_error=str(exc))
                    else:
                        row['output_voltage_v']=mvm_output(adc,board,converters)
                        if row['output_voltage_v']>.8*board['vol_ref_adc']:
                            raise ValueError('Observed output exceeds 80% ADC reference')
                    return row
                except (Exception,KeyboardInterrupt) as exc:
                    row.update(status='error',error=f'{type(exc).__name__}: {exc}')
                    raise
                finally:
                    emit({**row,'event':'result','latency_s':time.monotonic()-started})

            if not connector.open_port():
                raise ConnectionError('Connector.open_port returned false')
            for stage in ('pre','post'):
                for wl,bl in CELLS:
                    cell=state[wl,bl]
                    for repeat in range(3):
                        row=impact(9,wl,bl,.1,stage=stage,repeat=repeat)
                        if row['status']=='invalid_readback_model':
                            reason=f'invalid_{stage}_readback_model'
                            if reason not in cell['reasons']: cell['reasons'].append(reason)
                        else:
                            cell[f'{stage}_g'].append(row['conductance_siemens'])
                if stage=='post': break
                eligible={}
                for wl,bl in CELLS:
                    cell=state[wl,bl]
                    if len(cell['pre_g'])==3:
                        g=statistics.mean(cell['pre_g'])
                        current=.1*g
                        if current>.005 or current*board['sum_gain']>.8*board['vol_ref_adc'] or .1*round(board['sum_gain']*g,4)>=3.3:
                            cell['reasons'].append('single_cell_model_bound')
                    eligible[wl,bl]=not cell['reasons']
                for repeat in range(5):
                    samples=[(None,0.0)]+[(bl,level) for bl in range(32) for level in (.05,.1)]
                    for bl,level in samples:
                        for wl in range(8):
                            if bl is not None and not eligible[wl,bl]:
                                summary['skipped_mvm']+=1
                                emit(dict(event='skipped',source='hardware',session_id=session_id,wl=wl,bl=bl,
                                     nominal_v=level,repeat=repeat,reasons=state[wl,bl]['reasons'],timestamp_utc=now()))
                                continue
                            row=impact(10,wl,bl,level,stage='mvm',repeat=repeat,bl_basis=bl,nominal_v=level)
                            if bl is None: zeros[wl].append(row['raw_adc'])
                            else: state[wl,bl]['low_adc' if level==.05 else 'high_adc'].append(row['raw_adc'])
            summary['status']='completed'
    except (Exception,KeyboardInterrupt) as exc:
        summary['error']=f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if not connector.close_port(): raise ConnectionError('Connector.close_port returned false')
        except Exception as exc:
            summary.update(status='failed',close_error=f'{type(exc).__name__}: {exc}')
        cells=[classify(state[wl,bl],zeros[wl],criteria) for wl,bl in CELLS]
        summary.update(finished_utc=now(),counts={status:sum(c['status']==status for c in cells)
                      for status in ('candidate','reject','unmeasured')})
        summary['mask_usable']=summary['status']=='completed'
        export_map(output,cells,summary['status'])
        save_json(output/'summary.json',summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--run-hardware',action='store_true')
    mode.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    if not(args.run_hardware or args.dry_run): parser.print_help(); return 0
    if sha256(local_path('read_all.py'))!=READ_ALL_SHA256: raise ValueError('Unreviewed read_all source')
    config,snapshot,hashes=preflight(INI)
    criteria_document=load_criteria(local_path(CONFIG))
    for name in ('experiment.py','cell_map.py','native_conversions.py','experiment_hardware.py',CONFIG):
        hashes[name]=sha256(local_path(name))
    description=dict(maximum_impacts=4136,mode9_maximum=1536,mode10_maximum=2600,
        cells=256,nominal_voltages=[0,.05,.1],source_sha256=hashes,criteria=criteria_document,
        protocol='pre:256x3; repeat5(zero then BL0..31 x .05/.1, WL0..7 inner); post:256x3',
        model_is_not_calibration=True,programmability_tested=False,pair_additivity_tested=False)
    if args.dry_run: print(json.dumps(dict(hardware_access=False,**description))); return 0
    provenance=launch_provenance(argv)
    lock=local_path(LOCK)
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x',encoding='utf-8') as stream: stream.write(f'Cell map {now()}\n')
    try:
        session_id=str(uuid.uuid4())
        output=local_path('results/cell_maps')/session_id
        output.mkdir(parents=True,exist_ok=False)
        (output/'board.ini').write_bytes(snapshot)
        save_json(output/'criteria.json',criteria_document)
        manifest=dict(session_id=session_id,source='hardware',started_utc=now(),status='failed',invocation=provenance,**description)
        save_json(output/'start_manifest.json',manifest)
        try:
            connector=native_connector(config)
            from manager.service import converters
            result=collect(connector,converters,output,session_id,criteria_document['criteria'])
            manifest.update(status=result['status'],attempted=result['attempted'],error=result.get('error'))
        except (Exception,KeyboardInterrupt) as exc: manifest['error']=f'{type(exc).__name__}: {exc}'
        finally:
            manifest['finished_utc']=now()
            manifest['result_sha256']={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
            save_json(output/'manifest.json',manifest)
        print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'])))
        return 0 if manifest['status']=='completed' else 2
    finally: lock.unlink()


if __name__=='__main__':
    raise SystemExit(main())
