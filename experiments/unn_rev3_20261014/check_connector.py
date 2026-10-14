"""Short native simulator SI/DAC contract check; no hardware or long experiment.

Run only with explicit simulator authorization:
python -m experiments.unn_rev3_20261014.check_connector --run-simulator

Each command loads the same newly created, persisted array through a separate
native Connector. No running experiment array is opened. Mode 9's native
returned ID is explicitly expected to be zero, regardless of requested ID.
"""

import argparse
from configparser import ConfigParser
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import logging
import math
from pathlib import Path
import random
import sys
import uuid

from .config import parse_protocol
from .paths import repo_path, relative_path, use_local_dependencies


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def json_value(value):
    """Keep malformed numeric results representable in the error log."""
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return repr(value)


def run_check():
    session_id = str(uuid.uuid4())
    output_dir = repo_path("results/connector_checks") / session_id
    output_dir.mkdir(parents=True, exist_ok=False)
    root = repo_path("vendor/MemriBoard")
    report = {"session_id": session_id, "source": "simulator", "passed": False,
              "hardware_qualification": False, "output_dir": relative_path(output_dir),
              "started_utc": datetime.now(timezone.utc).isoformat(), "cases": [], "errors": [],
              "scope": "7 cases, 14 native impacts on fresh copies in memory of one fresh persisted array"}
    path_inserted = False
    try:
        snapshot = repo_path("experiments/unn_rev3_20261014/protocol.json").read_bytes()
        protocol = parse_protocol(snapshot)
        if protocol.connection["cb_type"] != "simulator":
            raise ValueError("The protocol must select simulator; no hardware fallback")
        if protocol.connection["board_type"] != "rp5_fpga_python":
            raise ValueError("Unexpected board_type for the reviewed native Connector contract")
        with (output_dir/"protocol_reference.json").open("xb") as stream:
            stream.write(snapshot)
        report["protocol_sha256"] = hashlib.sha256(snapshot).hexdigest()
        ini_path = root/"simulator/simulator_settings.ini"
        ini = ConfigParser()
        with ini_path.open(encoding="utf-8-sig") as stream:
            ini.read_file(stream)
        report["board_comparison"] = {}
        for key, expected in vars(protocol.board).items():
            observed = ini.getfloat("board", key)
            equal = math.isfinite(observed) and math.isclose(observed, expected, rel_tol=0, abs_tol=1e-12)
            report["board_comparison"][key] = {"protocol": expected, "ini": observed, "equal": equal}
        if not all(item["equal"] for item in report["board_comparison"].values()):
            raise ValueError("Protocol board settings do not match the native simulator INI")
        report["source_sha256"] = {relative_path(path): sha256(path) for path in (
            repo_path("experiments/unn_rev3_20261014/check_connector.py"),
            root/"manager/board.py", root/"manager/service/drivers.py",
            root/"manager/service/converters.py", root/"simulator/src.py",
            root/"simulator/memristor.py", ini_path,
        )}
        use_local_dependencies()
        sys.path.insert(0, str(root))
        path_inserted = True
        native = importlib.import_module("manager.board")
        simulator = importlib.import_module("simulator.src")
        converters = importlib.import_module("manager.service.converters")
        for module, relative in ((native, "manager/board.py"), (simulator, "simulator/src.py"),
                                 (converters, "manager/service/converters.py")):
            if (root/relative).resolve() != Path(module.__file__).resolve():
                raise RuntimeError("A different MemriBoard installation is already imported")
        import numpy as np
        seed = 20261014
        random.seed(seed)
        np.random.seed(seed)
        report.update(seed=seed, numpy=np.__version__)
        serial = output_dir/"fresh_array"
        simulator.create_crossbar_array(str(serial), 32, 8)
        array_file = output_dir/"fresh_array.cb"
        report["array_file"] = relative_path(array_file)
        report["array_sha256_before"] = sha256(array_file)
        board = protocol.board

        def encode(voltage):
            return converters.convert_volt_to_dac(board.dac_bit, board.vol_ref_dac, voltage)

        cases = [("mode7_zero_read", 7, 0.0)]
        cases += [(f"mode9_{voltage:.1f}V", 9, voltage) for voltage in (0.0, 0.1, 0.2, 0.3)]
        cases += [("mode10_zero", 10, [0.0]*32),
                  ("mode10_onehot_0.1V", 10, [0.1 if bl == 2 else 0.0 for bl in range(32)])]
        with (output_dir/"commands.jsonl").open("x", encoding="utf-8") as log:
            for case_index, (name, mode, voltage) in enumerate(cases):
                records = []
                for format_index, task_format in enumerate(("SI", "dac")):
                    task_id = 2*case_index+format_index+1
                    task = {"mode_flag": mode, "vol": copy.deepcopy(voltage), "sign": 0,
                            "id": task_id, "wl": 0, "bl": 2}
                    if task_format == "SI":
                        task["pulse_width"] = 0.0
                    else:
                        task["vol"] = [encode(v) for v in voltage] if isinstance(voltage, list) else encode(voltage)
                        task.update(t_ms=0, t_us=0)
                    record = {"case": name, "source": "simulator", "session_id": session_id,
                              "task_format": task_format, "task": copy.deepcopy(task),
                              "expected_returned_id": 0 if mode == 9 else task_id, "passed": False}
                    connector = None
                    try:
                        connector = native.Connector(
                            silent=True, logger=logging.getLogger(__name__), cb_type="simulator",
                            board_type="rp5_fpga_python", task_format=task_format,
                            config=protocol.board_config(), crossbar_serial=str(serial))
                        if connector.cb_type != "simulator" or not connector.open_port():
                            raise RuntimeError("Native simulator connection failed")
                        connector.interface.config = connector.config
                        result = connector.impact(copy.deepcopy(task))
                        record["native_result"] = json_value(result)
                        if not isinstance(result, (tuple, list)) or len(result) != 3:
                            raise ValueError("Unexpected native result format")
                        _, returned_id, adc = result
                        record.update(returned_id=json_value(returned_id), raw_adc=json_value(adc))
                        if type(returned_id) is not int or returned_id != record["expected_returned_id"]:
                            raise ValueError("Native returned ID differs from the explicit mode contract")
                        if type(adc) is not int or not (1 if mode == 7 else 0) <= adc <= 16382:
                            raise ValueError("Invalid raw ADC for this mode")
                        if mode in (9, 10) and not any(voltage if isinstance(voltage, list) else [voltage]) and adc != 0:
                            raise ValueError("Zero-input mode9/MVM must return ADC zero in this simulator")
                        record["passed"] = True
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    finally:
                        if connector is not None:
                            try:
                                connector.close_port()
                            except Exception as exc:
                                record.update(passed=False, close_error=f"{type(exc).__name__}: {exc}")
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                    records.append(record)
                matched = all(row["passed"] for row in records) and records[0]["raw_adc"] == records[1]["raw_adc"]
                report["cases"].append({"case": name, "passed": matched,
                                        "si_adc": records[0].get("raw_adc"), "dac_adc": records[1].get("raw_adc"),
                                        "si_returned_id": records[0].get("returned_id"),
                                        "dac_returned_id": records[1].get("returned_id"),
                                        "id_contract": "native mode9 always returns 0" if mode == 9 else "echo requested ID"})
        code = encode(0.3)
        decoded = converters.convert_dac_to_volt(board.dac_bit, board.vol_ref_dac, code)
        report["voltage_contract"] = {
            "nominal_v": 0.3, "native_dac_code": code, "native_decoded_v": decoded,
            "ideal_dac_ratio_v": code*board.vol_ref_dac/(2**board.dac_bit-1),
            "expected_code_and_rounded_decode": code == 246 and decoded == 0.3,
            "hardware_ceiling_validated": False,
            "note": "0.3 V nominal -> DAC246 -> native rounded decode0.300 V. Ideal 5V DAC ratio is about0.300366 V; no physical voltage or hardware ceiling was measured.",
        }
        report["array_sha256_after"] = sha256(array_file)
        report["array_unchanged"] = report["array_sha256_before"] == report["array_sha256_after"]
        report["passed"] = (len(report["cases"]) == 7 and all(case["passed"] for case in report["cases"])
                            and report["array_unchanged"] and report["voltage_contract"]["expected_code_and_rounded_decode"])
        report["interpretation"] = "Raw ADC equality checks native SI/DAC conversion only. Native R is logged without physical interpretation for mode9/MVM. This is not hardware qualification."
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if path_inserted:
            sys.path.pop(0)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-simulator", action="store_true")
    args = parser.parse_args()
    if not args.run_simulator:
        parser.print_help()
        return
    report = run_check()
    print(json.dumps({"output_dir": report["output_dir"], "passed": report["passed"],
                      "case_count": len(report["cases"])}))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
