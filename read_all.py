"""First physical read: native Connector mode 9, one cell or all 256 cells.

No hardware access without --run-hardware. --dry-run checks files only.
The two resistance estimates are circuit-model estimates, not calibration.
"""

import argparse
from configparser import ConfigParser
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path, PureWindowsPath
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
INI = "configs/raspberry_board.ini"
VENDOR = "vendor/MemriBoard"
READ_VOLTAGE = 0.1
EXPECTED_BOARD = {
    "dac_bit": 12, "vol_ref_dac": 5, "res_load": 3000, "vol_read": 0.3,
    "adc_bit": 14, "vol_ref_adc": 5, "res_switches": 10, "gain": 11,
    "sum_gain": 330, "read_time": 1200, "blank_time": 5,
    "blank_time_between": 55000, "soft_cc": 0.002,
}
EXPECTED_SOURCES = {
    "manager/service/converters.py": "28c6495c56afa225501dd6b0dea999215f08b73f30e016ff7ea34d6c5412701c",
    "manager/service/__init__.py": "116c5e49243fbeed84f1807f938901bcb24f84415ed8ac84568126ca187b1098",
    "manager/board.py": "5e7076651001532e409287f1cd7d17a95ba26d3d382c44e1865de56e45477df0",
    "manager/service/drivers.py": "e4e47cd172196f6e1b6d2f22e9481a8d69cf2fcbcedb78760c43daebc8f0cdc9",
    "MemriCORE/rp5_fpga_python/rpi_FPGAed.py": "bc717659fa5d4d36d2037eef978bf1cd8dcf0fe92f9bbfad51f1eb1d0d27df9c",
}
FPGA_SOURCE = "MemriCORE/rp5_fpga_python/FPGA_SIM.py"
FPGA_SHA256 = "5bcb02c04bc185e332b2ec96acabc2d4303d21724c9c052de540ef92d603b634"


def local_path(value):
    path, windows = Path(value), PureWindowsPath(value)
    if not value or path.is_absolute() or windows.drive or windows.root or ".." in path.parts or ".." in windows.parts:
        raise ValueError("Use a repository-relative path")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Path escapes this repository")
    return resolved


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def json_value(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return repr(value)


def preflight(ini_name):
    ini_path = local_path(ini_name)
    snapshot = ini_path.read_bytes()
    config = ConfigParser()
    config.read_string(snapshot.decode("utf-8-sig"))
    if config.get("board", "board_type") != "rp5_fpga_python":
        raise ValueError("Only the reviewed rp5_fpga_python board is supported")
    for key, expected in EXPECTED_BOARD.items():
        actual = config.getfloat("board", key)
        if not math.isfinite(actual) or actual != expected:
            raise ValueError(f"Unreviewed board setting: {key}={actual}")
    if FPGA_SOURCE is None or FPGA_SHA256 is None:
        raise ValueError("FPGA encoder source review/fingerprint is still required")
    required = {**EXPECTED_SOURCES, FPGA_SOURCE: FPGA_SHA256}
    hashes = {}
    for relative, expected in required.items():
        name = f"{VENDOR}/{relative}"
        hashes[name] = sha256(local_path(name))
        if hashes[name] != expected:
            raise ValueError(f"Source fingerprint differs: {name}")
    # Record the remaining local runtime sources as provenance, without importing them.
    for path in local_path(VENDOR).rglob("*.py"):
        name = path.relative_to(ROOT).as_posix()
        hashes[name] = sha256(local_path(name))
    hashes[Path(ini_name).as_posix()] = hashlib.sha256(snapshot).hexdigest()
    hashes["read_all.py"] = sha256(local_path("read_all.py"))
    return config, snapshot, hashes


def native_connector(config):
    """Ordinary import only; the experiment never imports a physical driver."""
    for relative in (".deps", VENDOR):
        directory = local_path(relative)
        if directory.is_dir():
            sys.path.insert(0, str(directory))
    from manager.board import Connector
    return Connector(silent=True, logger=logging.getLogger(__name__),
                     cb_type="real", board_type="rp5_fpga_python",
                     task_format="SI", config=config)


def model_estimates(adc, board):
    flags = []
    output = {"resistance_nominal_ohm": None, "resistance_dac_estimate_ohm": None,
              "conductance_nominal_siemens": None, "conductance_dac_estimate_siemens": None}
    if type(adc) is not int or not 0 <= adc < 2 ** int(board["adc_bit"]):
        return output, ["invalid_adc"]
    if adc == 0 or adc == 2 ** int(board["adc_bit"]) - 1:
        return output, ["adc_zero" if adc == 0 else "adc_saturated"]
    code = round(READ_VOLTAGE * (2 ** int(board["dac_bit"]) - 1) / board["vol_ref_dac"])
    estimated_voltage = code * board["vol_ref_dac"] / (2 ** int(board["dac_bit"]) - 1)
    for name, voltage in (("nominal", READ_VOLTAGE), ("dac_estimate", estimated_voltage)):
        resistance = (board["gain"] * board["res_load"] * voltage * 2 ** int(board["adc_bit"])
                      / (adc * board["vol_ref_adc"]) - board["res_switches"] - board["res_load"])
        if resistance <= 0 or not math.isfinite(resistance):
            flags.append(f"nonpositive_resistance_{name}")
        else:
            output[f"resistance_{name}_ohm"] = resistance
            output[f"conductance_{name}_siemens"] = 1 / resistance
    return output, flags


def collect(connector, cells, output_dir, session_id, board):
    """All physical interaction is confined to the three public Connector calls."""
    if (not cells or len(set(cells)) != len(cells)
            or any(type(wl) is not int or type(bl) is not int or not 0 <= wl < 8 or not 0 <= bl < 32
                   for wl, bl in cells)):
        raise ValueError("Expected distinct valid WL/BL addresses")
    audit = {"session_id": session_id, "source": "hardware", "status": "failed",
             "expected_cells": len(cells), "attempted": 0, "received": 0,
             "valid_model_estimates": 0, "flagged": 0, "cells": [],
             "calibration_verified": False, "finished_utc": None}
    try:
        with (output_dir / "raw.jsonl").open("x", encoding="utf-8") as stream:
            def emit(row):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            if not connector.open_port():
                raise ConnectionError("Connector.open_port returned false")
            for task_id, (wl, bl) in enumerate(cells, 1):
                task = {"mode_flag": 9, "vol": READ_VOLTAGE, "pulse_width": 0.0,
                        "sign": 0, "id": task_id, "wl": wl, "bl": bl}
                row = {"session_id": session_id, "source": "hardware", "task_id": task_id,
                       "wl": wl, "bl": bl, "timestamp_utc": now(), "command": task.copy()}
                emit({**row, "event": "request"})
                audit["attempted"] += 1
                start = time.monotonic()
                try:
                    result = connector.impact(task.copy())
                    row["native_result"] = json_value(result)
                    if not isinstance(result, (tuple, list)) or len(result) != 3:
                        raise ValueError("Expected native (R, id, ADC)")
                    native_r, echoed_id, adc = result
                    row.update(raw_adc=json_value(adc), returned_id=json_value(echoed_id),
                               native_r_uses_ini_vol_read=json_value(native_r))
                    if type(echoed_id) is not int or echoed_id != 0:
                        raise ValueError("Mode 9 returned ID must be zero")
                    estimates, flags = model_estimates(adc, board)
                    row.update(estimates, flags=flags, status="flagged" if flags else "ok")
                    audit["received"] += 1
                    audit["cells"].append([wl, bl])
                    audit["flagged" if flags else "valid_model_estimates"] += 1
                except (Exception, KeyboardInterrupt) as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    emit({**row, "event": "result", "latency_s": time.monotonic() - start})
            audit["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        audit["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if not connector.close_port():
                raise ConnectionError("Connector.close_port returned false")
        except Exception as exc:
            audit.update(status="failed", close_error=f"{type(exc).__name__}: {exc}")
        audit["finished_utc"] = now()
        audit["unique_cells"] = len(set(map(tuple, audit["cells"])))
        audit["missing_cells"] = [list(cell) for cell in cells if list(cell) not in audit["cells"]]
        if (output_dir / "raw.jsonl").is_file():
            audit["raw_sha256"] = sha256(output_dir / "raw.jsonl")
        save_json(output_dir / "audit.json", audit)
    return audit


def parse_cell(text):
    try:
        wl, bl = map(int, text.split(","))
        if not 0 <= wl < 8 or not 0 <= bl < 32:
            raise ValueError
        return wl, bl
    except ValueError:
        raise argparse.ArgumentTypeError("Use WL,BL with WL=0..7 and BL=0..31")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run-hardware", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ini", default=INI, help="Reviewed repository-relative board INI")
    parser.add_argument("--cell", type=parse_cell, help="Read exactly one cell WL,BL; default all 256")
    args = parser.parse_args(argv)
    if not (args.run_hardware or args.dry_run):
        parser.print_help()
        return 0
    config, snapshot, hashes = preflight(args.ini)
    cells = [args.cell] if args.cell is not None else [(wl, bl) for wl in range(8) for bl in range(32)]
    if args.dry_run:
        print(json.dumps({"hardware_access": False, "cell_count": len(cells),
                          "mode": 9, "nominal_v": READ_VOLTAGE, "source_sha256": hashes}))
        return 0
    # Import/config validation precedes open_port, which performs hardware initialization.
    connector = native_connector(config)
    base = local_path("results/read_all")
    base.mkdir(parents=True, exist_ok=True)
    lock = base / ".hardware.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"{now()}\n")
    try:
        session_id = str(uuid.uuid4())
        output_dir = base / session_id
        output_dir.mkdir()
        (output_dir / "board.ini").write_bytes(snapshot)
        save_json(output_dir / "manifest.json", {
            "session_id": session_id, "source": "hardware", "started_utc": now(),
            "board_type": "rp5_fpga_python", "source_sha256": hashes,
            "requested_cells": [list(cell) for cell in cells], "mode": 9,
            "nominal_read_voltage_v": READ_VOLTAGE, "dac_code": 82,
            "ideal_dac_voltage_v": 82 * 5 / 4095,
            "native_r_is_not_the_0_1v_resistance": True,
            "resistance_estimates_are_uncalibrated_models": True,
            "driver_reads_ini": False, "python": sys.version,
        })
        audit = collect(connector, cells, output_dir, session_id, EXPECTED_BOARD)
        print(json.dumps({"output_dir": output_dir.relative_to(ROOT).as_posix(),
                          "status": audit["status"], "received": audit["received"],
                          "flagged": audit["flagged"]}))
        return 0 if audit["status"] == "completed" else 1
    finally:
        lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
