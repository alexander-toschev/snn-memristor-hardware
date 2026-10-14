"""Explicitly approved simulator readback baseline through native Connector.

No programming, reset, mapping, MVM, hardware backend, or array creation.
Both the CLI approval and protocol execution/read_baseline flags are required.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import time
import uuid

from config import parse_protocol
from connector_session import connect
from measurements import adc_to_resistance, resistance_to_conductance


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_record(session, protocol, *, session_id, array_id, phase, task_id, wl, bl):
    start = time.monotonic()
    row = {
        "timestamp_utc": utc_now(), "monotonic_time_s": start,
        "session_id": session_id, "array_id": array_id, "source": "simulator",
        "wl": wl, "bl": bl, "task_id": task_id, "raw_adc": None,
        "status": "error", "latency_s": None, "phase": phase,
        "cycle": None, "target_conductance_siemens": None,
    }
    try:
        result = session.read_cell(wl, bl, task_id)
        row["native_result_repr"] = repr(result)
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise ValueError("Expected native (resistance, request_id, raw_adc) result")
        _, returned_id, raw_adc = result
        # Preserve even invalid ADC values in valid JSON. Exact native repr remains
        # available for malformed types, NaN, and infinity without fabricating R.
        if raw_adc is None or type(raw_adc) in (str, bool, int):
            row["raw_adc"] = raw_adc
        elif type(raw_adc) is float and math.isfinite(raw_adc):
            row["raw_adc"] = raw_adc
        else:
            row["raw_adc"] = repr(raw_adc)
        if type(returned_id) is not int or returned_id != task_id:
            raise ValueError(f"Request ID mismatch: expected {task_id}, received {returned_id!r}")
        resistance = adc_to_resistance(raw_adc, protocol.board)
        row["resistance_ohm"] = resistance
        row["conductance_siemens"] = resistance_to_conductance(resistance)
        row["status"] = "ok"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["latency_s"] = time.monotonic() - start
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New session directory")
    parser.add_argument("--approved-run", action="store_true")
    parser.add_argument("--stage", choices=("first_check", "repeatability"), default="first_check")
    args = parser.parse_args()
    try:
        snapshot = args.protocol.read_bytes()
        protocol = parse_protocol(snapshot)
        if not args.approved_run or not protocol.execution["approved"]:
            raise PermissionError("Explicit approval is required for this simulator run")
        if not protocol.execution["simulator_enabled"] or protocol.execution["hardware_enabled"]:
            raise PermissionError("Enable only simulator execution in the approved protocol")
        if not protocol.read_baseline["enabled"]:
            raise PermissionError("read_baseline is disabled")
        if protocol.connection["cb_type"] != "simulator":
            raise NotImplementedError("Only the native simulator backend is implemented")
        count_key = ("first_check_reads_per_cell" if args.stage == "first_check"
                     else "repeatability_reads_per_cell_after_first_check")
        repeats = protocol.read_baseline[count_key]
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except (OSError, KeyError, TypeError, ValueError, PermissionError, NotImplementedError) as exc:
        parser.exit(2, f"Preflight failed: {exc}\n")

    session_id = str(uuid.uuid4())
    manifest = {
        "session_id": session_id, "protocol_id": protocol.protocol_id, "source": "simulator",
        "protocol_sha256": hashlib.sha256(snapshot).hexdigest(),
        "stage": args.stage, "reads_per_cell": repeats, "started_utc": utc_now(),
        "cell_order": [{"wl": wl, "bl": bl} for wl, bl in protocol.initial_cells],
        "status": "failed", "records_saved": 0,
    }
    session = None
    try:
        with (args.output_dir / "protocol.json").open("xb") as stream:
            stream.write(snapshot)
        root = Path(protocol.connection["memriboard_root"])
        manifest["source_sha256"] = {
            name: file_sha256(root / name) for name in (
                "manager/board.py", "manager/service/drivers.py",
                "manager/service/converters.py", "simulator/src.py", "simulator/memristor.py",
            )
        }
        array_file = Path(str(protocol.connection["crossbar_serial"]) + ".cb")
        manifest["array_file"] = str(array_file)
        manifest["array_sha256_before"] = file_sha256(array_file)
        session = connect(protocol, logger=logging.getLogger(__name__), approved=args.approved_run)
        array_id = Path(protocol.connection["crossbar_serial"]).name
        with (args.output_dir / "raw_readback.jsonl").open("x", encoding="utf-8") as stream:
            for wl, bl in protocol.initial_cells:
                for _ in range(repeats):
                    task_id = manifest["records_saved"] + 1
                    row = read_record(session, protocol, session_id=session_id, array_id=array_id,
                                      phase=args.stage, task_id=task_id, wl=wl, bl=bl)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    manifest["records_saved"] += 1
                    if row["status"] != "ok":
                        raise RuntimeError(f"Stopped after task {task_id}: {row['error']}")
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                manifest["status"] = "failed"
                manifest["close_error"] = f"{type(exc).__name__}: {exc}"
        if "array_sha256_before" in manifest:
            try:
                manifest["array_sha256_after"] = file_sha256(array_file)
            except OSError as exc:
                manifest["status"] = "failed"
                manifest["array_hash_error"] = str(exc)
        manifest["finished_utc"] = utc_now()
        with (args.output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
    if manifest["status"] != "completed":
        parser.exit(2, f"Experiment stopped; inspect {args.output_dir / 'manifest.json'}\n")
    print(f"Saved {manifest['records_saved']} simulator reads to {args.output_dir}")


if __name__ == "__main__":
    main()
