"""Четыре фазы TWO_CROSSBAR_PLAN: настройки и порядок запуска в одном файле.

Одна точка входа: --run-simulator, --run-hardware или --dry-run.
Без флагов только справка. --dry-run не подключает плату.
"""

import argparse
from datetime import datetime, timezone
from configparser import ConfigParser
import json
import logging
import random
import sys
import uuid

from experiments.unn_rev3_20261014.config import parse_protocol
from experiments.unn_rev3_20261014.paths import repo_path, relative_path, use_local_dependencies
from experiments.unn_rev3_20261014.readback import file_sha256, utc_now

# ================= НАСТРОЙКИ ЭКСПЕРИМЕНТА =================
CB_TYPE = "simulator"
BOARD_TYPE = "rp5_fpga_python"  # Имя драйвера; backend остаётся simulator.
SETTINGS_FILE = "vendor/MemriBoard/simulator/simulator_settings.ini"
WEIGHTS_FILE = "weights/conv1_weights.npy"
CHECKPOINT_FILE = "weights/best_fs.pt"
PROTOCOL_FILE = "experiments/unn_rev3_20261014/protocol.json"
RESULTS_DIR = "results/simulator"
SEED = 20261014
CELLS = ((0, 2), (2, 6), (2, 4))  # WL, BL; WL=1 исключена.
PHASE2_FILTERS = (63, 41, 32, 30, 38, 43, 60, 47, 2, 9, 46, 12, 0, 1)
PHASE3_FILTERS = tuple(range(64))
PHASE4_FILTERS = (41,)  # Заранее выбранный трудный фильтр; K=1,2,3.
WEIGHTS_SHA256 = "8be312b9eafb9de23f8340b17f848e7e0e2e7884293c826a6d9f6cbb5125d440"
CHECKPOINT_SHA256 = "ee4fc4f58a3d010218116e9ad4c7ba808787959f9ed7687cd58cbb32d90f62fb"


def save_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def run_simulator():
    if CB_TYPE != "simulator":
        raise ValueError("Этот сценарий разрешает только simulator")

    # 1. Проверить локальные веса и зафиксировать фактический протокол.
    use_local_dependencies()
    import numpy as np
    from manager.board import Connector
    from simulator.src import create_crossbar_array
    from experiments.unn_rev3_20261014.suite import SimulatorSuite
    for filename, expected in ((WEIGHTS_FILE, WEIGHTS_SHA256), (CHECKPOINT_FILE, CHECKPOINT_SHA256)):
        if file_sha256(repo_path(filename)) != expected:
            raise ValueError(f"Хеш весов не совпал: {filename}")
    weights = np.load(repo_path(WEIGHTS_FILE), allow_pickle=False)
    if weights.shape != (64, 3, 5, 5) or weights.dtype != np.float32 or not np.isfinite(weights).all():
        raise ValueError("Ожидаются конечные float32 веса Conv1 [64,3,5,5]")
    settings = json.loads(repo_path(PROTOCOL_FILE).read_text(encoding="utf-8-sig"))
    board_config = ConfigParser()
    with repo_path(SETTINGS_FILE).open(encoding="utf-8") as stream:
        board_config.read_file(stream)
    settings["board"] = {
        key: (board_config.getint("board", key) if type(value) is int
              else board_config.getfloat("board", key))
        for key, value in settings["board"].items()
    }
    settings["status"] = "approved_simulator_run"
    settings["connection"].update(cb_type=CB_TYPE, board_type=BOARD_TYPE,
                                   task_format="SI", memriboard_root="vendor/MemriBoard")
    settings["geometry"]["initial_cells"] = [{"wl": wl, "bl": bl} for wl, bl in CELLS]
    settings["execution"].update(approved=True, simulator_enabled=True, hardware_enabled=False,
                                  approval_scope="User approved all four phases with simulator backend")
    settings["read_baseline"]["enabled"] = True
    settings["simulator_characterization"]["enabled"] = True
    settings["suite"] = {
        "seed": SEED, "phase2_filters": list(PHASE2_FILTERS),
        "phase3_filters": list(PHASE3_FILTERS), "phase4_filters": list(PHASE4_FILTERS),
        "phase4_repetitions": 3, "tile_arrays": ["A", "B", "A"],
        "mapping_wl_pair": [0, 2], "mapping_q_window": [0.01, 0.05],
        "weights_file": WEIGHTS_FILE, "weights_sha256": WEIGHTS_SHA256,
        "checkpoint_file": CHECKPOINT_FILE, "checkpoint_sha256": CHECKPOINT_SHA256,
        "scope": "Two simulated arrays; synthetic MVM; no classifier or physical measurements",
    }
    session_id = str(uuid.uuid4())
    settings["suite"]["session_id"] = session_id
    settings["suite"]["input_voltage"] = 0.1
    settings["suite"]["reset_endpoint_q"] = 0.005
    settings["suite"]["max_pulses_per_reset_or_set"] = 271
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = repo_path(RESULTS_DIR) / f"{stamp}_{session_id[:8]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "arrays").mkdir()
    settings["connection"]["crossbar_serial"] = relative_path(output_dir / "arrays" / "array_A")
    save_json(output_dir / "protocol.json", settings)
    protocol = parse_protocol((output_dir / "protocol.json").read_bytes())
    manifest = {
        "session_id": session_id, "source": CB_TYPE, "started_utc": utc_now(),
        "status": "failed", "completed_phases": [], "seed": SEED,
        "protocol_sha256": file_sha256(output_dir / "protocol.json"),
        "weights_sha256": WEIGHTS_SHA256, "checkpoint_sha256": CHECKPOINT_SHA256,
        "python": sys.version, "numpy": np.__version__,
    }
    source_files = [repo_path("experiment.py"), repo_path("requirements.txt")]
    source_files += list(repo_path("experiments/unn_rev3_20261014").glob("*.py"))
    source_files += list(repo_path("vendor/MemriBoard").rglob("*.py"))
    source_files += list(repo_path("vendor/MemriBoard").rglob("*.ini"))
    manifest["source_sha256"] = {relative_path(p): file_sha256(p) for p in source_files}
    connectors, summary = {}, {}
    suite = None
    try:
        # 2. ДВА новых массива и штатные Connector с backend=simulator.
        random.seed(SEED)
        np.random.seed(SEED)
        for name in ("A", "B"):
            serial = relative_path(output_dir / "arrays" / f"array_{name}")
            create_crossbar_array(str(repo_path(serial)), 32, 8)
            connector = Connector(
                silent=True, logger=logging.getLogger(__name__),
                cb_type=CB_TYPE, board_type=BOARD_TYPE, task_format="SI",
                config=board_config, crossbar_serial=str(repo_path(serial)),
            )
            connectors[name] = connector
            if not connector.open_port():
                raise ConnectionError(f"Не удалось подключить массив {name}")
        suite = SimulatorSuite(connectors, protocol, output_dir, settings["suite"])

        # Входная проверка: один отсчёт, затем 20 чтений трёх ячеек.
        summary["baseline"] = suite.run_baseline()
        manifest["completed_phases"].append("baseline")

        # ФАЗА 1. 3 ячейки × 5 целей × 5 циклов: запись и bias/SD/RMSE.
        summary["phase1"] = suite.run_characterization()
        manifest["completed_phases"].append("phase1")

        # ФАЗА 2. 14 фильтров K=1: RESET → G+/G− → readback → MVM.
        summary["phase2"] = suite.run_mapping(weights, filters=PHASE2_FILTERS, label="phase2")
        manifest["completed_phases"].append("phase2")

        # ФАЗА 3. Все 64 фильтра Conv1; те же два массива A/B.
        summary["phase3"] = suite.run_mapping(weights, filters=PHASE3_FILTERS, label="phase3")
        manifest["completed_phases"].append("phase3")

        # ФАЗА 4. Трудный фильтр: три записи, средние K=1,2,3.
        summary["phase4"] = suite.run_mapping(weights, filters=PHASE4_FILTERS,
                                               label="phase4", repetitions=3)
        manifest["completed_phases"].append("phase4")
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if suite is not None:
            manifest["raw_task_count"] = suite.summary["raw_task_count"]
            suite.close()
        for name, connector in connectors.items():
            try:
                connector.close_port()
            except Exception as exc:
                manifest.update(status="failed", close_error=f"{name}: {exc}")
        manifest["finished_utc"] = utc_now()
        manifest["array_sha256"] = {relative_path(p): file_sha256(p) for p in (output_dir / "arrays").glob("*.cb")}
        save_json(output_dir / "summary.json", summary)
        save_json(output_dir / "manifest.json", manifest)
        print(f"Результаты: {relative_path(output_dir)}; status={manifest['status']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-simulator", action="store_true", help="Существующие четыре фазы симулятора")
    mode.add_argument("--run-hardware", action="store_true", help="Физический запуск; требуется разрешение оператора")
    mode.add_argument("--dry-run", action="store_true", help="Аппаратный preflight без подключения")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--phase", choices=("phase1", "phase2", "phase3", "phase4"))
    phase.add_argument("--all", action="store_true")
    parser.add_argument("--phase1-session")
    parser.add_argument("--max-pulses", type=int)
    parser.add_argument("--check-mvm", action="store_true", help="Mode9/10 diagnostic on existing conductances; no writes")
    args = parser.parse_args(argv)
    hardware_options = args.phase is not None or args.all or args.phase1_session is not None or args.max_pulses is not None
    if args.check_mvm and (hardware_options or not (args.run_hardware or args.dry_run)):
        parser.error("--check-mvm requires --run-hardware or --dry-run and cannot combine with phase options")
    if hardware_options and not (args.run_hardware or args.dry_run):
        parser.error("--phase/--all/--phase1-session/--max-pulses require --run-hardware or --dry-run")
    if args.run_simulator:
        return run_simulator()
    if args.run_hardware or args.dry_run:
        if args.check_mvm:
            from mvm_check import main as check_mvm
            return check_mvm(["--run-hardware" if args.run_hardware else "--dry-run"])
        from experiment_hardware import main as run_hardware
        forwarded = ["--run-hardware" if args.run_hardware else "--dry-run"]
        if args.phase is not None:
            forwarded += ["--phase", args.phase]
        if args.all:
            forwarded.append("--all")
        if args.phase1_session is not None:
            forwarded += ["--phase1-session", args.phase1_session]
        if args.max_pulses is not None:
            forwarded += ["--max-pulses", str(args.max_pulses)]
        return run_hardware(forwarded)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
