"""Четыре фазы TWO_CROSSBAR_PLAN: настройки и порядок запуска в одном файле.

Разрешённый прогон: python experiment.py --run-simulator
Без этого аргумента подключения и измерений нет. Все пути внутри репозитория.
"""

import argparse
from datetime import datetime, timezone
import importlib
import json
import logging
import random
import sys
import uuid

from experiments.unn_rev3_20261014.config import parse_protocol
from experiments.unn_rev3_20261014.connector_session import connect
from experiments.unn_rev3_20261014.paths import repo_path, relative_path, use_local_dependencies
from experiments.unn_rev3_20261014.readback import file_sha256, utc_now

# ================= НАСТРОЙКИ ЭКСПЕРИМЕНТА =================
BACKEND = "simulator"
BOARD_DRIVER = "rp5_fpga_python"  # Имя драйвера; backend остаётся simulator.
MEMRIBOARD_ROOT = "vendor/MemriBoard"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-simulator", action="store_true",
                        help="Выполнить разрешённые пользователем четыре фазы")
    if not parser.parse_args().run_simulator:
        parser.print_help()
        return
    if BACKEND != "simulator":
        raise ValueError("Этот сценарий разрешает только simulator")

    # 1. Проверить локальные веса и зафиксировать фактический протокол.
    use_local_dependencies()
    import numpy as np
    from experiments.unn_rev3_20261014.suite import SimulatorSuite
    for filename, expected in ((WEIGHTS_FILE, WEIGHTS_SHA256), (CHECKPOINT_FILE, CHECKPOINT_SHA256)):
        if file_sha256(repo_path(filename)) != expected:
            raise ValueError(f"Хеш весов не совпал: {filename}")
    weights = np.load(repo_path(WEIGHTS_FILE), allow_pickle=False)
    if weights.shape != (64, 3, 5, 5) or weights.dtype != np.float32 or not np.isfinite(weights).all():
        raise ValueError("Ожидаются конечные float32 веса Conv1 [64,3,5,5]")
    settings = json.loads(repo_path(PROTOCOL_FILE).read_text(encoding="utf-8-sig"))
    settings["status"] = "approved_simulator_run"
    settings["connection"].update(cb_type=BACKEND, board_type=BOARD_DRIVER,
                                   task_format="SI", memriboard_root=MEMRIBOARD_ROOT)
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
        "session_id": session_id, "source": BACKEND, "started_utc": utc_now(),
        "status": "failed", "completed_phases": [], "seed": SEED,
        "protocol_sha256": file_sha256(output_dir / "protocol.json"),
        "weights_sha256": WEIGHTS_SHA256, "checkpoint_sha256": CHECKPOINT_SHA256,
        "python": sys.version, "numpy": np.__version__,
    }
    source_files = [repo_path("experiment.py"), repo_path("requirements.txt")]
    source_files += list(repo_path("experiments/unn_rev3_20261014").glob("*.py"))
    source_files += list(repo_path(MEMRIBOARD_ROOT).rglob("*.py"))
    source_files += list(repo_path(MEMRIBOARD_ROOT).rglob("*.ini"))
    manifest["source_sha256"] = {relative_path(p): file_sha256(p) for p in source_files}
    sessions, summary = {}, {}
    suite = None
    try:
        # 2. ДВА новых массива и штатные Connector с backend=simulator.
        random.seed(SEED)
        np.random.seed(SEED)
        sys.path.insert(0, str(repo_path(MEMRIBOARD_ROOT)))
        simulator = importlib.import_module("simulator.src")
        for name in ("A", "B"):
            serial = relative_path(output_dir / "arrays" / f"array_{name}")
            simulator.create_crossbar_array(str(repo_path(serial)), 32, 8)
            array_settings = json.loads(json.dumps(settings))
            array_settings["connection"]["crossbar_serial"] = serial
            array_protocol = parse_protocol(json.dumps(array_settings).encode("utf-8"))
            sessions[name] = connect(array_protocol, logger=logging.getLogger(__name__), approved=True)
        suite = SimulatorSuite({name: session.connector for name, session in sessions.items()},
                               protocol, output_dir, settings["suite"])

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
        for name, session in sessions.items():
            try:
                session.close()
            except Exception as exc:
                manifest.update(status="failed", close_error=f"{name}: {exc}")
        manifest["finished_utc"] = utc_now()
        manifest["array_sha256"] = {relative_path(p): file_sha256(p) for p in (output_dir / "arrays").glob("*.cb")}
        save_json(output_dir / "summary.json", summary)
        save_json(output_dir / "manifest.json", manifest)
        print(f"Результаты: {relative_path(output_dir)}; status={manifest['status']}", flush=True)


if __name__ == "__main__":
    main()
