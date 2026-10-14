"""Thin adapter to MemriBoard's native Connector; no new board driver.

Importing this file does not import MemriBoard or open a connection.
connect() requires explicit approval AND the protocol execution flags.
Only an existing trusted simulator .cb file is supported in this revision.
"""

import importlib
from pathlib import Path
import sys

from .config import Protocol
from .paths import repo_path, use_local_dependencies


class Session:
    def __init__(self, connector, protocol: Protocol):
        self.connector = connector
        self.protocol = protocol

    def read_cell(self, wl: int, bl: int, task_id: int):
        """One native mode-7 read; return (native resistance, request ID, raw ADC).

        Caller must save raw ADC before validation/conversion. Native resistance
        is retained for comparison; measurements.py defines the analysis units.
        Zero write amplitude does not suppress the simulator's measurement pulse.
        """
        self.protocol.check_cell(wl, bl)
        if type(task_id) is not int or task_id < 0:
            raise ValueError("task_id must be a nonnegative integer")
        return self.connector.impact({
            "mode_flag": 7, "vol": 0.0, "pulse_width": 0.0, "sign": 0,
            "id": task_id, "wl": wl, "bl": bl,
        })

    def close(self):
        return self.connector.close_port()


def connect(protocol: Protocol, *, logger, approved: bool = False) -> Session:
    """Open the native simulator explicitly. Never create or replace an array.

    MemriBoard must have its normal dependencies installed. No import stubs,
    fake Application, HardCore, full-board scan, or alternate hardware protocol.
    """
    connection = protocol.connection
    if connection["cb_type"] != "simulator":
        raise NotImplementedError("Real backend is not enabled in this revision")
    if approved is not True or protocol.execution["approved"] is not True:
        raise PermissionError("Explicit user approval is required before connecting")
    if protocol.execution["simulator_enabled"] is not True:
        raise PermissionError("Simulator execution is disabled in protocol.json")
    if protocol.execution["hardware_enabled"] is not False:
        raise ValueError("hardware_enabled must remain false for the simulator session")
    root_value, serial_value = connection["memriboard_root"], connection["crossbar_serial"]
    if not root_value or not serial_value:
        raise ValueError("Set memriboard_root and crossbar_serial explicitly")
    root, serial = repo_path(root_value), repo_path(serial_value)
    if serial.suffix.lower() == ".cb":
        raise ValueError("crossbar_serial must be a repository path without the .cb suffix")
    if not repo_path(str(serial_value) + ".cb").is_file():
        raise FileNotFoundError(f"Existing simulator array required: {serial}.cb")
    connector_file = root / "manager" / "board.py"
    if not connector_file.is_file():
        raise FileNotFoundError(connector_file)
    # Normal import from the explicitly selected checkout, only at connect time.
    use_local_dependencies()
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("manager.board")
    finally:
        sys.path.pop(0)
    if Path(module.__file__).resolve() != connector_file.resolve():
        raise RuntimeError("Another MemriBoard checkout is already imported; use a fresh process")
    config = protocol.board_config()
    connector = module.Connector(
        silent=True, logger=logger, cb_type="simulator",
        board_type=connection["board_type"], task_format="SI",
        config=config, crossbar_serial=str(serial),
    )
    # open_port imports simulator.src lazily, so retain its checkout for the call.
    sys.path.insert(0, str(root))
    try:
        if not connector.open_port():
            raise ConnectionError("Native simulator could not load the array")
    finally:
        sys.path.pop(0)
    # Native BoardSimulator reads a cwd-relative ini; explicitly use our board
    # settings before any impact() to keep simulation and conversion consistent.
    connector.interface.config = config
    return Session(connector, protocol)

