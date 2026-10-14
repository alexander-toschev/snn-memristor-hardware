"""Local configuration only. Loading a protocol never connects to a device."""

from dataclasses import dataclass
from configparser import ConfigParser
import json
import math
from pathlib import Path


def positive_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


@dataclass(frozen=True)
class Board:
    dac_bit: int
    vol_ref_dac: float
    res_load: float
    vol_read: float
    adc_bit: int
    vol_ref_adc: float
    res_switches: float
    gain: float
    sum_gain: float
    read_time: int
    blank_time: int
    blank_time_between: int
    soft_cc: float

    def __post_init__(self):
        for name, value in vars(self).items():
            positive_number(value, name)
        for name in ("dac_bit", "adc_bit", "read_time", "blank_time", "blank_time_between"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.adc_bit != 14 or self.dac_bit != 12:
            raise ValueError("This revision expects the 14-bit ADC / 12-bit DAC board")
        if self.vol_read > 0.3:
            raise ValueError("vol_read exceeds the 0.3 V readback limit")


@dataclass(frozen=True)
class Protocol:
    board: Board
    initial_cells: tuple[tuple[int, int], ...]  # (WL, BL); never implicit row/column
    excluded_wl: tuple[int, ...]
    wl_count: int
    bl_count: int
    connection: dict
    execution: dict
    read_baseline: dict
    protocol_id: str

    def board_config(self) -> ConfigParser:
        config = ConfigParser()
        config.read_dict({"board": {name: str(value) for name, value in vars(self.board).items()}})
        return config

    def check_cell(self, wl, bl):
        if type(wl) is not int or type(bl) is not int:
            raise ValueError("WL and BL must be integers")
        if not (0 <= wl < self.wl_count and 0 <= bl < self.bl_count):
            raise ValueError(f"Cell WL={wl}, BL={bl} is outside the board")
        if wl in self.excluded_wl:
            raise ValueError(f"WL={wl} is excluded")
        if (wl, bl) not in self.initial_cells:
            raise ValueError(f"Cell WL={wl}, BL={bl} is outside the initial readback set")


def load_protocol(path: Path) -> Protocol:
    return parse_protocol(Path(path).read_bytes())


def parse_protocol(content: bytes) -> Protocol:
    data = json.loads(content.decode("utf-8-sig"))
    execution = data["execution"]
    for key in ("approved", "simulator_enabled", "hardware_enabled"):
        if type(execution[key]) is not bool:
            raise ValueError(f"execution.{key} must be a boolean")
    geometry = data["geometry"]
    if geometry["wl_count"] != 8 or geometry["bl_count"] != 32:
        raise ValueError("Expected 8 WL and 32 BL")
    excluded = tuple(geometry["excluded_wl"])
    if any(type(wl) is not int or not 0 <= wl < 8 for wl in excluded) or 1 not in excluded:
        raise ValueError("excluded_wl must contain WL=1 and valid integer WL indices")
    cells = tuple((cell["wl"], cell["bl"]) for cell in geometry["initial_cells"])
    if len(cells) != 3 or len(set(cells)) != 3:
        raise ValueError("Expected three distinct initial cells")
    connection = {
        "cb_type": "simulator", "board_type": "rp5_fpga_python", "task_format": "SI",
        "crossbar_serial": None, "memriboard_root": None,
        **data.get("connection", {}),
    }
    if connection["cb_type"] not in ("simulator", "real") or connection["task_format"] != "SI":
        raise ValueError("Expected cb_type simulator|real and task_format SI")
    baseline = data["read_baseline"]
    if type(baseline["enabled"]) is not bool:
        raise ValueError("read_baseline.enabled must be a boolean")
    for key in ("first_check_reads_per_cell", "repeatability_reads_per_cell_after_first_check"):
        if type(baseline[key]) is not int or not 1 <= baseline[key] <= 20:
            raise ValueError(f"read_baseline.{key} must be an integer in [1, 20]")
    protocol_id = data["protocol_id"]
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError("protocol_id must be a nonempty string")
    protocol = Protocol(Board(**data["board"]), cells, excluded, 8, 32,
                        connection, execution, baseline, protocol_id)
    for wl, bl in cells:
        protocol.check_cell(wl, bl)
    return protocol
