"""Общие repo-relative зависимости, INI и label-free шкала проводимости.

Только публичные источники записываются для воспроизводимости. Нет private
receipts, новых approval gates или импортов из других экспериментов.
"""
from __future__ import annotations
from configparser import ConfigParser
import json
import math
from pathlib import Path, PureWindowsPath
import sys
from .durable import sha
from read_all import EXPECTED_BOARD, EXPECTED_SOURCES

ROOT = Path(__file__).resolve().parents[1]
INI = 'configs/raspberry_board.ini'
SIM_INI = 'vendor/MemriBoard/simulator/simulator_settings.ini'
CHARACTERIZATION = 'experiments/mnist_archive64_20261020/calibration'


def repo_path(value: str | Path) -> Path:
    raw = str(value)
    path = Path(raw)
    windows = PureWindowsPath(raw)
    if not raw or path.is_absolute() or windows.drive or windows.root:
        raise ValueError('Use a nonempty path relative to the repository')
    if '..' in path.parts or '..' in windows.parts:
        raise ValueError('Parent-directory traversal is not allowed')
    resolved = (ROOT/path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError('Path must remain inside the repository')
    return resolved


def use_local_dependencies(*, prefer_environment: bool = False) -> None:
    """Добавить public vendor и необязательный локальный dependency fallback.

    prefer_environment сохраняет приоритет пакетов активного интерпретатора:
    например, Windows .deps с NumPy cp312 нельзя ставить перед venv cp311.
    Старый default сохранён для уже существующих сценариев.
    """
    for relative in ('.deps', 'vendor/MemriBoard'):
        directory = repo_path(relative)
        if directory.is_dir() and str(directory) not in sys.path:
            if prefer_environment and relative == '.deps':
                sys.path.append(str(directory))
            else:
                sys.path.insert(0, str(directory))


def preflight(source: str) -> tuple[ConfigParser, dict[str, str]]:
    if source != 'simulator':
        raise ValueError('This MNIST runtime supports native simulator only')
    config = ConfigParser()
    config.read_string((ROOT/INI).read_text(encoding='utf-8-sig'))
    if config.get('board', 'board_type') != 'rp5_fpga_python':
        raise ValueError('Unexpected INI board driver')
    other = ConfigParser()
    other.read_string((ROOT/SIM_INI).read_text(encoding='utf-8-sig'))
    for key, value in EXPECTED_BOARD.items():
        if config.getfloat('board', key) != value or other.getfloat('board', key) != value:
            raise ValueError('Board/native simulator setting mismatch: '+key)
    paths = ['vendor/MemriBoard/'+key for key in EXPECTED_SOURCES if not key.startswith('MemriCORE/')]
    paths += [INI, SIM_INI, 'read_all.py', 'experiment_hardware.py',
              'vendor/MemriBoard/simulator/src.py', 'vendor/MemriBoard/simulator/memristor.py',
              CHARACTERIZATION+'/manifest.json', CHARACTERIZATION+'/observations.json']
    paths += [path.relative_to(ROOT).as_posix() for path in Path(__file__).parent.glob('*.py')]
    return config, {name: sha(ROOT/name) for name in sorted(set(paths))}


def conductance_mapping() -> tuple[float, float]:
    # Это существующие измеренные данные SIM, не импорт кода эксперимента.
    folder = ROOT/CHARACTERIZATION
    meta = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    path = folder/'observations.json'
    if meta['status'] != 'completed' or sha(path) != meta['result_sha256']['observations.json']:
        raise ValueError('Invalid label-free characterization')
    data = json.loads(path.read_text(encoding='utf-8'))['mapping']
    low = float(data['low_g_s'])
    span = float(data['span_g_s'])
    if not all(math.isfinite(value) for value in (low, span)) or min(low, span) <= 0:
        raise ValueError('Invalid G scale')
    return low, span


def query_cost(keys: list[int]) -> int:
    """Три mode10 прохода восьмиWL плюс freshR каждого активногоBL."""
    return 24*sum(1+mask.bit_count() for mask in set(keys) | {0})
