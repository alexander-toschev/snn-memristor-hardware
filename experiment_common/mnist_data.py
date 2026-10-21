"""Общие официальные MNIST IDX, без импорта другого эксперимента."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
# Существующий репозиторный набор данных; не дублируем его во всех опытах.
DATA = ROOT/'experiments/mnist_baseline_20261019/data/MNIST/raw'
SOURCES = {
    'train-images-idx3-ubyte.gz': '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609',
    'train-labels-idx1-ubyte.gz': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c',
    't10k-images-idx3-ubyte.gz': '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6',
    't10k-labels-idx1-ubyte.gz': 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6',
}


def checked(name: str) -> bytes:
    path = DATA/name
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCES[name]:
        raise ValueError('Official IDX SHA differs: '+name)
    return data
