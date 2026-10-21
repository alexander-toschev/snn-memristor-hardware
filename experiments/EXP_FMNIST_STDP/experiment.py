"""EXP_FMNIST_STDP — пользовательское имя опыта MNIST block-STDP.

FMNIST здесь является именем эксперимента, а не переключением на Fashion-MNIST.
Тонкая точка входа передаёт неизменённые CLI-аргументы существующей реализации.
Ни схема опыта, ни пути результатов, ни работающий процесс не изменяются.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.mnist_block_stdp64_20261021.experiment import main as implementation_main


def main() -> int:
    """Разбор sys.argv и выполнение остаются в прежнем typed entrypoint."""
    return implementation_main()


if __name__ == '__main__':
    raise SystemExit(main())
