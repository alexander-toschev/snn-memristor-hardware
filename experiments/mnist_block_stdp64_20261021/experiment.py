"""Совместимый исторический CLI; фактический сценарий находится в EXP_FMNIST_STDP."""

from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.EXP_FMNIST_STDP.experiment import (
    BASE as BASE,
    SPEC as SPEC,
    main as main,
    run as run,
)

if __name__ == "__main__":
    raise SystemExit(main())
