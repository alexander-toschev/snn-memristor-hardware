from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    value = re.sub(r"-{2,}", "-", value).strip("-._")
    return value or "run"


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_config_for_hash(config: Mapping[str, Any]) -> Dict[str, Any]:
    def _norm(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _norm(v[k]) for k in sorted(v)}
        if isinstance(v, (list, tuple)):
            return [_norm(x) for x in v]
        return v

    return _norm(dict(config))



def config_hash(config: Mapping[str, Any], length: int = 12) -> str:
    normalized = normalize_config_for_hash(config)
    blob = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]



def parse_scalar(value: str) -> Any:
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null" or low == "none":
        return None
    try:
        if value.startswith("0") and value not in {"0", "0.0"} and not value.startswith("0."):
            raise ValueError
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value



def parse_overrides(items: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item}")
        key, raw = item.split("=", 1)
        out[key.strip()] = parse_scalar(raw.strip())
    return out



def merge_dicts(*parts: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for part in parts:
        merged.update(dict(part))
    return merged



def pick_metrics(summary: Mapping[str, Any], names: Iterable[str]) -> Dict[str, Any]:
    return {name: summary.get(name) for name in names if name in summary}



def best_readout(eval_summary: Mapping[str, Any] | None) -> tuple[str | None, float | None]:
    if not eval_summary:
        return None, None
    numeric = {k: float(v) for k, v in eval_summary.items() if isinstance(v, (int, float))}
    if not numeric:
        return None, None
    name = max(numeric, key=numeric.get)
    return name, float(numeric[name])
