"""Copy two hash-verified operator-provided files locally; never import a driver."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = {
    "rpi_FPGAed.py": "bc717659fa5d4d36d2037eef978bf1cd8dcf0fe92f9bbfad51f1eb1d0d27df9c",
    "FPGA_SIM.py": "5bcb02c04bc185e332b2ec96acabc2d4303d21724c9c052de540ef92d603b634",
}


def copy_verified(source_root, destination):
    prepared = {}
    for name, expected in FILES.items():
        content = (source_root / "MemriCORE/rp5_fpga_python" / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Unreviewed installed driver: {name}")
        target = destination / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Existing destination differs; nothing overwritten: {name}")
        prepared[name] = content
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name, content in prepared.items():
        target = destination / name
        if target.exists():
            continue
        with target.open("xb") as stream:
            stream.write(content)
        copied.append(name)
    return {"copied": copied, "source_sha256": FILES, "hardware_access": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="../MemriBoard", help="Operator's installed MemriBoard, relative to this repository")
    args = parser.parse_args(argv)
    destination = (ROOT / "vendor/MemriBoard/MemriCORE/rp5_fpga_python").resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("Destination escapes this repository")
    result = copy_verified((ROOT / args.source).resolve(), destination)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
