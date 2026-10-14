"""Lossless, bounded-size gzip parts for committing large simulator traces."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

if __package__:
    from .paths import repo_path, relative_path
else:
    from paths import repo_path, relative_path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


PART_BYTES = 64*1024*1024


def load_storage(directory):
    """Validate the manifest shape and declared totals, without decompressing."""
    storage = json.loads((directory / "trace_storage.json").read_text(encoding="utf-8"))
    if not isinstance(storage, dict) or storage.get("format") != "ordered_gzip_jsonl_parts":
        raise ValueError("Unknown trace storage format")
    parts = storage.get("parts")
    if not isinstance(parts, list) or not parts or any(not isinstance(p, dict) for p in parts):
        raise ValueError("Trace storage requires a nonempty part list")
    for metadata in [storage, *parts]:
        for key in ("records", "uncompressed_bytes"):
            if type(metadata.get(key)) is not int or metadata[key] <= 0:
                raise ValueError(f"Invalid trace metadata: {key}")
        digest = metadata.get("uncompressed_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid uncompressed SHA256")
    for part in parts:
        if not isinstance(part.get("path"), str) or not part["path"]:
            raise ValueError("Invalid trace part path")
        if type(part.get("compressed_bytes")) is not int or part["compressed_bytes"] <= 0:
            raise ValueError("Invalid compressed size")
        digest = part.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid compressed SHA256")
    for key in ("records", "uncompressed_bytes"):
        if sum(part[key] for part in parts) != storage[key]:
            raise ValueError(f"Trace part totals disagree: {key}")
    return storage


def raw_parts(directory, *, storage=None):
    """Prefer stored parts when present, so a retained raw file cannot hide damage."""
    directory = Path(directory).resolve()
    if storage is None and not (directory / "trace_storage.json").exists():
        raw = directory / "raw_tasks.jsonl"
        if not raw.is_file():
            raise FileNotFoundError("No raw trace or storage manifest")
        return [raw]
    storage = load_storage(directory) if storage is None else storage
    parts = [repo_path(part["path"]) for part in storage["parts"]]
    if len(set(parts)) != len(parts) or any(not p.is_relative_to(directory) or p.suffix != ".gz" for p in parts):
        raise ValueError("Distinct gzip trace parts must remain in the session directory")
    for path, metadata in zip(parts, storage["parts"]):
        if path.stat().st_size != metadata["compressed_bytes"] or sha256(path) != metadata["sha256"]:
            raise ValueError("Compressed trace size or hash mismatch")
    return parts


def raw_lines(parts, *, storage=None, verification=None):
    """Stream exact bytes into hashes before UTF-8 decoding; never normalize CRLF.

    The caller must consume the generator completely for final hash validation.
    Compressed input requires its manifest; raw_parts alone checks compressed bytes.
    """
    if not parts or (storage is None and any(path.suffix == ".gz" for path in parts)):
        raise ValueError("A storage manifest is required for compressed trace validation")
    if storage is not None and len(parts) != len(storage["parts"]):
        raise ValueError("Trace part count mismatch")
    total_digest, total_size, total_records = hashlib.sha256(), 0, 0
    for index, path in enumerate(parts):
        digest, size, records = hashlib.sha256(), 0, 0
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    raise ValueError("Incomplete raw JSONL record")
                digest.update(line)
                total_digest.update(line)
                size += len(line)
                records += 1
                total_size += len(line)
                total_records += 1
                yield line.decode("utf-8-sig" if total_records == 1 else "utf-8")
        if storage is not None:
            metadata = storage["parts"][index]
            if (digest.hexdigest(), size, records) != (metadata["uncompressed_sha256"], metadata["uncompressed_bytes"], metadata["records"]):
                raise ValueError("Decompressed trace part hash, size or record count mismatch")
    if storage is not None and (total_digest.hexdigest(), total_size, total_records) != (
        storage["uncompressed_sha256"], storage["uncompressed_bytes"], storage["records"]
    ):
        raise ValueError("Reconstructed trace hash, size or record count mismatch")
    if verification is not None:
        verification.update(sha256=total_digest.hexdigest(), bytes=total_size, records=total_records)


def pack(directory):
    directory = Path(directory).resolve()
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if audit.get("passed") is not True or audit.get("run_complete") is not True:
        raise ValueError("A completed, passing integrity audit is required")
    if (directory / "trace_storage.json").exists():
        raise FileExistsError("Trace storage manifest already exists")
    if type(audit.get("raw_records")) is not int or audit["raw_records"] <= 0:
        raise ValueError("A nonempty audited trace is required")
    expected_digest = audit.get("trace_uncompressed_sha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError("A fresh audit with the exact raw byte hash is required")
    original = directory / "raw_tasks.jsonl"
    source_stat = original.stat()
    audit_digest = sha256(directory / "audit.json")
    output = directory / "trace"
    output.mkdir(exist_ok=False)
    original_digest = hashlib.sha256()
    parts, total_rows = [], 0
    source_size = source_stat.st_size
    with original.open("rb") as source:
        line = source.readline(PART_BYTES+1)
        while line:
            if len(line) > PART_BYTES:
                raise ValueError("A single record exceeds the 64 MiB part limit")
            part_path = output / f"part_{len(parts)+1:04d}.jsonl.gz"
            digest, size, count = hashlib.sha256(), 0, 0
            with part_path.open("xb") as destination:
                with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressed:
                    while line and size + len(line) <= PART_BYTES:
                        if not line.endswith(b"\n"):
                            raise ValueError("Incomplete raw JSONL record")
                        compressed.write(line)
                        digest.update(line)
                        original_digest.update(line)
                        size += len(line)
                        count += 1
                        line = source.readline(PART_BYTES+1)
            parts.append({"path": relative_path(part_path), "records": count,
                          "uncompressed_bytes": size, "uncompressed_sha256": digest.hexdigest(),
                          "compressed_bytes": part_path.stat().st_size, "sha256": sha256(part_path)})
            total_rows += count
            print(f"Packed part {len(parts)}: {total_rows} records", flush=True)
    if (original.stat().st_size, original.stat().st_mtime_ns) != (source_size, source_stat.st_mtime_ns) or total_rows != audit["raw_records"]:
        raise ValueError("Source changed or audited count differs")
    if original_digest.hexdigest() != expected_digest:
        raise ValueError("Source differs from the exact bytes checked by the audit")
    restored = hashlib.sha256()
    for part in parts:
        digest = hashlib.sha256()
        with gzip.open(repo_path(part["path"]), "rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
                restored.update(block)
        if digest.hexdigest() != part["uncompressed_sha256"]:
            raise ValueError("Decompression verification failed")
    if restored.hexdigest() != original_digest.hexdigest() or sha256(original) != restored.hexdigest():
        raise ValueError("Reconstructed trace does not exactly match source")
    if sha256(directory / "audit.json") != audit_digest:
        raise ValueError("Audit changed during packing")
    result = {"format": "ordered_gzip_jsonl_parts", "records": total_rows,
              "uncompressed_bytes": source_size, "uncompressed_sha256": restored.hexdigest(),
              "audit_sha256": audit_digest,
              "packer_sha256": sha256(Path(__file__)), "parts": parts}
    with (directory / "trace_storage.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    # Removal is deliberately a separate, verified repository cleanup operation.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Session path relative to the repository")
    args = parser.parse_args()
    result = pack(repo_path(args.session))
    print(json.dumps({"records": result["records"], "parts": len(result["parts"]),
                      "lossless_verified": True}))


if __name__ == "__main__":
    main()
