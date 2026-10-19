"""Lossless bounded gzip JSONL chunks. No hardware and no deletion of input."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path, PureWindowsPath


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


class Writer:
    def __init__(self, output, limit=64*1024**2):
        if type(limit) is not int or not 1 <= limit <= 64*1024**2:
            raise ValueError("Uncompressed part limit must be 1..64 MiB")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.limit, self.parts, self.stream = limit, [], None
        self.total = 0
        self.expanded_total = 0
        self.all_hash = hashlib.sha256()

    def _close(self):
        if self.stream is None:
            return
        self.stream.close()
        self.file.close()
        path = self.output / self.name
        if path.stat().st_size > 128*1024**2:
            raise ValueError("Compressed part exceeds 128 MiB")
        self.parts.append(dict(path=self.name, compressed_bytes=path.stat().st_size,
                               compressed_sha256=digest(path), bytes=self.size,
                               sha256=self.part_hash.hexdigest(), records=self.records))
        self.stream = None

    def write(self, record):
        line = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8")
        if len(line) > self.limit:
            raise ValueError("Single record exceeds chunk limit")
        if self.expanded_total+len(line)>32*1024**3:
            raise ValueError("Whole raw trace exceeds 32 GiB expanded budget")
        if self.total%4096==0 and shutil.disk_usage(self.output).free<8*1024**3:
            raise ValueError("Less than 8 GiB free; stop before losing raw data")
        if self.stream and self.size+len(line) > self.limit:
            self._close()
        if self.stream is None:
            self.name = f"part-{len(self.parts):06d}.jsonl.gz"
            self.file = (self.output/self.name).open("xb")
            self.stream = gzip.GzipFile(filename="", mode="wb", fileobj=self.file, mtime=0, compresslevel=1)
            self.size = self.records = 0
            self.part_hash = hashlib.sha256()
        self.stream.write(line)
        self.size += len(line)
        self.records += 1
        self.total += 1
        self.expanded_total += len(line)
        self.part_hash.update(line)
        self.all_hash.update(line)

    def finish(self):
        self._close()
        result = dict(schema="lossless-chunked-jsonl-v1", parts=self.parts,
                      records=self.total, sha256=self.all_hash.hexdigest(),gzip_compression_level=1)
        with (self.output/"index.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        return result


def records(directory):
    directory = Path(directory)
    index = json.loads((directory/"index.json").read_text(encoding="utf-8"))
    if index.get("schema") != "lossless-chunked-jsonl-v1":
        raise ValueError("Unexpected raw index")
    total, combined = 0, hashlib.sha256()
    for number, part in enumerate(index["parts"]):
        name = part["path"]
        if name != f"part-{number:06d}.jsonl.gz" or PureWindowsPath(name).drive:
            raise ValueError("Invalid or reordered raw part")
        path = directory/name
        if path.stat().st_size != part["compressed_bytes"] or path.stat().st_size > 128*1024**2 or digest(path) != part["compressed_sha256"]:
            raise ValueError("Compressed raw hash/size mismatch")
        count, size, h = 0, 0, hashlib.sha256()
        with gzip.open(path, "rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    raise ValueError("Truncated raw line")
                count += 1
                size += len(line)
                if size > 64*1024**2:
                    raise ValueError("Expanded raw part exceeds bound")
                h.update(line)
                combined.update(line)
                yield json.loads(line)
        if count != part["records"] or size != part["bytes"] or h.hexdigest() != part["sha256"]:
            raise ValueError("Expanded raw hash/count/size mismatch")
        total += count
    if total != index["records"] or combined.hexdigest() != index["sha256"]:
        raise ValueError("Complete raw stream mismatch")
