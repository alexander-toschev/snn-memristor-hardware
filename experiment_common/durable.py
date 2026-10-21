"""Общий typed writer: закрытые gzip-части, atomic JSON, fsync и disk reserve.

Формат сохранён от прежнего durable.Writer. Завершённый поток не означает
завершённый научный эксперимент: этот статус задаёт experiment.py.
"""
from __future__ import annotations
from collections.abc import Mapping
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import BinaryIO


def sha(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name != 'nt':
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class Writer:
    def __init__(self, directory: str | Path, limit: int = 8*1024**2,
                 maximum_expanded_bytes: int = 2*1024**3,
                 minimum_free_bytes: int = 8*1024**3) -> None:
        if not 1 <= limit <= 64*1024**2:
            raise ValueError('Invalid lossless chunk limit')
        directory = Path(directory)
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.limit = limit
        self.maximum = maximum_expanded_bytes
        self.reserve = minimum_free_bytes
        self.parts: list[dict[str, object]] = []
        self.stream: gzip.GzipFile | None = None
        self.file: BinaryIO | None = None
        self.total = self.expanded = self.size = self.records = 0
        self.all_hash = hashlib.sha256()
        self.part_hash = hashlib.sha256()
        self.name = ''
        self.finished = False
        self.checkpoint(reason='writer_created')

    def _close(self) -> None:
        if self.stream is None:
            return
        if self.file is None:
            raise ValueError('Missing backing raw file')
        self.stream.close()
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        path = self.directory/self.name
        if path.stat().st_size > 128*1024**2:
            raise ValueError('Compressed raw chunk too large')
        self.parts.append(dict(path=self.name, compressed_bytes=path.stat().st_size,
            compressed_sha256=sha(path), bytes=self.size,
            sha256=self.part_hash.hexdigest(), records=self.records))
        self.stream = None
        self.file = None

    def checkpoint(self, **progress: object) -> dict[str, object]:
        self._close()
        index: dict[str, object] = dict(schema='lossless-chunked-jsonl-v1', parts=self.parts,
            records=self.total, sha256=self.all_hash.hexdigest(), gzip_compression_level=1,
            stream_finalized=self.finished,
            checkpoint_semantics='Durable closed parts only; not an acquisition-completion or within-stage-resume claim')
        atomic_json(self.directory/'index.json', index)
        atomic_json(self.directory.parent/'progress.json', dict(records=self.total,
            expanded_bytes=self.expanded, closed_parts=len(self.parts),
            raw_index_sha256=sha(self.directory/'index.json'), **progress))
        return index

    def write(self, record: Mapping[str, object]) -> None:
        if self.finished:
            raise ValueError('Raw writer already finalized')
        line = (json.dumps(dict(record), sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
        if len(line) > self.limit or self.expanded+len(line) > self.maximum:
            raise ValueError('Finite raw storage budget exhausted')
        if self.total % 4096 == 0 and shutil.disk_usage(self.directory).free < self.reserve:
            raise ValueError('Raw disk reserve reached')
        if self.stream is not None and self.size+len(line) > self.limit:
            self.checkpoint(reason='raw_chunk_boundary')
        if self.stream is None:
            self.name = 'part-%06d.jsonl.gz' % len(self.parts)
            self.file = (self.directory/self.name).open('xb')
            self.stream = gzip.GzipFile(filename='', mode='wb', fileobj=self.file, mtime=0, compresslevel=1)
            self.size = self.records = 0
            self.part_hash = hashlib.sha256()
        self.stream.write(line)
        self.size += len(line)
        self.records += 1
        self.total += 1
        self.expanded += len(line)
        self.part_hash.update(line)
        self.all_hash.update(line)

    def finish(self) -> dict[str, object]:
        self.finished = True
        return self.checkpoint(reason='stream_finalized_not_stage_qualification')
