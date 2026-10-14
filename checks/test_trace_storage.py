"""Independent lossless-storage tests; only disposable synthetic files are used."""

import copy
import gzip
import gc
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest

REVIEW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVIEW_ROOT / "experiments" / "unn_rev3_20261014"))
import trace_storage as storage
import verify_run


class TraceStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="synthetic-trace-", dir=REVIEW_ROOT)
        cls.directory = Path(cls.temp.name).resolve()
        assert cls.directory.is_relative_to(REVIEW_ROOT.resolve())
        cls.original = cls.directory / "raw_tasks.jsonl"
        rng, count, size = random.Random(12345), 0, 0
        cls.unicode_text = "Привет, плата — 器件 🧪 café"
        with cls.original.open("xb") as stream:
            while size < 66*1024*1024:
                count += 1
                row = {"task_id": count, "session_id": "synthetic", "source": "simulator",
                       "array_id": "A", "phase": "synthetic", "wl": 0, "bl": 2,
                       "task": {"mode_flag": 7, "id": count, "wl": 0, "bl": 2,
                                "vol": 0, "sign": 0, "pulse_width": 0},
                       "raw_adc": 500, "resistance_ohm": 1000, "status": "ok", "latency_s": .001,
                       "unicode": cls.unicode_text, "payload": rng.randbytes(4096).hex()}
                line = json.dumps(row, ensure_ascii=False).encode("utf-8") + (b"\r\n" if count % 3 == 0 else b"\n")
                if count == 1:
                    line = b"\xef\xbb\xbf" + line
                stream.write(line)
                size += len(line)
        cls.count, cls.size = count, size
        cls.original_sha = storage.sha256(cls.original)
        (cls.directory/"audit.json").write_text(json.dumps({
            "passed": True, "run_complete": True, "raw_records": count, "synthetic_fixture": True,
            "trace_uncompressed_sha256": cls.original_sha}), encoding="utf-8")
        cls.packed = storage.pack(cls.directory)
        assert cls.original.is_file(), "pack must not delete raw input"
        assert len(cls.packed["parts"]) == 2
        (cls.directory/"manifest.json").write_text(json.dumps({
            "session_id": "synthetic", "source": "simulator", "status": "completed", "raw_task_count": count}), encoding="utf-8")
        (cls.directory/"protocol.json").write_text(json.dumps({"suite": {"session_id": "synthetic"}}), encoding="utf-8")
        (cls.directory/"summary.json").write_text("{}", encoding="utf-8")
        cls.metadata_path = cls.directory/"trace_storage.json"
        cls.saved_metadata = cls.metadata_path.read_bytes()

    @classmethod
    def tearDownClass(cls):
        # Verify the resolved deletion target explicitly before TemporaryDirectory cleanup.
        assert cls.directory.is_relative_to(REVIEW_ROOT.resolve())
        gc.collect()
        for attempt in range(5):
            try:
                cls.temp.cleanup()
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.25 * (attempt+1))

    def tearDown(self):
        self.metadata_path.write_bytes(self.saved_metadata)

    def changed_metadata(self, change):
        metadata = copy.deepcopy(self.packed)
        change(metadata)
        self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return metadata

    def consume(self):
        metadata = storage.load_storage(self.directory)
        return sum(1 for _ in storage.raw_lines(storage.raw_parts(self.directory, storage=metadata), storage=metadata))

    def test_01_roundtrip_two_parts_exact_unicode_crlf_bom(self):
        self.assertGreater(self.size, 66*1024*1024)
        self.assertEqual(len(self.packed["parts"]), 2)
        self.assertTrue(all(p["uncompressed_bytes"] <= 64*1024*1024 for p in self.packed["parts"]))
        restored = self.directory/"restored.jsonl"
        with restored.open("xb") as destination:
            for part in self.packed["parts"]:
                digest, size, records = hashlib.sha256(), 0, 0
                part_path = storage.repo_path(part["path"])
                self.assertEqual(storage.sha256(part_path), part["sha256"])
                self.assertEqual(part_path.stat().st_size, part["compressed_bytes"])
                with gzip.open(part_path, "rb") as stream:
                    for block in iter(lambda: stream.read(1024*1024), b""):
                        destination.write(block)
                        digest.update(block)
                        size += len(block)
                        records += block.count(b"\n")
                self.assertEqual((digest.hexdigest(), size, records),
                                 (part["uncompressed_sha256"], part["uncompressed_bytes"], part["records"]))
        self.assertEqual(storage.sha256(restored), self.original_sha)
        self.assertEqual(self.packed["uncompressed_sha256"], self.original_sha)
        with self.original.open("rb") as left, restored.open("rb") as right:
            for a, b in itertools.zip_longest(iter(lambda: left.read(1024*1024), b""),
                                             iter(lambda: right.read(1024*1024), b"")):
                self.assertEqual(a, b)
        metadata = storage.load_storage(self.directory)
        count = 0
        for line in storage.raw_lines(storage.raw_parts(self.directory), storage=metadata):
            count += 1
            row = json.loads(line)
            self.assertEqual(row["task_id"], count)
            self.assertEqual(row["unicode"], self.unicode_text)
        self.assertEqual(count, self.count)
        print(json.dumps({"synthetic_bytes": self.size, "records": self.count,
                          "parts": len(self.packed["parts"]), "sha256": self.original_sha,
                          "compressed_bytes": sum(p["compressed_bytes"] for p in self.packed["parts"])}), flush=True)

    def test_02_audit_verifies_archive_with_and_without_raw(self):
        self.assertTrue(all(path.suffix == ".gz" for path in storage.raw_parts(self.directory)))
        for without_raw in (False, True):
            saved = self.directory/"saved-original.jsonl"
            if without_raw:
                self.original.rename(saved)
            try:
                audit, _ = verify_run.audit_run(self.directory)
                self.assertEqual(audit["raw_records"], self.count)
                self.assertTrue(audit["trace_storage_verified"])
                self.assertEqual(audit["trace_uncompressed_sha256"], self.original_sha)
                self.assertNotIn("artifact", audit["issue_counts"])
                # Synthetic data lacks scientific phase coverage; never call it a passing experiment.
                self.assertFalse(audit["passed"])
            finally:
                if without_raw:
                    saved.rename(self.original)

    def test_03_uncompressed_metadata_corruption(self):
        changes = [lambda m: m["parts"][0].update(uncompressed_sha256="0"*64),
                   lambda m: m.update(uncompressed_sha256="0"*64),
                   lambda m: (m["parts"][0].update(records=m["parts"][0]["records"]+1), m.update(records=m["records"]+1)),
                   lambda m: (m["parts"][0].update(uncompressed_bytes=m["parts"][0]["uncompressed_bytes"]+1),
                              m.update(uncompressed_bytes=m["uncompressed_bytes"]+1)),
                   lambda m: m["parts"].reverse()]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self.changed_metadata(change)
                with self.assertRaises(ValueError):
                    self.consume()

    def test_04_compressed_corruption_detected_even_with_raw_retained(self):
        path = storage.repo_path(self.packed["parts"][0]["path"])
        with path.open("r+b") as stream:
            stream.seek(100)
            original_byte = stream.read(1)
            stream.seek(100)
            stream.write(bytes([original_byte[0] ^ 1]))
        try:
            with self.assertRaises(ValueError):
                storage.raw_parts(self.directory)
            audit, _ = verify_run.audit_run(self.directory)
            self.assertFalse(audit["passed"])
            self.assertFalse(audit["trace_storage_verified"])
            self.assertIn("artifact", audit["issue_counts"])
        finally:
            with path.open("r+b") as stream:
                stream.seek(100)
                stream.write(original_byte)

    def test_05_truncated_gzip_with_updated_compressed_hash_is_audit_failure(self):
        path = storage.repo_path(self.packed["parts"][-1]["path"])
        size = path.stat().st_size
        with path.open("r+b") as stream:
            stream.seek(size-8)
            trailer = stream.read(8)
            stream.truncate(size-8)
        try:
            self.changed_metadata(lambda m: m["parts"][-1].update(
                compressed_bytes=path.stat().st_size, sha256=storage.sha256(path)))
            audit, _ = verify_run.audit_run(self.directory)
            self.assertFalse(audit["passed"])
            self.assertFalse(audit["trace_storage_verified"])
            self.assertIn("artifact", audit["issue_counts"])
        finally:
            with path.open("ab") as stream:
                stream.write(trailer)

    def test_06_duplicate_and_escaping_parts_rejected(self):
        for change in (lambda m: m["parts"][1].update(path=m["parts"][0]["path"]),
                       lambda m: m["parts"][0].update(path="../outside.gz")):
            self.changed_metadata(change)
            with self.assertRaises(ValueError):
                storage.raw_parts(self.directory)

    def test_07_repacking_refused_without_touching_original(self):
        with self.assertRaises(FileExistsError):
            storage.pack(self.directory)
        self.assertEqual(storage.sha256(self.original), self.original_sha)

    def test_08_partial_jsonl_never_gets_storage_manifest(self):
        tiny = self.directory/"partial"
        tiny.mkdir()
        (tiny/"raw_tasks.jsonl").write_bytes(b'{"id":1}\n{"id":2}')
        (tiny/"audit.json").write_text(json.dumps({"passed": True, "run_complete": True, "raw_records": 2,
                                                   "trace_uncompressed_sha256": storage.sha256(tiny/"raw_tasks.jsonl")}))
        with self.assertRaises(ValueError):
            storage.pack(tiny)
        self.assertFalse((tiny/"trace_storage.json").exists())
        self.assertEqual((tiny/"raw_tasks.jsonl").read_bytes(), b'{"id":1}\n{"id":2}')

    def test_09_same_count_source_change_invalidates_audit_binding(self):
        tiny = self.directory/"stale_audit"
        tiny.mkdir()
        raw = tiny/"raw_tasks.jsonl"
        raw.write_bytes(b'{"value":1}\n')
        (tiny/"audit.json").write_text(json.dumps({"passed": True, "run_complete": True, "raw_records": 1,
                                                   "trace_uncompressed_sha256": storage.sha256(raw)}))
        raw.write_bytes(b'{"value":2}\n')
        with self.assertRaisesRegex(ValueError, "exact bytes"):
            storage.pack(tiny)
        self.assertFalse((tiny/"trace_storage.json").exists())
        self.assertEqual(raw.read_bytes(), b'{"value":2}\n')

    def test_10_unpacked_audit_records_exact_byte_identity(self):
        tiny = self.directory/"raw_only"
        tiny.mkdir()
        original = b'\xef\xbb\xbf' + json.dumps({"unicode": self.unicode_text}, ensure_ascii=False).encode("utf-8") + b"\r\n"
        (tiny/"raw_tasks.jsonl").write_bytes(original)
        audit, _ = verify_run.audit_run(tiny)
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["trace_storage_verified"])
        self.assertEqual(audit["trace_uncompressed_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(audit["trace_uncompressed_bytes"], len(original))


if __name__ == "__main__":
    unittest.main(verbosity=2)
