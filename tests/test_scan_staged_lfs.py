"""Offline fixtures only. No repository mutation, network, or real credentials."""
import hashlib
import gzip
import io
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'security'))
import scan_staged_lfs as scan


def pointer(data):
    return ('version https://git-lfs.github.com/spec/v1\n'
            'oid sha256:%s\nsize %d\n' % (hashlib.sha256(data).hexdigest(), len(data))).encode()


def cache(root, data):
    oid = hashlib.sha256(data).hexdigest()
    path = Path(root) / oid[:2] / oid[2:4] / oid
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return oid, path


class PayloadTests(unittest.TestCase):
    @staticmethod
    def scanner_process(code):
        process = mock.Mock()
        process.stdin.write.side_effect = len
        process.stdin.closed = False
        process.wait.return_value = code
        process.poll.return_value = code
        return process

    def test_framing_preserves_all_original_bytes_and_counts(self):
        payload = b'\x1f\x8b\0binary\xff\n' * 100000
        process = self.scanner_process(0)
        chunks = [payload[:123], payload[123:]]
        with mock.patch.object(scan.subprocess, 'Popen', return_value=process):
            result = scan.scan_chunks('scanner', iter(chunks), '.')
        written = b''.join(c.args[0] for c in process.stdin.write.call_args_list)
        self.assertEqual(written, scan.SCAN_FRAME + payload)
        self.assertGreaterEqual(len(scan.SCAN_FRAME), 1024)
        self.assertTrue(scan.SCAN_FRAME.isascii())
        self.assertIn(b'\n', scan.SCAN_FRAME)
        self.assertEqual(result['payload_bytes'], len(payload))
        self.assertEqual(result['scanner_input_bytes'], len(written))

    def test_early_close_zero_exit_and_short_writes_block(self):
        for failure in (BrokenPipeError('synthetic'), 0):
            process = self.scanner_process(0)
            process.stdin.write.side_effect = [len(scan.SCAN_FRAME), failure]
            with mock.patch.object(scan.subprocess, 'Popen', return_value=process):
                with self.assertRaises(scan.Blocked):
                    scan.scan_chunks('scanner', [b'payload'], '.')

    def test_scanner_timeout_blocks_and_kills(self):
        process = self.scanner_process(0)
        process.wait.side_effect = [subprocess.TimeoutExpired('scanner', 300), 0]
        process.poll.return_value = None
        with mock.patch.object(scan.subprocess, 'Popen', return_value=process):
            with self.assertRaises(scan.Blocked):
                scan.scan_chunks('scanner', [b'payload'], '.')
        process.kill.assert_called_once()

    def test_chunk_source_failure_blocks_without_acceptance(self):
        def chunks():
            yield b'prefix'
            raise OSError('unread tail')
        with mock.patch.object(scan.subprocess, 'Popen', return_value=self.scanner_process(0)):
            with self.assertRaises(scan.Blocked):
                scan.scan_chunks('scanner', chunks(), '.')

    def test_pointer_and_crlf(self):
        data = b'example\0binary\xff'
        for blob in (pointer(data), pointer(data).replace(b'\n', b'\r\n')):
            self.assertEqual(scan.parse_pointer(blob), (hashlib.sha256(data).hexdigest(), len(data)))

    def test_malformed_or_extended_lfs_fails_closed(self):
        for blob in (b'ordinary bytes', pointer(b'a').replace(b'size 1', b'size -1'),
                     pointer(b'a') + b'ext-0-custom unknown\n'):
            with self.assertRaises(scan.Blocked):
                scan.parse_pointer(blob, True)
        self.assertIsNone(scan.parse_pointer(b'ordinary bytes', False))

    def test_large_binary_snapshot_preserves_every_byte(self):
        data = b'\0\xffbinary\n' * 1500000 + b'END-OF-CONTENT'
        with tempfile.TemporaryDirectory() as root:
            oid, _ = cache(root, data)
            with tempfile.TemporaryFile() as dest:
                scan.copy_verified_object(root, oid, len(data), dest)
                self.assertEqual(hashlib.sha256(dest.read()).hexdigest(), oid)

    def test_missing_cache_never_uses_worktree(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, 'payload.bin').write_bytes(b'worktree would look harmless')
            with self.assertRaises(scan.Blocked):
                scan.copy_verified_object(root, '0' * 64, 10, io.BytesIO())

    def test_size_and_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            oid, path = cache(root, b'original')
            for size in (7, 9):
                with self.assertRaises(scan.Blocked):
                    scan.copy_verified_object(root, oid, size, io.BytesIO())
            path.write_bytes(b'changed!')  # same size, wrong SHA
            with self.assertRaises(scan.Blocked):
                scan.copy_verified_object(root, oid, 8, io.BytesIO())

    def test_cache_env_handles_spaces_and_linked_worktree(self):
        with mock.patch.object(scan, 'git', return_value=b'Endpoint=DO-NOT-PRINT\nLocalMediaDir=/a path/store\n'):
            self.assertEqual(scan.local_media_dir('.'), Path('/a path/store'))

    def test_index_paths_are_nul_separated_not_line_split(self):
        names = [b'space name.bin', b'line\nbreak\tname.bin']
        oid = b'a' * 40
        def query(repo, *args, data=None):
            if args[0] == 'diff':
                self.assertIn('--no-renames', args)
                return b'\0'.join(names) + b'\0'
            if args[0] == 'check-attr':
                self.assertEqual(set(data.split(b'\0')[:-1]), set(names))
                return b''.join(n + b'\0filter\0lfs\0' for n in names)
            return b''.join(b'100644 ' + oid + b' 0\t' + n + b'\0' for n in names)
        with mock.patch.object(scan, 'git', side_effect=query):
            actual = scan.staged_entries('.')
        self.assertEqual({x[0] for x in actual}, set(names))

    def test_staged_object_not_unstaged_payload_is_scanned_and_deduplicated(self):
        payload = b'only staged bytes\0\xff'
        entries = [(b'a.bin', b'100644', b'a' * 40, b'lfs'),
                   (b'b.bin', b'100644', b'a' * 40, b'lfs')]
        with tempfile.TemporaryDirectory() as root:
            cache(root, payload)
            Path(root, 'a.bin').write_bytes(b'unstaged harmless replacement')
            captured = []
            def query(repo, *args, data=None):
                return str(len(pointer(payload))).encode() if args[1] == '-s' else pointer(payload)
            with mock.patch.object(scan, 'staged_entries', return_value=entries), \
                 mock.patch.object(scan, 'git', side_effect=query), \
                 mock.patch.object(scan, 'local_media_dir', return_value=Path(root)), \
                 mock.patch.object(scan, 'find_scanner', return_value='scanner'), \
                 mock.patch.object(scan, 'scan_stream', side_effect=lambda exe, stream, cwd: captured.append(stream.read())):
                self.assertEqual(scan.scan(root), 1)
            self.assertEqual(captured, [payload])

    def test_index_change_blocks(self):
        with mock.patch.object(scan, 'staged_entries', side_effect=[[], [(b'new',)]]):
            with self.assertRaises(scan.Blocked):
                scan.scan('.')

    def test_scanner_mandatory_flags_and_isolated_config(self):
        with mock.patch.dict(os.environ, {'GITLEAKS_CONFIG':'suppress.toml', 'GITLEAKS_CONFIG_TOML':'anything'}), \
             mock.patch.object(scan.subprocess, 'Popen', return_value=self.scanner_process(0)) as call:
            scan.scan_stream('scanner', io.BytesIO(b'bytes'), 'scratch')
        args, kwargs = call.call_args
        for flag in ('stdin', '--redact=100', '--ignore-gitleaks-allow', '--max-target-megabytes=0'):
            self.assertIn(flag, args[0])
        self.assertEqual(kwargs['cwd'], 'scratch')
        self.assertNotIn('GITLEAKS_CONFIG', kwargs['env'])
        self.assertNotIn('GITLEAKS_CONFIG_TOML', kwargs['env'])
        self.assertEqual(kwargs['stdout'], subprocess.DEVNULL)
        self.assertEqual(kwargs['stderr'], subprocess.DEVNULL)

    def test_scanner_finding_and_error_both_block(self):
        for code in (1, 2, 127):
            with mock.patch.object(scan.subprocess, 'Popen', return_value=self.scanner_process(code)):
                with self.assertRaises(scan.Blocked):
                    scan.scan_stream('scanner', io.BytesIO(b'anything'), '.')

    def test_main_never_prints_arbitrary_exception(self):
        with mock.patch.object(scan, 'git', side_effect=ValueError('SENSITIVE-EXCEPTION')), \
             mock.patch('sys.stderr', new_callable=io.StringIO) as err:
            self.assertEqual(scan.main(), 1)
        self.assertNotIn('SENSITIVE-EXCEPTION', err.getvalue())

    def test_hook_retains_original_staged_scan_before_lfs(self):
        hook = (Path(__file__).resolve().parents[1] / '.githooks' / 'pre-commit').read_text()
        self.assertLess(hook.index('"$scanner" git --staged'), hook.index('tools/security/scan_staged_lfs.py'))
        self.assertIn('set -eu', hook)
        self.assertIn('sys.version_info >= (3, 8)', hook)
        self.assertIn("else 1)' >/dev/null 2>&1; then", hook)
        self.assertIn('if [ -z "$python" ]; then', hook)

    def test_hook_skips_nonworking_python3_alias(self):
        shell = shutil.which('sh')
        if shell is None and Path('C:/Program Files/Git/bin/sh.exe').is_file():
            shell = 'C:/Program Files/Git/bin/sh.exe'
        if shell is None:
            self.skipTest('POSIX shell unavailable')
        hook = (Path(__file__).resolve().parents[1] / '.githooks' / 'pre-commit').read_text()
        # Exercise the exact selection block using fake shell functions; no
        # installation, PATH mutation, Store alias launch or actual scanner.
        selection = hook[hook.index('python=\n'):hook.index('exec "$python"')]
        fake = "set -eu\npython3() { echo SHIM-ERROR >&2; return 127; }\npython() { return 0; }\n"
        with tempfile.TemporaryDirectory() as root:
            result = subprocess.run([shell, '-c', fake + selection + 'printf "%s" "$python"'],
                                    cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'python')
        self.assertEqual(result.stderr, b'')

    def test_hook_no_working_python_blocks(self):
        shell = shutil.which('sh')
        if shell is None and Path('C:/Program Files/Git/bin/sh.exe').is_file():
            shell = 'C:/Program Files/Git/bin/sh.exe'
        if shell is None:
            self.skipTest('POSIX shell unavailable')
        hook = (Path(__file__).resolve().parents[1] / '.githooks' / 'pre-commit').read_text()
        selection = hook[hook.index('python=\n'):hook.index('exec "$python"')]
        fake = 'set -eu\npython3() { return 127; }\npython() { return 1; }\n'
        with tempfile.TemporaryDirectory() as root:
            result = subprocess.run([shell, '-c', fake + selection], cwd=root,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b'')
        self.assertIn(b'working Python 3.8+', result.stderr)

    @unittest.skipUnless(os.environ.get('LFS_SCAN_TEST_GITLEAKS'), 'Optional installed Gitleaks integration')
    def test_real_gzip_magic_and_begin_middle_end_canaries(self):
        scanner = os.environ['LFS_SCAN_TEST_GITLEAKS']
        # Generated noncredential, never printed or saved.
        token = b'gh' + b'p_' + hashlib.sha256(b'framed-stream-noncredential').hexdigest()[:36].encode()
        harmless = b''.join(hashlib.sha256(str(i).encode()).digest() for i in range(10000))
        packed = gzip.compress(harmless, mtime=0)
        original = hashlib.sha256(packed).hexdigest()
        with tempfile.TemporaryDirectory() as root:
            result = scan.scan_stream(scanner, io.BytesIO(packed), root)
            self.assertEqual(result['payload_bytes'], len(packed))
            for position in (0, len(packed)//2, len(packed)):
                with self.subTest(kind='raw', position=position):
                    data = packed[:position] + b'\nfixture=' + token + b'\n' + packed[position:]
                    with self.assertRaisesRegex(scan.Blocked, 'detected a potential secret'):
                        scan.scan_stream(scanner, io.BytesIO(data), root)
            text = b'{"safe":"measurement"}\n' * 30000
            for position in (0, len(text)//2, len(text)):
                with self.subTest(kind='decoded', position=position):
                    data = text[:position] + b'\nfixture=' + token + b'\n' + text[position:]
                    compressed = gzip.compress(data, mtime=0)
                    with self.assertRaisesRegex(scan.Blocked, 'detected a potential secret'):
                        scan.scan_stream(scanner, io.BytesIO(gzip.decompress(compressed)), root)
        self.assertEqual(hashlib.sha256(packed).hexdigest(), original)

    @unittest.skipUnless(os.environ.get('LFS_SCAN_TEST_GITLEAKS'), 'Optional installed Gitleaks integration')
    def test_real_scanner_detects_synthetic_token_after_large_binary_prefix(self):
        # Fake, generated token; no usable credential. Split literal avoids test
        # source itself looking like a committed secret to other scanners.
        token = b'gh' + b'p_' + hashlib.sha256(b'noncredential-test-fixture').hexdigest()[:36].encode()
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryFile() as stream:
            stream.write(b'\0' * (12 * 1024 * 1024) + b'\nfixture=' + token + b'\n')
            stream.seek(0)
            with self.assertRaisesRegex(scan.Blocked, 'detected a potential secret'):
                scan.scan_stream(os.environ['LFS_SCAN_TEST_GITLEAKS'], stream, root)
            stream.seek(0); stream.truncate(); stream.write(b'ordinary innocuous fixture\0\xff\n'); stream.seek(0)
            scan.scan_stream(os.environ['LFS_SCAN_TEST_GITLEAKS'], stream, root)


if __name__ == '__main__':
    unittest.main()
