"""Pure synthetic IDX parsing tests; no real fixture extraction or model."""
import io
import struct
import unittest
from tools import prepare_mnist_conductance_fixture as fixture


class FixtureTests(unittest.TestCase):
    def test_exact_selection_and_full_length_validation(self):
        data = struct.pack('>IIII', 2051, 3, 28, 28) + bytes(784) + bytes([27]) * 784 + bytes([255]) * 784
        result = fixture.selected_images(io.BytesIO(data), [2, 0], expected_count=3)
        self.assertEqual(result, {0: bytes(784), 2: bytes([255]) * 784})
        for bad in (data[:-1], data + b'x', bytes(16) + data[16:]):
            with self.assertRaises(ValueError):
                fixture.selected_images(io.BytesIO(bad), [0], expected_count=3)
        with self.assertRaises(ValueError):
            fixture.selected_images(io.BytesIO(data), [0, 0], expected_count=3)

    def test_fixed_disjoint_coverage_and_no_labels(self):
        ids = [i for group in fixture.GROUPS.values() for i in group]
        self.assertEqual(len(set(ids)), 176)
        images = {i: bytes([i % 256]) * 784 for i in ids}
        result = fixture.build_fixture(images)
        self.assertEqual([row['train_index'] for row in result['images']], ids)
        self.assertFalse(result['labels_read'])
        self.assertFalse(result['test_split_read'])
        self.assertTrue(all(set(row) == {'train_index', 'shape', 'image_sha256', 'pixels_uint8',
                                       'blocks', 'schedule', 'schedule_sha256'} for row in result['images']))
        self.assertEqual(result['images_sha256'], fixture.sha_bytes(b''.join(images[i] for i in ids)))
        del images[0]
        with self.assertRaises(ValueError):
            fixture.build_fixture(images)

    def test_repository_relative_path_only(self):
        for path in ('../fixture.json', 'D:/outside.json', '/outside.json', ''):
            with self.assertRaises(ValueError):
                fixture.local(path)

    def test_integer_pooling_ttfs_boundaries_and_exact_area(self):
        for pixel, expected in ((0, None), (1, 31), (128, 15), (255, 0)):
            output = fixture.pooled_schedule(bytes([pixel]) * 784)
            self.assertEqual(sum(block['area'] for block in output['blocks']), 784)
            self.assertEqual(sum(block['sum_uint8'] for block in output['blocks']), 784 * pixel)
            self.assertTrue(all(block['spike_t'] == expected for block in output['blocks']))
            self.assertEqual(sum(map(len, output['schedule'])), 0 if expected is None else 32)
            if expected is not None:
                self.assertEqual(output['schedule'][expected], list(range(32)))
        image = bytearray(784); image[0] = 255
        output = fixture.pooled_schedule(image)
        self.assertEqual(output['schedule'][30], [0])
        self.assertEqual(output['blocks'][0]['area'], 21)


if __name__ == '__main__':
    unittest.main()
