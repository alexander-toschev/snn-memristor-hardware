"""Offline CPU-only golden and adversarial measured-contract checks."""
import copy
import json
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F
import full_cifar_classifier as c
from full_cifar_reference.encoders import LatencyEncoder


class FullClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.repo = Path(__file__).resolve().parent
        cls.state, cls.model, cls.case, cls.image = c.assets(cls.repo)

    def contract(self):
        p = {"partials": torch.zeros(c.SHAPE), "coverage": torch.ones(c.SHAPE, dtype=torch.bool)}
        m = dict(schema="full-conv1-measured-v1", source="hardware", stage="pre_bn",
                 units="checkpoint_conv1", shape=c.SHAPE.copy(), tile_tap_ranges_chw=copy.deepcopy(c.TILES),
                 bn_applied=False, image_sha256="synthetic-test", checkpoint_sha256=c.PINS["weights/best_fs.pt"],
                 calibration_frozen_before_acquisition=True, cpu_replacement_used=False)
        return p, m

    def test_ten_real_images_against_original_forward_and_encoder(self):
        cases = json.loads((self.repo / "weights/cifar100_conv1_fixture.json").read_text())["cases"]
        with torch.no_grad():
            for case in cases:
                image = torch.tensor(case["image_chw_uint8"], dtype=torch.uint8).reshape(3,32,32)
                x = c.encode(image)
                self.assertTrue(torch.equal(x, LatencyEncoder(T=12, x_min=.1)(image.float()/255)))
                original = self.model(x[:, None])
                direct = F.conv2d(x, self.state["conv_layers.0.weight"])
                tiled = c.cpu_tiles(x, self.state).sum(-1)
                self.assertLess(float((direct-tiled).abs().max()), 2e-6)
                a, b = c.downstream(direct, self.state), c.downstream(tiled, self.state)
                for direct_out, tiled_out, golden in zip(a, b, original):
                    self.assertTrue(torch.equal(direct_out, golden))
                    self.assertTrue(torch.allclose(tiled_out, golden, atol=1e-5, rtol=0))
                self.assertTrue(torch.equal(b[1], original[1]))

    def test_chw75_and_all_spatial_positions_independent_slice(self):
        x = c.encode(self.image)
        parts = c.cpu_tiles(x, self.state)
        w = self.state["conv_layers.0.weight"]
        # Explicit channel/kernel slices, not unfold or the tile generator.
        for t, f, y, xx in [(0,0,0,0), (4,63,27,27), (8,17,13,2)]:
            patch = x[t,:,y:y+5,xx:xx+5]
            expected = (patch*w[f]).sum()
            self.assertAlmostEqual(float(parts[t,f,y,xx].sum()), float(expected), places=6)

    def test_zero_late_input_still_runs_trained_bn_and_lif(self):
        x = c.encode(self.image)
        self.assertEqual(int(x[10:].count_nonzero()), 0)
        seen = []
        hooks = [self.model.bn_layers[0][t].register_forward_hook(
            lambda module, args, out, t=t: seen.append((t, args[0].clone(), out.clone()))) for t in (10,11)]
        try:
            with torch.no_grad():
                output = self.model(x[:,None])
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual([v[0] for v in seen], [10,11])
        for _, incoming, outgoing in seen:
            self.assertEqual(int(incoming.count_nonzero()), 0)
            self.assertGreater(int(outgoing.count_nonzero()), 0)
        self.assertEqual(output[0].shape[0], 12)

    def test_lif_negative_preservation_threshold_and_hard_reset(self):
        s, v = c.lif(torch.tensor([-2., 1., 1.5, 0.]), torch.tensor([0.,0.,0.,-2.]))
        self.assertEqual(s.tolist(), [0.,1.,1.,0.])
        self.assertEqual(v.tolist(), [-2.,0.,0.,-1.])

    def test_contract_incomplete_or_nan_rejected(self):
        for mutation in ("missing", "nan", "partial_shape"):
            p, m = self.contract()
            if mutation == "missing": p["coverage"][11,63,27,27,2] = False
            elif mutation == "nan": p["partials"][0,0,0,0,0] = float("nan")
            else: p["partials"] = p["partials"][:,:4]
            with self.assertRaises(ValueError): c.validate_payload(p,m,"synthetic-test")

    def test_contract_no_cpu_bn_wrong_image_or_repeated_tap(self):
        for key, value in [("source","cpu_reference"),("bn_applied",True),
                           ("image_sha256","wrong"),("cpu_replacement_used",True),
                           ("tile_tap_ranges_chw",[[0,32],[31,64],[64,75]])]:
            p,m = self.contract()
            m[key] = value
            with self.assertRaises(ValueError): c.validate_payload(p,m,"synthetic-test")

    def test_structural_contract_sum_before_bn(self):
        p,m = self.contract()
        p["partials"][...,0],p["partials"][...,1],p["partials"][...,2] = 1.,2.,3.
        raw = c.validate_payload(p,m,"synthetic-test")
        self.assertTrue(torch.equal(raw, torch.full(c.SHAPE[:-1],6.)))

    def test_hardware_import_unavailable_even_with_self_declared_flags(self):
        with self.assertRaises(NotImplementedError):
            c.load_measured(self.repo,"nonexistent.json","unattested")

    def test_paths_cannot_escape_repository(self):
        for path in ("../outside.json", "C:/outside.json", "C:relative.json", "\\\\host\\share\\x.json"):
            with self.assertRaises(ValueError): c.local(self.repo,path)


if __name__ == "__main__":
    unittest.main()
