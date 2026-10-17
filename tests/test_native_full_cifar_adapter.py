"""Pure synthetic scope refusal tests: never qualify historical spot data as full."""
import copy
import itertools
import unittest
import full_cifar_plan as p
import native_full_cifar_adapter as a


class NativeScopeTests(unittest.TestCase):
    def setUp(self):
        self.plan=p.make_plan([dict(case_id="synthetic",image_sha256="a"*64)])
        self.plan["source_kind"]="synthetic_test"
        self.route=self.plan["physical_microtiles"][0]
        self.slots=[dict(tap_index=t) for t in self.route["taps_chw"]]

    def test_actual_spot_shape_refused_fullimage_and_no_self_attestation(self):
        # Historical position13,13 acrossT12 is not all28x28 positions.
        with self.assertRaises(ValueError):
            a.verify_native_block_scope(self.plan,self.route["micro_id"],self.slots,[(0,0,t,13,13) for t in range(12)])
        with self.assertRaises(ValueError):
            a.require_enabled_hardware_source(dict(status="passed",source="hardware",full_image_coverage=True))

    def test_complete_synthetic_scope_and_duplicate_missing_tile_rejection(self):
        keys=itertools.product(range(1),range(1),range(12),range(28),range(28))
        scope=a.verify_native_block_scope(self.plan,self.route["micro_id"],self.slots,keys)
        self.assertEqual(scope["native_key_count"],9408)
        self.assertFalse(scope["hardware_trust_granted"])
        with self.assertRaises(ValueError):a.verify_all_blocks(self.plan,[scope])
        keys=list(itertools.product(range(1),range(1),range(12),range(28),range(28)))
        keys[-1]=keys[0]
        with self.assertRaises(ValueError):a.verify_native_block_scope(self.plan,self.route["micro_id"],self.slots,keys)

    def test_changed_adc_or_task_reference_cannot_pass_comparison(self):
        expected=dict(event="measured_partial",terms=[dict(positive_adc=[1,2,3],raw_task_refs=["source#task=1"])])
        for field in ("adc","ref"):
            actual=copy.deepcopy(expected)
            if field=="adc":actual["terms"][0]["positive_adc"][1]=4
            else:actual["terms"][0]["raw_task_refs"][0]="source#task=2"
            with self.assertRaises(ValueError):a.compare_normalized(actual,expected)

    def test_windows_backslash_origin_traversal_rejected(self):
        for session,folder in (("results\\..\\outside","tiles/ok"),
                               ("results/ok","tiles\\..\\outside")):
            with self.assertRaises(ValueError):
                a.normalized_record(self.plan,self.route["micro_id"],{},session,folder,"b"*64,"c"*64,1)


if __name__=="__main__":unittest.main()
