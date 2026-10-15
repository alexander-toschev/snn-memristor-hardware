import copy
import hashlib
import json
import struct
import unittest

import plan_cifar75_routing as planner


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared=json.loads(planner.local('weights/prepared_cifar100_conv1.json').read_bytes())
        cls.mapping=json.loads(planner.local('configs/mapped_sequential_mvm.json').read_bytes())
        cls.weights=planner.read_weights(planner.local('weights/conv1_weights.npy').read_bytes())

    def test_pins_and_all_300_coefficients_preserved_without_replication(self):
        for name,pin in planner.INPUTS.items():self.assertEqual(hashlib.sha256(planner.local(name).read_bytes()).hexdigest(),pin)
        result=planner.build(self.prepared,self.mapping,self.weights)
        self.assertEqual(result['tile_counts'],[6,9,7,6]);self.assertEqual(result['total_filter_tiles'],28);self.assertEqual(result['shared_stages'],9)
        for fid,entry in enumerate(result['filters']):
            active=[s for tile in entry['tiles'] for s in tile['slots'] if not s['padding']]
            self.assertEqual([s['tap_index'] for s in active],list(range(75)))
            self.assertEqual([s['trained_weight'] for s in active],self.weights[fid])
            for s in active:
                c,h,w=s['tap_chw'];self.assertEqual(25*c+5*h+w,s['tap_index'])
                self.assertEqual(bytes.fromhex(s['float32_le_hex']),struct.pack('<f',self.weights[fid][s['tap_index']]))
            self.assertTrue(all(s['trained_weight'] is None and s['padding_input']==0 for tile in entry['tiles'] for s in tile['slots'] if s['padding']))
            self.assertEqual(entry['bntt_t12']['bias'],[self.prepared['bntt']['parameters_t12x64']['bias'][t][fid] for t in range(12)])
        self.assertFalse(result['cifar_executed']);self.assertFalse(result['weights_programmed']);self.assertFalse(result['mapped_states_qualified'])

    def test_inactive_stages_do_not_duplicate_last_tile(self):
        result=planner.build(self.prepared,self.mapping,self.weights)
        scheduled=[(e['filter_id'],e['tile_id']) for s in result['stages'] for e in s['filter_tiles'] if e['tile_id'] is not None]
        self.assertEqual(len(scheduled),28);self.assertEqual(len(set(scheduled)),28)
        self.assertEqual([e['tile_id'] for e in result['stages'][-1]['filter_tiles']],[None,8,None,None])

    def test_invalid_input_and_scope_rejected(self):
        with self.assertRaises(ValueError):planner.read_weights(b'bad-data')
        changed=copy.deepcopy(self.prepared);changed['hardware_representation_qualified']=True
        with self.assertRaises(ValueError):planner.build(changed,self.mapping,self.weights)
        with self.assertRaises(ValueError):planner.local('../outside.json')


if __name__=='__main__':unittest.main()
