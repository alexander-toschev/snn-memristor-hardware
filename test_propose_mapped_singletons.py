import unittest
from propose_mapped_singletons import lexical_matching,edge


class MappedProposalTests(unittest.TestCase):
    def test_matching_avoids_greedy_loss_and_has_fixed_tie_break(self):
        self.assertEqual(lexical_matching({0:[0,1],1:[0]},[0,1],[0,1]),[(0,1),(1,0)])
        self.assertEqual(lexical_matching({0:[0,1],1:[0,1]},[0,1],[0,1]),[(0,0),(1,1)])

    def test_unknown_covariance_does_not_get_quadrature_discount(self):
        def cell(wl,value):
            level=dict(signal_adc=value,single=dict(mean=value,sd=2.),zero=dict(mean=0.,sd=0.))
            return dict(wl=wl,bl=0,basis=level,selection=level)
        result=edge(cell(0,150),cell(1,80))
        self.assertEqual(result['phases']['basis']['sd_bound_adc'],4.)
        self.assertFalse(result['contrast_supported'])  # 4/70 >5%, although quadrature would pass.
        self.assertFalse(result['physical_cell_defect_claim'])


if __name__=='__main__':unittest.main()
