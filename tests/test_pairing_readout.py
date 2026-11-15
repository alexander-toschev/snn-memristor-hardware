"""Offline selection/cardinality, read envelope and current guard regression tests."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING,cast
import unittest
from experiment_common.runtime import use_local_dependencies
use_local_dependencies(prefer_environment=True)
from manager.service import converters
from experiment_common.block_readout_validation import Block,Leg
from experiment_common.durable import Writer
from experiment_common.pairing_readout import Cell,DELTAS,Edge,WideCollector,choose_map,maximum_matching,wide_profiles
from experiment_common.selective_metrology import Reading
if TYPE_CHECKING:
    from manager.board import Connector


def leg(wl: int,bl: int,delta: float) -> Leg:
    def samples(adc: int) -> tuple[Reading,...]:
        return tuple(Reading(task_id=i,resistance_ohm=10000.,adc=adc) for i in range(4))
    baseline=tuple([.025]*32);signal=tuple(.025+delta if b==bl else .025 for b in range(32))
    codes=tuple(converters.convert_volt_to_dac(12,5,v) for v in signal)
    return Leg(wl,(),samples(100),samples(100+round(delta*1000)),samples(100),tuple([20]*32),codes,baseline,signal,.0001)


class FakeConnector:
    def __init__(self) -> None:
        self.modes: list[int]=[]
    def impact(self,command: dict[str,int|float|list[float]]) -> tuple[float,int,int]:
        mode=command['mode_flag'];task=command['id']
        if not isinstance(mode,int) or not isinstance(task,int):
            raise ValueError('Invalid fake call')
        self.modes.append(mode)
        return 100.,task,100


class PairingTests(unittest.TestCase):
    def test_cardinality_precedes_score(self) -> None:
        edges={(0,1):Edge(0,1,0,1.,2.,100.,1.,1.),(0,2):Edge(0,2,0,1.,2.,1.,1.,1.),(1,3):Edge(1,3,0,1.,2.,1.,1.,1.)}
        selected=maximum_matching((0,1,2,3),edges)
        self.assertEqual([(e.plus,e.minus) for e in selected],[(0,2),(1,3)])

    def test_pairing_uses_positive_overlap_and_keeps_multiinput(self) -> None:
        cells=tuple(Cell(w,b,(1 if w in (0,2) else 3)*1e-6,(2 if w in (0,2) else 4)*1e-6) for b in (0,31) for w in range(4))
        basis={(c.wl,c.bl,d):[leg(c.wl,c.bl,d) for _ in range(3)] for c in cells for d in DELTAS}
        result=choose_map(cells,basis,1e-6,5e-6,5/16383,330.)
        self.assertEqual(result['physical_cells'],8)
        self.assertEqual(result['selected_delta_v'],.075)
        self.assertEqual(result['multiinput_blocks'],2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'map.json';path.write_text(json.dumps(result))
            profiles=wide_profiles(path)
            self.assertEqual({(p.block.plus_wl,p.block.minus_wl) for p in profiles},{(0,2),(1,3)})
            self.assertTrue(all(len(p.block.bls)==2 for p in profiles))

    def test_current_guard_stops_before_any_mvm(self) -> None:
        fake=FakeConnector()
        with tempfile.TemporaryDirectory() as directory:
            writer=Writer(Path(directory)/'raw',maximum_expanded_bytes=1024**2,minimum_free_bytes=0)
            reader=WideCollector(cast('Connector',fake),converters,writer,'simulator')
            with self.assertRaisesRegex(ValueError,'Fresh summed current'):
                reader.measure_leg(Block('guard',0,1,(31,)),0,'basis',0,'fixed_positive',.075,(31,),global_bias=True)
            self.assertEqual(fake.modes,[7]*32)
            writer.finish()

    def test_write_and_above_envelope_rejected_before_connector(self) -> None:
        fake=FakeConnector()
        with tempfile.TemporaryDirectory() as directory:
            writer=Writer(Path(directory)/'raw',maximum_expanded_bytes=1024**2,minimum_free_bytes=0)
            reader=WideCollector(cast('Connector',fake),converters,writer,'simulator')
            with self.assertRaises(ValueError):
                reader.call(dict(mode_flag=7,wl=0,bl=31,vol=.3,sign=0,pulse_width=.0001))
            with self.assertRaises(ValueError):
                reader.call(dict(mode_flag=10,wl=0,vol=[.1001]*32))
            self.assertEqual(fake.modes,[])
            writer.finish()


if __name__=='__main__':
    unittest.main()
