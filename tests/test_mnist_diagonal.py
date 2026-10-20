"""Focused diagonal selection and physical-event replay; no native execution."""
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
from experiments.mnist_diagonal_20261020 import core,replay,experiment
from experiments.mnist_conductance_20261019 import core as old
from tools import audit_mnist_conductance as numeric
from tests.test_audit_mnist_conductance import fixture,FakeConnector,FakeConverters,Clock

class Sink:
    def __init__(self):self.events=[]
    def write(self,r):self.events.append(copy.deepcopy(r))
    def checkpoint(self,**kw):pass
    def finish(self):pass

class Tests(unittest.TestCase):
    def test_protocol_and_unchanged_physical_methods(self):
        self.assertEqual(core.SPEC,replay.expected_protocol())
        self.assertEqual(core.SPEC['maximum_impacts'],172288);self.assertEqual(core.SPEC['maximum_pulses'],2560)
        for name in ('call','vector','pulse','checkpoint'):
            self.assertIs(getattr(core.Collector,name),getattr(old.Collector,name))
            self.assertIs(getattr(replay.Replay,name if name!='checkpoint' else 'checkpoint'),getattr(numeric.Replay,name if name!='checkpoint' else 'checkpoint'))
        _,_,f,_=experiment.preflight('simulator');self.assertEqual(len(f['images']),176)

    def test_individual_selection_tie_and_missing_row(self):
        candidates=[]
        for gain in core.GAINS:
            rows=[dict(eligible=gain in ((10,20),(100,200),(400,800),(4,8))[w],total_posts=32) for w in range(4)]
            candidates.append(dict(gain=gain,per_wl=rows))
        gains,rows=core.select_diagonal(candidates)
        self.assertEqual(gains,[10,100,400,4]*2)
        for c in candidates:c['per_wl'][2]['eligible']=False
        self.assertIsNone(core.select_diagonal(candidates)[0])

    def test_independent_gain_and_image_replay(self):
        f=fixture();images=[]
        for i,index in enumerate(core.GROUPS['fit']):
            schedule=f['images'][0]['schedule']
            volts=[[0.]*32 for _ in range(8)]
            for w,voltage in enumerate([.06,.006,.003,.2]*2):
                for t,mask in enumerate(schedule):
                    if mask:volts[w][t]=voltage
            images.append(dict(phase='fit',ordinal=i,train_index=index,schedule=schedule,voltages=volts))
        sink=Sink();clock=Clock()
        def emit(event,**fields):sink.write(dict(event=event,source='simulator',utc='2026-10-20T00:00:00+00:00',monotonic_s=clock.now(),**fields))
        calibration=core.choose_gain(images,emit);self.assertIsNotNone(calibration['selected_gain'])
        emit('gain_frozen',last_native_task_id=0,**calibration)
        checker=replay.Replay(iter(sink.events),f,'simulator');checker.gain_fit(images)
        self.assertEqual(checker.calibration,calibration);self.assertEqual(checker.gain[:4],checker.gain[4:])
        bad=copy.deepcopy(sink.events);bad[-1]['selected_gain'][4]*=2
        with self.assertRaises(ValueError):replay.Replay(iter(bad),f,'simulator').gain_fit(images)
        sink=Sink();runner=core.Collector(FakeConnector(),FakeConverters,sink,'simulator',f,sleep=clock.sleep,monotonic=clock.now)
        runner.gain=[10,20,40,80]*2;runner.image('preprobe',core.GROUPS['probe'][0])
        checker=replay.Replay(iter(json.loads(json.dumps(sink.events))),f,'simulator');checker.gain=runner.gain
        checker.image('preprobe',core.GROUPS['probe'][0])
        self.assertEqual(checker.outcomes,runner.outcomes)

if __name__=='__main__':unittest.main()
