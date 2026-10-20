"""Offline protocol tests. No native driver, Connector, or model is imported."""
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.deps')]
from experiments.stable_cell_calibration_20261020 import core, metrics, plan, experiment


class Clock:
    def __init__(self):self.t=0.
    def __call__(self):return self.t
    def sleep(self,n):self.t+=n


class Sink:
    def __init__(self):self.count=0;self.events={};self.last=None
    def write(self,row):
        self.count+=1;self.events[row['event']]=self.events.get(row['event'],0)+1
        self.last=row
    def checkpoint(self,**kwargs):pass


class Converters:
    @staticmethod
    def convert_volt_to_dac(bits,ref,v):return int(v/ref*((1<<bits)-1))
    @staticmethod
    def convert_dac_to_volt(bits,ref,v):return v*ref/((1<<bits)-1)
    @staticmethod
    def convert_adc_to_volt(gain,bits,ref,v):return v*ref/((1<<bits)-1)/gain


class Board:
    def __init__(self):self.r=[[8000.]*32 for _ in range(8)];self.closed=False;self.writes=0
    def open_port(self):return True
    def close_port(self):self.closed=True;return True
    def impact(self,c):
        w=c['wl']
        if c['mode_flag']==7:
            b=c['bl']
            if c['vol']:
                self.writes+=1
                self.r[w][b]=8000. if c['sign']==0 else 8000/(1+4*c['vol']/3)
            return self.r[w][b],c['id'],100
        v=sum(x/r for x,r in zip(c['vol'],self.r[w]))*330
        return 0.,0,100+round(v/5*16383)


def runner(board=None):
    clock=Clock();sink=Sink();board=board or Board()
    return core.Collector(board,Converters,sink,'simulator',clock,clock.sleep)


class ProtocolTests(unittest.TestCase):
    def test_full_bounded_schedule_and_real_pair_signs(self):
        c=runner();r=c.run()
        self.assertEqual(r['status'],'completed',r.get('error'))
        self.assertEqual(len(r['cells']),256)
        self.assertTrue(all(x['eligible'] for x in r['cells']))
        self.assertEqual(r['selection']['selected']['coefficient_slots'],128)
        self.assertEqual(r['phase_impacts']['A'],282624)
        self.assertEqual(r['phase_impacts']['B'],106496)
        self.assertLessEqual(r['phase_impacts']['C'],22416)
        self.assertLessEqual(r['impacts'],411536)
        self.assertEqual(r['pulses'],20480)
        self.assertEqual(len(r['checkpoints']),6656)
        self.assertEqual(r['explicit_wait_requested_s'],600.)
        self.assertTrue(r['pair_qualification']['qualified'])
        self.assertEqual(c.writer.events['matrix_frozen'],1)
        self.assertEqual(len(c.calibration['fit']),144)
        self.assertEqual(len(c.calibration['selection']),72)
        self.assertEqual(len(c.calibration['heldout']),72)
        self.assertTrue(c.connector.closed)

    def test_current_skip_retains_zero_brackets(self):
        c=runner();c.connector.r[0][0]=1.
        b=c.bracket(0,[0])
        self.assertEqual(b['status'],'guard_skipped')
        self.assertEqual(c.calls,3)  # zero, fresh native R, zero; no signal impact.
        self.assertIsNone(b['signal'])
        c.pulse(0,0,1,3.,cycle=0,state='H',step=0)
        self.assertFalse(c.updates[-1]['admitted']);self.assertEqual(c.pulses,0)

    def test_invalid_native_result_stops_and_closes(self):
        class Bad(Board):
            def impact(self,c):return float('nan'),c['id'],100
        c=runner(Bad());r=c.run()
        self.assertEqual(r['status'],'failed');self.assertTrue(c.connector.closed)
        self.assertEqual(len(r['cells']),256)
        self.assertTrue(all(not x['eligible'] for x in r['cells']))

    def test_open_failure_best_effort_close(self):
        class Bad(Board):
            def open_port(self):return False
        c=runner(Bad());r=c.run()
        self.assertEqual(r['status'],'failed');self.assertEqual(c.calls,0);self.assertTrue(c.connector.closed)

    def test_masks_fixed_disjoint_and_no_labels(self):
        a=plan.masks(list(range(32)));b=plan.masks(list(range(32)))
        self.assertEqual(a,b)
        flat=[tuple(m) for group in a.values() for m in group]
        self.assertEqual(len(flat),96);self.assertEqual(len(set(flat)),96)
        self.assertEqual([len(a[k]) for k in ('fit','selection','heldout')],[48,24,24])
        self.assertTrue(all(1<=len(x)<=4 for x in flat))

    def test_rank_deficiency_excludes_full_matrix(self):
        rows=[dict(mask=[0],centered_v=[.1,.1]) for _ in range(4)]
        m=metrics.fit_matrix({0:[.1,.1]},rows,rows)
        self.assertNotIn('full',[c['name'] for c in m['candidates']])
        self.assertEqual(m['name'],'identity')

    def test_file_only_preflight_without_private_checks(self):
        _,hashes=experiment.preflight('hardware')
        self.assertFalse(any('MemriCORE/' in name for name in hashes))

    def test_diagnostic_fallback_fixed_capacity_score_and_no_primary_override(self):
        cells=[dict(wl=w,bl=b,eligible=False,hard_eligible=w<2,
                    worst_normalized_error=2.+b/100) for w,b in plan.CELLS]
        s=metrics.select_rectangle(cells)
        self.assertEqual(s['status'],'diagnostic_best_available')
        self.assertEqual(s['selected']['pairs'],[[0,1]])
        self.assertEqual(s['selected']['bls'],list(range(8)))
        self.assertFalse(s['quality_qualified'])
        cells[0]['eligible']=True;cells[32]['eligible']=True
        s=metrics.select_rectangle(cells)
        self.assertEqual(s['status'],'qualified')
        self.assertEqual(s['selected']['bls'],[0])

    def test_b_soft_drift_diagnostic_but_wrong_sign_mandatory_fail(self):
        cells=[dict(wl=w,bl=0,span_adc=20) for w in (0,1)]
        rows=[]
        for config in ('plus','minus','zero','checkerboard'):
            for role in ('immediate','delayed'):
                base=10 if config!='minus' else -10
                rows.append(dict(pair=[0,1],bl=0,configuration=config,role=role,status='measured',
                                 differences_adc=[base+(4 if role=='delayed' else 0)]*10))
        q=metrics.qualify_pairs(dict(pairs=[[0,1]],bls=[0]),cells,rows)
        self.assertTrue(q['mandatory_pass']);self.assertFalse(q['qualified'])
        rows[0]['differences_adc']=[-10]*10
        q=metrics.qualify_pairs(dict(pairs=[[0,1]],bls=[0]),cells,rows)
        self.assertFalse(q['mandatory_pass'])

    def test_recipe_counts(self):
        self.assertEqual(sum(len(plan.recipe(s,True)) for s in ('R','M','H')),16)
        for state in ('plus','minus','zero','checkerboard'):
            self.assertEqual(sum(len(plan.recipe(s)) for s in plan.pair_states(state,0)),24)


if __name__=='__main__':unittest.main()
