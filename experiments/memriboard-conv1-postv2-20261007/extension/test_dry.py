"""Synthetic/no-simulator tests. These must not call runner.execute or simulator_api."""
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import runner
from extension_core import TILES, assemble, predrive, require, sha, summarize, validation_gate


class Safety(unittest.TestCase):
    def test_dual_view_strict_current(self):
        v=np.ones(32)*.3
        r=np.ones(32)*1000
        a=predrive(v,v,r,r)
        self.assertTrue(a['permitted'])
        # 32*.3/640 = exactly .015, strictly disallowed.
        b=predrive(v,v,r,np.ones(32)*640)
        self.assertFalse(b['permitted'])
        self.assertAlmostEqual(b['actual_current_a'],.015)
        self.assertAlmostEqual(b['margin_a'],0.,places=14)
    def test_invalid_voltage_or_resistance_no_clipping(self):
        v=np.zeros(32);r=np.ones(32)*1000
        for requested,decoded,observed,actual in [
            (v,np.ones(32)*.301,r,r), (np.ones(32)*.301,v,r,r),
            (v,v,np.zeros(32),r),(v,v,r,np.full(32,np.nan)),
            (np.full(32,np.inf),v,r,r)]:
            with self.assertRaises(ValueError):predrive(requested,decoded,observed,actual)
    def test_drive_passes_scaled_voltage_not_binary(self):
        class Cv:
            @staticmethod
            def convert_volt_to_dac(bit, ref, v): return int(round(v*1000))
            @staticmethod
            def convert_dac_to_volt(bit, ref, n): return n/1000
        class Core:
            dac_bit=12;vol_ref_dac=1.;max_dac_value=300
            current_resistances=np.ones((32,8))*1000
            def dot_product(self,x,col):
                self.captured=x
                return 0.,0
        with tempfile.TemporaryDirectory() as td:
            c=Core();b=SimpleNamespace(crossbar=[[SimpleNamespace(last_resistance=1000) for _ in range(8)] for _ in range(32)])
            x=np.zeros(32);x[3]=1.
            _,dec,_=runner.drive(b,c,Cv,x,0,{'filter':0},pathlib.Path(td)/'predrive.jsonl')
            self.assertEqual(c.captured[3],.3)
            self.assertEqual(max(c.captured),.3)
            self.assertEqual(dec[3],.3)
            self.assertEqual(json.loads((pathlib.Path(td)/'predrive.jsonl').read_text())['guard'],'predrive')
            x[3]=2.
            with self.assertRaisesRegex(ValueError,'requested >0.3V'):
                runner.drive(b,c,Cv,x,0,{},pathlib.Path(td)/'predrive.jsonl')
            self.assertEqual(c.captured[3],.3)

    def test_guard_stops_before_mvm(self):
        class Base:
            calls=0
            def mode_mvm(self,*args):
                self.calls+=1
                return (123,)
        class Cv:
            @staticmethod
            def convert_dac_to_volt(*args):return .3
        with tempfile.TemporaryDirectory() as td:
            Guard=runner.create_guarded(Base,Cv,runner.P['budgets_per_filter'],__import__('time').monotonic(),
                                        pathlib.Path(td)/'audit.jsonl',[0])
            b=Guard();b.core=SimpleNamespace(dac_bit=12,vol_ref_dac=1.,adc_bit=12,
                                          current_resistances=np.ones((32,8))*1000)
            b.crossbar=[[SimpleNamespace(last_resistance=640 if col==0 else 1000) for col in range(8)] for _ in range(32)]
            b.meta={'filter':7}
            with self.assertRaisesRegex(ValueError,'15mA'):
                b.mode_mvm([1]*32,0,0,0,0,0,0)
            self.assertEqual(b.calls,0)
            self.assertFalse(json.loads((pathlib.Path(td)/'audit.jsonl').read_text())['permitted'])


class Aggregation(unittest.TestCase):
    def records(self):
        rows=[]
        for rep in range(3):
            for split in ('train','val'):
                for position in range(8):
                    for spatial in range(4):
                        for t in range(4):
                            for read in range(3):
                                x=float(position+spatial+t+1)
                                for tile in range(3):
                                    rows.append(dict(filter=5,replicate=rep,split=split,
                                      split_position=position,top=spatial,left=spatial,t=t*3,read=read,
                                      tile=tile,noise_flag=1,ideal=x/3,quantized_ideal=x/3,
                                      programmed_readback=x/3,quantized_programmed=x/3,
                                      sim_mvm=x/3))
        return rows
    def test_assembled_and_summary(self):
        rows=self.records();self.assertEqual(len(rows),6912)
        out=assemble(rows,5);self.assertEqual(len(out),2304)
        report=summarize(out,runner.P['gates']);self.assertTrue(report['all_passed'])
        self.assertEqual(report['replicates']['0']['validation']['count'],384)
    def test_incomplete_duplicate_crossfilter(self):
        rows=self.records()
        for changed in (rows[:-1],rows+[rows[0]], [dict(rows[0],filter=6)]+rows[1:]):
            with self.assertRaises(ValueError):assemble(changed,5)
    def test_schedule_and_geometry(self):
        self.assertEqual(TILES,((0,32),(32,64),(64,75)))
        self.assertEqual([len(x) for x in runner.STAGES.values()],[1,3,12,48])
        self.assertEqual(sorted(sum(runner.STAGES.values(),[])),list(range(64)))
        seeds={50000+100000*f+1000*r+101*t for f in range(64) for r in range(3) for t in range(3)}
        self.assertEqual(len(seeds),576)
        # Fresh seeds must NOT reuse v2's 27001-27003.
        self.assertTrue({27001,27002,27003}.isdisjoint(seeds))
    def test_approval_hash_and_no_forged_stage(self):
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/'approval.json'
            record=dict(decision='APPROVED_BY_PARENT',stage='pilot',filters=[0],
                        source_sha256=runner.source_hashes(),checkpoint_sha256=runner.P['checkpoint_sha256'],
                        no_concurrent_training_confirmed=True)
            p.write_text(json.dumps(record));self.assertEqual(runner.approval('pilot',p),sha(p))
            record['filters']=[0,1];p.write_text(json.dumps(record))
            with self.assertRaises(ValueError):runner.approval('pilot',p)
            record['filters']=[0];record['source_sha256']['runner.py']='0'*64
            p.write_text(json.dumps(record))
            with self.assertRaises(ValueError):runner.approval('pilot',p)
    def test_interruption_never_replays_array(self):
        with tempfile.TemporaryDirectory() as td:
            (pathlib.Path(td)/'filter_00').mkdir()
            with self.assertRaisesRegex(ValueError,'interrupted filter'):
                runner.completed(pathlib.Path(td),0)


if __name__=='__main__':unittest.main()
