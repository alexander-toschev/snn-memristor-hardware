"""Synthetic evidence only: independent audit replay and corruption rejection, no native driver."""
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from tools import audit_mnist_conductance as audit
from experiments.mnist_conductance_20261019 import core


class Clock:
    def __init__(self):self.value=0.
    def now(self):self.value+=.00001;return self.value
    def sleep(self,value):self.value+=value


class FakeConverters:
    @staticmethod
    def convert_volt_to_dac(bits,ref,value):return int(round(value*((1<<bits)-1)/ref,0))
    @staticmethod
    def convert_dac_to_volt(bits,ref,value):return round(value*ref/((1<<bits)-1),3)
    @staticmethod
    def convert_adc_to_volt(gain,bits,ref,value):return round(value*ref/((1<<bits)-1)/gain,5)


class FakeConnector:
    cb_type='simulator'
    def __init__(self,silent=False):self.silent=silent
    def open_port(self):return True
    def close_port(self):return True
    def impact(self,command):
        active=sum(v!=0 for v in command['vol']) if command['mode_flag']==10 else 0
        return 10000.,command['id'],100+(0 if self.silent else 10*active)


class TestWriter:
    """Same external durable format; no acquisition helper import."""
    def __init__(self,session):
        self.directory=session/'raw';self.directory.mkdir()
        self.parts=[];self.file=None;self.records=self.expanded=0;self.hash=hashlib.sha256()
    def _close(self):
        if self.file is None:return
        self.file.close()
        self.parts.append(dict(path=self.path.name,compressed_bytes=self.path.stat().st_size,compressed_sha256=audit.sha(self.path),bytes=self.part_bytes,sha256=self.part_hash.hexdigest(),records=self.part_records))
        self.file=None
    def write(self,row):
        line=(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode()
        if self.file is not None and self.part_bytes+len(line)>8*1024**2:self._close()
        if self.file is None:
            self.path=self.directory/('part-%06d.jsonl.gz'%len(self.parts));self.file=gzip.open(self.path,'wb',compresslevel=1)
            self.part_bytes=self.part_records=0;self.part_hash=hashlib.sha256()
        self.file.write(line);self.hash.update(line);self.records+=1;self.expanded+=len(line)
        self.part_hash.update(line);self.part_bytes+=len(line);self.part_records+=1
    def checkpoint(self,**fields):pass
    def finish(self):
        self._close()
        (self.directory/'index.json').write_text(json.dumps(dict(schema='lossless-chunked-jsonl-v1',parts=self.parts,records=self.records,sha256=self.hash.hexdigest(),stream_finalized=True)),encoding='utf-8')


def fixture():
    pixels=[240 if row<14 else 236 for row in range(28) for column in range(28)]
    blocks,schedule=audit.schedule_from_pixels(pixels)
    rows=[]
    for index in audit.GROUPS['train']+audit.GROUPS['fit']+audit.GROUPS['probe']:
        rows.append(dict(train_index=index,shape=[28,28],pixels_uint8=pixels,image_sha256=hashlib.sha256(bytes(pixels)).hexdigest(),blocks=blocks,schedule=schedule,schedule_sha256=audit.digest(schedule)))
    return dict(schema='mnist-conductance-label-free176-v1',source_gzip_sha256=audit.SOURCE_IDX_SHA,groups=audit.GROUPS,split='train',image_count=176,images=rows,images_sha256=hashlib.sha256(bytes(pixels)*176).hexdigest(),labels_included=False,labels_read=False,test_split_read=False,model_executed=False,board_access=False)


class IndependentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp_root=audit.ROOT/'.tmp';temp_root.mkdir(exist_ok=True)
        cls.tmp=tempfile.TemporaryDirectory(prefix='mnist-audit-fake-',dir=temp_root)
        cls.root=Path(cls.tmp.name);cls.fixture=fixture();cls.bundles={}
        for name,silent,reference in [('main',False,None),('silent',True,None)]:
            session=cls.root/name;session.mkdir();writer=TestWriter(session);clock=Clock()
            runner=core.Collector(FakeConnector(silent),FakeConverters,writer,'simulator',cls.fixture,sleep=clock.sleep,monotonic=clock.now)
            with redirect_stdout(io.StringIO()):result=runner.run()
            cls.bundles[name]=(session,result,None)
        reference=dict(gain=cls.bundles['main'][1]['gain'],session='synthetic-not-a-native-session')
        session=cls.root/'reference';session.mkdir();writer=TestWriter(session);clock=Clock()
        runner=core.Collector(FakeConnector(),FakeConverters,writer,'simulator',cls.fixture,sleep=clock.sleep,monotonic=clock.now,reference=reference)
        with redirect_stdout(io.StringIO()):result=runner.run()
        cls.bundles['reference']=(session,result,reference)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def replay(self,name,transform=None):
        session,result,reference=self.bundles[name]
        base=audit.durable_records(session)
        records=transform(base) if transform is not None else base
        try:
            actual=audit.Replay(records,self.fixture,'simulator',reference).run(audit.expected_protocol())
        finally:
            base.close()
        audit.same(actual,result,'entire synthetic observations')
        return actual

    def test_main_full_natural_fake(self):
        result=self.replay('main')
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['pulses'],2560)
        self.assertEqual(result['diagnostics']['training_admitted_pulses'],512)
        self.assertEqual(result['impacts'],132352)
        self.assertEqual(result['measured_images'],208)

    def test_no_eligible_prefix(self):
        result=self.replay('silent')
        self.assertEqual(result['status'],'uninformative_calibration')
        self.assertEqual(result['measured_images'],32)
        self.assertEqual(result['pulses'],2048)
        self.assertEqual(len(result['checkpoints']),3)
        self.assertFalse(result['outcomes'])

    def test_reference_no_training_pulses(self):
        result=self.replay('reference')
        self.assertEqual(result['pulses'],2048)
        self.assertEqual(result['impacts'],131840)
        self.assertEqual(result['diagnostics']['reference_suppressed'],512)
        with self.assertRaisesRegex(ValueError,'SIM only'):
            audit.Replay([],self.fixture,'hardware',dict(gain=20))

    def test_semantic_rehashed_compatible_mutations_rejected(self):
        # Mutations happen after the external-container reader: equivalent to rehashing
        # all container metadata. No integrity hash can rescue invalid semantic evidence.
        cases=[
            ('checkpoint references',lambda r:r['event']=='checkpoint_cell',lambda r:r['reads'][1].update(task_id=r['reads'][0]['task_id'])),
            ('wrong mask',lambda r:r['event']=='request' and r['phase']=='checkpoint_signal',lambda r:r['command_si']['vol'].__setitem__(2,.05)),
            ('candidate trajectory',lambda r:r['event']=='fit_candidate_lif_step',lambda r:r.update(u_after=r['u_after']+.1)),
            ('freeze selection',lambda r:r['event']=='gain_frozen',lambda r:r.update(selected_gain=1600)),
            ('control gain',lambda r:r['event']=='lif_step' and r['wl']==4,lambda r:r.update(gain=1600)),
            ('training sign',lambda r:r['event']=='pulse_decision' and r['phase']=='training_update',lambda r:r.update(sign=1-r['sign'])),
            ('wrong post read',lambda r:r['event']=='pulse_complete' and r['phase']=='training_update',lambda r:r['post'].update(task_id=r['guard']['task_id'])),
            ('short retention',lambda r:r['event']=='retention_end',lambda r:r.update(elapsed_s=59.)),
        ]
        for name,predicate,change in cases:
            def transform(rows,predicate=predicate,change=change):
                changed=False
                for row in rows:
                    if not changed and predicate(row):change(row);changed=True
                    yield row
                self.assertTrue(changed)
            with self.subTest(name=name),self.assertRaises(ValueError):self.replay('main',transform)

    def test_missing_native_command_and_tail_rejected(self):
        def missing(rows):
            removed=False
            for row in rows:
                if not removed and row['event']=='request':removed=True;continue
                yield row
        with self.assertRaises(ValueError):self.replay('main',missing)
        def extra(rows):
            last=None
            for last in rows:yield last
            yield last
        with self.assertRaisesRegex(ValueError,'after terminal'):self.replay('silent',extra)

    def test_fixture_integer_ttfs_and_pixels_bound(self):
        sample=copy.deepcopy(self.fixture)
        sample['images'][0]['schedule'][1]=[]
        sample['images'][0]['schedule_sha256']=audit.digest(sample['images'][0]['schedule'])
        with self.assertRaises(ValueError):audit.verify_fixture(sample)
        n=audit.replay_neuron([[]]*32,[-.001]*32,10)
        self.assertLess(n[1]['u_after'],n[0]['u_after'])
        self.assertEqual(audit.choose_update([-.5,.5]+[0.]*30),dict(bl=0,eligibility=-.5,sign=0))

    def test_durable_hash_crc_and_finalization(self):
        for variant in ('not_finalized','wrong_expanded_hash','truncated_crc'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory(dir=self.root) as folder:
                session=Path(folder);writer=TestWriter(session)
                writer.write({'synthetic':True});writer.finish()
                index_path=session/'raw/index.json';index=json.loads(index_path.read_text())
                if variant=='not_finalized':index['stream_finalized']=False
                elif variant=='wrong_expanded_hash':index['parts'][0]['sha256']='0'*64
                else:
                    path=session/'raw'/index['parts'][0]['path'];path.write_bytes(path.read_bytes()[:-5])
                    index['parts'][0]['compressed_bytes']=path.stat().st_size;index['parts'][0]['compressed_sha256']=audit.sha(path)
                index_path.write_text(json.dumps(index))
                with self.assertRaises((ValueError,EOFError,gzip.BadGzipFile)):
                    list(audit.durable_records(session))



class FocusedAuditTests(unittest.TestCase):
    """Small hardening cases without constructing the expensive full fake bundles."""
    def test_exact_request_and_result_commands(self):
        command=dict(mode_flag=7,wl=0,bl=0,vol=0.,sign=0,pulse_width=0.)
        task=dict(command,id=1)
        common=dict(source='simulator',utc='2026-10-19T12:00:00+00:00',task_id=1,phase='focused',command_si=task)
        request=dict(common,event='request',monotonic_s=1.)
        result=dict(common,event='result',monotonic_s=2.,native_result=[10000.,1,100],latency_s=.01)
        result['command_si']=dict(task)
        self.assertEqual(audit.Replay([request,result],fixture(),'simulator').call(command,'focused')['task_id'],1)
        for target in (0,1):
            for key,value in [('wl',1e-13),('wl',False),('bl',0.0),('id',True),('vol',1e-13),('vol',False),('sign',0.0),('pulse_width',1e-13)]:
                rows=copy.deepcopy([request,result]);rows[target]['command_si'][key]=value
                with self.subTest(target=target,key=key,value=value),self.assertRaises(ValueError):
                    audit.Replay(rows,fixture(),'simulator').call(command,'focused')

    def test_pinned_executed_helper(self):
        relative='experiments/unn_rev3_20261014/paths.py'
        self.assertEqual(audit.sha(audit.ROOT/relative),audit.HELPER_PINS[relative])

    def test_physical_duration_cannot_be_claimed(self):
        good={'physical_pulse_duration_s':None}
        audit.unknown_physical_duration(good,good)
        for bad in ({},{'physical_pulse_duration_s':0},{'physical_pulse_duration_s':.0001},{'physical_pulse_duration_s':False}):
            for pair in ((bad,good),(good,bad)):
                with self.subTest(pair=pair),self.assertRaises(ValueError):audit.unknown_physical_duration(*pair)

    def test_bounded_raw_read(self):
        class BoundedStream:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def readline(self,maximum):
                self.maximum=maximum
                return b'x'*maximum
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=audit.ROOT/'.tmp') as folder:
            session=Path(folder);writer=TestWriter(session);writer.write({'synthetic':True});writer.finish()
            stream=BoundedStream()
            with patch.object(audit.gzip,'open',return_value=stream),self.assertRaises(ValueError):
                list(audit.durable_records(session))
            self.assertEqual(stream.maximum,8*1024**2+1)


if __name__=='__main__':unittest.main()
