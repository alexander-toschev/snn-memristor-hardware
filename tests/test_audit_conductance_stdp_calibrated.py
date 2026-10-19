"""Offline synthetic acquisition bundles; never imports/opens native Connector."""
import contextlib
import copy
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools import audit_conductance_stdp_calibrated as a
from experiments.conductance_stdp_calibrated_20261019 import core
from experiments.conductance_stdp_20261019 import raw
from tests.test_conductance_stdp_calibrated import VariedConnector
from tests.test_conductance_stdp import Converters, Clock


def bundle(path, silent=False):
    from cell_calibration import check_sources
    _, ini, hashes=check_sources('simulator')
    freeze=json.loads((a.REVIEW/'SOURCE_FREEZE.json').read_text())
    hashes.update(a.HELPERS)
    hashes.update({a.EXP+'/'+n:p for n,p in freeze['file_sha256'].items()})
    connector=VariedConnector();impact=connector.impact
    if silent:
        def quiet(task):
            result=impact(task)
            return result if task['mode_flag']==7 else (0.,0,100)
        connector.impact=quiet
    clock=Clock();runner=core.Collector(connector,Converters,raw.Writer(path/'raw'),'simulator',clock.sleep,clock.now)
    with contextlib.redirect_stdout(io.StringIO()):result=runner.run()
    (path/'protocol.json').write_text(json.dumps(core.SPEC))
    (path/'board.ini').write_bytes(ini)
    manifest=dict(status=result['status'],source='simulator',session_id='synthetic-offline-only',
        source_sha256=hashes,protocol_sha256=freeze['protocol_sha256'],invocation={'test_only':True},
        impacts=result['impacts'],pulses=result['pulses'],decisions=result['decisions'])
    (path/'start_manifest.json').write_text(json.dumps(manifest))
    (path/'observations.json').write_text(json.dumps(result))
    manifest['result_sha256']={p.relative_to(path).as_posix():a.digest(p.read_bytes()) for p in path.rglob('*') if p.is_file()}
    (path/'manifest.json').write_text(json.dumps(manifest))


def rehash_rows(path, mutate):
    data=gzip.decompress((path/'raw/raw.jsonl.gz').read_bytes())
    rows=[json.loads(line) for line in data.splitlines()];mutate(rows)
    data=('\n'.join(json.dumps(r) for r in rows)+'\n').encode();packed=gzip.compress(data,mtime=0)
    (path/'raw/raw.jsonl.gz').write_bytes(packed)
    index=json.loads((path/'raw/index.json').read_text())
    index.update(lines=len(rows),compressed_bytes=len(packed),expanded_bytes=len(data),
        compressed_sha256=a.digest(packed),expanded_sha256=a.digest(data))
    (path/'raw/index.json').write_text(json.dumps(index))
    m=json.loads((path/'manifest.json').read_text())
    for n in ('raw/raw.jsonl.gz','raw/index.json'):m['result_sha256'][n]=a.digest((path/n).read_bytes())
    (path/'manifest.json').write_text(json.dumps(m))


class AuditorTests(unittest.TestCase):
    def test_complete_and_prefix(self):
        for silent in (False,True):
            with self.subTest(silent=silent),tempfile.TemporaryDirectory(dir=a.ROOT/'.tmp') as d:
                path=Path(d);bundle(path,silent);result=a.audit(path)
                self.assertEqual(result['status'],'pass')
                self.assertEqual(result['selected_gain'],None if silent else 400.)
                self.assertEqual(result['lif_steps'],0 if silent else 832)

    def test_rehashed_semantic_tampering(self):
        def selected(rows):next(r for r in rows if r['event']=='gain_frozen')['selected_gain']=800.
        def candidate(rows):next(r for r in rows if r['event']=='gain_frozen')['candidates'][0]['windows'][0]['steps'][0]['u_after']+=.1
        def control(rows):next(r for r in rows if r['event']=='gain_frozen')['control_used_for_selection']=True
        def gain(rows):next(r for r in rows if r['event']=='lif_step')['gain']=800.
        def sign(rows):next(r for r in rows if r['event']=='pulse_complete' and r['phase']=='training_update')['sign']^=1
        def fitbank(rows):next(r for r in rows if r['event']=='fit_measurement_step')['measurement']['wl']=7
        def duplicate(rows):
            r=next(r for r in rows if r['event']=='bracket');r['after']=copy.deepcopy(r['before'])
        def wrongread(rows):
            points=[r for r in rows if r['event']=='checkpoint'];points[0]['reads']=copy.deepcopy(points[1]['reads'])
        def eligibility(rows):
            for event in ('pulse_decision','pulse_complete'):
                next(r for r in rows if r['event']==event and r['phase']=='training_update')['eligibility']=999.
        def sham(rows):
            for event in ('pulse_decision','pulse_complete'):
                next(r for r in rows if r['event']==event and r['phase']=='training_update')['sham']['r']=999.
        def repeatedpulse(rows):
            for event in ('pulse_decision','pulse_complete'):
                values=[r for r in rows if r['event']==event and r['phase']=='training_update']
                for name in ('guard','sham','pulse_task_id','pulse_native_r','pulse_adc'):
                    if name in values[0]:values[1][name]=copy.deepcopy(values[0][name])
        def latefreeze(rows):
            freeze=next(r for r in rows if r['event']=='gain_frozen');rows.remove(freeze)
            rows.insert(next(i for i,r in enumerate(rows) if r['event']=='lif_step'),freeze)
        for mutate in (selected,candidate,control,gain,sign,fitbank,duplicate,wrongread,eligibility,sham,repeatedpulse,latefreeze):
            with self.subTest(case=mutate.__name__),tempfile.TemporaryDirectory(dir=a.ROOT/'.tmp') as d:
                path=Path(d);bundle(path);rehash_rows(path,mutate)
                with self.assertRaises(ValueError):a.audit(path)


if __name__=='__main__':unittest.main()
