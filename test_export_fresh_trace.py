import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import export_fresh_trace as e


class Tests(unittest.TestCase):
    def test_canonical_route_formats_equal_but_changes_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
            a,b=Path(name)/'a.json',Path(name)/'b.json';value={'z':[{'cell':[0,1]}],'a':3}
            a.write_bytes(json.dumps(value,indent=2).replace('\n','\r\n').encode())
            b.write_bytes(json.dumps(value,sort_keys=True,separators=(',',':')).encode())
            self.assertNotEqual(e.sha(a),e.sha(b));self.assertEqual(e.canonical_route_sha(a),e.canonical_route_sha(b))
            value['z'][0]['cell']=[0,2];b.write_text(json.dumps(value))
            self.assertNotEqual(e.canonical_route_sha(a),e.canonical_route_sha(b))
    def fixture(self):
        fixture={'cases':[{'original_test_index':100+c} for c in range(10)]}
        trajectories=[dict(filter_id=f,case=c,repeat=r,original_test_index=100+c,
            actual=[dict(t=t,conv_sum=.12345678901234567,bntt_current=-.2,lif_before_reset=-.3,
                         lif_after_reset=-.3,lif_spike=0) for t in range(12)])
            for f in range(4) for c in range(10) for r in range(5)]
        return trajectories,fixture
    def test_complete2400_sorted_roundtrip_no_negative_clipping(self):
        trajectories,fixture=self.fixture();rows=e.trace_rows(list(reversed(trajectories)),fixture)
        self.assertEqual(len(rows),2400);self.assertEqual(rows[0]['original_test_index'],100)
        parsed=list(csv.DictReader(io.StringIO(e.csv_bytes(rows).decode())))
        self.assertEqual(float(parsed[0]['conv']),trajectories[0]['actual'][0]['conv_sum'])
        self.assertEqual(float(parsed[0]['v_after']),-.3)
        self.assertEqual(set(parsed[0]),set(e.FIELDS))
    def test_duplicate_missing_and_wrong_case_fail(self):
        trajectories,fixture=self.fixture()
        with self.assertRaises(ValueError):e.trace_rows(trajectories[:-1],fixture)
        duplicate=copy.deepcopy(trajectories);duplicate[-1]=duplicate[0]
        with self.assertRaisesRegex(ValueError,'Duplicate'):e.trace_rows(duplicate,fixture)
        trajectories[0]['original_test_index']=999
        with self.assertRaisesRegex(ValueError,'case'):e.trace_rows(trajectories,fixture)
    def test_nan_and_bad_timestep_fail(self):
        trajectories,fixture=self.fixture();trajectories[0]['actual'][0]['t']=1
        with self.assertRaisesRegex(ValueError,'ordering'):e.trace_rows(trajectories,fixture)
        trajectories[0]['actual'][0]['t']=0;trajectories[0]['actual'][0]['conv_sum']=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):e.trace_rows(trajectories,fixture)

if __name__=='__main__':unittest.main()
