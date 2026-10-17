"""Pure offline synthetic contract mutations, never native backend data."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import full_cifar_plan as p
import full_cifar_export as e
import audit_full_cifar_export as a


def plan():
    v=p.make_plan([dict(case_id="synthetic",image_sha256="a"*64)],repeats=1)
    v["source_kind"]="synthetic_test"
    return v


def freeze(spec,ordinal=1,micro=None):
    route=spec["physical_microtiles"][0] if micro is None else next(v for v in spec["physical_microtiles"] if v["micro_id"]==micro)
    row=dict(event="calibration_frozen",ordinal=ordinal,micro_id=route["micro_id"],raw_manifest_sha256="b"*64,
             matrix_C2=[[1.,.25],[.5,2.]],fixed_gain_v_per_weight=1.,fitted_weight_scale_used=False,
             calibration_uses_cifar=False,tap_route_sha256=p.digest_object(route))
    row["calibration_sha256"]=e.calibration_digest(row)
    return row


def measured(f,ordinal=2):
    return dict(event="measured_partial",ordinal=ordinal,micro_id=f["micro_id"],raw_manifest_sha256="b"*64,
                calibration_sha256=f["calibration_sha256"],case_id="synthetic",image_sha256="a"*64,
                repeat=0,t=0,y=0,x=0,
                terms=[dict(positive_adc=[100,400,200],negative_adc=[50,150,100],
                            raw_task_refs=[f"synthetic:task{i}" for i in range(6)])])


class BridgeTests(unittest.TestCase):
    def both(self,spec=None):
        v=spec or plan()
        return e.Accumulator(v,"b"*64),a.Replay(v,"b"*64)

    def test_lazy_complete_schedule_count_and_boundaries(self):
        v=plan();keys=p.iter_keys(v)
        self.assertIs(iter(keys),keys)
        first=next(keys);count=1;last=first
        for last in keys:count+=1
        self.assertEqual(count,1806336)
        self.assertEqual(first,("synthetic",0,"f0_k0_a0",0,0,0))
        self.assertEqual(last,("synthetic",0,"f63_k2_a64",11,27,27))
        self.assertIsNone(p.estimate(v)["native_impacts"])

    def test_arbitrary_physical_split_and_exact4800_partition(self):
        v=p.make_plan([dict(case_id="synthetic",image_sha256="a"*64)],maximum_taps=7)
        p.validate(v);a.Replay(v,"b"*64)
        self.assertEqual(sum(len(r["taps_chw"]) for r in v["physical_microtiles"]),4800)
        self.assertGreater(len(v["physical_microtiles"]),192)

    def test_holes_overlap_and_cross_logical_tile_rejected(self):
        for change in ("hole","overlap","cross"):
            v=plan()
            if change=="hole":v["physical_microtiles"][0]["taps_chw"].pop()
            elif change=="overlap":v["physical_microtiles"].append({**v["physical_microtiles"][0],"micro_id":"duplicate"})
            else:v["physical_microtiles"][0]["taps_chw"][-1]=32
            with self.assertRaises(ValueError):p.validate(v)
            with self.assertRaises(ValueError):a.Replay(v,"b"*64)

    def test_duplicate_case_rejected(self):
        v=plan();v["cases"].append(copy.deepcopy(v["cases"][0]))
        with self.assertRaises(ValueError):p.validate(v)
        with self.assertRaises(ValueError):a.Replay(v,"b"*64)

    def test_hand_computed_native_rounding_zero_brackets_matrix(self):
        v=plan();f=freeze(v);m=measured(f)
        export,replay=self.both(v)
        export.add(f);export.add(m);replay.accept(f);replay.accept(m)
        # Independent explicit rounded voltages: p=.12208-(.03052+.06104)/2=.07630;
        # n=.04578-(.01526+.03052)/2=.02289; signed C=[.5,-1.75].
        expected=.5*.07630-1.75*.02289
        self.assertAlmostEqual(export.values[0],expected,places=12)
        self.assertAlmostEqual(replay.dense[0],expected,places=12)
        self.assertEqual(export.measured,1)
        with self.assertRaises(ValueError):export.finish()
        with self.assertRaises(ValueError):replay.complete()

    def test_duplicate_partial_rejected_not_summed_twice(self):
        v=plan();f=freeze(v);m=measured(f)
        for obj in self.both(v):
            add=obj.add if isinstance(obj,e.Accumulator) else obj.accept
            add(f);add(m)
            duplicate={**m,"ordinal":3}
            with self.assertRaises(ValueError):add(duplicate)

    def test_no_freeze_or_changed_freeze_or_gain_rejected(self):
        v=plan();f=freeze(v);m=measured(f)
        for obj in self.both(v):
            add=obj.add if isinstance(obj,e.Accumulator) else obj.accept
            with self.assertRaises(ValueError):add({**m,"ordinal":1})
        for field,value in (("calibration_sha256","c"*64),("raw_manifest_sha256","c"*64),("image_sha256","c"*64)):
            for obj in self.both(v):
                add=obj.add if isinstance(obj,e.Accumulator) else obj.accept
                add(f)
                with self.assertRaises(ValueError):add({**m,field:value})
        changed={**f,"fixed_gain_v_per_weight":2.}
        changed["calibration_sha256"]=e.calibration_digest(changed)
        for obj in self.both(v):
            with self.assertRaises(ValueError):(obj.add if isinstance(obj,e.Accumulator) else obj.accept)(changed)

    def test_refreeze_cannot_change_matrix_mid_cifar(self):
        v=plan();f=freeze(v);later=freeze(v,2)
        later["matrix_C2"]=[[2.,0.],[0.,1.]]
        later["calibration_sha256"]=e.calibration_digest(later)
        for obj in self.both(v):
            add=obj.add if isinstance(obj,e.Accumulator) else obj.accept
            add(f)
            with self.assertRaises(ValueError):add(later)

    def test_invalid_adc_missing_measurement_and_out_of_range_coordinates(self):
        v=plan();f=freeze(v)
        mutations=[]
        m=measured(f);m["terms"][0]["positive_adc"][1]=16383;mutations.append(m)
        m=measured(f);m["terms"]=[];mutations.append(m)
        mutations.append({**measured(f),"x":28})
        mutations.append({**measured(f),"case_id":"missing"})
        for m in mutations:
            for obj in self.both(v):
                add=obj.add if isinstance(obj,e.Accumulator) else obj.accept
                add(f)
                with self.assertRaises(ValueError):add(m)

    def test_chunked_stream_hash_order_and_missing_line_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            repo=Path(tmp);lines=[b'{"ordinal":1}\n',b'{"ordinal":2}\n']
            parts=[]
            for i,line in enumerate(lines):
                path=repo/f"p{i}.jsonl";path.write_bytes(line)
                parts.append(dict(path=path.name,sha256=hashlib.sha256(line).hexdigest(),bytes=len(line),records=1))
            index=dict(parts=parts,records=2,sha256=hashlib.sha256(b"".join(lines)).hexdigest())
            self.assertEqual(list(e.records(repo,index)),[{"ordinal":1},{"ordinal":2}])
            bad=copy.deepcopy(index);bad["parts"].reverse()
            with self.assertRaises(ValueError):list(e.records(repo,bad))
            (repo/"p1.jsonl").write_bytes(lines[1][:-1])
            with self.assertRaises(ValueError):list(e.records(repo,index))


if __name__=="__main__":unittest.main()
