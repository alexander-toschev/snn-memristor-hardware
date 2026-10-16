"""Software-only normalized ADC ledger to dense pre-BN Conv1 export.

No model weights, CPU W*x, driver, acquisition or trusted hardware attestation.
Native raw -> normalized ledger is deliberately an unresolved external boundary.
"""
from array import array
import argparse
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import struct
import sys

import full_cifar_plan as plan_api


def relative(repo,name):
    if Path(name).is_absolute() or PureWindowsPath(name).drive or PureWindowsPath(name).root:
        raise ValueError("Only repository-relative paths")
    out=(repo/name).resolve()
    if not out.is_relative_to(repo.resolve()):raise ValueError("Path escape")
    return out


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def scalar_adc(adc):
    if type(adc) is not int or not 0<=adc<16383:
        raise ValueError("Invalid or saturated14bit ADC")
    # Native MemriBoard convert_adc_to_volt(gain=1,adc_bit=14,ref=5).
    return round(adc*5/16383,5)


def calibration_digest(event):
    return plan_api.digest_object({k:v for k,v in event.items() if k!="calibration_sha256"})


class Accumulator:
    """Strict exact physical-key coverage; no acquisition order assumption."""
    def __init__(self,plan,source_manifest_sha):
        self.routes=plan_api.validate(plan)
        self.plan=plan
        self.case_index={c["case_id"]:i for i,c in enumerate(plan["cases"])}
        self.micro_index={key:i for i,key in enumerate(self.routes)}
        self.n=len(self.routes)
        self.r=plan["repeats"]
        samples=len(self.case_index)*self.r*12*784
        self.coverage=bytearray(samples*self.n)
        self.values=array("d",[0.])*(samples*64*3)
        self.calibrations={}
        self.source_sha=source_manifest_sha
        self.records=0
        self.measured=0

    def add(self,row):
        self.records+=1
        if row.get("ordinal")!=self.records:raise ValueError("Nonconsecutive ledger ordinal")
        if row.get("raw_manifest_sha256")!=self.source_sha:raise ValueError("Raw provenance changed")
        micro=row.get("micro_id")
        if micro not in self.routes:raise ValueError("Unknown microtile")
        if row.get("event")=="calibration_frozen":
            if micro in self.calibrations:raise ValueError("Calibration changed or duplicated")
            if row.get("calibration_sha256")!=calibration_digest(row):raise ValueError("Calibration hash mismatch")
            matrix=row.get("matrix_C2")
            if (not isinstance(matrix,list) or len(matrix)!=2 or any(len(v)!=2 for v in matrix)
                or any(not math.isfinite(float(v)) for r in matrix for v in r)):
                raise ValueError("Invalid C2")
            gain=row.get("fixed_gain_v_per_weight")
            if type(gain) not in (float,int) or not math.isfinite(gain) or gain<=0:raise ValueError("Invalid fixed gain")
            if gain!=self.plan["fixed_decoder_gain_v_per_weight"]:raise ValueError("Gain differs from fixed plan")
            if row.get("fitted_weight_scale_used") is not False or row.get("calibration_uses_cifar") is not False:
                raise ValueError("Calibration contaminated or fitted scale")
            if row.get("tap_route_sha256")!=plan_api.digest_object(self.routes[micro]):
                raise ValueError("Frozen route mismatch")
            self.calibrations[micro]=row
            return
        if row.get("event")!="measured_partial" or micro not in self.calibrations:
            raise ValueError("Measurement before calibration freeze or unknown event")
        frozen=self.calibrations[micro]
        if row.get("calibration_sha256")!=frozen["calibration_sha256"]:raise ValueError("Changed calibration reference")
        if row.get("case_id") not in self.case_index:raise ValueError("Unknown image/case")
        case=self.case_index[row["case_id"]]
        if row.get("image_sha256")!=self.plan["cases"][case]["image_sha256"]:raise ValueError("Image mismatch")
        vals=[row.get(k) for k in ("repeat","t","y","x")]
        if any(type(v) is not int or not 0<=v<limit for v,limit in zip(vals,(self.r,12,28,28))):
            raise ValueError("Index outside complete plan")
        repeat,t,y,x=vals
        spatial=t*784+y*28+x
        i=((case*self.r+repeat)*self.n+self.micro_index[micro])*9408+spatial
        if self.coverage[i]:raise ValueError("Duplicate physical partial")
        self.coverage[i]=1
        terms=row.get("terms")
        if not isinstance(terms,list) or not terms:raise ValueError("No measured terms; missing zeros cannot be synthesized")
        total=0.
        for term in terms:
            if set(term)!={"positive_adc","negative_adc","raw_task_refs"}:raise ValueError("Unexpected measured term fields")
            refs=term["raw_task_refs"]
            if not isinstance(refs,list) or len(refs)!=6 or any(not isinstance(r,str) or not r for r in refs) or len(set(refs))!=6:
                raise ValueError("Six distinct raw task references required per paired bracket")
            rows=[]
            for key in ("positive_adc","negative_adc"):
                if len(term[key])!=3:raise ValueError("Expected zero-before/signal/zero-after")
                before,signal,after=map(scalar_adc,term[key])
                rows.append(signal-(before+after)/2)
            c=frozen["matrix_C2"]
            total+=((c[0][0]-c[1][0])*rows[0]+(c[0][1]-c[1][1])*rows[1])/frozen["fixed_gain_v_per_weight"]
        if not math.isfinite(total):raise ValueError("Nonfinite measured contribution")
        route=self.routes[micro]
        j=(((((case*self.r+repeat)*12+t)*64+route["filter_id"])*28+y)*28+x)*3+route["logical_tile"]
        self.values[j]+=total
        self.measured+=1

    def finish(self):
        if len(self.calibrations)!=self.n or not all(self.coverage):
            raise ValueError(f"Incomplete physical coverage: {self.measured}/{len(self.coverage)}; no CPU/zero fill")
        result=array("f",self.values)
        if not all(math.isfinite(v) for v in result):raise ValueError("float32 overflow")
        if sys.byteorder!="little":result.byteswap()
        return result.tobytes()


def records(repo,index):
    seen=set()
    stream_hash=hashlib.sha256()
    count=0
    for part in index["parts"]:
        name=part["path"]
        if name in seen:raise ValueError("Duplicate ledger part")
        seen.add(name)
        path=relative(repo,name)
        if sha(path)!=part["sha256"] or path.stat().st_size!=part["bytes"]:raise ValueError("Ledger part hash/bytes mismatch")
        part_count=0
        with path.open("rb") as f:
            for line in f:
                if not line.endswith(b"\n"):raise ValueError("Truncated JSONL")
                stream_hash.update(line);part_count+=1;count+=1
                yield json.loads(line)
        if part_count!=part["records"]:raise ValueError("Part record count mismatch")
    if count!=index["records"] or stream_hash.hexdigest()!=index["sha256"]:raise ValueError("Concatenated ledger hash/count mismatch")


def export(repo,manifest_name,output_name):
    inp=relative(repo,manifest_name)
    m=json.loads(inp.read_text())
    if m.get("schema")!="full-conv1-adc-ledger-v1":raise ValueError("Wrong normalized-ledger schema")
    plan_path=relative(repo,m["plan"]["path"])
    if sha(plan_path)!=m["plan"]["sha256"]:raise ValueError("Plan hash mismatch")
    raw_path=relative(repo,m["raw_manifest"]["path"])
    if sha(raw_path)!=m["raw_manifest"]["sha256"]:raise ValueError("Raw manifest hash mismatch")
    plan=json.loads(plan_path.read_text())
    if m.get("source_kind")!=plan.get("source_kind"):raise ValueError("Source label changed")
    acc=Accumulator(plan,m["raw_manifest"]["sha256"])
    for row in records(repo,m["ledger"]):acc.add(row)
    data=acc.finish()
    output=relative(repo,output_name)
    if output.exists():raise ValueError("Preserve previous export")
    output.mkdir(parents=True)
    (output/"conv1_partials.f32le").write_bytes(data)
    report=dict(schema="full-conv1-dense-export-draft-v1",source_kind=m["source_kind"],
                hardware_provenance_verified=False,hardware_import_enabled=False,
                boundary="Native raw to normalized ledger is NOT independently implemented or attested",
                input_manifest=dict(path=manifest_name,sha256=sha(inp)),plan_sha256=m["plan"]["sha256"],
                raw_manifest_sha256=m["raw_manifest"]["sha256"],shape=[len(plan["cases"]),plan["repeats"],12,64,28,28,3],
                dtype="float32-little-endian",axis_order="case,repeat,t,filter,y,x,logical_tile",
                stage="pre_bn",units="checkpoint_conv1",checkpoint_sha256=plan["checkpoint_sha256"],
                bn_applied=False,coverage_complete=True,measured_physical_partials=acc.measured,
                calibration_events=len(acc.calibrations),ledger_records=acc.records,
                payload_sha256=hashlib.sha256(data).hexdigest(),payload_bytes=len(data),
                exporter_sha256=sha(Path(__file__)))
    (output/"export.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path.cwd())
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args();print(json.dumps(export(a.repo,a.manifest,a.output)))


if __name__=="__main__":main()
