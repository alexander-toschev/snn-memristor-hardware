"""Independent stdlib replay of normalized ADC ledger and dense export.

Never imports exporter, plan generator, classifier, torch or any driver.
PASS is arithmetic/coverage consistency, not native acquisition attestation.
"""
from array import array
import argparse
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import struct

CHECKPOINT="ee4fc4f58a3d010218116e9ad4c7ba808787959f9ed7687cd58cbb32d90f62fb"


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def path_in(repo,name):
    if Path(name).is_absolute() or PureWindowsPath(name).drive or PureWindowsPath(name).root:
        raise ValueError("Absolute path rejected")
    p=(repo/name).resolve()
    if not p.is_relative_to(repo.resolve()):raise ValueError("Path traversal")
    return p


def file_sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b=f.read(1048576)
            if not b:break
            h.update(b)
    return h.hexdigest()


class Replay:
    def __init__(self,plan,raw_sha):
        if plan.get("schema")!="full-conv1-plan-v1" or plan.get("checkpoint_sha256")!=CHECKPOINT:
            raise ValueError("Wrong architecture/checkpoint contract")
        if plan.get("dimensions")!={"timesteps":12,"filters":64,"height":28,"width":28,"logical_tiles":3,"taps":75}:
            raise ValueError("Incomplete dimensions")
        self.plan=plan;self.raw_sha=raw_sha;self.repeats=plan["repeats"]
        if plan.get("source_kind") not in ("hardware","simulator","synthetic_test") or plan.get("fixed_decoder_gain_v_per_weight")!=(.03 if plan["source_kind"]=="simulator" else 1.):
            raise ValueError("Fixed source/gain contract")
        if type(self.repeats) is not int or not 1<=self.repeats<=5:raise ValueError("Repeat count")
        self.cases={v["case_id"]:(i,v["image_sha256"]) for i,v in enumerate(plan["cases"])}
        if not self.cases or len(self.cases)!=len(plan["cases"]):raise ValueError("Duplicate/empty cases")
        self.routes={}
        covered=set()
        for i,m in enumerate(plan["physical_microtiles"]):
            f,k=m["filter_id"],m["logical_tile"]
            if type(f) is not int or type(k) is not int or f not in range(64) or k not in range(3):raise ValueError("Route dimension")
            if m["micro_id"] in self.routes or not m["taps_chw"]:raise ValueError("Repeated/empty route")
            for tap in m["taps_chw"]:
                if type(tap) is not int or tap not in range(75):raise ValueError("Tap outside Conv1")
                logical=0 if tap<32 else 1 if tap<64 else 2
                if logical!=k or (f,tap) in covered:raise ValueError("Wrong tile or overlapping coefficient")
                covered.add((f,tap))
            self.routes[m["micro_id"]]=(i,m)
        if covered!={(f,tap) for f in range(64) for tap in range(75)}:raise ValueError("Not all4800 coefficients")
        self.n=len(self.routes)
        self.observed=bytearray(len(self.cases)*self.repeats*self.n*12*28*28)
        self.dense=array("d",[0.])*(len(self.cases)*self.repeats*12*64*28*28*3)
        self.frozen={};self.ordinal=0;self.count=0

    def accept(self,row):
        self.ordinal+=1
        if row.get("ordinal")!=self.ordinal or row.get("raw_manifest_sha256")!=self.raw_sha:
            raise ValueError("Ledger order/provenance")
        micro=row.get("micro_id")
        if micro not in self.routes:raise ValueError("Unknown route")
        _,route=self.routes[micro]
        if row.get("event")=="calibration_frozen":
            if micro in self.frozen:raise ValueError("Refreeze forbidden")
            if row.get("calibration_sha256")!=digest({k:v for k,v in row.items() if k!="calibration_sha256"}):
                raise ValueError("Calibration hash")
            if row.get("tap_route_sha256")!=digest(route):raise ValueError("Route freeze identity")
            gain=row["fixed_gain_v_per_weight"];c=row["matrix_C2"]
            if type(gain) not in (int,float) or not math.isfinite(gain) or gain<=0:raise ValueError("Gain")
            if gain!=self.plan["fixed_decoder_gain_v_per_weight"]:raise ValueError("Gain changed from plan")
            if not isinstance(c,list) or len(c)!=2 or any(len(v)!=2 for v in c) or any(not math.isfinite(float(v)) for a in c for v in a):
                raise ValueError("Matrix")
            if row.get("fitted_weight_scale_used") is not False or row.get("calibration_uses_cifar") is not False:
                raise ValueError("Posthoc fitted/calibration source")
            self.frozen[micro]=row;return
        if row.get("event")!="measured_partial" or micro not in self.frozen:raise ValueError("Unfrozen measurement")
        frozen=self.frozen[micro]
        if row.get("calibration_sha256")!=frozen["calibration_sha256"]:raise ValueError("Changed freeze")
        if row.get("case_id") not in self.cases:raise ValueError("Unknown case")
        case,image=self.cases[row["case_id"]]
        if row.get("image_sha256")!=image:raise ValueError("Different image")
        repeat,t,y,x=(row.get(k) for k in ("repeat","t","y","x"))
        for v,n in zip((repeat,t,y,x),(self.repeats,12,28,28)):
            if type(v) is not int or v not in range(n):raise ValueError("Coverage index")
        mi=self.routes[micro][0]
        key=case*(self.repeats*self.n*9408)+repeat*self.n*9408+mi*9408+t*784+y*28+x
        if self.observed[key]:raise ValueError("Duplicate partial")
        self.observed[key]=1
        terms=row.get("terms")
        if not isinstance(terms,list) or not terms:raise ValueError("Unmeasured empty partial")
        decoded=0.
        for term in terms:
            if set(term)!={"positive_adc","negative_adc","raw_task_refs"}:raise ValueError("Unexpected term")
            refs=term["raw_task_refs"]
            if not isinstance(refs,list) or len(refs)!=6 or len(set(refs))!=6 or any(not isinstance(v,str) or not v for v in refs):
                raise ValueError("Unbound task reference")
            corrected=[]
            for side in ("positive_adc","negative_adc"):
                adcs=term[side]
                if len(adcs)!=3 or any(type(v) is not int or v<0 or v>=16383 for v in adcs):raise ValueError("ADC saturation/invalid")
                volts=[round(v*5.0/((1<<14)-1),5) for v in adcs]
                corrected.append(volts[1]-.5*volts[0]-.5*volts[2])
            c=frozen["matrix_C2"]
            plus=sum(a*b for a,b in zip(c[0],corrected))
            minus=sum(a*b for a,b in zip(c[1],corrected))
            decoded+=(plus-minus)/frozen["fixed_gain_v_per_weight"]
        if not math.isfinite(decoded):raise ValueError("Nonfinite decoded value")
        index=case*self.repeats*1806336+repeat*1806336+t*150528+route["filter_id"]*2352+y*84+x*3+route["logical_tile"]
        self.dense[index]+=decoded;self.count+=1

    def complete(self):
        if len(self.frozen)!=self.n or sum(self.observed)!=len(self.observed):
            raise ValueError(f"Incomplete physical coverage: {self.count}/{len(self.observed)}")


def audit(repo,manifest_name,export_name):
    manifest_path=path_in(repo,manifest_name);export_path=path_in(repo,export_name)
    m=json.loads(manifest_path.read_text());e=json.loads((export_path/"export.json").read_text())
    if m.get("schema")!="full-conv1-adc-ledger-v1":raise ValueError("Normalized schema")
    if e["input_manifest"]!={"path":manifest_name,"sha256":file_sha(manifest_path)}:raise ValueError("Export input binding")
    plan_file=path_in(repo,m["plan"]["path"]);raw_file=path_in(repo,m["raw_manifest"]["path"])
    if file_sha(plan_file)!=m["plan"]["sha256"] or file_sha(raw_file)!=m["raw_manifest"]["sha256"]:
        raise ValueError("Plan/raw SHA mismatch")
    plan=json.loads(plan_file.read_text());r=Replay(plan,m["raw_manifest"]["sha256"])
    if m.get("source_kind")!=plan.get("source_kind") or e.get("source_kind")!=m.get("source_kind"):raise ValueError("Source label")
    h=hashlib.sha256();names=set();total=0
    for p in m["ledger"]["parts"]:
        if p["path"] in names:raise ValueError("Duplicate ledger part")
        names.add(p["path"]);file=path_in(repo,p["path"])
        if file_sha(file)!=p["sha256"] or file.stat().st_size!=p["bytes"]:raise ValueError("Part commitment")
        lines=0
        with file.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):raise ValueError("Incomplete line")
                h.update(line);lines+=1;total+=1;r.accept(json.loads(line))
        if lines!=p["records"]:raise ValueError("Part count")
    if total!=m["ledger"]["records"] or h.hexdigest()!=m["ledger"]["sha256"]:raise ValueError("Whole stream commitment")
    r.complete()
    payload=export_path/"conv1_partials.f32le"
    if file_sha(payload)!=e["payload_sha256"] or payload.stat().st_size!=len(r.dense)*4 or e["payload_bytes"]!=len(r.dense)*4:
        raise ValueError("Dense payload commitment/size")
    if (e.get("coverage_complete") is not True or e.get("hardware_provenance_verified") is not False or
        e.get("hardware_import_enabled") is not False or e.get("bn_applied") is not False or
        e.get("measured_physical_partials")!=r.count or e.get("calibration_events")!=r.n or
        e.get("ledger_records")!=r.ordinal or e.get("shape")!=[len(r.cases),r.repeats,12,64,28,28,3] or
        e.get("dtype")!="float32-little-endian" or e.get("stage")!="pre_bn" or e.get("units")!="checkpoint_conv1" or
        e.get("checkpoint_sha256")!=CHECKPOINT or e.get("plan_sha256")!=m["plan"]["sha256"] or
        e.get("raw_manifest_sha256")!=m["raw_manifest"]["sha256"] or
        e.get("axis_order")!="case,repeat,t,filter,y,x,logical_tile" or
        e.get("schema")!="full-conv1-dense-export-draft-v1"):
        raise ValueError("Export coverage/status declarations")
    max_error=0.
    with payload.open("rb") as f:
        for expected in r.dense:
            actual=struct.unpack("<f",f.read(4))[0]
            if not math.isfinite(actual) or not math.isclose(actual,expected,abs_tol=2e-7,rel_tol=1e-6):
                raise ValueError("Dense raw-derived arithmetic mismatch")
            max_error=max(max_error,abs(actual-expected))
    return dict(status="passed",errors=[],scope="normalized-ledger arithmetic and complete coverage only",
                hardware_provenance_verified=False,hardware_import_enabled=False,
                unresolved_boundary="Native collector/raw-task values -> normalized ADC ledger is not attested",
                physical_partials=r.count,dense_values=len(r.dense),maximum_float32_absolute_error=max_error,
                input_manifest_sha256=file_sha(manifest_path),export_manifest_sha256=file_sha(export_path/"export.json"),
                auditor_sha256=file_sha(Path(__file__)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path.cwd());p.add_argument("--manifest",required=True)
    p.add_argument("--export",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();out=path_in(a.repo,a.output)
    if out.exists():raise ValueError("Preserve audit")
    report=audit(a.repo,a.manifest,a.export)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report))


if __name__=="__main__":main()
