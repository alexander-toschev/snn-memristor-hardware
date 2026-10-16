"""Offline native request/result -> measured partial linkage, with pinned ancestry.

The reviewed historical300-coefficient dataset is a compatibility test, never
full-image coverage. No driver imports, backend calls, or CPU weight evaluation.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import struct

TOP_SESSION="results/cifar_feedback_remaining_reserves/58ca618b-c37f-454e-be25-db21d1d25f60"
TOP_MANIFEST="5fb5041cbdba148ff0a02182512a98ca0784fed074e2830d6fc06b230f933171"
TOP_AUDIT="results/cifar_feedback_reviews/remaining_hardware_20261016/audit.json"
TOP_AUDIT_SHA="62a3f84f10ce55eb55be4930c605108a31440359d7e46c086d5f594a48ae162e"
TOP_AUDITOR="4f2f73eb96fd286aabde4e0513b7aac2ef96e57ef873ce36573a73a29a3b0525"
PINS={
 "vendor/MemriBoard/manager/service/converters.py":"28c6495c56afa225501dd6b0dea999215f08b73f30e016ff7ea34d6c5412701c",
 "weights/best_fs.pt":"ee4fc4f58a3d010218116e9ad4c7ba808787959f9ed7687cd58cbb32d90f62fb",
 "weights/prepared_cifar100_conv1.json":"86113ff1d2fdbe155056d2e1377cf380e7395120fd93ccf2e1efc7a0a933167c",
 "weights/cifar100_conv1_fixture.json":"8cb9ec0fe30242328527073bb076f8019295d73a748343f58fc2a8671bf7c154",
 "weights/conv1_weight.npy":"8be312b9eafb9de23f8340b17f848e7e0e2e7884293c826a6d9f6cbb5125d440",
}


def path(repo,name):
    if Path(name).is_absolute() or PureWindowsPath(name).drive or PureWindowsPath(name).root:
        raise ValueError("Absolute path forbidden")
    p=(repo/name).resolve()
    if not p.is_relative_to(repo.resolve()):raise ValueError("Path escape")
    return p


def sha(file):
    h=hashlib.sha256()
    with file.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()


def bound_json(repo,name,digest):
    f=path(repo,name)
    if sha(f)!=digest:raise ValueError("Source SHA mismatch: "+name)
    return json.loads(f.read_text())


def close(a,b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(a,b,abs_tol=1e-11,rel_tol=1e-10)


def volts(adc):
    if type(adc) is not int or not 0<=adc<16383:raise ValueError("Invalid/saturated native ADC")
    return round(adc*5/16383,5)


def flat(v):
    if isinstance(v,list):
        for x in v:yield from flat(x)
    else:yield v


def input_images(repo):
    fixture=bound_json(repo,"weights/cifar100_conv1_fixture.json",PINS["weights/cifar100_conv1_fixture.json"])
    out=[]
    for c in fixture["cases"]:
        image=list(flat(c["image_chw_uint8"]))
        if len(image)!=3072 or any(type(v) is not int or not 0<=v<=255 for v in image):raise ValueError("Image shape/bytes")
        if hashlib.sha256(bytes(image)).hexdigest()!=c["image_chw_uint8_sha256"]:raise ValueError("Image SHA")
        # Exact integer equivalent of float32 TTFS for uint8 inputs: no internal
        # boundary except0/255 since gcd(255,11)=1. Threshold requires byte>=26.
        times=[((255-v)*11)//255 if v>=26 else None for v in image]
        y,x=c["patch_top_left_yx"]
        expected=[times[(tap//25)*1024+(y+(tap%25)//5)*32+x+tap%5] for tap in range(75)]
        for t in range(12):
            if [int(v==t) for v in expected]!=c["spikes_t12x75"][t]:raise ValueError("Pinned TTFS disagreement")
        out.append(dict(case=c,image=image,times=times))
    if len(out)!=10:raise ValueError("Expected ten pinned real images")
    return out


def npy_weights(repo):
    name="weights/conv1_weight.npy"
    # Historical filename is found by exact pinned bytes, not guessed contents.
    candidates=[repo/name,repo/"weights/conv1_weights.npy"]
    found=[f for f in candidates if f.exists() and sha(f)==PINS[name]]
    if len(found)!=1:raise ValueError("Pinned extracted Conv1 NPY unavailable/ambiguous")
    data=found[0].read_bytes()
    if data[:6]!=b"\x93NUMPY":raise ValueError("NPY header")
    if data[6]==1:n=struct.unpack("<H",data[8:10])[0];start=10
    elif data[6] in (2,3):n=struct.unpack("<I",data[8:12])[0];start=12
    else:raise ValueError("NPY version")
    meta=ast.literal_eval(data[start:start+n].decode("latin1"))
    if meta.get("descr")!="<f4" or meta.get("fortran_order") is not False or meta.get("shape")!=(64,3,5,5):raise ValueError("NPY dtype/order/shape")
    payload=data[start+n:]
    if len(payload)!=4800*4:raise ValueError("NPY payload")
    return payload


def trusted_manifests(repo):
    root=bound_json(repo,TOP_SESSION+"/manifest.json",TOP_MANIFEST)
    audit=bound_json(repo,TOP_AUDIT,TOP_AUDIT_SHA)
    if (audit.get("audit_status")!="passed" or audit.get("errors")!=[] or
        audit.get("auditor_sha256")!=TOP_AUDITOR or audit.get("manifest_sha256")!=TOP_MANIFEST):
        raise ValueError("Pinned full acquisition audit invalid")
    manifests={TOP_SESSION:root}
    def collect(v):
        if isinstance(v,dict):
            if "session" in v and "manifest_sha256" in v:
                session=v["session"]
                m=bound_json(repo,session+"/manifest.json",v["manifest_sha256"])
                if session in manifests and manifests[session]!=m:raise ValueError("Conflicting ancestry")
                manifests[session]=m
                if "audit_path" in v:
                    proof=bound_json(repo,v["audit_path"],v["audit_sha256"])
                    if proof.get("errors")!=[] or proof.get("manifest_sha256")!=v["manifest_sha256"]:
                        raise ValueError("Ancestor proof inconsistency")
            for k,x in v.items():
                if k not in ("result_sha256","source_sha256"):collect(x)
        elif isinstance(v,list):
            for x in v:collect(x)
    collect(root["parent_import"])
    for session,m in manifests.items():
        if m.get("source")!="hardware":raise ValueError("Cross-backend ancestry")
        for k in ("vendor/MemriBoard/manager/service/converters.py","weights/best_fs.pt","weights/prepared_cifar100_conv1.json"):
            if m["source_sha256"].get(k)!=PINS[k]:raise ValueError("Recorded converter/checkpoint/TTFS pin mismatch")
    return manifests


def resolve_wrapper(repo,wrapper,manifests):
    expected=wrapper["result"]
    current=wrapper;seen=set()
    while True:
        origin=current["origin"];session=origin["session"];name=origin["path"]
        key=(session,name,origin.get("wrapper_index"))
        if key in seen:raise ValueError("Ancestry cycle")
        seen.add(key)
        if session not in manifests:raise ValueError("Unbound source session")
        if manifests[session]["result_sha256"].get(name)!=origin["sha256"]:raise ValueError("Origin not committed in manifest")
        value=bound_json(repo,session+"/"+name,origin["sha256"])
        if "wrapper_index" in origin:
            current=value[origin["wrapper_index"]]
            if current["result"]!=expected:raise ValueError("Imported result changed")
        else:
            if value!=expected:raise ValueError("Native tile result changed")
            return session,str(Path(name).parent).replace("\\","/"),expected


def tile_json(repo,session,folder,name,manifest):
    relative=folder+"/"+name
    return bound_json(repo,session+"/"+relative,manifest["result_sha256"][relative])


def native_rows(repo,session,folder,manifest):
    """Stream and hash all parts, preserving request/result pairs across splits."""
    index=tile_json(repo,session,folder,"raw_parts.json",manifest)
    all_hash=hashlib.sha256();totalbytes=records=0
    for i,part in enumerate(index["parts"]):
        name="raw.jsonl" if i==0 else f"raw.{i:05d}.jsonl"
        if part["path"]!=name:raise ValueError("Raw part order/name")
        rel=folder+"/"+name
        if manifest["result_sha256"].get(rel)!=part["sha256"]:raise ValueError("Uncommitted raw part")
        h=hashlib.sha256();n=size=0
        with path(repo,session+"/"+rel).open("rb") as f:
            for line in f:
                if not line.endswith(b"\n"):raise ValueError("Raw truncated line")
                h.update(line);all_hash.update(line);n+=1;size+=len(line)
                yield json.loads(line)
        if h.hexdigest()!=part["sha256"] or n!=part["records"] or size!=part["bytes"]:raise ValueError("Raw part bytes/hash/count")
        records+=n;totalbytes+=size
    if all_hash.hexdigest()!=index["sha256"] or records!=index["records"] or totalbytes!=index["bytes"]:
        raise ValueError("Raw concatenated integrity")


class NativeLink:
    """Independent stream parser for reviewed one-hot paired native CIFAR rows.

    historical_xy is only supplied by the pinned spot-test adapter. A future
    full-image adapter must pass None and provide output_y/output_x in every
    native request. The programming/source/safety audit remains separately bound.
    """
    def __init__(self,session_id,source,filter_id,slots,images,matrix,gain,
                 route_sha,calibration_sha,matrix_sha,historical_xy=None):
        self.sid=session_id;self.source=source;self.fid=filter_id;self.slots=slots;self.images=images
        self.matrix=matrix;self.gain=gain;self.route_sha=route_sha;self.cal_sha=calibration_sha;self.mat_sha=matrix_sha
        self.xy=historical_xy;self.pending=None;self.last_id=0;self.route_frozen=False;self.cal_frozen=False
        self.group_key=None;self.group=[];self.completed=set();self.cifar_tasks=0;self.pairs=0
        self.expected_first=None;self.native_matrix_before=False
        if source not in ("hardware","synthetic_test"):raise ValueError("Unsupported adapter source")
        if gain!=1. or len(matrix)!=2 or any(len(r)!=2 for r in matrix) or any(not math.isfinite(v) for r in matrix for v in r):
            raise ValueError("Fixed hardware gain/C contract")
        if not slots or len({s["tap_index"] for s in slots})!=len(slots):raise ValueError("Route taps duplicate/empty")
        for side in ("positive_cell","negative_cell"):
            cells=[s[side] for s in slots]
            if len({v[0] for v in cells})!=1 or len({tuple(v) for v in cells})!=len(cells):raise ValueError("Row route ambiguity")
            if any(type(a) is not int or a not in range(n) for v in cells for a,n in zip(v,(8,32))):raise ValueError("Address range")

    def accept(self,row):
        emitted=[]
        if row.get("session_id")!=self.sid:raise ValueError("Session mismatch")
        event=row.get("event")
        if event=="route_frozen":
            if self.pending or self.route_frozen or row.get("sha256")!=self.route_sha or row.get("before_task_id")!=self.last_id+1:
                raise ValueError("Route freeze placement/hash")
            self.route_frozen=True;return emitted
        if event=="calibration_frozen":
            if self.pending or not self.route_frozen or self.cal_frozen or row.get("sha256")!=self.cal_sha or row.get("matrix_sha256")!=self.mat_sha:
                raise ValueError("Calibration freeze identity/order")
            if row.get("before_task_id")!=self.last_id+1:raise ValueError("Calibration freeze task boundary")
            self.cal_frozen=True;self.expected_first=self.last_id+1;return emitted
        if event=="skipped":
            if self.pending:raise ValueError("Skip interrupts request/result")
            return emitted  # source-pinned acquisition auditor verifies current/pulse guard
        if event=="request":
            if self.pending or row.get("task_id")!=self.last_id+1:raise ValueError("Missing/duplicate request ID")
            if row.get("source")!=self.source:raise ValueError("Cross-source raw")
            self.pending=row;return emitted
        if event!="result" or self.pending is None:raise ValueError("Orphan result/unreviewed event")
        req=self.pending
        if any(row.get(k)!=v for k,v in req.items() if k!="event"):raise ValueError("Request/result metadata mutation")
        if row.get("status")!="ok":raise ValueError("Native task failed")
        self.last_id=req["task_id"];self.pending=None;self.pairs+=1
        if req.get("phase")!="cifar":
            if self.group:
                emitted.append(self.finish_group())
            return emitted
        if not self.cal_frozen:raise ValueError("CIFAR precedes frozen C/gain")
        if not self.cifar_tasks and req["task_id"]!=self.expected_first:raise ValueError("Tasks intervened after final freeze")
        self.cifar_tasks+=1
        if req.get("filter_id")!=self.fid:raise ValueError("Wrong filter")
        ci,rep,t=(req.get(k) for k in ("case","repeat","t"))
        if any(type(v) is not int or not 0<=v<n for v,n in zip((ci,rep,t),(len(self.images),5,12))):raise ValueError("Case/time/repeat")
        case=self.images[ci]["case"]
        if req.get("original_test_index")!=case["original_test_index"]:raise ValueError("Wrong original image")
        if self.xy is None:
            y,x=req.get("output_y"),req.get("output_x")
        else:
            y,x=self.xy
            if ("output_y" in req and req["output_y"]!=y) or ("output_x" in req and req["output_x"]!=x):raise ValueError("Changed historical location")
        if any(type(v) is not int or not 0<=v<28 for v in (y,x)):raise ValueError("Explicit full-image spatial coordinate required")
        key=ci,rep,t,y,x
        if self.group_key is not None and key!=self.group_key:
            emitted.append(self.finish_group())
        if key in self.completed:raise ValueError("Repeated logical CIFAR record")
        self.group_key=key
        cmd=req.get("command_si",{})
        if cmd.get("mode_flag")!=10 or cmd.get("id")!=0 or set(cmd)!={"mode_flag","vol","wl","id"}:
            raise ValueError("CIFAR must be native mode10 id0")
        if req.get("native_adc_samples")!=1 or req.get("returned_id",0)!=0:raise ValueError("Native averaging/ID")
        native=row.get("native_result")
        if not isinstance(native,list) or len(native)!=3 or native[1]!=0 or row.get("returned_id")!=0 or native[2]!=row.get("raw_adc"):
            raise ValueError("Native result/ADC/returnedID mismatch")
        voltage=volts(native[2])
        if not close(row.get("output_voltage_v",float("nan")),voltage):raise ValueError("Native voltage conversion mismatch")
        if voltage>4.:raise ValueError("Measured output exceeds reviewed4V guard")
        conversions=req.get("native_voltage_conversion")
        vector=cmd.get("vol")
        if not isinstance(vector,list) or len(vector)!=32 or any(v not in (0.,.1) for v in vector) or sum(v!=0 for v in vector)>1:
            raise ValueError("Unreviewed non-one-hot CIFAR command")
        expected=[dict(requested_v=v,dac_code=82 if v else 0,native_decoded_v=.1 if v else 0.) for v in vector]
        if conversions!=expected:raise ValueError("DAC/source conversion mismatch")
        self.group.append((req,row))
        return emitted

    def finish_group(self):
        ci,repeat,t,y,x=self.group_key
        times=self.images[ci]["times"]
        active=[s for s in self.slots if times[(s["tap_index"]//25)*1024+(y+(s["tap_index"]%25)//5)*32+x+s["tap_index"]%5]==t]
        expected=[]
        for side in ("positive_cell","negative_cell"):
            expected.append((side,"zero_before",None))
            expected.extend((side,"signal",s) for s in active)
            if not active:expected.append((side,"empty_zero_signal",None))
            expected.append((side,"zero_after",None))
        if len(self.group)!=len(expected):raise ValueError("Missing/extra native zero/signal terms")
        by_side={};used=[]
        for (req,result),(side,role,slot) in zip(self.group,expected):
            if req.get("side")!=side or req.get("role")!=role:raise ValueError("CIFAR row/zero bracket order")
            cell=(slot or self.slots[0])[side]
            if req["wl"]!=cell[0] or req["command_si"]["wl"]!=cell[0]:raise ValueError("Wrong physical WL")
            vector=[0.]*32
            if slot:
                vector[cell[1]]=.1
                if req.get("tap")!=slot["tap_index"]:raise ValueError("Signal tap/source mismatch")
            elif "tap" in req:raise ValueError("Zero row has signal tap")
            if req["command_si"]["vol"]!=vector:raise ValueError("TTFS input/BL routing mismatch")
            by_side.setdefault(side,[]).append((req["task_id"],result["raw_adc"]))
            used.append(req["task_id"])
        if used!=list(range(used[0],used[-1]+1)):raise ValueError("Nonfresh/intervening task sequence")
        pos=by_side["positive_cell"];neg=by_side["negative_cell"]
        terms=[]
        for i in range(max(1,len(active))):
            ps=[pos[0],pos[i+1],pos[-1]];ns=[neg[0],neg[i+1],neg[-1]]
            terms.append(dict(positive_adc=[v[1] for v in ps],negative_adc=[v[1] for v in ns],
                              raw_task_ids=[v[0] for v in ps+ns],signal_tap=None if not active else active[i]["tap_index"]))
        row_sums=[]
        for side in (pos,neg):
            zero=(volts(side[0][1])+volts(side[-1][1]))/2
            row_sums.append(sum(volts(v[1])-zero for v in side[1:-1]))
        corrected=[sum(c*v for c,v in zip(r,row_sums)) for r in self.matrix]
        output=dict(case=ci,repeat=repeat,t=t,output_y=y,output_x=x,
                    image_sha256=self.images[ci]["case"]["image_chw_uint8_sha256"],
                    active_taps=[s["tap_index"] for s in active],terms=terms,raw_row_sums_v=row_sums,
                    corrected_row_sums_v=corrected,trained_unit_contribution=(corrected[0]-corrected[1])/self.gain,
                    first_task_id=used[0],last_task_id=used[-1])
        self.completed.add(self.group_key);self.group=[];self.group_key=None
        return output

    def finish(self):
        if self.pending:raise ValueError("Truncated request without result")
        out=[self.finish_group()] if self.group else []
        if not self.cal_frozen or not self.route_frozen:raise ValueError("Missing freezes")
        return out


def historical_check(repo,output_name):
    output=path(repo,output_name)
    if output.exists():raise ValueError("Preserve existing review")
    manifests=trusted_manifests(repo);root=manifests[TOP_SESSION]
    images=input_images(repo);weights=npy_weights(repo)
    for name in ("vendor/MemriBoard/manager/service/converters.py","weights/best_fs.pt","weights/prepared_cifar100_conv1.json"):
        if sha(repo/name)!=PINS[name]:raise ValueError("Current source/input differs from pinned numerical contract")
    wrappers=bound_json(repo,TOP_SESSION+"/completed_microtiles.json",root["result_sha256"]["completed_microtiles.json"])
    all_taps=set();blocks=[];ledger_hash=hashlib.sha256();total=tasks=0;maxerror=0.
    for wrapper in wrappers:
        session,folder,tile=resolve_wrapper(repo,wrapper,manifests)
        m=manifests[session]
        if not tile.get("technical_completed"):raise ValueError("Incomplete contributing tile")
        route=tile_json(repo,session,folder,"frozen_route.json",m)
        cal=tile_json(repo,session,folder,"frozen_calibration.json",m)
        matrix=tile_json(repo,session,folder,"frozen_matrix.json",m)
        slots=route["slots"]
        if slots!=tile["effective_route"]:raise ValueError("Effective route changed")
        fid=tile["filter_id"]
        if fid not in range(4):raise ValueError("Historical scope must be4filters")
        for s in slots:
            tap=s["tap_index"]
            if type(tap) is not int or tap not in range(75) or (fid,tap) in all_taps:raise ValueError("Historical tap duplicate/outside")
            all_taps.add((fid,tap))
            raw=weights[(fid*75+tap)*4:(fid*75+tap+1)*4]
            if s["float32_le_hex"]!=raw.hex() or s["trained_weight"]!=struct.unpack("<f",raw)[0] or s["tap_chw"]!=[tap//25,(tap%25)//5,tap%5]:
                raise ValueError("Trained coefficient/CHW mismatch")
        if cal.get("decoder_gain_v_per_weight")!=1. or cal.get("decoder_gain_is_fixed_before_acquisition") is not True:
            raise ValueError("Fixed hardware decoder changed")
        if matrix.get("offset_v")!=[0.,0.] or matrix.get("uses_cifar_or_trained_targets") is not False:
            raise ValueError("C calibration context")
        parser=NativeLink(m["session_id"],"hardware",fid,slots,images,matrix["matrix"],1.,
                          m["result_sha256"][folder+"/frozen_route.json"],m["result_sha256"][folder+"/frozen_calibration.json"],
                          m["result_sha256"][folder+"/frozen_matrix.json"],historical_xy=(13,13))
        saved_path=folder+"/partials.jsonl"
        if sha(path(repo,session+"/"+saved_path))!=m["result_sha256"][saved_path]:raise ValueError("Saved partials changed")
        saved=[json.loads(line) for line in path(repo,session+"/"+saved_path).read_text().splitlines()]
        reconstructed=[]
        for row in native_rows(repo,session,folder,m):reconstructed.extend(parser.accept(row))
        reconstructed.extend(parser.finish())
        if len(saved)!=600 or len(reconstructed)!=600 or parser.completed!={(c,r,t,13,13) for c in range(10) for r in range(5) for t in range(12)}:
            raise ValueError("Historical600points incomplete/duplicate")
        for expected,actual in zip(saved,reconstructed):
            if any(expected[k]!=actual[k] for k in ("case","repeat","t","active_taps")):raise ValueError("Partial identity/input mismatch")
            if not close(expected["trained_unit_contribution"],actual["trained_unit_contribution"]):raise ValueError("Native contribution mismatch")
            for key in ("raw_row_sums_v","corrected_row_sums_v"):
                if any(not close(a,b) for a,b in zip(expected[key],actual[key])):raise ValueError("Raw/calibrated rows mismatch")
            # Check the saved ADC/task references independently against reconstructed raw.
            sides=expected["measured_rows"]
            for term_i,term in enumerate(actual["terms"]):
                for side_i,key in enumerate(("positive_adc","negative_adc")):
                    s=sides[side_i];refs=[s["zero_before"],s["signals"][term_i],s["zero_after"]]
                    if [r["raw_adc"] for r in refs]!=term[key] or [r["task_id"] for r in refs]!=term["raw_task_ids"][side_i*3:side_i*3+3]:
                        raise ValueError("Saved partial copied/replaced raw ADC/reference")
            maxerror=max(maxerror,abs(expected["trained_unit_contribution"]-actual["trained_unit_contribution"]))
            ledger_hash.update((json.dumps(dict(origin_session=session,origin_folder=folder,filter_id=fid,**actual),sort_keys=True,separators=(",",":"))+"\n").encode())
        blocks.append(dict(origin_session=session,origin_folder=folder,filter_id=fid,taps=[s["tap_index"] for s in slots],
                           native_pairs=parser.pairs,cifar_tasks=parser.cifar_tasks,partial_points=len(reconstructed),
                           logical32_boundary_crossed=len({0 if s["tap_index"]<32 else 1 if s["tap_index"]<64 else 2 for s in slots})>1))
        total+=len(reconstructed);tasks+=parser.cifar_tasks
    if all_taps!={(f,t) for f in range(4) for t in range(75)}:raise ValueError("Historical300coeff coverage")
    report=dict(status="passed",scope="actual historical4filter/one-position300tap native request/result linkage",
                source="hardware_records_offline_replay",hardware_calls=0,backend_calls=0,
                current_hardware_state_claim=False,full_image_coverage=False,full_classifier_import_enabled=False,
                root_manifest_sha256=TOP_MANIFEST,root_acquisition_audit_sha256=TOP_AUDIT_SHA,
                root_acquisition_auditor_sha256=TOP_AUDITOR,ancestry_sessions=list(manifests),
                trained_coefficients=300,filters=4,spatial_positions_per_image=1,position=[13,13],
                real_images=10,repeats=5,timesteps=12,contributing_microtiles=len(blocks),partial_points=total,
                fresh_cifar_native_tasks=tasks,maximum_saved_contribution_error=maxerror,
                normalized_historical_ledger_sha256=ledger_hash.hexdigest(),blocks=blocks,
                native_request_result_adc_ttfs_route_link_verified=True,
                source_and_safety_basis="Explicit pinned historical acquisition audits plus bound ancestor manifests; this replay does not re-run their programming/current-gate mathematics",
                limitation="Old spot-patch and cross-logical-boundary tiles cannot be passed as full image export; no future full-image collector is registered",
                verifier_sha256=sha(Path(__file__)))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--repo",type=Path,default=Path.cwd())
    ap.add_argument("--check-historical-300",action="store_true",required=True);ap.add_argument("--output",required=True)
    args=ap.parse_args();result=historical_check(args.repo,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!="blocks"}))
