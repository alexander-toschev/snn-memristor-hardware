"""Complete1.8M-key in-memory synthetic software check; no backend or files of ADC data."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import full_cifar_plan as p
import full_cifar_export as e
import audit_full_cifar_export as a


def run():
    start=time.monotonic()
    plan=p.make_plan([dict(case_id="synthetic-software-test",image_sha256="a"*64)])
    plan["source_kind"]="synthetic_test"
    exporter=e.Accumulator(plan,"b"*64);auditor=a.Replay(plan,"b"*64)
    ordinal=0;expected_by_micro={}
    for micro in plan["physical_microtiles"]:
        ordinal+=1
        freeze=dict(event="calibration_frozen",ordinal=ordinal,micro_id=micro["micro_id"],raw_manifest_sha256="b"*64,
                    matrix_C2=[[1.,0.],[0.,1.]],fixed_gain_v_per_weight=1.,
                    fitted_weight_scale_used=False,calibration_uses_cifar=False,tap_route_sha256=p.digest_object(micro))
        freeze["calibration_sha256"]=e.calibration_digest(freeze)
        exporter.add(freeze);auditor.accept(freeze)
        adc=100+micro["filter_id"]*3+micro["logical_tile"]
        expected_by_micro[micro["micro_id"]]=round(adc*5/16383,5)
        for t in range(12):
            for y in range(28):
                for x in range(28):
                    ordinal+=1
                    row=dict(event="measured_partial",ordinal=ordinal,micro_id=micro["micro_id"],raw_manifest_sha256="b"*64,
                             calibration_sha256=freeze["calibration_sha256"],case_id="synthetic-software-test",image_sha256="a"*64,
                             repeat=0,t=t,y=y,x=x,terms=[dict(positive_adc=[0,adc,0],negative_adc=[0,0,0],
                             raw_task_refs=[f"synthetic-only:{ordinal}:{i}" for i in range(6)])])
                    exporter.add(row);auditor.accept(row)
    payload=exporter.finish();auditor.complete()
    maximum=0.
    for index,(actual,) in enumerate(struct.iter_unpack("<f",payload)):
        k=index%3;f=(index//2352)%64
        expected=round((100+f*3+k)*5/16383,5)
        if abs(actual-expected)>1e-7 or abs(auditor.dense[index]-expected)>1e-12:
            raise ValueError("Full dense axis/arithmetic mismatch")
        maximum=max(maximum,abs(actual-expected))
    return dict(status="passed",source_kind="synthetic_software_test",backend_calls=0,hardware_calls=0,
                measurement_claim=False,physical_partial_keys=exporter.measured,dense_values=len(auditor.dense),
                logical_coverage="T12,F64,Y28,X28,three logical tiles,all4800 coefficient positions",
                records=ordinal,calibration_events=192,maximum_float32_error=maximum,
                runtime_seconds=time.monotonic()-start,
                source_sha256={name:e.sha(Path(__file__).parent/name) for name in
                               ("full_cifar_plan.py","full_cifar_export.py","audit_full_cifar_export.py",Path(__file__).name)},
                limitations="In-memory synthetic data only; native raw normalization/export attestation is unimplemented")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    args=parser.parse_args();out=e.relative(Path.cwd(),args.output)
    if out.exists():raise ValueError("Preserve previous proof")
    report=run();out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report))
