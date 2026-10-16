"""Pure lazy full Conv1 coverage plan. Does not choose a physical write policy."""
import argparse
import hashlib
import json
from pathlib import Path

CHECKPOINT_SHA = "ee4fc4f58a3d010218116e9ad4c7ba808787959f9ed7687cd58cbb32d90f62fb"
FIXTURE_SHA = "8cb9ec0fe30242328527073bb076f8019295d73a748343f58fc2a8671bf7c154"
RANGES = ((0,32),(32,64),(64,75))
DIMENSIONS = {"timesteps":12,"filters":64,"height":28,"width":28,"logical_tiles":3,"taps":75}


def digest_object(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def validate(plan):
    if plan.get("schema") != "full-conv1-plan-v1" or plan.get("dimensions") != DIMENSIONS:
        raise ValueError("Only the complete T12/64-filter/28x28/75-tap plan is supported")
    if plan.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise ValueError("Wrong checkpoint")
    if plan.get("source_kind") not in ("hardware","simulator","synthetic_test"):
        raise ValueError("Explicit source kind required")
    expected_gain=.03 if plan["source_kind"]=="simulator" else 1.
    if plan.get("fixed_decoder_gain_v_per_weight")!=expected_gain:
        raise ValueError("Fixed source gain must be declared before acquisition")
    if type(plan.get("repeats")) is not int or not 1 <= plan["repeats"] <= 5:
        raise ValueError("Require1..5 explicitly separate repeats")
    cases = plan.get("cases", [])
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and unique")
    for case in cases:
        if not isinstance(case["case_id"], str) or not case["case_id"] or len(case["image_sha256"]) != 64:
            raise ValueError("Invalid case identity")
    routes = plan.get("physical_microtiles", [])
    if len({m["micro_id"] for m in routes}) != len(routes):
        raise ValueError("Duplicate microtile identity")
    partition = {(f,k):set() for f in range(64) for k in range(3)}
    for m in routes:
        f,k = m["filter_id"],m["logical_tile"]
        if type(f) is not int or type(k) is not int or (f,k) not in partition:
            raise ValueError("Invalid filter/logical tile")
        taps=m["taps_chw"]
        lo,hi=RANGES[k]
        if not taps or any(type(t) is not int or not lo<=t<hi for t in taps) or len(set(taps))!=len(taps):
            raise ValueError("Microtile taps outside its logical tile, duplicated or empty")
        if partition[f,k].intersection(taps):
            raise ValueError("Overlapping physical taps")
        partition[f,k].update(taps)
    if any(v!=set(range(*RANGES[k])) for (f,k),v in partition.items()):
        raise ValueError("Missing physical taps: require all4800 exactly once")
    return {m["micro_id"]:m for m in routes}


def make_plan(cases, repeats=1, maximum_taps=32):
    if type(maximum_taps) is not int or not 1<=maximum_taps<=32:
        raise ValueError("maximum_taps must be1..32")
    routes=[]
    for f in range(64):
        for k,(lo,hi) in enumerate(RANGES):
            for a in range(lo,hi,maximum_taps):
                routes.append(dict(micro_id=f"f{f}_k{k}_a{a}",filter_id=f,logical_tile=k,
                                   taps_chw=list(range(a,min(a+maximum_taps,hi)))))
    plan=dict(schema="full-conv1-plan-v1",dimensions=DIMENSIONS.copy(),checkpoint_sha256=CHECKPOINT_SHA,
              source_kind="hardware",fixed_decoder_gain_v_per_weight=1.,
              cases=cases,repeats=repeats,physical_microtiles=routes,
              physical_write_strategy="UNIMPLEMENTED: no hardware acquisition authorized by this plan",
              normalized_ledger_contract="full-conv1-adc-ledger-v1")
    validate(plan)
    return plan


def iter_keys(plan):
    """Yield keys lazily; no list of millions of command or spatial records."""
    validate(plan)
    for case in plan["cases"]:
        for repeat in range(plan["repeats"]):
            for m in plan["physical_microtiles"]:
                for t in range(12):
                    for y in range(28):
                        for x in range(28):
                            yield case["case_id"],repeat,m["micro_id"],t,y,x


def estimate(plan):
    validate(plan)
    count=len(plan["cases"])*plan["repeats"]
    return dict(trained_coefficients=4800,spatial_positions=784,
                logical_values=count*12*64*784,logical_partials=count*12*64*784*3,
                physical_microtile_partials=count*12*784*len(plan["physical_microtiles"]),
                native_impacts=None,programming_pulses=None,
                reason="Native command composition, JIT write policy and fresh calibration strategy remain unimplemented")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path.cwd())
    p.add_argument("--output",required=True)
    p.add_argument("--repeats",type=int,default=1)
    p.add_argument("--maximum-taps",type=int,default=32)
    args=p.parse_args()
    from full_cifar_export import relative
    fixture=args.repo/"weights/cifar100_conv1_fixture.json"
    if hashlib.sha256(fixture.read_bytes()).hexdigest()!=FIXTURE_SHA: raise ValueError("Wrong fixture")
    cases=[dict(case_id=f"test_{c['original_test_index']}",image_sha256=c["image_chw_uint8_sha256"],
                original_test_index=c["original_test_index"],fine_label=c["fine_label"])
           for c in json.loads(fixture.read_text())["cases"]]
    plan=make_plan(cases,args.repeats,args.maximum_taps)
    out=relative(args.repo,args.output)
    if out.exists(): raise ValueError("Preserve existing plan")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(estimate(plan)))


if __name__=="__main__":main()
