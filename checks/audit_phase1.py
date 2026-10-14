"""Independent offline phase-1 statistics from one mean per write attempt."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

if not __debug__:
    raise RuntimeError('Audit assertions require normal Python; do not use -O or -OO')


def audit(run):
    saved = json.loads((run/'characterization.json').read_text())
    with (run/'characterization_attempts.csv').open(newline='') as stream:
        attempts = list(csv.DictReader(stream))
    assert len(attempts) == saved['attempt_count'] == 75
    assert saved['post_read_count'] == 375
    assert sum(r['target_reached'] == 'True' for r in attempts) == saved['target_reached_count']
    groups, failures = [], []
    differences = []
    for group in saved['groups']:
        wl,bl,q = group['wl'],group['bl'],group['target_q']
        rows = [r for r in attempts if (int(r['wl']),int(r['bl']),float(r['target_q'])) == (wl,bl,q)]
        assert len(rows) == 5 and {int(r['cycle']) for r in rows} == set(range(5))
        target = q/330
        assert all(float(r['target_conductance_siemens']) == target for r in rows)
        valid = [float(r['mean_conductance_siemens']) for r in rows if r['reset_ok'] == 'True']
        errors = [value-target for value in valid]
        bias = math.fsum(errors)/len(errors)
        sd = math.sqrt(math.fsum((e-bias)**2 for e in errors)/(len(errors)-1)) if len(errors)>1 else None
        result = dict(wl=wl,bl=bl,target_q=q,target_conductance_siemens=target,
                      attempt_count=5,reset_success_count=len(valid),
                      target_reached_count=sum(r['target_reached']=='True' for r in rows),
                      cycle_count=len(valid),bias_s=bias,error_std_s_ddof1=sd,
                      rmse_s=math.sqrt(math.fsum(e*e for e in errors)/len(errors)))
        for key,value in result.items():
            same = value == group[key] if value is None else math.isclose(value,group[key],rel_tol=1e-12,abs_tol=1e-15)
            if not same:
                failures.append(dict(group=[wl,bl,q],field=key,recomputed=value,saved=group[key]))
            if value is not None:
                differences.append(abs(value-group[key]))
        groups.append(result)
    assert len(groups) == 15
    return dict(source='simulator',phase='phase1',passed=not failures,errors=failures,
                method='Independent arithmetic from characterization_attempts.csv, one mean per programming cycle; failed RESETs excluded.',
                limits='CSV consistency check; raw ADC reconstruction audited separately. Five cycles per cell/target; target_reached means first crossing.',
                attempt_count=len(attempts),groups=groups,max_absolute_difference=max(differences),
                input_hashes={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in
                              ('characterization.json','characterization_attempts.csv')})


if __name__ == '__main__':
    result = audit(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('phase','passed','attempt_count','max_absolute_difference')}))
    raise SystemExit(0 if result['passed'] else 1)
