"""Explicit native-SIM component check, not a production CIFAR session or unit test."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import random
import uuid

ROOT=Path(__file__).resolve().parent
CORE_SHA='a46eb7b94891d1adf48d10f126227e10ecfa696aba61a98f293199a8dcb9f2ba'
CONFIG_SHA='c788a91a1dc51e4a9f6070354c9597249a45b4770decb07847030f683ef09663'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-native-simulator-component',action='store_true')
    args=parser.parse_args(argv)
    if not args.run_native_simulator_component:parser.print_help();return 0
    for name,pin in [('cifar_feedback_adaptive.py',CORE_SHA),('configs/cifar_feedback_adaptive.json',CONFIG_SHA)]:
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=pin:raise ValueError('Component target changed: '+name)
    import cifar_feedback_adaptive as a
    plan,routing,fixture=a.load_inputs();config,snapshot,hashes=a.check_sources('simulator')
    f,parent=a.tiles(routing)[0];pending=[a.slots(parent)[-1]]
    count=sum(map(len,a.candidate_pools(parent).values()))
    bounds=dict(maximum_impacts=count*244+811+a.measurement_tail(pending,fixture),
        maximum_programming_pulses=count*55+109,maximum_programming_decisions=count*55+109)
    sid=str(uuid.uuid4());output=ROOT/'.tmp/native-adaptive-n1'/sid;output.mkdir(parents=True)
    for name in (*a.PINS,a.CONFIG,'cifar_feedback_adaptive.py','run_native_adaptive_n1.py'):
        hashes[name]=a.sha256(ROOT/name)
    manifest=dict(session_id=sid,status='failed',source='simulator',hardware_io=False,
        component_only=True,production_cifar_execution=False,full300_coverage_claim=False,
        partition_setup='pending=[original f0_t0 last tap13], micro=1; no prefix acquisition is claimed',
        native_state_policy='fresh standard native array; no hidden state edits',
        original_candidate_count=count,pending_taps=[s['tap_index'] for s in pending],
        started_utc=a.now(),source_sha256=hashes,bounds=bounds,invocation=a.launch_provenance(argv))
    a.save_json(output/'start_manifest.json',manifest);a.save_json(output/'protocol.json',{**plan,**bounds})
    (output/'board.ini').write_bytes(snapshot)
    connector=run=None
    try:
        from experiments.unn_rev3_20261014.paths import use_local_dependencies
        use_local_dependencies()
        import numpy as np
        from manager.board import Connector
        from manager.service import converters
        from simulator.src import create_crossbar_array
        random.seed(plan['simulator_seed']);np.random.seed(plan['simulator_seed'])
        serial=output/'array';create_crossbar_array(str(serial),32,8)
        manifest['array_initial_sha256']=a.sha256(serial.with_suffix('.cb'))
        connector=Connector(silent=True,logger=logging.getLogger(__name__),cb_type='simulator',
            board_type='rp5_fpga_python',task_format='SI',config=config,crossbar_serial=str(serial))
        if not connector.open_port():raise ConnectionError('Native simulator open_port failed')
        folder=output/'microtile';folder.mkdir()
        run=a.AdaptiveRun(connector,converters,folder,sid,{**plan,**bounds},'simulator')
        result=run.collect_microtile(f,parent,pending,1,fixture)
        matrix=result['matrix']
        if (len(result['partials'])!=600 or len(result['effective_route'])!=1 or
                matrix['reference_design_rank']>1 or matrix['selected_model'] not in ('identity','diagonal') or
                any(c['eligible'] for c in matrix['candidates'] if c['name'].startswith('full_'))):
            raise ValueError('Native N1 component invariant failed')
        manifest.update(status='completed',partial_points=600,selected_model=matrix['selected_model'],
            reference_design_rank=matrix['reference_design_rank'],full_matrix_eligible=False)
    except (Exception,KeyboardInterrupt) as exc:
        manifest['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if run is not None:run.log.close()
        if connector is not None:
            try:
                if not connector.close_port():raise ConnectionError('Native simulator close_port failed')
            except Exception as exc:manifest.update(status='failed',close_error=str(exc))
        if run is not None:manifest.update(impact_attempts=run.sequence,programming_pulse_attempts=run.pulses)
        if (output/'array.cb').exists():manifest['array_final_file_sha256']=a.sha256(output/'array.cb')
        manifest.update(finished_utc=a.now(),result_sha256={p.relative_to(output).as_posix():a.sha256(p) for p in output.rglob('*') if p.is_file()})
        a.save_json(output/'manifest.json',manifest)
    print(json.dumps(dict(session=output.relative_to(ROOT).as_posix(),status=manifest['status'],
        impacts=manifest.get('impact_attempts'),matrix=manifest.get('selected_model'))))
    return 0 if manifest['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
