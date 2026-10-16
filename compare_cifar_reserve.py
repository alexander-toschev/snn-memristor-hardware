"""Draft offline reserve-A comparison. No device, collector, auditor or Git execution.

Only a reviewed audit PASS permits derived metrics; failed acquisition remains partial.
B/C deliberately need an independently reviewed contract before being accepted here.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

HELPER_SHA = '7fd0aeddea06bbed6286750e236c272c536f1f372ed165b685d6b97232c971c0'
RESERVE_AUDITOR_SHA = '372f48a05abc652cda0f2c9d5891660a9c23429f53d18ea84749c587f5e10459'
PROTOCOL = 'cifar-feedback-reserve-wl7-bl15-v1'
TARGET = {(3, t) for t in range(52, 65)}
ALL = {(f, t) for f in range(4) for t in range(75)}
RESERVES = ((6, 2), (6, 11), (7, 15))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def local(repo, value):
    require(not Path(value).is_absolute(), 'Path must be repository-relative')
    result = (repo / value).resolve()
    require(result.is_relative_to(repo), 'Path escapes repository')
    return result


def helper(repo):
    path = repo / 'compare_cifar_adaptive.py'
    require(sha(path) == HELPER_SHA, 'Reviewed pure comparison helper changed')
    namespace = {'__name__': '_reviewed_comparison_helper', '__file__': str(path)}
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), namespace)
    return SimpleNamespace(**namespace)


def bound_json(directory, manifest, name):
    path = local(directory, name)
    require(sha(path) == manifest['result_sha256'].get(name), 'Unbound or changed artifact: ' + name)
    return load(path)


def indexed_weights(wrappers):
    """Reject duplicates and missing finite measured coefficients; no zero fill."""
    result = {}
    identities = set()
    for w in wrappers:
        identity = (w['filter_id'], w['parent_tile_id'], w['microtile_id'])
        require(identity not in identities, 'Duplicate microtile')
        identities.add(identity)
        r = w['result']
        require(r['technical_completed'] is True, 'Incomplete contributing microtile')
        route = r['effective_route']
        cal = r['calibration']
        require(len(route) == len(cal['effective_weights']) == len(cal['trained_weights']), 'Weight geometry differs')
        for slot, effective, trained in zip(route, cal['effective_weights'], cal['trained_weights']):
            key = (w['filter_id'], slot['tap_index'])
            require(key not in result, 'Duplicate trained tap')
            require(trained == slot['trained_weight'], 'Trained coefficient changed')
            result[key] = dict(trained=trained, effective=effective)
    return result


def weight_stats(wrappers, h):
    values = indexed_weights(wrappers)
    for row in values.values():
        require(h.finite(row['trained']) and h.finite(row['effective']), 'Nonfinite weight')
    keys = sorted(values)
    return dict(coefficients=len(keys), keys=[list(k) for k in keys],
                effective_vs_trained=h.error_stats([values[k]['effective'] for k in keys], [values[k]['trained'] for k in keys]))


def address_sets(wrappers):
    sets = {k: set() for k in ('reset_probed', 'reset_eligible', 'assigned', 'set_attempted')}
    for w in wrappers:
        r = w['result']
        for c in r['reset_candidates']:
            sets['reset_probed'].add(tuple(c['cell']))
            if c['eligible']:
                sets['reset_eligible'].add(tuple(c['cell']))
        for slot in r['effective_route']:
            for side in ('positive_cell', 'negative_cell'):
                sets['assigned'].add(tuple(slot[side]))
        for p in r['pairs']:
            if p['programming_pulse_attempts']:
                require(p['active_set_side'] in ('positive_cell', 'negative_cell'), 'SET side missing')
                sets['set_attempted'].add(tuple(p[p['active_set_side']]))
    return {k: dict(count=len(v), cells=[list(c) for c in sorted(v)]) for k, v in sets.items()}


def dispositions(tile, complete):
    candidates = {tuple(c['cell']): c for c in tile['reset_candidates']}
    require(len(candidates) == len(tile['reset_candidates']), 'Duplicate candidate')
    slots = tile.get('effective_route', [])
    out = []
    for cell in RESERVES:
        c = candidates.get(cell)
        used = [s['tap_index'] for s in slots if list(cell) in (s['positive_cell'], s['negative_cell'])]
        if c is None:
            disposition = 'not_assessed_technical_stop'
        elif not c['eligible']:
            disposition = 'not_freshly_quiet_under_bounded_protocol'
        elif complete and used:
            disposition = 'freshly_quiet_assigned_and_measured'
        else:
            disposition = 'freshly_quiet_without_complete_permitted_route'
        set_count = sum(p['programming_pulse_attempts'] for p in tile.get('pairs', [])
                        if p.get('active_set_side') in ('positive_cell', 'negative_cell')
                        and p[p['active_set_side']] == list(cell))
        out.append(dict(cell=list(cell), disposition=disposition, assigned_taps=used,
                        set_pulses=set_count, accurate_weight_claim=False,
                        reset_evidence=None if c is None else {k: c.get(k) for k in
                        ('eligible', 'reset_stop', 'baseline_mvm_mean_v', 'baseline_mvm_sd_v', 'mode10_quiet_limit_v')}))
    return out


def partial_metrics(tile, h):
    """Fresh13 conv contributions only. BN/LIF is meaningful only after all75."""
    weights = {s['tap_index']: s['trained_weight'] for s in tile['effective_route']}
    effective = dict(zip(weights, tile['calibration']['effective_weights']))
    rows = {}
    for p in tile['partials']:
        key = (p['case'], p['repeat'], p['t'])
        require(key not in rows, 'Duplicate fresh partial')
        require(len(set(p['active_taps'])) == len(p['active_taps']), 'Duplicate active tap')
        require(set(p['active_taps']) <= set(weights), 'Partial contains other taps')
        value = p['trained_unit_contribution']
        require(h.finite(value), 'Nonfinite fresh actual')
        rows[key] = dict(actual=value, trained=sum(weights[t] for t in p['active_taps']),
                         effective=sum(effective[t] for t in p['active_taps']))
    require(set(rows) == {(c, r, t) for c in range(10) for r in range(5) for t in range(12)}, 'Fresh600 partial coverage missing')
    order = sorted(rows)
    return dict(points=600, scope='fresh13_only_before_BN_LIF',
                actual_vs_trained=h.error_stats([rows[k]['actual'] for k in order], [rows[k]['trained'] for k in order]),
                actual_vs_effective=h.error_stats([rows[k]['actual'] for k in order], [rows[k]['effective'] for k in order]),
                aligned=[dict(case=k[0], repeat=k[1], t=k[2], **rows[k]) for k in order])


def acquisition_complete(manifest, audit, directory):
    """Failure omits the flag in native A; only audited failure may do so."""
    complete = audit['acquisition_complete']
    require(type(complete) is bool, 'Invalid audited acquisition flag')
    require(manifest['status'] == audit['acquisition_status'] == ('completed' if complete else 'failed'), 'Completion status differs')
    if complete:
        require(manifest.get('acquisition_complete') is True, 'Completed manifest lacks explicit acquisition flag')
    else:
        require('acquisition_complete' not in manifest or manifest['acquisition_complete'] is False, 'Failed manifest acquisition flag differs')
        require(not (directory/'summary.json').exists() and not (directory/'completed_microtiles.json').exists(), 'Failed route restored old13/full output')
    return complete


def verify_session(repo, session, audit_path, source, indices, h):
    require(RESERVE_AUDITOR_SHA is not None, 'Reserve audit SHA not frozen')
    directory = local(repo, session)
    manifest = load(directory / 'manifest.json')
    audit_file = local(repo, audit_path)
    a = load(audit_file)
    require(manifest['source'] == source and manifest['hardware_access'] is (source == 'hardware'), 'Wrong backend')
    require(manifest['protocol_id'] == PROTOCOL, 'Unreviewed reserve protocol (B/C need own contract)')
    require(a['audit_status'] == 'passed' and a['errors'] == [] and a['checks'] > 0, 'Audit not PASS')
    require(a['auditor_sha256'] == RESERVE_AUDITOR_SHA, 'Unreviewed reserve auditor')
    require(a['manifest_sha256'] == sha(directory / 'manifest.json'), 'Audit manifest mismatch')
    require(all(a[k] == manifest[k] for k in ('session_id', 'source', 'protocol_id')), 'Audit identity mismatch')
    require(manifest['source_sha256'][h.FIXTURE] == h.FIXTURE_SHA, 'Different fixture')
    gain = 1.0 if source == 'hardware' else .03
    require(manifest['fixed_decoder_gain_v_per_weight'] == gain, 'Fixed gain mismatch')
    new = h.counters(manifest['new_counters'])
    require(new == a['new_counters'], 'New counters differ')
    parent = manifest['parent_import']
    pd = local(repo, parent['session'])
    require(sha(pd / 'manifest.json') == parent['manifest_sha256'], 'Parent manifest changed')
    require(sha(local(repo, parent['audit_path'])) == parent['audit_sha256'], 'Parent audit changed')
    verified_parent = h.verify_session(repo, parent['session'], parent['audit_path'], source, indices)
    pm = load(pd / 'manifest.json')
    require(parent['result_sha256'] == pm['result_sha256'], 'Parent result hash dictionary differs')
    require(parent['ancestry42'] == pm['parent_import'], 'Nested42 metadata differs')
    require(parent['imported_trained_coefficients'] == 287, 'Parent import count differs')
    require(a['parent_import_audit']['manifest_sha256'] == parent['manifest_sha256'] and
            a['parent_import_audit']['audit_sha256'] == parent['audit_sha256'], 'Audited parent binding differs')
    originals = bound_json(pd, pm, 'completed_microtiles.json')
    imported = bound_json(directory, manifest, 'imported_microtiles.json')
    expected = {(w['filter_id'], w['parent_tile_id'], w['microtile_id']): w for w in originals
                if (w['filter_id'], w['parent_tile_id']) != (3, 4)}
    require(len(imported) == len(expected), 'Historical microtile count differs')
    for w in imported:
        identity = (w['filter_id'], w['parent_tile_id'], w['microtile_id'])
        require(identity in expected and w['result'] == expected[identity]['result'], 'Historical result replaced')
        origin = w['origin']
        require(origin['session'] == parent['session'] and origin['path'] == 'completed_microtiles.json'
                and origin['sha256'] == pm['result_sha256']['completed_microtiles.json'], 'Historical wrapper binding changed')
        require(origin['ancestry_original_origin'] == expected[identity]['origin'], 'Nested historical origin changed')
        require(originals[origin['wrapper_index']] == expected[identity], 'Historical wrapper index changed')
    require(set(indexed_weights(imported)) == ALL - TARGET, 'Historical287 are not exact complement')
    for key, manifest_key, parent_key in (
            ('impact_attempts', 'cumulative_impacts_including_all_ancestry', 'cumulative_impacts_including_all_parent_attempts'),
            ('programming_pulse_attempts', 'cumulative_pulses_including_all_ancestry', 'cumulative_pulses_including_all_parent_attempts')):
        require(manifest[manifest_key] == a[manifest_key] == new[key] + pm[parent_key], 'Ancestry omitted/double counted')
    tile = bound_json(directory, manifest, 'tiles/f3_p4_reserve/tile.json')
    complete = acquisition_complete(manifest, a, directory)
    provenance = dict(session=session, session_id=manifest['session_id'], source=source,
        manifest_sha256=sha(directory / 'manifest.json'), independent_audit=audit_path,
        independent_audit_sha256=sha(audit_file), auditor_sha256=a['auditor_sha256'],
        source_commit=manifest['invocation']['git']['commit'], fixed_gain=gain,
        acquisition_complete=complete, new_counters=new, parent=verified_parent['provenance'],
        parent_import_audit=a['parent_import_audit'], result_sha256=manifest['result_sha256'],
        cumulative_impacts=manifest['cumulative_impacts_including_all_ancestry'],
        cumulative_pulses=manifest['cumulative_pulses_including_all_ancestry'],
        reserve_dispositions=dispositions(tile, complete), gate=a['fresh_tile'].get('gate'),
        new_coefficients=13 if complete else 0, retained_historical_coefficients=287,
        full_output_points=2400 if complete else 0, fresh_output_points=600 if complete else 0,
        historical287_metrics=weight_stats(imported, h))
    if not complete:
        require(a['full_time_outputs'] == a['new_partial_points'] == 0, 'Failed route reports outputs')
        require(not (directory / 'summary.json').exists() and not (directory / 'completed_microtiles.json').exists(), 'Failed route restored old13/full output')
        provenance['contributing_address_union'] = None
        provenance['historical287_address_union'] = address_sets(imported)
        return dict(provenance=provenance)
    wrappers = bound_json(directory, manifest, 'completed_microtiles.json')
    fresh = [w for w in wrappers if not w['origin']['reused']]
    require(len(fresh) == 1 and fresh[0]['result'] == tile, 'Fresh13 result mismatch')
    require(set(indexed_weights(fresh)) == TARGET and set(indexed_weights(wrappers)) == ALL, '300/fresh13 geometry mismatch')
    require([w for w in wrappers if w['origin']['reused']] == imported, 'Combined historical287 changed')
    require(a['full_time_outputs'] == 2400 and a['new_partial_points'] == 600, 'Audited output count differs')
    summary = bound_json(directory, manifest, 'summary.json')
    require(summary['new_trained_coefficients'] == 13 and summary['imported_trained_coefficients'] == 287, 'Summary new/history counts differ')
    require(summary['parent_import'] == parent and summary['fixed_decoder_gain_v_per_weight'] == gain, 'Summary binding/gain differs')
    for k in ('physical_weight_quality_pass', 'fresh_weight_quality_pass'):
        require(type(summary[k]) is bool and summary[k] == a[k], 'Quality differs from audit')
        provenance[k] = summary[k]
    # Reserve summary omits this redundant count; normalize only a local view.
    require(summary.get('microtiles_completed', len(wrappers)) == len(wrappers), 'Summary microtile count differs')
    actual, effective = h.indexed_summary({**summary, 'microtiles_completed': len(wrappers)}, indices)
    union = address_sets(wrappers)
    old = address_sets(originals)
    newcells = {tuple(c) for c in union['assigned']['cells']}
    oldcells = {tuple(c) for c in old['assigned']['cells']}
    provenance.update(summary_sha256=sha(directory / 'summary.json'), all300_metrics=weight_stats(wrappers, h),
        fresh13_metrics=weight_stats(fresh, h), fresh13_partials=partial_metrics(tile, h),
        fresh_matrix={k: tile['matrix'].get(k) for k in ('matrix','selected_model','selection_pass','reference_design_rank','training_design_rank','full_matrix_identifiable_from_reference')},
        fresh_physical_fidelity=tile['physical_fidelity'], fresh_final_physical_fidelity=tile['final_physical_fidelity'],
        pair_stop_counts=dict(Counter(p['stop'] for p in tile['pairs'])), contributing_address_union=union,
        assigned_addresses_added=[list(c) for c in sorted(newcells-oldcells)],
        assigned_addresses_displaced=[list(c) for c in sorted(oldcells-newcells)],
        fresh_address_union=address_sets(fresh), inherited_contributing_counters=a['parent_import_audit']['imported_contributing_counters'])
    return dict(actual=actual, effective=effective, provenance=provenance)


def make_report(sim, hardware, trained, h):
    complete = all(d['provenance']['acquisition_complete'] for d in (sim, hardware))
    result = dict(schema='reserve-A-offline-comparison-draft-v1', hardware_io=False,
        helper_sha256=HELPER_SHA, audit_pin=RESERVE_AUDITOR_SHA,
        comparison_complete=complete, sessions={s:d['provenance'] for s,d in [('simulator',sim),('hardware',hardware)]},
        comparisons=h.build_comparison(sim, hardware, trained) if complete else None,
        interpretation='13 new + 287 historical; no classifier accuracy, no fresh300 claim; consistency audit is not weight fidelity.')
    if complete:
        sp = sim['provenance']['fresh13_partials']['aligned']
        hp = hardware['provenance']['fresh13_partials']['aligned']
        require([(r['case'],r['repeat'],r['t']) for r in sp] == [(r['case'],r['repeat'],r['t']) for r in hp], 'Fresh partial keys differ')
        result['fresh13_hardware_vs_simulator'] = h.error_stats([r['actual'] for r in hp], [r['actual'] for r in sp])
    return result


def markdown(report):
    lines = ['# Reserve A: offline evidence comparison', '', report['interpretation'], '',
             'Full comparison available: ' + str(report['comparison_complete']) + '.', '',
             '| Source | New coefficients | Historical coefficients | Derived outputs | New calls | New writes | All ancestry calls |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for source, p in report['sessions'].items():
        lines.append(f"| {source} | {p['new_coefficients']} | {p['retained_historical_coefficients']} | {p['full_output_points']} | {p['new_counters']['impact_attempts']} | {p['new_counters']['programming_pulse_attempts']} | {p['cumulative_impacts']} |")
    if report['comparison_complete']:
        lines += ['', '| Comparison | Conv RMSE | BN RMSE | Pre-LIF RMSE | Post-LIF RMSE | Spike differences /2400 |', '|---|---:|---:|---:|---:|---:|']
        for key, value in report['comparisons'].items():
            if key == 'repeat_variation':
                continue
            row = value['overall']
            lines.append('| ' + key + ' | ' + ' | '.join(f"{row['numeric'][f]['rmse']:.8g}" for f in ('conv_sum','bntt_current','lif_before_reset','lif_after_reset')) + f" | {row['spike_mismatches']} |")
    else:
        lines += ['', 'A failed route has no complete300 replacement or derived2400 output. The historical baseline remains a separate valid result; excluded old13 are never restored here.']
    lines += ['', '| Source | Reserve | Fresh disposition | Assigned taps | SET pulses |', '|---|---|---|---|---:|']
    for source, p in report['sessions'].items():
        for r in p['reserve_dispositions']:
            lines.append(f"| {source} | {r['cell']} | {r['disposition']} | {r['assigned_taps']} | {r['set_pulses']} |")
    lines += ['', 'comparison.json separates fresh13/old287/all300 weight metrics, fresh600 partials before BN/LIF, current skips, gains, immutable ancestry, audit/manifest/result hashes and contributing address unions/displacements. Assigned is not accurately programmed. Finite reserves do not prove absolute array optimality.', '',
              'Audit PASS is trusted through a frozen auditor hash and bound artifacts consumed here; this report does not replay raw commands. Sequential one-hot sums are not simultaneous analog MVM. Five repeats are not assumed independent; HW and SIM have different states and fixed gains. No runtime code is imported.', '']
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='.')
    for name in ('sim-session','hardware-session','sim-audit','hardware-audit','output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args(argv)
    repo = Path(args.repo).resolve()
    out = local(repo, args.output)
    require(not out.exists(), 'Output must be a fresh directory')
    h = helper(repo)
    fixture_path = local(repo, h.FIXTURE)
    require(sha(fixture_path) == h.FIXTURE_SHA, 'Fixture changed')
    fixture = load(fixture_path)
    indices = [c['original_test_index'] for c in fixture['cases']]
    sim = verify_session(repo, args.sim_session, args.sim_audit, 'simulator', indices, h)
    hardware = verify_session(repo, args.hardware_session, args.hardware_audit, 'hardware', indices, h)
    for data in (sim, hardware):
        require(not out.is_relative_to(local(repo, data['provenance']['session'])), 'Output would alter session')
        require(not out.is_relative_to(local(repo, data['provenance']['parent']['session'])), 'Output would alter parent')
    report = make_report(sim, hardware, h.trained_reference(fixture), h)
    report['comparison_script_sha256'] = sha(Path(__file__))
    out.mkdir(parents=True, exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    (out/'REPORT.md').write_text(markdown(report), encoding='utf-8')
    print(json.dumps(dict(status='completed' if report['comparison_complete'] else 'partial', output=args.output)))


if __name__ == '__main__':
    main()
