"""Explicit exploratory measured-LUT transfer; strict diagnostic failures survive.

Two baseline policies use exactly the same physical samples. Neither changes
weights, replaces failed cells, fits a classifier, nor uses MNIST test labels.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from experiment_common.block_readout_validation import Collector as ReadCollector, Leg, Pair, patterns
from experiment_common.durable import Writer, atomic_json, sha
from experiment_common.native_memriboard import Converters
from experiment_common.selective_metrology import Source, number, object_dict
from experiment_common.tiled_transfer import AdaptiveProgrammer, Profile, Assignment, acquire_load, compose, schedule

if TYPE_CHECKING:
    from manager.board import Connector

Baseline = Literal['before_only', 'bracket_centered']


class ScopedProgrammer(AdaptiveProgrammer):
    """Allow every selected BL0..31 without changing the historical BL<25 API."""
    def __init__(self, connector: Connector, cv: Converters, writer: Writer, source: Source,
                 span: float, cells: frozenset[tuple[int, int]]) -> None:
        super().__init__(connector, cv, writer, source, span)
        self.allowed = cells

    def call(self, command: dict[str, int | float | list[float]], **metadata: object) -> tuple[int, float, int]:
        wl, bl = command.get('wl'), command.get('bl')
        if type(wl) is not int or type(bl) is not int or (wl, bl) not in self.allowed:
            raise ValueError('Programming/read outside pinned physical scope')
        if command.get('mode_flag') != 7 or not 0 <= number(command['vol']) <= 3.:
            raise ValueError('Only bounded public mode7 programming')
        return super().call(command, **metadata)

    def read_r(self, wl: int, bl: int, **metadata: object) -> float:
        return self.call(dict(mode_flag=7, wl=wl, bl=bl, vol=0., sign=0, pulse_width=0.), **metadata)[1]

    def program(self, wl: int, bl: int, goal: float, **metadata: object) -> dict[str, object]:
        scale = number(metadata.get('conductance_scale', 1.))
        original = self.span
        self.span = original*scale
        try:
            result = super().program(wl, bl, goal, **metadata)
            result['target_tolerance_s'] = .02*self.span
            result['conductance_scale'] = scale
            result['source_target_g_s'] = metadata.get('source_target_g_s')
            return result
        finally:
            self.span = original


def precheck_load(reader: ReadCollector, selected: tuple[Profile, ...], assigned: tuple[Assignment, ...],
                  goals: NDArray[np.float64], out: Path, load: int) -> None:
    """Fresh all32 background plus conservative actual/planned conductance guard."""
    reader.calls = 0
    reader.started = time.monotonic()
    actual: dict[tuple[int, int], float] = {}
    for wl in sorted({w for p in selected for w in (p.block.plus_wl, p.block.minus_wl)}):
        for bl in range(32):
            result = reader.call(dict(mode_flag=7, wl=wl, bl=bl, vol=0., sign=0, pulse_width=0.), role='prewrite_all32_current_guard', load=load)
            actual[wl,bl] = 1/result['resistance_ohm']
    worst = dict(actual)
    for p in selected:
        for a in (a for a in assigned if a.block == p.block.name):
            for leg, wl in enumerate((p.block.plus_wl,p.block.minus_wl)):
                target = float(goals[a.filter,a.tap,leg])*p.conductance_scale+p.conductance_offset
                worst[wl,p.block.bls[a.slot]] = max(actual[wl,p.block.bls[a.slot]], target)
    checks: list[dict[str, object]] = []
    for p in selected:
        for wl in (p.block.plus_wl,p.block.minus_wl):
            current = sum(max(.025+p.delta if bl in p.block.bls else .025,
                reader.cv.convert_dac_to_volt(12,5,reader.cv.convert_volt_to_dac(12,5,.025+p.delta if bl in p.block.bls else .025))) * worst[wl,bl] for bl in range(32))
            admitted = current <= .005 and current*330 < 3.3
            checks.append(dict(block=p.block.name, wl=wl, current_a=current, admitted=admitted))
    atomic_json(out/f'load_{load:03d}_prewrite_guard.json', dict(checks=checks, rule='max(fresh actualG, planned targetG) at each programmed cell; all32 background'))
    if not all(c['admitted'] for c in checks):
        raise ValueError('Prewrite planned all32 current/headroom guard')


def centered(leg: Leg, cv: Converters, method: Baseline, adc: bool = False) -> float:
    def mean(samples: tuple[dict[str, object], ...]) -> float:
        return statistics.mean(number(s['adc']) if adc else cv.convert_adc_to_volt(1, 14, 5, int(number(s['adc']))) for s in samples)
    # TypedDict samples are copied into plain JSON records at this boundary.
    before = mean(tuple(dict(s) for s in leg.before))
    signal = mean(tuple(dict(s) for s in leg.signal))
    after = mean(tuple(dict(s) for s in leg.after))
    return signal-before if method == 'before_only' else signal-(before+after)/2


def operators_from_pairs(pairs: list[Pair], selected: tuple[Profile, ...], cv: Converters,
                         method: Baseline) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    operators: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for profile in selected:
        b = profile.block
        lut = [0.]
        for mask, targets in enumerate(patterns(b), 1):
            basis = [p for p in pairs if p.block == b.name and p.targets == targets and p.phase == 'basis']
            validation = [p for p in pairs if p.block == b.name and p.targets == targets and p.phase == 'validation']
            if len(basis) != 3 or len(validation) != 5:
                raise ValueError('Incomplete fixed basis/independent validation')
            leg = basis[0].plus
            delta = leg.signal_decoded_v[targets[0]]-leg.baseline_decoded_v[targets[0]]
            lut.append(statistics.mean(centered(p.plus, cv, method)-centered(p.minus, cv, method) for p in basis)/(330*delta*profile.conductance_scale))
            pmean = statistics.mean(centered(p.plus, cv, method, True) for p in basis)
            mmean = statistics.mean(centered(p.minus, cv, method, True) for p in basis)
            held = [centered(p.plus, cv, method, True)-centered(p.minus, cv, method, True) for p in validation]
            residual = abs(statistics.mean(held)-(pmean-mmean))
            limit = max(5., .1*(abs(pmean)+abs(mmean)))
            diagnostics.append(dict(block=b.name, mask=mask, method=method,
                heldout_error_adc=residual, heldout_limit_adc=limit, signed_sd_adc=statistics.stdev(held),
                heldout_mean_pass=residual <= limit, signed_sd_pass=statistics.stdev(held) <= 5,
                fitted_from='three basis repeats only', validation_used_for_operator=False))
        operators.append(dict(block=b.name, lut_difference_g_s=lut))
    return operators, diagnostics


def acquire_exploratory(reader: ReadCollector, selected: tuple[Profile, ...], out: Path,
                        load: int) -> tuple[list[dict[str, object]], bool]:
    operators, strict_pass = acquire_load(reader, selected, out, load)
    atomic_json(out/f'load_{load:03d}_validation.json', [asdict(p) for p in reader.pairs if p.phase == 'validation'])
    comparisons: dict[str, object] = dict(strict_pass=strict_pass, exploratory=True, no_test_labels=True)
    for method in ('before_only', 'bracket_centered'):
        policy: Baseline = 'before_only' if method == 'before_only' else 'bracket_centered'
        alternate, checks = operators_from_pairs(reader.pairs, selected, reader.cv, policy)
        atomic_json(out/f'load_{load:03d}_{method}_operators.json', dict(operators=alternate, diagnostics=checks))
        comparisons[method] = checks
    atomic_json(out/f'load_{load:03d}_baseline_comparison.json', comparisons)
    return operators, strict_pass


def measured_lut(folder: Path, keys: NDArray[np.uint32], selected: tuple[Profile, ...],
                 span: float, cv: Converters, method: Baseline) -> NDArray[np.float32]:
    """Rebuild from immutable measured basis; strict false is recorded, not hidden."""
    manifest = object_dict(json.loads((folder/'manifest.json').read_text(encoding='utf-8')))
    if object_dict(manifest['protocol']).get('exploratory_measured_lut') is not True:
        raise ValueError('Explicit exploratory acquisition required')
    hashes = object_dict(manifest['result_sha256'])
    result = np.zeros((len(keys), 64), dtype=np.float32)
    plan = schedule(selected)
    for load in range(max(a.load for a in plan)+1):
        name = f'load_{load:03d}_frozen.json'
        if sha(folder/name) != hashes[name]:
            raise ValueError('Frozen evidence checksum mismatch')
        raw: object = json.loads((folder/name).read_text(encoding='utf-8'))
        if not isinstance(raw, list):
            raise ValueError('Frozen measured rows required')
        rows = [object_dict(r) for r in raw]
        operators: list[dict[str, object]] = []
        def from_raw(value: object) -> float:
            leg = object_dict(value)
            means: dict[str, float] = {}
            for phase in ('before', 'signal', 'after'):
                samples = leg[phase]
                if not isinstance(samples, list) or len(samples) != 4:
                    raise ValueError('Four measured samples required')
                means[phase] = statistics.mean(cv.convert_adc_to_volt(1, 14, 5, int(number(object_dict(s)['adc']))) for s in samples)
            return means['signal']-means['before'] if method == 'before_only' else means['signal']-(means['before']+means['after'])/2
        for profile in selected:
            values = [0.]
            for targets in patterns(profile.block):
                found = [r for r in rows if r['block'] == profile.block.name and r['targets'] == list(targets)]
                if len(found) != 3 or any(r['phase'] != 'basis' for r in found):
                    raise ValueError('Incomplete basis snapshot')
                leg = object_dict(found[0]['plus'])
                baseline, signal = leg['baseline_decoded_v'], leg['signal_decoded_v']
                if not isinstance(baseline, list) or not isinstance(signal, list):
                    raise ValueError('Native decoded voltages missing')
                delta = number(signal[targets[0]])-number(baseline[targets[0]])
                values.append(statistics.mean(from_raw(r['plus'])-from_raw(r['minus']) for r in found)/(330*delta*profile.conductance_scale))
            operators.append(dict(block=profile.block.name, lut_difference_g_s=values))
        compose(keys, tuple(a for a in plan if a.load == load), operators, span, result)
    return result
