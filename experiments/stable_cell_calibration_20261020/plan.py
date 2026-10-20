"""Finite fresh-map schedule. No device imports or CIFAR data."""
import hashlib
import itertools
import json
import random

CELLS = [(wl, bl) for wl in range(8) for bl in range(32)]
LEVELS = [.3, .6, .9, 1.2, 1.5, 2., 2.5, 3.]
SPEC = dict(
    schema='fresh-stable-cell-calibration-v1', seed=20261020,
    scientific_design_sha256='f404bab23f1d9be6d4019dd89110475bd27826e3dd9ca73bc3cad248637b6e3f',
    cells=[list(c) for c in CELLS], amplitude_v=.05, maximum_active_bl=4,
    cycles=2, states=['R', 'M', 'H'], repeats=10, delays_s=[0, 10, 60],
    reset_levels_v=LEVELS, set_levels_v=LEVELS,
    phase_a_checkpoint_count=26*256, phase_a_maximum_impacts=282624,
    phase_b_maximum_impacts=106496, phase_c_maximum_impacts=22416,
    maximum_impacts=411536, maximum_pulses=20480,
    phase_a_maximum_pulses=8192, phase_b_maximum_pulses=12288,
    explicit_wait_limit_s=600., soft_deadline_s=3600.,
    simulator_pulse_width_s=.000125, hardware_submitted_pulse_width_s=0.,
    physical_pulse_duration_s=None, read_current_limit_a=.005,
    output_prediction_limit_v=3.3, output_measured_limit_v=4.,
    write_current_limits_a={'0': .04, '1': .002},
    span_min_adc=10., span_snr=5., span_se_margin=3.,
    sd_floor_adc=2., sd_span_fraction=.05,
    drift_floor_adc=3., drift_span_fraction=.05,
    cycle_floor_adc=4., cycle_span_fraction=.10,
    zero_shift_floor_adc=2., zero_shift_se_margin=3.,
    pair_sign_min_adc=5., pair_sign_snr=5.,
    matrix_ridge_v2=.0001, matrix_gain_limit=10., matrix_condition_limit=1e6,
    rank_margin=16., heldout_floor_adc=4., heldout_fraction=.10,
    raw_maximum_bytes=1024**3, raw_chunk_bytes=8*1024**2,
    initial_free_bytes=4*1024**3, reserve_bytes=1024**3,
    private_runtime_status='private_runtime_not_independently_reverified',
    cifar_labels_used=False, cpu_replacement_used=False,
    deployment='Pi finite A/B/C with lightweight ranking and C fit; PC readout and independent audit',
    phase_c_collection='Fit/selection then matrix freeze before heldout acquisition',
    phase_b_failure='Mandatory signed/missing fail stops; quality fail diagnostic C only; no rerank',
    fallback='If no quality rectangle: one pair, best up to8 common hardeligible BL; maxK,minworstscore,lex',
    hard_eligibility='complete valid states, both admitted polarities percycle, lower span>=max(10ADC,5SD)',
    quality_failures='noise,drift,zero_shift,crosscycle,middle_nonmonotonic; thresholds unchanged',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def recipe(state, incremental=False):
    if state == 'R':
        return [(0, v) for v in LEVELS]
    if state not in ('M', 'H'):
        raise ValueError('Unknown recipe')
    if incremental:
        return [(1, v) for v in (LEVELS[:4] if state == 'M' else LEVELS[4:])]
    return recipe('R') + [(1, v) for v in LEVELS[:4 if state == 'M' else 8]]


def pair_states(configuration, bl_index):
    if configuration == 'zero':
        return 'M', 'M'
    if configuration == 'plus' or configuration == 'checkerboard' and bl_index % 2 == 0:
        return 'H', 'R'
    if configuration in ('minus', 'checkerboard'):
        return 'R', 'H'
    raise ValueError('Unknown pair configuration')


def masks(bls):
    pool = [list(c) for k in range(1, min(4, len(bls))+1)
            for c in itertools.combinations(sorted(bls), k)]
    random.Random(SPEC['seed']).shuffle(pool)
    selected = pool[:96]
    n = len(selected)
    a, b = n//2, n//4
    return dict(fit=selected[:a], selection=selected[a:a+b], heldout=selected[a+b:])
