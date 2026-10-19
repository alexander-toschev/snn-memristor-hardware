"""Pure full-image schedule and conservative impact budget; no I/O."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = "weights/cifar100_conv1_fixture.json"
FIXTURE_SHA = "8cb9ec0fe30242328527073bb076f8019295d73a748343f58fc2a8671bf7c154"
CHECKPOINT_SHA = "ee4fc4f58a3d010218116e9ad4c7ba808787959f9ed7687cd58cbb32d90f62fb"
RANGES = ((0, 32), (32, 64), (64, 75))
MAX_ACTIVE = 4
AMPLITUDE_V = .05
SPEC = {
    "schema": "full-case0-v1", "review_status": "requires_registered_protocol_and_source_review",
    "fixture_case": 0, "groups": 16, "stages": 48, "filters": 64,
    "height": 28, "width": 28, "timesteps": 12,
    "max_active_inputs": MAX_ACTIVE, "input_voltage_v": AMPLITUDE_V,
    "fresh_mode7_before_every_active_cell_vector": True,
    "readout": "eight sequential WL reads for every actual vector",
    "zero_policy": "eight before and eight after each spatial position; shared across T12",
    "gain_v_per_weight": {"hardware": .125, "simulator": .015},
    "programming": {
        "ramp_v": [.3,.6,.9,1.2,1.5,2.,2.5,3.],
        "max_reset_pulses_per_leg": 8, "max_set_pulses_one_signed_leg": 8,
        "max_nonzero_decisions_per_coefficient": 24,
        "max_nonzero_decisions": 115200,
        "fresh_mode7_before_each_nonzero": True,
        "sign0_limit_a": .04, "sign1_limit": "unchanged board.soft_cc",
        "hardware_pulse_duration_s": None,
        "status": "bounded feedback; approximate weights retained, not corrected per coefficient",
    },
    "calibration": {
        "basis_vectors_by_tile": [33,33,12], "fit_vectors": 64, "selection_vectors": 32,
        "heldout_vectors": 32, "final_basis_vectors_by_tile": [33,33,12], "repeats": 3,
        "reference": "independent fresh singleton sums; never learned weight targets",
        "status": "fresh fit/selection/heldout, independently replayed native provenance",
    },
    "raw_uncompressed_chunk_bytes": 64 * 1024**2,
    "raw_compressed_chunk_limit_bytes": 128 * 1024**2,
    "hardware_qualification_required": True, "native_simulator_qualification_required": True,
}


def fixture(repo=ROOT):
    data = (repo / FIXTURE).read_bytes()
    if hashlib.sha256(data).hexdigest() != FIXTURE_SHA:
        raise ValueError("Fixture SHA mismatch")
    case = json.loads(data)["cases"][0]
    nested = case["image_chw_uint8"]
    if len(nested) != 3 or any(len(channel) != 32 or any(len(row) != 32 for row in channel) for channel in nested):
        raise ValueError("Expected 3x32x32 CHW image")
    image = [v for channel in nested for row in channel for v in row]
    if len(image) != 3072 or any(type(v) is not int or not 0 <= v <= 255 for v in image):
        raise ValueError("Expected flattened uint8 CHW image")
    if hashlib.sha256(bytes(image)).hexdigest() != case["image_chw_uint8_sha256"]:
        raise ValueError("Image SHA mismatch")
    return {**case, "image_chw_uint8": image}


def masks(image, tile, y, x):
    if tile not in range(3) or y not in range(28) or x not in range(28):
        raise ValueError("Invalid spatial or tile index")
    lo, hi = RANGES[tile]
    times = []
    for tap in range(lo, hi):
        value = image[(tap//25)*1024+(y+(tap%25)//5)*32+x+tap%5]
        times.append(((255-value)*11)//255 if value >= 26 else -1)
    result = []
    for t in range(12):
        active = [bl for bl, time in enumerate(times) if time == t]
        result.append([active[a:a+MAX_ACTIVE] for a in range(0, len(active), MAX_ACTIVE)] or [[]])
    return result


def position_commands(image, stage, y, x):
    if type(stage) is not int or stage not in range(48):
        raise ValueError("Invalid stage")
    common = dict(stage=stage, y=y, x=x)
    for wl in range(8):
        yield dict(**common, role="zero_before", t=None, split=None, wl=wl, mask=[])
    for t, parts in enumerate(masks(image, stage % 3, y, x)):
        for split, mask in enumerate(parts):
            for wl in range(8):
                yield dict(**common, role="signal", t=t, split=split, wl=wl, mask=mask)
    for wl in range(8):
        yield dict(**common, role="zero_after", t=None, split=None, wl=wl, mask=[])


def bounds(image):
    signals = sum(len(parts)*8 for tile in range(3) for y in range(28) for x in range(28)
                  for parts in masks(image, tile, y, x)) * 16
    zeros = 48*784*16
    guard_reads = sum(len(mask)*8 for tile in range(3) for y in range(28) for x in range(28)
                      for parts in masks(image, tile, y, x) for mask in parts) * 16
    # Exact fixed schedule: basis/final basis and bracketed fit/selection/heldout.
    calibration = 16*sum(2*(n+1)*3*8+128*3*24 for n in (32,32,11))
    # Seeded masks are fixed before data; this is their exact fresh-R call count.
    calibration_guards = 466608
    # Per coefficient: <=32 RESET +16 SET guard/pulse calls and9*32 feedback calls.
    programming = 4800*(32+16+9*32)
    return dict(signal_mode10=signals, zero_mode10=zeros,
                inference_mode10=signals+zeros, calibration_mode10=calibration,
                calibration_fresh_guard_mode7=calibration_guards,
                inference_fresh_guard_mode7=guard_reads,
                programming_all_impacts_upper=programming,
                all_impacts_upper=signals+zeros+guard_reads+calibration+calibration_guards+programming,
                dense_tensor_shape=[12,64,28,28,3], dense_tensor_bytes=12*64*28*28*3*4,
                native_latency_extrapolation_is_not_eta=True,
                maximum_programming_pulses=115200, maximum_impacts=15000000)


if __name__ == "__main__":
    case = fixture()
    print(json.dumps(dict(protocol=SPEC, case_original_test_index=case["original_test_index"],
                         image_sha256=case["image_chw_uint8_sha256"], bounds=bounds(case["image_chw_uint8"])), indent=2))
