"""Draft raw-to-tensor reconstruction; DOES NOT attest hardware readiness.

Independently derives CHW/TTFS masks, expected commands and native ADC conversion.
The missing write/calibration/safety auditor prevents classifier promotion.
"""
from array import array
import hashlib
import json
import math
from pathlib import Path
import sys


def expected_signals(image, tile, y, x):
    start, stop = ((0,32), (32,64), (64,75))[tile]
    times = []
    for coefficient in range(start, stop):
        channel, kernel = divmod(coefficient, 25)
        dy, dx = divmod(kernel, 5)
        pixel = image[channel*1024+(y+dy)*32+x+dx]
        times.append((255-pixel)*11//255 if pixel >= 26 else -1)
    for t in range(12):
        active = [i for i, timing in enumerate(times) if timing == t]
        for split in range(max(1, (len(active)+3)//4)):
            yield t, split, active[4*split:4*split+4]


class Replay:
    def __init__(self, records):
        self.records = iter(records)
        self.task_id = 0

    def read(self, stage, y, x, role, t, split, wl, mask):
        row = next(self.records)
        self.task_id += 1
        expected = dict(event="native_mode10", task_id=self.task_id, stage=stage,
                        y=y, x=x, role=role, t=t, split=split, wl=wl, mask=mask)
        if any(row.get(k) != value for k, value in expected.items()):
            raise ValueError("Raw order, coverage or identity mismatch")
        command = dict(mode_flag=10, vol=[.05 if bl in mask else 0. for bl in range(32)], wl=wl, id=0)
        if row.get("command_si") != command or row.get("status") != "ok":
            raise ValueError("Command mismatch or failed acquisition")
        result = row.get("native_result")
        if not isinstance(result, list) or len(result) != 3 or type(result[1]) is not int or result[1] != 0:
            raise ValueError("Invalid native response/echo")
        adc = result[2]
        if type(adc) is not int or not 0 <= adc <= 16382:
            raise ValueError("Missing/noninteger/saturated ADC")
        voltage = round(adc*5/16383, 5)
        if voltage > 4.:
            raise ValueError("ADC headroom exceeded")
        return voltage

    def position(self, image, stage, y, x, matrix, gain):
        if len(matrix) != 8 or any(len(row) != 8 for row in matrix):
            raise ValueError("Expected eight-row correction matrix")
        if any(not math.isfinite(v) for row in matrix for v in row) or any(sum(abs(v) for v in row) > 10 for row in matrix):
            raise ValueError("Nonfinite/excessive correction matrix")
        if gain not in (.125, .015):
            raise ValueError("Unknown fixed source gain")
        before = [self.read(stage,y,x,"zero_before",None,None,wl,[]) for wl in range(8)]
        signals = []
        for t, split, mask in expected_signals(image, stage % 3, y, x):
            values = [self.read(stage,y,x,"signal",t,split,wl,mask) for wl in range(8)]
            signals.append((t, values))
        after = [self.read(stage,y,x,"zero_after",None,None,wl,[]) for wl in range(8)]
        zero = [(a+b)/2 for a,b in zip(before, after)]
        decoded = [[0.]*4 for _ in range(12)]
        for t, values in signals:
            centered = [v-z for v,z in zip(values,zero)]
            corrected = [sum(c*v for c,v in zip(row,centered)) for row in matrix]
            for pair in range(4):
                decoded[t][pair] += (corrected[2*pair]-corrected[2*pair+1])/gain
        if any(not math.isfinite(v) for row in decoded for v in row):
            raise ValueError("Nonfinite decoded contribution")
        return decoded


def reconstruct(records, image, image_sha256, source):
    if source not in ("hardware", "simulator", "synthetic_test"):
        raise ValueError("Explicit source required")
    if len(image) != 3072 or hashlib.sha256(bytes(image)).hexdigest() != image_sha256:
        raise ValueError("Image identity mismatch")
    expected_gain = .015 if source == "simulator" else .125
    replay = Replay(records)
    size = 12*64*28*28*3
    values, coverage = array("f", [0.])*size, bytearray(size)
    stages = []
    try:
        for stage in range(48):
            freeze = next(replay.records)
            if (freeze.get("event") != "stage_frozen" or freeze.get("stage") != stage or
                    freeze.get("image_sha256") != image_sha256 or freeze.get("source") != source or
                    freeze.get("gain_v_per_weight") != expected_gain or
                    freeze.get("calibration_uses_cifar_or_target_weights") is not False):
                raise ValueError("Missing/reordered stage freeze or changed source/gain")
            # Fields and hashes preserve declared evidence; they cannot prove it.
            for name in ("programming_evidence_sha256", "calibration_evidence_sha256"):
                value = freeze.get(name)
                if not isinstance(value, str) or len(value) != 64 or any(v not in "0123456789abcdef" for v in value):
                    raise ValueError("Missing evidence binding")
            matrix = freeze["matrix_C8"]
            stages.append(freeze)
            for y in range(28):
                for x in range(28):
                    decoded = replay.position(image,stage,y,x,matrix,expected_gain)
                    for t in range(12):
                        for pair in range(4):
                            f = (stage//3)*4+pair
                            index = (((t*64+f)*28+y)*28+x)*3+stage%3
                            if coverage[index]:
                                raise ValueError("Duplicate physical contribution")
                            values[index] = decoded[t][pair]
                            coverage[index] = 1
    except StopIteration as exc:
        raise ValueError("Incomplete raw coverage; no CPU/zero fill") from exc
    if next(replay.records, None) is not None:
        raise ValueError("Unexpected trailing raw data")
    if not all(coverage) or not all(math.isfinite(v) for v in values):
        raise ValueError("Incomplete/nonfinite dense tensor")
    if sys.byteorder != "little":
        values.byteswap()
    payload = values.tobytes()
    report = dict(schema="draft-full-case0-reconstruction-v1", source=source,
                  shape=[12,64,28,28,3], native_mode10_records=replay.task_id,
                  payload_sha256=hashlib.sha256(payload).hexdigest(), payload_bytes=len(payload),
                  coverage_complete=True, hardware_provenance_verified=False,
                  classifier_import_enabled=False, stage_freezes=stages,
                  limitations=["Programming/current guards and physical route have not been audited.",
                               "Frozen matrix fitting, heldout qualification and drift have not been audited.",
                               "Shared-zero policy requires fresh dense protocol evidence.",
                               "Recorded native request/response data are structurally checked, not authenticated by declared SHA alone."])
    return payload, report
