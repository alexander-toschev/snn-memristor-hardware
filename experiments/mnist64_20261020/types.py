"""Structural types for the existing MNIST64 event and public-I/O contracts.

These declarations do not validate or transform measurements. Runtime checks
remain in Collector.call and inherited public-MemriBoard read/vector methods.
"""
from collections.abc import Callable, Mapping
from typing import Literal, NotRequired, Protocol, TypedDict

Source = Literal['hardware', 'simulator']
Scope = Literal['train', 'accuracy']
Progress = Callable[[str], None]
class Emitter(Protocol):
    def __call__(self, event: str, **fields: object) -> None: ...

class Converters(Protocol):
    def convert_volt_to_dac(self, bit: int, reference: float, voltage: float) -> int: ...
    def convert_dac_to_volt(self, bit: int, reference: float, dac: int) -> float: ...
    def convert_adc_to_volt(self, mode: int, bit: int, reference: float, adc: int) -> float: ...

class RawWriter(Protocol):
    def write(self, value: Mapping[str, object]) -> None: ...
    def checkpoint(self, **metadata: object) -> None: ...
    def finish(self) -> None: ...

class NativeCommand(TypedDict):
    mode_flag: int
    wl: int
    vol: float | list[float]
    bl: NotRequired[int]
    sign: NotRequired[int]
    pulse_width: NotRequired[float]
    id: NotRequired[int]

class NativeReading(TypedDict):
    r: float
    adc: int
    task_id: int

class VectorReading(TypedDict):
    task_id: int
    adc: int
    voltage_v: float
    guards: list[NativeReading]

class Signal(VectorReading):
    wl: int
    neuron: int
    tile: int
    t: int
    micro: int
    mask: list[int]

class PooledBlock(TypedDict):
    feature: int
    sum_uint8: int
    area: int
    spike_t: int | None

class InputImage(TypedDict):
    train_index: int
    dataset_split: str
    dataset_index: int
    image_sha256: str
    pixels_uint8: list[int]
    blocks: list[PooledBlock]
    schedule: list[list[int]]
    schedule_sha256: str

class Fixture(TypedDict):
    schema: str
    scope: Scope
    groups: dict[str, list[int]]
    labels_included: bool
    source_images: dict[str, str]
    images: list[InputImage]

class Limits(TypedDict):
    image_impacts: int
    maximum_impacts: int
    maximum_pulses: int
    noeligible_maximum_impacts: NotRequired[int]

class Parent(TypedDict):
    session_id: str
    manifest_sha256: str
    observations_sha256: str
    gain: list[float]
    final_array_sha256: str | None

class Choice(TypedDict):
    feature: int
    eligibility: float
    sign: int

class LifStep(TypedDict):
    t: int
    pre: list[int]
    voltage_v: float
    gain: float
    u_before: float
    u_unreset: float
    u_after: float
    post: int
    pre_trace: list[float]
    post_trace: float
    eligibility_increment: list[float]
    eligibility: list[float]

class NeuronOutcome(TypedDict):
    neuron: int
    posts: list[int]
    eligibility: list[float]
    choice: Choice | None

class ImageOutcome(TypedDict):
    phase: str
    ordinal: int
    train_index: int
    neurons: list[NeuronOutcome]
    zero_drift_v: list[float]

class ImageMeasurement(TypedDict):
    phase: str
    ordinal: int
    train_index: int
    schedule: list[list[int]]
    voltages: list[list[float]]
    zero_before: dict[int, VectorReading]
    zero_after: dict[int, VectorReading]
    signals: list[Signal]
    zero_drift_v: list[float]

class Checkpoint(TypedDict):
    phase: str
    mean_inverse_r: list[list[float]]

class PulseMetadata(TypedDict, total=False):
    preparation_step: int
    image_ordinal: int
    train_index: int
    neuron: int
    feature: int
    eligibility: float

class PulseDecision(TypedDict):
    wl: int
    bl: int
    sign: int
    voltage_v: float
    phase: str
    decision: int
    guard: NativeReading
    admitted: bool
    pulse: NotRequired[NativeReading]
    post: NotRequired[NativeReading]
    delta_g_s: NotRequired[float]
    preparation_step: NotRequired[int]
    image_ordinal: NotRequired[int]
    train_index: NotRequired[int]
    neuron: NotRequired[int]
    feature: NotRequired[int]
    eligibility: NotRequired[float]

class GainNeuron(TypedDict):
    neuron: int
    post_counts: list[int]
    total_posts: int
    active_images: int
    empty_input_posts: int
    eligible: bool

class GainCandidate(TypedDict):
    gain: float
    per_neuron: list[GainNeuron]

class Calibration(TypedDict):
    gains: list[float] | None
    candidates: list[GainCandidate]
    selection: str
    labels_used: bool

class Summary(TypedDict):
    status: str
    scope: Scope
    impacts: int
    pulses: int
    decisions: int
    measured_images: int
    gain: list[float] | None
    calibration: Calibration | None
    outcomes: list[ImageOutcome]
    updates: list[PulseDecision]
    checkpoints: list[Checkpoint]
    image_cell_participation: list[list[int]]
    input_event_counts: dict[str, list[int]]
    separate_control_bank: bool
    classification_accuracy_measured: bool

class ProtocolSpec(TypedDict):
    schema: str
    physical_shape: list[int]
    logical_shape: list[int]
    pooling: str
    timesteps: int
    amplitude_v: float
    maximum_mask: int
    mapping: str
    tile_sum_before_lif: bool
    separate_control_bank: bool
    lif_leak: float
    lif_threshold: float
    eligibility_features: int
    trace_decay: float
    depression: float
    eligibility_threshold: float
    updates: str
    gain_grid: list[float]
    gain_fit: str
    gain_selection: str
    preparation_signs: list[int]
    reset_v: float
    set_v: float
    reset_current_a: float
    set_current_a: float
    simulator_pulse_width_s: float
    hardware_pulse_width_s: float
    hardware_physical_duration_s: None
    train_upper_impacts: int
    train_max_pulses: int
    accuracy_upper_impacts: int
    accuracy_max_pulses: int
    train_checkpoints: list[str]
    accuracy_checkpoints: list[str]
    retention_seconds: float
    readout: str
    accuracy: str
    labels_in_acquisition: bool
