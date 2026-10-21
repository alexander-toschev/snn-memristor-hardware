"""Совместимый путь shared writer; CIFAR fault hook остаётся здесь."""
import os
from experiment_common.durable import Writer, atomic_json, sha


def simulated_process_loss(writer,source,stage,positions,impacts):
    """Explicit qualification fault: SIM only; never a hardware recovery operation."""
    if source!='simulator' or stage!=1 or positions!=17:
        raise ValueError('Process-loss injection is fixed to SIM stage1 after17positions')
    atomic_json(writer.directory.parent/'fault_injection.json',dict(schema='cifar-sim-process-loss-v1',
        source=source,stage=stage,software_observed_positions=positions,software_observed_impacts=impacts,
        raw_index_sha256=sha(writer.directory/'index.json'),exit_code=86,
        interpretation='Marker is a software observation; only the closed-index raw is durable evidence. No close or final manifest is claimed.'))
    os._exit(86)
