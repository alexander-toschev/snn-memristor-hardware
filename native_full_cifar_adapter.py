"""Structural adapter after native linkage; never grants hardware trust.

Public importer remains disabled. Full-source approval and a collector-specific
audit must be added by review, not through an input JSON flag or CLI override.
"""
import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath

import full_cifar_plan as p

# Intentionally empty: current reviewed sessions cover only4filters/oneposition.
APPROVED_FULL_HARDWARE_AUDITORS = frozenset()


def require_enabled_hardware_source(evidence):
    """No user-supplied `passed`/hardware/coverage flag can open this boundary."""
    if not APPROVED_FULL_HARDWARE_AUDITORS:
        raise ValueError("No reviewed full-image native collector/export auditor is registered")
    raise ValueError("Full source registration requires implemented raw-audit invocation, not self declarations")


def verify_native_block_scope(plan,micro_id,actual_slots,actual_native_keys):
    """Exact coverage of keys actually parsed from native request/result pairs.

    Each key is (case_index,repeat,t,output_y,output_x). No declared count or
    boolean mask is accepted in place of enumerated native identities.
    """
    routes=p.validate(plan)
    if micro_id not in routes:raise ValueError("Unknown planned microtile")
    expected=routes[micro_id]
    taps=[s["tap_index"] for s in actual_slots]
    if len(taps)!=len(set(taps)) or sorted(taps)!=sorted(expected["taps_chw"]):
        raise ValueError("Native route is not the exact planned microtile partition")
    limits=(len(plan["cases"]),plan["repeats"],12,28,28)
    total=1
    for n in limits:total*=n
    coverage=bytearray(total);observed=0
    for key in actual_native_keys:
        if len(key)!=5 or any(type(v) is not int or not 0<=v<n for v,n in zip(key,limits)):
            raise ValueError("Native key outside full-image plan")
        index=0
        for v,n in zip(key,limits):index=index*n+v
        if coverage[index]:raise ValueError("Duplicate native full-image identity")
        coverage[index]=1;observed+=1
    if observed!=total:raise ValueError(f"Native full-image coverage incomplete: {observed}/{total}")
    return dict(micro_id=micro_id,filter_id=expected["filter_id"],logical_tile=expected["logical_tile"],
                taps_chw=taps,native_key_count=observed,structural_scope_complete=True,
                hardware_trust_granted=False)


def verify_all_blocks(plan,block_scopes):
    routes=p.validate(plan)
    scopes=list(block_scopes)
    if len(scopes)!=len(routes) or {s["micro_id"] for s in scopes}!=set(routes):
        raise ValueError("Missing/duplicate physical microtile; full64filters required")
    for scope in scopes:
        expected=routes[scope["micro_id"]]
        if (scope.get("structural_scope_complete") is not True or scope.get("hardware_trust_granted") is not False or
            scope.get("filter_id")!=expected["filter_id"] or scope.get("logical_tile")!=expected["logical_tile"] or
            sorted(scope.get("taps_chw",[]))!=sorted(expected["taps_chw"]) or
            scope.get("native_key_count")!=len(plan["cases"])*plan["repeats"]*12*784):
            raise ValueError("Inconsistent block scope")
    return dict(structural_scope_complete=True,hardware_import_enabled=False,
                full_native_acquisition_attested=False,
                reason="Structural results must be recreated from source-bound raw by a reviewed full-collector auditor")


def normalized_record(plan,micro_id,linked,origin_session,origin_folder,raw_manifest_sha,calibration_sha,ordinal):
    """Translate a NativeLink-derived row, not a caller's predicted contribution.

    The independent audit must re-run NativeLink and compare this translation.
    This helper cannot by itself establish physical provenance.
    """
    route=p.validate(plan)[micro_id]
    for value in (origin_session,origin_folder):
        if (PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive or PureWindowsPath(value).root
            or ".." in PurePosixPath(value).parts or ".." in PureWindowsPath(value).parts):
            raise ValueError("Origin must remain repo-relative")
    ci=linked["case"]
    if linked["image_sha256"]!=plan["cases"][ci]["image_sha256"]:raise ValueError("Linked image differs from planned image")
    allowed=set(route["taps_chw"])
    if not set(linked["active_taps"])<=allowed:raise ValueError("Linked term outside planned microtile")
    if not linked["terms"]:raise ValueError("No native measurement, including no empty-zero measurement")
    terms=[]
    for term in linked["terms"]:
        if term["signal_tap"] is not None and term["signal_tap"] not in allowed:raise ValueError("Signal from another tile")
        terms.append(dict(positive_adc=term["positive_adc"],negative_adc=term["negative_adc"],
                          raw_task_refs=[f"{origin_session}/{origin_folder}#task={i}" for i in term["raw_task_ids"]]))
    return dict(event="measured_partial",ordinal=ordinal,micro_id=micro_id,
                raw_manifest_sha256=raw_manifest_sha,calibration_sha256=calibration_sha,
                case_id=plan["cases"][ci]["case_id"],image_sha256=linked["image_sha256"],
                repeat=linked["repeat"],t=linked["t"],y=linked["output_y"],x=linked["output_x"],terms=terms)


def compare_normalized(actual,expected):
    """Exact ADC/reference/metadata equality after independent raw reconstruction."""
    if actual!=expected:raise ValueError("Normalized record differs from source-bound native reconstruction")
    return hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
