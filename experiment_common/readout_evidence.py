"""Verified source identities for CPU adaptation; no board imports or I/O."""
from __future__ import annotations
import json
from pathlib import Path
from .durable import sha
from .selective_metrology import Source,object_dict


def terminal_manifest(base: Path,sid: str,source: Source) -> dict[str,object]:
    folder=base/sid
    receipt=object_dict(json.loads((base/'archives'/f'{sid}.json').read_text(encoding='utf-8')))
    expected=object_dict(object_dict(receipt['files'])['manifest.json'])['sha256']
    if receipt['session']!=sid or sha(folder/'manifest.json')!=expected:
        raise ValueError('Source terminal manifest differs from archive receipt')
    manifest=object_dict(json.loads((folder/'manifest.json').read_text(encoding='utf-8')))
    if manifest['session_id']!=sid or manifest['status']!='completed' or manifest['source']!=source:
        raise ValueError('Completed evidence with matching UUID and source required')
    return manifest
