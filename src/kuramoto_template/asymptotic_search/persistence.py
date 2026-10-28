"""Portable JSON persistence and structural shortlist helpers for joint research."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np

from .models import candidate_from_data


def json_safe(value):
    """Convert NumPy/Path/nonfinite values into standard JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _signature(c):
    digest = hashlib.sha256()
    for name in ("x","s","groups","p0","allowed_mask"):
        digest.update(getattr(c,name).tobytes())
    digest.update(str(c.backbone_cycle).encode())
    return digest.hexdigest()

def _code_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parents[1].rglob("*.py")):
        digest.update(path.relative_to(Path(__file__).resolve().parents[1]).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

def _write(path, data):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(json_safe(data),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)

def _shortlist(records, cap):
    # Round-robin structural strata, density and dipole headroom within strata.
    strata = {}
    seen = set()
    for index,item in enumerate(records):
        if item["metrics"]["status"] != "stage1_screened":
            continue
        c = candidate_from_data(item["candidate"])
        signature = _signature(c)
        if signature in seen:
            continue
        seen.add(signature)
        strata.setdefault((c.m,c.mask_hash),[]).append(index)
    for group in strata.values():
        group.sort(key=lambda i:(records[i]["metrics"]["mu0"],min(records[i]["metrics"]["dipole_slacks"])),reverse=True)
    # Interleave masks WITHIN each size first, then round-robin sizes. A single
    # size with many masks must not occupy the entire bounded shortlist.
    per_size = {}
    for size in sorted({key[0] for key in strata}):
        groups = [strata[key] for key in sorted(strata) if key[0]==size]
        queue = []
        while any(groups):
            for group in groups:
                if group:
                    queue.append(group.pop(0))
        per_size[size] = queue
    selected = []
    while len(selected)<cap and any(per_size.values()):
        for size in sorted(per_size):
            if per_size[size] and len(selected)<cap:
                selected.append(per_size[size].pop(0))
    return selected
