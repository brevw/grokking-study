"""Small helpers shared by all modules: seeding, devices, file I/O."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def stable_seed(*parts: Any) -> int:
    """Deterministic 31-bit seed derived from arbitrary parts.

    ``stable_seed("split", 97, 0)`` is the same on every machine and every Python run
    (unlike ``hash()``), so data splits are reproducible and *shared between
    conditions* that use the same parts.
    """
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` -> CUDA if available, otherwise CPU.

    MPS (Apple GPU) is *not* picked automatically: for these tiny models the
    per-kernel launch overhead makes it slower than the CPU. Ask for it explicitly
    with ``device="mps"`` if you want to try.
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ----------------------------------------------------------------------------- I/O


def _to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars / NaN so ``json.dump`` never fails."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(_to_jsonable(data), f, indent=2)
    tmp.replace(path)  # atomic: a crash never leaves a half-written summary


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a list of dicts; columns = union of keys in first-seen order."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
