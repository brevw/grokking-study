"""Configuration objects for every experiment.

Everything that defines a run lives in a small set of dataclasses so that

* a full experiment grid can be described by a single JSON file (see ``configs/``),
* every run directory stores the exact config it was produced with, and
* individual fields can be overridden from the command line
  (``--set train_transfer.max_steps=2000``).

The hierarchy is::

    ExperimentConfig
    ├── model:           ModelConfig    (architecture, identical for every run)
    ├── task:            TaskConfig     (operation + train fraction)
    ├── train_source:    TrainConfig    (phase A: learn Z_p)
    └── train_transfer:  TrainConfig    (phase B: continue on Z_q)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# Number of "=" tokens: one per task the network learns ("=A" for the first task,
# "=B" for the second). They come right after the number tokens.
N_EQ_TOKENS = 2

# --------------------------------------------------------------------------------------
# Leaf configs
# --------------------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Architecture of the transformer (Nanda et al. 2023 style, no LayerNorm)."""

    # Number tokens are 0 .. n_numbers-1. It must be larger than every modulus used in
    # the experiment so that *the same architecture* can represent all groups.
    n_numbers: int = 128
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    n_layers: int = 1
    # Input is always [a, b, "="], i.e. 3 positions.
    n_ctx: int = 3

    @property
    def vocab_size(self) -> int:
        """Numbers 0..n_numbers-1, then the "=" tokens "=A", "=B" (see data.py)."""
        return self.n_numbers + N_EQ_TOKENS


@dataclass
class TaskConfig:
    """What the network has to compute."""

    op: str = "add"  # one of grokking.data.OPS ("add", "sub", "mul")
    train_frac: float = 0.3  # fraction of the n^2 pairs used for training


@dataclass
class TrainConfig:
    """Optimisation, logging and stopping rules of one training phase."""

    # --- optimiser (AdamW, full batch) ---
    lr: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    warmup_steps: int = 10

    # --- budget and logging ---
    max_steps: int = 30_000
    eval_every: int = 50  # behavioural metrics (cheap)
    repr_every: int = 500  # representation metrics (Fourier, CKA, ...)
    save_checkpoints: bool = False  # also dump weights every repr_every steps

    # --- grokking detection ---
    # T_grok = first step with val acc >= grok_threshold that *stays* there for
    # grok_patience steps (avoids counting short spikes).
    grok_threshold: float = 0.99
    grok_patience: int = 1000
    # Stop this many steps after T_grok (None = always use the full budget).
    # Phase A uses it as "cleanup" time; phase B uses it to save compute.
    stop_after_grok: int | None = 2000

    # --- misc ---
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    # torch.compile the training forward pass: True | False | "auto".
    # "auto" = only on CUDA (NVIDIA GPUs). It is faster there, and it fails on some laptops (e.g. macOS CPU).
    compile: bool | str = "auto"
    print_every: int = 0  # print progress every k steps (0 = silent)


# --------------------------------------------------------------------------------------
# Experiment config
# --------------------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """A full grid: sources × targets × conditions × seeds."""

    name: str = "default"
    out_root: str = "results"

    sources: list[int] = field(default_factory=lambda: [97])
    # For each source modulus p: the list of target moduli q of phase B.
    targets: dict[int, list[int]] = field(default_factory=lambda: {97: [59, 89, 113, 91, 96, 121, 97]})
    conditions: list[str] = field(
        default_factory=lambda: ["scratch", "warm_grokked", "warm_memorized", "warm_permuted"]
    )
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    train_source: TrainConfig = field(default_factory=lambda: TrainConfig(max_steps=30_000, stop_after_grok=2000))
    train_transfer: TrainConfig = field(default_factory=lambda: TrainConfig(max_steps=50_000, stop_after_grok=1000))

    # Parameters of the shrink-and-perturb condition (Ash & Adams, 2020):
    # theta <- shrink * theta + perturb * theta_fresh_init
    shrink: float = 0.5
    perturb: float = 0.1

    # ------------------------------------------------------------------ helpers

    @property
    def exp_dir(self) -> Path:
        return Path(self.out_root) / self.name

    def all_moduli(self) -> list[int]:
        mods = set(self.sources)
        for qs in self.targets.values():
            mods.update(qs)
        return sorted(mods)

    def validate(self) -> None:
        """Fail early (before hours of compute) on inconsistent configs."""
        from .conditions import CONDITIONS  # local import to avoid a cycle
        from .data import OPS

        if self.task.op not in OPS:
            raise ValueError(f"Unknown op {self.task.op!r}; choose from {sorted(OPS)}")
        for n in self.all_moduli():
            if not 2 <= n < self.model.n_numbers:
                raise ValueError(f"Modulus {n} must satisfy 2 <= n < model.n_numbers={self.model.n_numbers}")
        for p in self.sources:
            if p not in self.targets:
                raise ValueError(f"Source {p} has no entry in 'targets'")
        unknown = [c for c in self.conditions if c not in CONDITIONS]
        if unknown:
            raise ValueError(f"Unknown conditions {unknown}; choose from {sorted(CONDITIONS)}")
        if self.model.d_model % self.model.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 < self.task.train_frac < 1:
            raise ValueError("train_frac must be in (0, 1)")
        for name in ("train_source", "train_transfer"):
            tc: TrainConfig = getattr(self, name)
            if tc.repr_every % tc.eval_every:
                raise ValueError(f"{name}.repr_every must be a multiple of {name}.eval_every")
            if tc.compile not in (True, False, "auto"):
                raise ValueError(f'{name}.compile must be true, false or "auto"')

    # ------------------------------------------------------------- (de)serialise

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["targets"] = {str(k): v for k, v in self.targets.items()}  # JSON keys are strings
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        cfg = cls()
        _update_dataclass(cfg, d)
        cfg.targets = {int(k): [int(q) for q in v] for k, v in cfg.targets.items()}
        return cfg

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "ExperimentConfig":
        """Load a JSON config and apply ``key.sub=value`` overrides."""
        with open(path) as f:
            cfg = cls.from_dict(json.load(f))
        for ov in overrides or []:
            apply_override(cfg, ov)
        cfg.targets = {int(k): [int(q) for q in v] for k, v in cfg.targets.items()}
        cfg.validate()
        return cfg

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------


def _update_dataclass(obj: Any, values: dict[str, Any]) -> None:
    """Recursively copy ``values`` into the dataclass ``obj`` (unknown keys are errors)."""
    known = {f.name for f in fields(obj)}
    for key, val in values.items():
        if key not in known:
            raise KeyError(f"Unknown config key {key!r} for {type(obj).__name__}; known: {sorted(known)}")
        cur = getattr(obj, key)
        if is_dataclass(cur) and isinstance(val, dict):
            _update_dataclass(cur, val)
        else:
            setattr(obj, key, val)


def apply_override(cfg: Any, override: str) -> None:
    """Apply one ``a.b.c=value`` override. ``value`` is parsed as JSON when possible.

    Examples: ``train_transfer.max_steps=2000``, ``seeds=[0,1]``, ``task.op="mul"``.
    """
    if "=" not in override:
        raise ValueError(f"Override must look like key=value, got {override!r}")
    key, raw = override.split("=", 1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw  # plain string
    *parents, leaf = key.strip().split(".")
    obj = cfg
    for p in parents:
        obj = getattr(obj, p)
    if not hasattr(obj, leaf):
        raise KeyError(f"Unknown config key {key!r}")
    setattr(obj, leaf, value)
