"""Behavioural metrics (accuracy and loss) and the timing definitions of the plan.

Timing definitions (PROJECT_PLAN.md, Section 5.2):

* ``T_mem``    first step with train acc >= 0.99
* ``T_grok``   first step with val acc >= 0.99 that *stays* >= 0.99 for ``patience`` steps
* ``T_95``     same with threshold 0.95
* ``T_forget`` first step at which the old task's val acc drops below 0.5

All of them share the same "sustained crossing" logic, implemented once in
``SustainedCrossing``. It is used online (to stop training) and offline (to
recompute timings from saved logs with other thresholds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .data import ModularTask

# ------------------------------------------------------------------------- logits


def mask_logits(logits: torch.Tensor, moduli: torch.Tensor) -> torch.Tensor:
    """Set logits of tokens >= n (not an element of Z_n) to -inf, per example.

    This keeps the cross-entropy comparable across moduli: a Z_59 example is a
    59-way classification even though the output layer has n_numbers units.
    """
    classes = torch.arange(logits.shape[-1], device=logits.device)
    invalid = classes[None, :] >= moduli[:, None]
    return logits.masked_fill(invalid, float("-inf"))


class TaskEvaluator:
    """Mean loss and accuracy of a model on subsets of one task, computed on the device.

    The subsets are the train split, the val split, and optional named parts of the
    val split (e.g. ``"agree"`` -> logged as ``val_acc_agree``). All tensors are cached
    on the device, and the result stays there: the trainer gathers the numbers of
    every evaluator and copies them to the CPU in one go, so an evaluation costs a
    single GPU -> CPU transfer of a few floats.
    """

    def __init__(self, task: ModularTask, device: torch.device, val_subsets: dict[str, np.ndarray] | None = None):
        self.task = task
        self.tokens = task.tokens.to(device)
        self.labels = task.labels.to(device)
        self.moduli = task.moduli.to(device)
        train = np.zeros(task.size, dtype=bool)
        train[task.train_idx] = True
        val = np.zeros(task.size, dtype=bool)
        val[task.val_idx] = True
        subsets = {"train": train, "val": val}
        for name, mask in (val_subsets or {}).items():
            subsets[f"val_{name}"] = val & mask
        self.subsets = list(subsets)
        # (n_subsets, n_examples) 0/1 weights: one matrix product gives every subset mean.
        self.weights = torch.tensor(np.stack(list(subsets.values())), dtype=torch.float32, device=device)

    @torch.no_grad()
    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        """Tensor (n_subsets, 2) on the device: (mean loss, accuracy) per subset (NaN if empty)."""
        logits = mask_logits(model(self.tokens), self.moduli)
        loss = F.cross_entropy(logits, self.labels, reduction="none")
        correct = (logits.argmax(-1) == self.labels).float()
        sums = self.weights @ torch.stack([loss, correct], dim=1)
        return sums / self.weights.sum(1, keepdim=True)

    @torch.no_grad()
    def hidden(self, model: torch.nn.Module) -> torch.Tensor:
        """MLP activations on every example, (n_examples, d_mlp), on the device."""
        return model(self.tokens, return_hidden=True)[1]

    def to_row(self, means: list[list[float]], prefix: str = "", with_loss: bool = True) -> dict:
        """Log-row entries from ``means`` (the output of ``__call__`` as a list):
        ``{prefix}train_loss``, ``{prefix}train_acc``, ``{prefix}val_loss``, ``{prefix}val_acc``,
        and ``{prefix}val_acc_{name}`` for each val subset."""
        row = {}
        for name, (loss, acc) in zip(self.subsets, means):
            if name in ("train", "val"):
                if with_loss:
                    row[f"{prefix}{name}_loss"] = loss
                row[f"{prefix}{name}_acc"] = acc
            else:
                row[f"{prefix}val_acc_{name[4:]}"] = acc
        return row


# ------------------------------------------------------------------ timing logic


@dataclass
class SustainedCrossing:
    """Detects the first step at which a metric crosses a threshold *and stays there*.

    ``update(step, value)`` must be called with increasing steps. The crossing is
    confirmed once the condition has held for ``patience`` steps; ``self.step`` is
    then the step at which it *started* to hold. If the condition breaks before
    confirmation, the candidate is discarded. Once confirmed, the result never changes.
    """

    threshold: float
    patience: int = 0
    above: bool = True  # True: value >= threshold;  False: value < threshold
    step: int | None = None
    _candidate: int | None = field(default=None, repr=False)

    def holds(self, value: float) -> bool:
        if value is None or np.isnan(value):
            return False
        return value >= self.threshold if self.above else value < self.threshold

    def update(self, step: int, value: float) -> bool:
        """Feed one observation; returns True if the crossing is confirmed."""
        if self.step is not None:
            return True
        if self.holds(value):
            if self._candidate is None:
                self._candidate = step
            if step - self._candidate >= self.patience:
                self.step = self._candidate
        else:
            self._candidate = None
        return self.step is not None

    @property
    def confirmed(self) -> bool:
        return self.step is not None


def first_sustained_crossing(
    steps: Iterable[int], values: Iterable[float], threshold: float, patience: int = 0, above: bool = True
) -> int | None:
    """Offline version of ``SustainedCrossing`` for a full logged curve."""
    tracker = SustainedCrossing(threshold, patience, above)
    for s, v in zip(steps, values):
        if tracker.update(int(s), float(v)):
            break
    return tracker.step
