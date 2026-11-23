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


@torch.no_grad()
def per_example(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    moduli: torch.Tensor,
    return_hidden: bool = False,
):
    """Loss and correctness for every example (all already on the model's device)."""
    if return_hidden:
        logits, hidden = model(tokens, return_hidden=True)
    else:
        logits, hidden = model(tokens), None
    logits = mask_logits(logits, moduli)
    loss = F.cross_entropy(logits, labels, reduction="none")
    correct = (logits.argmax(-1) == labels).float()
    return loss.cpu().numpy(), correct.cpu().numpy(), (None if hidden is None else hidden.cpu().numpy())


class TaskEvaluator:
    """Caches a task's tensors on the device and evaluates a model on it."""

    def __init__(self, task: ModularTask, device: torch.device):
        self.task = task
        self.tokens = task.tokens.to(device)
        self.labels = task.labels.to(device)
        self.moduli = task.moduli.to(device)

    def __call__(self, model: torch.nn.Module, return_hidden: bool = False):
        return per_example(model, self.tokens, self.labels, self.moduli, return_hidden)


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean over the masked entries, NaN if the mask is empty."""
    return float(values[mask].mean()) if mask.any() else float("nan")


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
