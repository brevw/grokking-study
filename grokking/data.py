"""Datasets for modular arithmetic, with symbols kept separate from group elements.

Vocabulary layout (fixed, independent of the moduli; ``V = ModelConfig.n_numbers``)::

    token 0 .. V-1     number symbols ("3" is token 3, so token id = number)
    token V            "=p"  the "=" of the first task  (phase A)
    token V + 1        "=q"  the "=" of the second task (phase B)

The "=" says *which task* a question belongs to, not which modulus. Example with
Z_5 then Z_7: ``[3, 4, =p] -> 2`` and ``[3, 4, =q] -> 0``. The same pair (3, 4) has a
different answer in each task, and the "=" tells the network which one is asked.
"=p" and "=q" are named after the roles p (old group) and q (new group), not after the
moduli: they are always tokens V and V + 1, whatever p and q are. Because phase B always
uses "=q", even the same-task control (Z_97 -> Z_97, i.e. q = p) shows the
network a symbol it has never seen.

An example of a task is the sequence ``[σ(a), σ(b), =]`` with target ``σ(c)``,
where ``c = a ∘ b mod n`` and ``σ : Z_n -> {0..n-1}`` is the *symbol map*:

* identity map (default): element i is written with token i;
* permuted map (control): element i is written with token π(i) for a random π.

The group element and the symbol are deliberately different objects. Keeping them
apart is what lets us ask whether transfer comes from token identity or from
structure (research question RQ3 in PROJECT_PLAN.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .config import N_EQ_TOKENS

# Binary operations on Z_n (vectorised over numpy arrays).
OPS: dict[str, Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = {
    "add": lambda a, b, n: (a + b) % n,
    "sub": lambda a, b, n: (a - b) % n,
    "mul": lambda a, b, n: (a * b) % n,
}


EQ_NAMES = ("=p", "=q")  # display names of the "=" tokens, by slot


def eq_token(slot: int, n_numbers: int) -> int:
    """Token id of the "=" of task slot ``slot`` (0 -> "=p", 1 -> "=q")."""
    if not 0 <= slot < N_EQ_TOKENS:
        raise ValueError(f"eq slot must be in 0..{N_EQ_TOKENS - 1}, got {slot}")
    return n_numbers + slot


@dataclass
class ModularTask:
    """All n² examples of ``a ∘ b mod n`` plus a fixed train/val split.

    Attributes
    ----------
    n, op:        modulus and operation name.
    eq_slot:      which "=" the task uses (0 = "=p", 1 = "=q").
    symbols:      ``symbols[e]`` = token used for group element e (shape (n,)).
    a, b, c:      group elements of every example (numpy, shape (n²,)).
    tokens:       model input, LongTensor (n², 3).
    labels:       target token σ(c), LongTensor (n²,).
    moduli:       n for every example. Used to mask logits to valid answers.
    train_idx, val_idx: indices into the n² examples.
    """

    n: int
    op: str
    eq_slot: int
    symbols: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    tokens: torch.Tensor
    labels: torch.Tensor
    moduli: torch.Tensor
    train_idx: np.ndarray
    val_idx: np.ndarray

    # ------------------------------------------------------------------ building

    @classmethod
    def build(
        cls,
        n: int,
        n_numbers: int,
        op: str = "add",
        train_frac: float = 0.3,
        split_seed: int = 0,
        perm_seed: int | None = None,
        eq_slot: int = 0,
    ) -> "ModularTask":
        """Create the task.

        ``split_seed`` fixes which pairs are in the training set.
        ``perm_seed`` (optional) draws a random symbol map π. ``None`` gives the identity map.
        ``eq_slot`` picks the "=" token: 0 ("=p") for phase A, 1 ("=q") for phase B.
        """
        if not 2 <= n <= n_numbers:
            raise ValueError(f"modulus {n} needs 2 <= n <= n_numbers={n_numbers}")
        # Every pair (a, b) with a, b in Z_n, in row-major order.
        a, b = (x.ravel() for x in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
        c = OPS[op](a, b, n)

        if perm_seed is None:
            symbols = np.arange(n)
        else:
            symbols = np.random.default_rng(perm_seed).permutation(n)

        tokens = np.stack([symbols[a], symbols[b], np.full_like(a, eq_token(eq_slot, n_numbers))], axis=1)
        labels = symbols[c]

        order = np.random.default_rng(split_seed).permutation(n * n)
        n_train = int(round(train_frac * n * n))
        train_idx, val_idx = np.sort(order[:n_train]), np.sort(order[n_train:])

        return cls(
            n=n,
            op=op,
            eq_slot=eq_slot,
            symbols=symbols,
            a=a,
            b=b,
            c=c,
            tokens=torch.as_tensor(tokens, dtype=torch.long),
            labels=torch.as_tensor(labels, dtype=torch.long),
            moduli=torch.full((n * n,), n, dtype=torch.long),
            train_idx=train_idx,
            val_idx=val_idx,
        )

    # ------------------------------------------------------------ analysis masks

    @property
    def size(self) -> int:
        return self.n * self.n

    @property
    def is_identity_map(self) -> bool:
        return bool(np.array_equal(self.symbols, np.arange(self.n)))

    def overlap_masks(self, p: int) -> dict[str, np.ndarray]:
        """Split all examples by how they relate to an *old* task Z_p (identity map).

        Everything is computed in **token space**: what would a network that perfectly
        solves the old task Z_p answer if it saw these tokens?

        * ``agree``:    both input tokens < p and the old rule gives the *correct* new label.
                        These pairs are "free" for a network that grokked Z_p.
        * ``disagree``: both input tokens < p but the old rule gives a *wrong* label.
        * ``newtok``:   at least one input token >= p (never seen during phase A).

        The three masks partition the n² examples.
        """
        ta, tb = self.tokens[:, 0].numpy(), self.tokens[:, 1].numpy()
        seen = (ta < p) & (tb < p)
        old_answer = np.where(seen, OPS[self.op](ta, tb, p), -1)
        agree = seen & (old_answer == self.labels.numpy())
        return {"agree": agree, "disagree": seen & ~agree, "newtok": ~seen}

    def describe(self) -> str:
        kind = "identity" if self.is_identity_map else "permuted"
        return (
            f"Z_{self.n} ({self.op}, {kind} symbols, '{EQ_NAMES[self.eq_slot]}'): {self.size} pairs, "
            f"{len(self.train_idx)} train / {len(self.val_idx)} val"
        )
