"""Cheap representation metrics (PROJECT_PLAN.md, Section 5.3).

All functions are plain NumPy on small matrices, so they can run during training
and offline on saved checkpoints.

Notation: ``E`` is an embedding matrix with **one row per group element in element
order** (row e = embedding of the token σ(e)). For the identity symbol map this is
just ``W_E[:n]``.

Fourier convention: for Z_n we use the real DFT along the element axis. Frequency
``k`` (0 <= k <= n//2) corresponds to the functions cos(2πka/n) and sin(2πka/n).
Frequencies k and n-k are the same "circle", so only k <= n//2 matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
import torch

from .data import ModularTask
from .model import GrokTransformer, flat_params

# ---------------------------------------------------------------------- Fourier


def group_embedding(W_E: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    """Rows of the embedding matrix re-ordered by group element: E[e] = W_E[σ(e)]."""
    return W_E[symbols]


def fourier_power(E: np.ndarray) -> np.ndarray:
    """Power per frequency k = 0..n//2, summed over embedding dimensions.

    ``E`` has shape (n, d). Returns an array of shape (n//2 + 1,).
    """
    coeffs = np.fft.rfft(E, axis=0)  # (n//2+1, d)
    return (np.abs(coeffs) ** 2).sum(axis=1)


def key_frequencies(power: np.ndarray, mass: float = 0.9, max_freqs: int | None = None) -> list[int]:
    """Smallest set of non-constant frequencies holding ``mass`` of the non-constant power."""
    p = power[1:]  # drop k = 0 (the mean embedding, not a circle)
    total = p.sum()
    if total <= 0:
        return []
    order = np.argsort(p)[::-1]
    cum = np.cumsum(p[order]) / total
    n_keep = int(np.searchsorted(cum, mass) + 1)
    if max_freqs is not None:
        n_keep = min(n_keep, max_freqs)
    return sorted(int(k) + 1 for k in order[:n_keep])


def fourier_sparsity(power: np.ndarray, top: int = 5) -> float:
    """Fraction of non-constant power in the ``top`` strongest frequencies.

    Near 1 means a clean "clock" representation built from a few circles. For random
    embeddings it is about top / (n/2).
    """
    p = power[1:]
    total = p.sum()
    return float(np.sort(p)[::-1][:top].sum() / total) if total > 0 else float("nan")


def fourier_basis(n: int, freqs: list[int], m: int | None = None) -> np.ndarray:
    """Design matrix [1, cos(2πka/n), sin(2πka/n) for k in freqs] for a = 0..m-1."""
    m = n if m is None else m
    a = np.arange(m)[:, None]
    cols = [np.ones((m, 1))]
    for k in freqs:
        cols += [np.cos(2 * np.pi * k * a / n), np.sin(2 * np.pi * k * a / n)]
    return np.hstack(cols)


def frequency_fit_r2(E_rows: np.ndarray, n: int, freqs: list[int]) -> float:
    """R² of a least-squares fit of the rows with the Fourier basis of ``freqs`` on Z_n.

    We use it for **old-frequency retention**: how much of the token embeddings
    0..p-1 is still explained by the key circles of the old group Z_p?
    """
    if len(freqs) == 0:
        return float("nan")
    X = fourier_basis(n, freqs, m=E_rows.shape[0])
    coef, *_ = np.linalg.lstsq(X, E_rows, rcond=None)
    resid = E_rows - X @ coef
    ss_tot = ((E_rows - E_rows.mean(axis=0)) ** 2).sum()
    return float(1.0 - (resid**2).sum() / ss_tot) if ss_tot > 0 else float("nan")


def frequency_match_distance(new_freqs: list[int], q: int, old_freqs: list[int], p: int) -> float:
    """Mean distance from each new key frequency k'/q to the closest old one k/p.

    Distances are in "turns per element" (in [0, 0.5]). Compare warm-started runs
    with scratch runs. If warm runs are systematically closer, the network re-tuned
    its old circles instead of building new ones.
    """
    if not new_freqs or not old_freqs:
        return float("nan")
    old = np.array(old_freqs) / p
    return float(np.mean([np.min(np.abs(old - k / q)) for k in new_freqs]))


def subgroup_fraction(freqs: list[int], n: int) -> float:
    """Fraction of frequencies k with gcd(k, n) > 1.

    Such a frequency is constant on cosets of a subgroup, so it cannot tell all the
    elements apart on its own. This is relevant for composite n (RQ4). It is always
    0 for prime n.
    """
    return float(np.mean([gcd(k, n) > 1 for k in freqs])) if freqs else float("nan")


# ------------------------------------------------------------- similarity, health


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., 2019) between two (n_examples, features) matrices."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    norm = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(hsic / norm) if norm > 0 else float("nan")


def dead_fraction(hidden: np.ndarray) -> float:
    """Fraction of MLP neurons that are never active on any example."""
    return float((hidden.max(axis=0) <= 0).mean())


# ------------------------------------------------------------- snapshot (online)


@dataclass
class OldTaskInfo:
    """What we remember about phase A, to measure retention in phase B."""

    p: int
    key_freqs: list[int]


@dataclass
class Reference:
    """State of the network at the start of the current run (step 0)."""

    E: np.ndarray  # group embedding of the primary task
    hidden: np.ndarray  # MLP activations on all primary-task examples
    theta: torch.Tensor  # flat parameter vector


def make_reference(model: GrokTransformer, task: ModularTask, hidden: np.ndarray) -> Reference:
    W = model.number_embeddings.detach().cpu().numpy()
    return Reference(E=group_embedding(W, task.symbols), hidden=hidden, theta=flat_params(model).cpu())


def snapshot(
    model: GrokTransformer,
    task: ModularTask,
    hidden: np.ndarray,
    ref: Reference,
    old: OldTaskInfo | None = None,
) -> dict:
    """All representation metrics of the current model for one log row."""
    W = model.number_embeddings.detach().cpu().numpy()
    E = group_embedding(W, task.symbols)
    power = fourier_power(E)
    freqs = key_frequencies(power)
    theta = flat_params(model).cpu()

    row = {
        "n_key_freqs": len(freqs),
        "key_freqs": ";".join(map(str, freqs)),
        "fourier_sparsity": fourier_sparsity(power),
        "subgroup_frac": subgroup_fraction(freqs, task.n),
        "emb_cka_to_start": linear_cka(ref.E, E),
        "act_cka_to_start": linear_cka(ref.hidden, hidden),
        "dead_frac": dead_fraction(hidden),
        "param_norm": float(theta.norm()),
        "rel_dist_from_start": float((theta - ref.theta).norm() / ref.theta.norm()),
    }
    if old is not None:
        # Old structure lives on tokens 0..p-1 (phase A always uses the identity map).
        row["old_freq_r2"] = frequency_fit_r2(W[: old.p], old.p, old.key_freqs)
        row["freq_match_dist"] = frequency_match_distance(freqs, task.n, old.key_freqs, old.p)
    return row
