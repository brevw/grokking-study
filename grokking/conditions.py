"""The experimental conditions of phase B (PROJECT_PLAN.md, Section 4).

A condition only says **how phase B starts**: which weights, which symbol map,
and whether the old task is trained jointly. All the mechanics live in
``experiments.run_transfer``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    name: str
    plan_id: str  # identifier used in PROJECT_PLAN.md
    description: str
    init: str  # "scratch" | "grokked" | "memorized"  (which weights phase B starts from)
    permute: bool = False  # phase-B task uses a random symbol map π
    reinit_embeddings: bool = False  # reset embed / pos / unembed after loading
    shrink_perturb: bool = False  # θ <- shrink·θ + perturb·θ_fresh (Ash & Adams, 2020)
    joint: bool = False  # train on Z_p ∪ Z_q at the same time


CONDITIONS: dict[str, Condition] = {
    c.name: c
    for c in [
        Condition("scratch", "C0", "random init, train on Z_q (baseline)", init="scratch"),
        Condition("warm_grokked", "C1", "start from the grokked Z_p network", init="grokked"),
        Condition("warm_memorized", "C2", "start from the Z_p network at T_mem (memorized only)", init="memorized"),
        Condition(
            "warm_permuted", "C3", "grokked Z_p network, Z_q written with permuted symbols", init="grokked", permute=True
        ),
        Condition(
            "warm_reinit_embed",
            "C4",
            "grokked Z_p network with embeddings/unembedding reset",
            init="grokked",
            reinit_embeddings=True,
        ),
        Condition("joint", "C5", "random init, train on Z_p and Z_q together", init="scratch", joint=True),
        Condition(
            "warm_shrink_perturb",
            "C6",
            "grokked Z_p network after shrink-and-perturb",
            init="grokked",
            shrink_perturb=True,
        ),
    ]
}

# Fixed display order (also fixes the colour of each condition in the plots).
CONDITION_ORDER = list(CONDITIONS)
