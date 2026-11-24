"""The experimental conditions of phase B (PROJECT_PLAN.md, Section 4).

A condition only says **how phase B starts**: which weights, which symbol map,
and whether the old task is trained jointly. All the mechanics live in
``experiments.run_transfer``.

Phase-A checkpoints a condition can start from (all saved by ``experiments.run_source``):

* ``final``        the grokked network after the clean-up phase (T_grok + stop_after_grok)
* ``memorized``    the end of the memorization plateau: the last checkpoint before T_grok
                   with train acc >= 0.99 and val acc < ``mem_val_threshold``
* ``progress_XX``  the checkpoint closest to XX% of T_grok (XX in PROGRESS)
"""

from __future__ import annotations

from dataclasses import dataclass

# Points along phase A (in % of the source's T_grok) used by the warm_progress_XX conditions.
PROGRESS = (10, 25, 50, 75, 100)


@dataclass(frozen=True)
class Condition:
    name: str
    plan_id: str  # identifier used in PROJECT_PLAN.md
    description: str
    init: str  # "scratch" or the phase-A checkpoint to start from ("final", "memorized", "progress_XX")
    permute: bool = False  # phase-B task uses a random symbol map π
    reinit_embeddings: bool = False  # reset embed / pos / unembed after loading
    shrink_perturb: bool = False  # θ <- shrink·θ + perturb·θ_fresh (Ash & Adams, 2020)
    joint: bool = False  # train on Z_p ∪ Z_q at the same time
    # Random init rescaled so that every weight matrix has the same norm as in the
    # warm_grokked start. Separates "useful structure" from "larger weights".
    norm_match: bool = False

    @property
    def needs_source(self) -> bool:
        return self.init != "scratch" or self.norm_match


CONDITIONS: dict[str, Condition] = {
    c.name: c
    for c in [
        Condition("scratch", "C0", "random init, train on Z_q (baseline)", init="scratch"),
        Condition("warm_grokked", "C1", "start from the grokked Z_p network (after clean-up)", init="final"),
        Condition(
            "warm_memorized",
            "C2",
            "start from the Z_p network at the end of the memorization plateau (memorized, not generalized)",
            init="memorized",
        ),
        Condition(
            "warm_permuted", "C3", "grokked Z_p network, Z_q written with permuted symbols", init="final", permute=True
        ),
        Condition(
            "warm_reinit_embed",
            "C4",
            "grokked Z_p network with embeddings/unembedding reset",
            init="final",
            reinit_embeddings=True,
        ),
        Condition("joint", "C5", "random init, train on Z_p and Z_q together", init="scratch", joint=True),
        Condition(
            "warm_shrink_perturb",
            "C6",
            "grokked Z_p network after shrink-and-perturb",
            init="final",
            shrink_perturb=True,
        ),
        Condition(
            "scratch_norm_matched",
            "C7",
            "random init with the per-matrix weight norms of the warm_grokked start",
            init="scratch",
            norm_match=True,
        ),
        *[
            Condition(
                f"warm_progress_{pct}",
                f"C8-{pct}",
                f"start from the Z_p network at {pct}% of its T_grok",
                init=f"progress_{pct}",
            )
            for pct in PROGRESS
        ],
    ]
}

# Fixed display order (also fixes the colour of each condition in the plots).
CONDITION_ORDER = list(CONDITIONS)
# The progress sweep has its own figure; the per-pair figures show the other conditions.
PROGRESS_CONDITIONS = [f"warm_progress_{pct}" for pct in PROGRESS]
MAIN_CONDITIONS = [c for c in CONDITION_ORDER if c not in PROGRESS_CONDITIONS]
