"""Figures of PROJECT_PLAN.md, Section 8 (static PNGs for the report).

``make_all_figures(exp_dir)`` writes every figure into ``<exp_dir>/figures``.
Each figure also has its own function, so it can be used in a notebook.

Style rules:
* every condition has a **fixed colour** (by condition, never by rank), so colours
  mean the same thing in every figure;
* training time is on a log axis (grokking times vary by orders of magnitude);
* median across seeds as a line, interquartile range as a band;
* the new task Z_q is drawn solid and the old task Z_p dashed;
* the per-pair figures show the main conditions; the progress sweep
  (warm_progress_XX) has its own figure (``progress_p{p}.png``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

from .analysis import Results  # noqa: E402
from .conditions import MAIN_CONDITIONS, PROGRESS, PROGRESS_CONDITIONS  # noqa: E402

# Categorical palette, assigned in fixed condition order. The norm-matched random
# init is a control, so it gets a neutral grey.
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#8c8c8c"]
COLORS = dict(zip(MAIN_CONDITIONS, _PALETTE))
# Progress sweep: one hue, light (early in phase A) to dark (at T_grok).
COLORS.update(zip(PROGRESS_CONDITIONS, ["#f6c9b3", "#f2a27e", "#eb6834", "#b84a1d", "#7a2e10"]))
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
    }
)


def is_prime(n: int) -> bool:
    return n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))


def _conditions(df: pd.DataFrame) -> list[str]:
    """Main conditions present in ``df``, in the fixed order (progress sweep excluded)."""
    present = set(df["condition"])
    return [c for c in MAIN_CONDITIONS if c in present]


def _median_band(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Median / quartiles of ``col`` across seeds at each step.

    Runs that stopped early (after grokking) keep their last value, so a finished
    run is not dropped from the median.
    """
    wide = df.pivot_table(index="step", columns="seed", values=col).sort_index().ffill()
    return pd.DataFrame({"med": wide.median(axis=1), "lo": wide.quantile(0.25, axis=1), "hi": wide.quantile(0.75, axis=1)})


def _band(ax, df: pd.DataFrame, col: str, color: str, label: str | None, ls: str = "-") -> None:
    if df.empty or col not in df or df[col].isna().all():
        return
    b = _median_band(df, col)
    ax.plot(b.index, b["med"], color=color, ls=ls, label=label)
    ax.fill_between(b.index, b["lo"], b["hi"], color=color, alpha=0.15, lw=0)


def _logx(ax, linthresh: float = 100) -> None:
    """Log-like x-axis that still shows step 0.

    ``symlog`` is linear on [0, linthresh] and logarithmic beyond, so the state at
    the start of phase B (e.g. 100% on the old task) stays visible.
    """
    ax.set_xscale("symlog", linthresh=linthresh, linscale=0.5)
    ax.set_xlim(left=0)
    ax.xaxis.set_minor_formatter(NullFormatter())


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- figures


def plot_source_curves(res: Results, out: Path) -> Path | None:
    logs = res.logs(kind="source")
    if logs.empty:
        return None
    ps = sorted(logs["p"].unique())
    fig, axes = plt.subplots(1, len(ps), figsize=(4.2 * len(ps), 3), squeeze=False)
    for ax, p in zip(axes[0], ps):
        d = logs[logs["p"] == p]
        _band(ax, d, "train_acc", INK_2, "train", ls="--")
        _band(ax, d, "val_acc", COLORS["scratch"], "validation")
        _logx(ax)
        ax.set(ylim=(-0.02, 1.02), xlabel="step", ylabel="accuracy", title=f"Phase A: Z_{p}")
        ax.legend(loc="lower right")
    return _save(fig, out / "source_curves.png")


def plot_transfer_curves(res: Results, p: int, q: int, out: Path) -> Path | None:
    """Main figure: phase-B val accuracy on Z_q per condition (median + IQR over seeds)."""
    logs = res.logs(p=p, q=q)
    if logs.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for c in _conditions(logs):
        d = logs[logs["condition"] == c]
        _band(axes[0], d, "train_acc", COLORS[c], c)
        _band(axes[1], d, "val_acc", COLORS[c], c)
    axes[0].set(title=f"Train accuracy on Z_{q}", ylabel="accuracy")
    axes[1].set(title=f"Validation accuracy on Z_{q}")
    for ax in axes:
        _logx(ax)
        ax.set(ylim=(-0.02, 1.02), xlabel="phase-B step")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Phase B: Z_{p} → Z_{q}", color=INK)
    return _save(fig, out / f"curves_p{p}_q{q}.png")


def plot_forgetting(res: Results, p: int, q: int, out: Path) -> Path | None:
    """Z_q (solid) and Z_p (dashed) val accuracy during phase B, one panel per condition."""
    logs = res.logs(p=p, q=q)
    if logs.empty or "old_val_acc" not in logs:
        return None
    conds = _conditions(logs)
    ncol = min(4, len(conds))
    nrow = int(np.ceil(len(conds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow), sharey=True, squeeze=False)
    disagree = "old_val_acc_disagree" in logs and logs["old_val_acc_disagree"].notna().any()
    for ax, c in zip(axes.flat, conds):
        d = logs[logs["condition"] == c]
        _band(ax, d, "val_acc", COLORS[c], None)
        if disagree:
            _band(ax, d, "old_val_acc_disagree", COLORS[c], None, ls="--")
            _band(ax, d, "old_val_acc", INK_2, None, ls=":")
        else:
            _band(ax, d, "old_val_acc", COLORS[c], None, ls="--")
        _logx(ax)
        ax.set(ylim=(-0.02, 1.02), xlabel="phase-B step", title=c)
    for ax in list(axes.flat)[len(conds):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel("validation accuracy")
    # One shared legend for the line styles (colour = condition, given by the titles).
    handles = [Line2D([], [], color=INK_2, ls="-", label=f"new task Z_{q}")]
    if disagree:
        handles += [Line2D([], [], color=INK_2, ls="--", label=f"old task Z_{p}, pairs where mod {q} is wrong"),
                    Line2D([], [], color=INK_2, ls=":", label=f"old task Z_{p}, all pairs")]
    else:
        handles += [Line2D([], [], color=INK_2, ls="--", label=f"old task Z_{p}")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.02 + 0.04 / nrow))
    fig.tight_layout()
    return _save(fig, out / f"forgetting_p{p}_q{q}.png")


def plot_overlap(res: Results, p: int, q: int, out: Path) -> Path | None:
    """Val accuracy split into agree / disagree / new-token pairs (identity-map conditions)."""
    logs = res.logs(p=p, q=q)
    cols = [c for c in ("val_acc_agree", "val_acc_disagree", "val_acc_newtok") if c in logs and logs[c].notna().any()]
    if logs.empty or not cols:
        return None
    fig, axes = plt.subplots(1, len(cols), figsize=(3.4 * len(cols), 3), sharey=True, squeeze=False)
    titles = {"val_acc_agree": "pairs where old answer is right",
              "val_acc_disagree": "pairs where old answer is wrong",
              "val_acc_newtok": "pairs with unseen tokens"}
    for ax, col in zip(axes[0], cols):
        for c in _conditions(logs):
            _band(ax, logs[logs["condition"] == c], col, COLORS[c], c)
        _logx(ax)
        ax.set(ylim=(-0.02, 1.02), xlabel="phase-B step", title=titles[col])
    axes[0][0].set_ylabel(f"val accuracy on Z_{q}")
    axes[0][-1].legend(loc="lower right", fontsize=7)
    return _save(fig, out / f"overlap_p{p}_q{q}.png")


def plot_representation(res: Results, p: int, q: int, out: Path) -> Path | None:
    """Representation dynamics: old-frequency R², Fourier sparsity, effective LR, weight norm."""
    rep = res.repr_logs(p=p, q=q)
    logs = res.logs(p=p, q=q)
    if rep.empty:
        return None
    panels = [
        (rep, "old_freq_r2", f"old Z_{p} circles retained (R²)"),
        (rep, "fourier_sparsity", f"Fourier sparsity on Z_{q} (top-5)"),
        (logs, "elr", "effective learning rate ‖Δθ‖/‖θ‖"),
        (rep, "param_norm", "parameter norm ‖θ‖"),
        (rep, "act_cka_to_start", "MLP activation CKA to start"),
        (rep, "dead_frac", "dead MLP neurons"),
    ]
    panels = [x for x in panels if x[1] in x[0]]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3 * nrow), squeeze=False)
    for ax, (df, col, title) in zip(axes.flat, panels):
        for c in _conditions(df):
            _band(ax, df[df["condition"] == c], col, COLORS[c], c)
        _logx(ax)
        ax.set(xlabel="phase-B step", title=title)
        if col == "elr":
            ax.set_yscale("log")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Representation during phase B: Z_{p} → Z_{q}", color=INK)
    fig.tight_layout()
    return _save(fig, out / f"representation_p{p}_q{q}.png")


def plot_survival(res: Results, p: int, q: int, out: Path) -> Path | None:
    surv = res.survival(p, q)
    if surv.empty:
        return None
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for c in _conditions(surv):
        d = surv[surv["condition"] == c]
        ax.step(d["step"], d["not_grokked"], where="post", color=COLORS[c], label=c)
    ax.set(xlabel="phase-B step", ylabel="fraction not yet grokked", ylim=(-0.02, 1.02), title=f"Z_{p} → Z_{q}")
    ax.legend(fontsize=7)
    return _save(fig, out / f"survival_p{p}_q{q}.png")


def plot_transfer_ratios(res: Results, out: Path) -> list[Path]:
    """Transfer ratio R per target modulus (median and 95% CI, one dot per seed)."""
    ratios = res.transfer_ratios()
    paired = res.paired_ratios()
    paths = []
    if ratios.empty:
        return paths
    for p, rp in ratios.groupby("p"):
        qs = sorted(rp["q"].unique(), key=lambda q: (q == p, not is_prime(q), q < p, q))
        conds = _conditions(rp)
        width = 0.8 / max(1, len(conds))
        fig, ax = plt.subplots(figsize=(max(4.5, 1.1 * len(qs) + 2), 3.4))
        for j, c in enumerate(conds):
            for i, q in enumerate(qs):
                x = i - 0.4 + width * (j + 0.5)
                r = rp[(rp["q"] == q) & (rp["condition"] == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(x, r["R_median"], yerr=[[r["R_median"] - r["R_ci_low"]], [r["R_ci_high"] - r["R_median"]]],
                            fmt="o", ms=6, color=COLORS[c], capsize=0, lw=1.5, label=c if i == 0 else None)
                s = paired[(paired["p"] == p) & (paired["q"] == q) & (paired["condition"] == c)]
                ax.scatter(np.full(len(s), x + width * 0.25), s["R"], s=8, color=COLORS[c], alpha=0.45, lw=0)
        ax.axhline(1.0, color=INK_2, lw=1, ls=":")
        labels = [f"{q}\n{'same' if q == p else ('prime' if is_prime(q) else 'composite')}" for q in qs]
        ax.set_xticks(range(len(qs)), labels)
        ax.set(yscale="log", xlabel="target modulus q", ylabel="R = T_grok(scratch) / T_grok(cond.)",
               title=f"Transfer from Z_{p}   (R > 1: faster than scratch)")
        ax.legend(fontsize=7, loc="best")
        paths.append(_save(fig, out / f"transfer_ratios_p{p}.png"))
    return paths


def plot_subgroup_frequencies(res: Results, out: Path) -> Path | None:
    """For composite q: fraction of learned key frequencies with gcd(k, q) > 1 (RQ4)."""
    rep = res.repr_logs()
    if rep.empty or "subgroup_frac" not in rep:
        return None
    last = rep.sort_values("step").groupby(["p", "q", "condition", "seed"]).tail(1)
    last = last[~last["q"].map(is_prime)]
    if last.empty:
        return None
    qs = sorted(last["q"].unique())
    conds = _conditions(last)
    width = 0.8 / len(conds)
    fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(qs) + 2), 3))
    for j, c in enumerate(conds):
        for i, q in enumerate(qs):
            vals = last[(last["q"] == q) & (last["condition"] == c)]["subgroup_frac"].dropna()
            if vals.empty:
                continue
            x = i - 0.4 + width * (j + 0.5)
            ax.bar(x, vals.median(), width * 0.9, color=COLORS[c], label=c if i == 0 else None)
            ax.scatter(np.full(len(vals), x), vals, s=8, color=INK_2, zorder=3, lw=0)
    ax.set_xticks(range(len(qs)), [str(q) for q in qs])
    ax.set(xlabel="composite target q", ylabel="fraction of key freqs with gcd(k,q)>1", ylim=(0, 1),
           title="Subgroup frequencies at the end of phase B")
    ax.legend(fontsize=7)
    return _save(fig, out / "subgroup_frequencies.png")


def plot_progress(res: Results, out: Path) -> list[Path]:
    """Transfer vs. how far phase A had progressed when phase B started.

    One panel per target q: transfer ratio R (median, 95% CI, one dot per seed) for a
    start at 0% (scratch, R = 1), 10..100% of the source's T_grok, and after clean-up
    (warm_grokked). Last panel: the source network at those points, its val accuracy
    and its Fourier sparsity (does the clock structure appear before val acc jumps?).
    """
    table = res.progress_table()
    paths: list[Path] = []
    if table.empty:
        return paths
    paired = res.paired_ratios()
    labels = ["0"] + [f"{pct}%" for pct in PROGRESS] + ["final"]
    xpos = {pct: i + 1 for i, pct in enumerate(PROGRESS)} | {"final": len(PROGRESS) + 1}
    cond_of = {pct: f"warm_progress_{pct}" for pct in PROGRESS} | {"final": "warm_grokked"}
    for p, tp in table.groupby("p"):
        qs = sorted(tp["q"].unique(), key=lambda q: (q == p, not is_prime(q), q < p, q))
        n = len(qs) + 1
        ncol = min(4, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.8 * nrow), squeeze=False)
        for ax, q in zip(axes.flat, qs):
            t = tp[tp["q"] == q]
            x = [0] + [xpos[v] for v in t["progress"]]
            ax.plot(x, [1.0] + list(t["R_median"]), color=COLORS["warm_grokked"], marker="o", ms=4)
            ax.vlines([xpos[v] for v in t["progress"]], t["R_ci_low"], t["R_ci_high"], color=COLORS["warm_grokked"], lw=1.2)
            for v in t["progress"]:
                s = paired[(paired["p"] == p) & (paired["q"] == q) & (paired["condition"] == cond_of[v])]
                ax.scatter(np.full(len(s), xpos[v] + 0.15), s["R"], s=6, color=INK_2, alpha=0.4, lw=0)
            ax.axhline(1.0, color=INK_2, lw=1, ls=":")
            kind = "same" if q == p else ("prime" if is_prime(q) else "composite")
            ax.set(yscale="log", title=f"Z_{p} → Z_{q} ({kind})")
            ax.set_xticks(range(len(labels)), labels, fontsize=7)
        for i in range(0, len(qs), ncol):
            axes.flat[i].set_ylabel("R = T_grok(scratch) / T_grok")
        # Source state along phase A.
        ax = axes.flat[len(qs)]
        t = tp.drop_duplicates("progress")
        x = [xpos[v] for v in t["progress"]]
        ax.plot(x, t["source_val_acc"], color=COLORS["scratch"], marker="o", ms=4, label="val accuracy")
        ax.plot(x, t["source_sparsity"], color=COLORS["warm_permuted"], marker="o", ms=4, label="Fourier sparsity")
        ax.set_xticks(range(len(labels)), labels, fontsize=7)
        ax.set(ylim=(-0.02, 1.02), title=f"source Z_{p} at that point")
        ax.legend(fontsize=7, loc="upper left")
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        fig.supxlabel("phase-A checkpoint (% of the source's T_grok)", fontsize=9)
        fig.tight_layout()
        paths.append(_save(fig, out / f"progress_p{p}.png"))
    return paths


# ------------------------------------------------------------------------ all


def make_all_figures(exp_dir: str | Path) -> list[Path]:
    res = Results(exp_dir)
    out = Path(exp_dir) / "figures"
    paths: list[Path | None] = [plot_source_curves(res, out)]
    if len(res.transfers):
        paths += plot_transfer_ratios(res, out)
        paths += plot_progress(res, out)
        paths.append(plot_subgroup_frequencies(res, out))
        for (p, q), _ in res.transfers.groupby(["p", "q"]):
            for fn in (plot_transfer_curves, plot_forgetting, plot_overlap, plot_representation, plot_survival):
                paths.append(fn(res, int(p), int(q), out))
    return [p for p in paths if p is not None]
