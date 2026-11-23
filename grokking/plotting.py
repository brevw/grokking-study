"""Figures of PROJECT_PLAN.md, Section 8 (static PNGs for the report).

``make_all_figures(exp_dir)`` writes every figure into ``<exp_dir>/figures``.
Each figure also has its own function, so it can be used in a notebook.

Style rules:
* every condition has a **fixed colour** (by condition, never by rank), so colours
  mean the same thing in every figure;
* training time is on a log axis (grokking times vary by orders of magnitude);
* median across seeds as a line, interquartile range as a band;
* the new task Z_q is drawn solid and the old task Z_p dashed.
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
from .conditions import CONDITION_ORDER  # noqa: E402

# Categorical palette, assigned in fixed condition order (validated reference palette).
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
COLORS = dict(zip(CONDITION_ORDER, _PALETTE))
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
    present = set(df["condition"])
    return [c for c in CONDITION_ORDER if c in present]


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
    fig, axes = plt.subplots(1, len(conds), figsize=(3.2 * len(conds), 3), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], conds):
        d = logs[logs["condition"] == c]
        _band(ax, d, "val_acc", COLORS[c], None)
        _band(ax, d, "old_val_acc", COLORS[c], None, ls="--")
        _logx(ax)
        ax.set(ylim=(-0.02, 1.02), xlabel="phase-B step", title=c)
    axes[0][0].set_ylabel("validation accuracy")
    # One shared legend for the line styles (colour = condition, given by the titles).
    handles = [Line2D([], [], color=INK_2, ls="-", label=f"new task Z_{q}"),
               Line2D([], [], color=INK_2, ls="--", label=f"old task Z_{p}")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
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


# ------------------------------------------------------------------------ all


def make_all_figures(exp_dir: str | Path) -> list[Path]:
    res = Results(exp_dir)
    out = Path(exp_dir) / "figures"
    paths: list[Path | None] = [plot_source_curves(res, out)]
    if len(res.transfers):
        paths += plot_transfer_ratios(res, out)
        paths.append(plot_subgroup_frequencies(res, out))
        for (p, q), _ in res.transfers.groupby(["p", "q"]):
            for fn in (plot_transfer_curves, plot_forgetting, plot_overlap, plot_representation, plot_survival):
                paths.append(fn(res, int(p), int(q), out))
    return [p for p in paths if p is not None]
