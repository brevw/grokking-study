"""Parse the results of an experiment into tidy tables (PROJECT_PLAN.md, Section 6).

Typical use, from a notebook::

    from grokking.analysis import Results
    res = Results("results/main")
    res.summaries          # one row per run (pandas DataFrame)
    res.condition_table()  # median T_grok, % grokked, forgetting ... per (p, q, condition)
    res.transfer_ratios()  # R = T_grok(scratch) / T_grok(condition), paired by seed, with CIs
    res.logs(p=97, q=113)  # all training curves of phase B for one pair, long format

or from the command line: ``python run.py analyze configs/main.json``.

Conventions:

* Censored runs (not grokked within the budget) get ``T_grok = budget`` and are
  counted in ``frac_grokked``. Medians that include censored runs are lower bounds.
* Grokking times are floored at ``eval_every`` before taking ratios, so a run that is
  solved at step 0 (e.g. the q = p control) does not divide by zero.
"""

from __future__ import annotations

import math
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .conditions import CONDITION_ORDER, PROGRESS
from .utils import read_json

KEYS = ["p", "q", "condition", "seed"]


# --------------------------------------------------------------------- statistics


def bootstrap_ci(values: np.ndarray, stat=np.median, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of ``stat`` over the given values (resampling seeds)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    boots = stat(rng.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def sign_test(diffs: np.ndarray) -> float:
    """Two-sided exact sign test p-value for H0: median(diffs) = 0 (ties dropped)."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[(diffs != 0) & ~np.isnan(diffs)]
    n = len(diffs)
    if n == 0:
        return float("nan")
    k = int(min((diffs > 0).sum(), (diffs < 0).sum()))
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * p))


def _order_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows in the fixed condition order (scratch first)."""
    order = {c: i for i, c in enumerate(CONDITION_ORDER + ["source"])}
    return df.assign(_o=df["condition"].map(order)).sort_values(["p", "q", "_o"]).drop(columns="_o")


# ---------------------------------------------------------------------- results


class Results:
    """Lazy loader around one experiment directory."""

    def __init__(self, exp_dir: str | Path):
        self.exp_dir = Path(exp_dir)
        if not self.exp_dir.exists():
            raise FileNotFoundError(self.exp_dir)

    # ---------------------------------------------------------------- loading

    @cached_property
    def summaries(self) -> pd.DataFrame:
        """One row per finished run (sources and transfers), from summary.json files."""
        rows = []
        for path in sorted(self.exp_dir.glob("*/**/summary.json")):
            s = read_json(path)
            s["run_dir"] = str(path.parent)
            rows.append(s)
        if not rows:
            raise FileNotFoundError(f"No finished runs (summary.json) under {self.exp_dir}")
        df = pd.DataFrame(rows)
        # Censored runs: use the budget as a (lower-bound) grokking time.
        df["T_grok_c"] = df["T_grok"].fillna(df["budget"]).astype(float)
        df["delay"] = df["T_grok"] - df["T_mem"]  # NaN if not grokked
        return df

    @property
    def sources(self) -> pd.DataFrame:
        return self.summaries[self.summaries["kind"] == "source"]

    @property
    def transfers(self) -> pd.DataFrame:
        return self.summaries[self.summaries["kind"] == "transfer"]

    def _load_curves(self, filename: str, kind: str, **filters) -> pd.DataFrame:
        df = self.summaries[self.summaries["kind"] == kind]
        for k, v in filters.items():
            if v is not None:
                df = df[df[k].isin(v if isinstance(v, (list, tuple, set)) else [v])]
        frames = []
        for _, r in df.iterrows():
            path = Path(r["run_dir"]) / filename
            if path.exists() and path.stat().st_size > 0:
                f = pd.read_csv(path)
                for k in KEYS:
                    f[k] = r[k]
                frames.append(f)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def logs(self, kind: str = "transfer", **filters) -> pd.DataFrame:
        """Behavioural curves (log.csv) in long format; filter with p=, q=, condition=, seed=."""
        return self._load_curves("log.csv", kind, **filters)

    def repr_logs(self, kind: str = "transfer", **filters) -> pd.DataFrame:
        """Representation curves (repr.csv) in long format."""
        return self._load_curves("repr.csv", kind, **filters)

    # --------------------------------------------------------------- tables

    def condition_table(self) -> pd.DataFrame:
        """Per (p, q, condition): the numbers of Table 5 in the plan."""
        df = self.summaries
        g = df.groupby(["p", "q", "condition"])
        out = pd.DataFrame(
            {
                "n_runs": g.size(),
                "frac_grokked": g["grokked"].mean(),
                "T_mem_med": g["T_mem"].median(),
                "T_grok_med": g["T_grok_c"].median(),
                "T_grok_q25": g["T_grok_c"].quantile(0.25),
                "T_grok_q75": g["T_grok_c"].quantile(0.75),
                "delay_med": g["delay"].median(),
                "step0_val_acc": g["step0_val_acc"].mean(),
            }
        )
        for col in ("forgetting", "forgetting_disagree", "T_forget", "step0_val_acc_agree", "step0_val_acc_disagree"):
            if col in df:
                out[col + "_med" if col.startswith("forgetting") else col] = g[col].median()
        return _order_conditions(out.reset_index())

    def progress_table(self) -> pd.DataFrame:
        """Transfer vs. how far phase A had progressed (the warm_progress_XX conditions).

        One row per (p, q, progress %): the transfer ratio (median, CI) and, for the
        source checkpoints at that point, the median val accuracy and Fourier sparsity.
        ``progress`` 0 is scratch (R = 1) and ``final`` is warm_grokked (after clean-up).
        """
        rt = self.transfer_ratios()
        if rt.empty:
            return pd.DataFrame()
        src = self.sources
        rows = []
        points = [(pct, f"warm_progress_{pct}", f"progress_{pct}") for pct in PROGRESS]
        points.append(("final", "warm_grokked", None))
        for (p, q), r in rt.groupby(["p", "q"]):
            s = src[src["p"] == p]
            for pct, cond, key in points:
                hit = r[r["condition"] == cond]
                if hit.empty:
                    continue
                h = hit.iloc[0]
                rows.append(
                    {
                        "p": p,
                        "q": q,
                        "progress": pct,
                        "R_median": h["R_median"],
                        "R_ci_low": h["R_ci_low"],
                        "R_ci_high": h["R_ci_high"],
                        "sign_test_p": h["sign_test_p"],
                        "n_censored": h["n_censored"],
                        "source_val_acc": s[f"{key}_val_acc"].median() if key and f"{key}_val_acc" in s else s["final_val_acc"].median(),
                        "source_sparsity": s[f"{key}_sparsity"].median() if key and f"{key}_sparsity" in s else s["final_fourier_sparsity"].median(),
                    }
                )
        return pd.DataFrame(rows)

    def transfer_ratios(self, baseline: str = "scratch") -> pd.DataFrame:
        """Paired transfer ratio R = T_grok(baseline) / T_grok(condition) per seed, then aggregated.

        R > 1: the condition grokks faster than the baseline (positive transfer).
        R < 1: slower (primacy bias). Returns one row per (p, q, condition) with the
        median R, a bootstrap 95% CI over seeds, a sign-test p-value, and the number
        of censored runs on either side.
        """
        df = self.transfers.copy()
        floor = df["eval_every"].astype(float)
        df["T_eff"] = np.maximum(df["T_grok_c"], floor)
        base = df[df["condition"] == baseline][["p", "q", "seed", "T_eff", "censored"]]
        base = base.rename(columns={"T_eff": "T_base", "censored": "base_censored"})
        other = df[df["condition"] != baseline][["p", "q", "condition", "seed", "T_eff", "censored"]]
        paired = other.merge(base, on=["p", "q", "seed"], how="inner")
        if paired.empty:
            return pd.DataFrame()
        paired["log2_R"] = np.log2(paired["T_base"] / paired["T_eff"])

        rows = []
        for (p, q, c), g in paired.groupby(["p", "q", "condition"]):
            lo, hi = bootstrap_ci(g["log2_R"].to_numpy())
            rows.append(
                {
                    "p": p,
                    "q": q,
                    "condition": c,
                    "n_pairs": len(g),
                    "R_median": 2 ** g["log2_R"].median(),
                    "R_ci_low": 2**lo,
                    "R_ci_high": 2**hi,
                    "sign_test_p": sign_test(g["log2_R"].to_numpy()),
                    "n_censored": int(g["censored"].sum()),
                    "n_base_censored": int(g["base_censored"].sum()),
                }
            )
        return _order_conditions(pd.DataFrame(rows))

    def paired_ratios(self, baseline: str = "scratch") -> pd.DataFrame:
        """Per-seed ratios (unaggregated), useful for strip plots."""
        df = self.transfers.copy()
        df["T_eff"] = np.maximum(df["T_grok_c"], df["eval_every"].astype(float))
        base = df[df["condition"] == baseline][["p", "q", "seed", "T_eff"]].rename(columns={"T_eff": "T_base"})
        paired = df[df["condition"] != baseline].merge(base, on=["p", "q", "seed"])
        paired["R"] = paired["T_base"] / paired["T_eff"]
        return paired

    def survival(self, p: int, q: int, steps: np.ndarray | None = None) -> pd.DataFrame:
        """Fraction of runs *not yet grokked* at each step, per condition.

        All runs share the same budget, so censoring only happens at the end and this
        empirical curve equals the Kaplan-Meier estimate.
        """
        df = self.transfers[(self.transfers["p"] == p) & (self.transfers["q"] == q)]
        if steps is None:
            steps = np.linspace(0, df["budget"].max(), 200)
        rows = []
        for c, g in df.groupby("condition"):
            t = g["T_grok"].astype(float).fillna(np.inf).to_numpy()
            for s in steps:
                rows.append({"condition": c, "step": s, "not_grokked": float((t > s).mean())})
        return pd.DataFrame(rows)

    # --------------------------------------------------------------- report

    def report(self, save: bool = True) -> str:
        """Text report for the terminal. Also writes the tables as CSV next to the results."""
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 30)
        parts = [f"=== Results in {self.exp_dir} ({len(self.summaries)} finished runs) ==="]

        src = self.sources
        if len(src):
            t = src.groupby("p").agg(
                n=("seed", "size"),
                frac_grokked=("grokked", "mean"),
                T_mem=("T_mem", "median"),
                T_grok=("T_grok_c", "median"),
                key_freqs=("final_key_freqs", lambda s: " | ".join(map(str, s))),
            )
            parts += ["", "--- Phase A (source runs) ---", t.to_string(float_format=lambda x: f"{x:.3g}")]

        if len(self.transfers):
            cols = ["p", "q", "condition", "n_runs", "frac_grokked", "T_mem_med", "T_grok_med", "delay_med",
                    "step0_val_acc", "forgetting_med", "forgetting_disagree_med"]
            ct = self.condition_table()
            ct = ct[ct["condition"] != "source"]
            parts += ["", "--- Phase B: per condition (T_grok of censored runs = budget) ---",
                      ct[[c for c in cols if c in ct]].to_string(index=False, float_format=lambda x: f"{x:.3g}")]
            rt = self.transfer_ratios()
            if len(rt):
                parts += ["", "--- Transfer ratio R = T_grok(scratch) / T_grok(condition)  (R>1: faster than scratch) ---",
                          rt.to_string(index=False, float_format=lambda x: f"{x:.3g}")]
            pt = self.progress_table()
            if len(pt):
                parts += ["", "--- Transfer vs. progress of phase A (0 = scratch, final = grokked + clean-up) ---",
                          pt.to_string(index=False, float_format=lambda x: f"{x:.3g}")]
            if save:
                ct.to_csv(self.exp_dir / "table_conditions.csv", index=False)
                rt.to_csv(self.exp_dir / "table_ratios.csv", index=False)
                if len(pt):
                    pt.to_csv(self.exp_dir / "table_progress.csv", index=False)
        if save:
            self.summaries.to_csv(self.exp_dir / "runs.csv", index=False)
            parts += ["", f"Tables written to {self.exp_dir}/(runs|table_conditions|table_ratios|table_progress).csv"]
        return "\n".join(parts)
