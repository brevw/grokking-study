"""The training loop: full-batch AdamW with logging, grokking detection and stopping.

One ``Trainer`` = one training phase of one network. It knows nothing about
"conditions" or "phase A / phase B". ``experiments.py`` sets those up and hands
the trainer:

* ``train_tasks``  the tasks whose *training splits* are optimised (1 task, or 2 for
                   joint training),
* ``primary``      the task whose val accuracy defines T_grok and stopping,
* ``eval_tasks``   extra tasks to monitor (e.g. ``{"old": Z_p}`` to measure forgetting),
* ``val_masks``    boolean masks over the primary task, used to split val accuracy
                   (agree / disagree / newtok),
* ``old``          phase-A info used by the old-frequency retention metric.

Files written to ``out_dir``::

    log.csv        one row every eval_every steps (behavioural metrics)
    repr.csv       one row every repr_every steps (representation metrics)
    summary.json   timings and final numbers of the run (the input of analysis.py)
    memorized.pt   weights at T_mem          (if save_memorized)
    final.pt       weights at the end        (if save_final)
    ckpt/step_*.pt periodic weights          (if cfg.save_checkpoints)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import TrainConfig
from .data import ModularTask
from .metrics import SustainedCrossing, TaskEvaluator, masked_mean
from .model import GrokTransformer, flat_params
from .representation import OldTaskInfo, make_reference, snapshot
from .utils import resolve_device, write_csv, write_json


class Trainer:
    def __init__(
        self,
        model: GrokTransformer,
        train_tasks: list[ModularTask],
        primary: ModularTask,
        cfg: TrainConfig,
        out_dir: str | Path,
        eval_tasks: dict[str, ModularTask] | None = None,
        val_masks: dict[str, np.ndarray] | None = None,
        old: OldTaskInfo | None = None,
        save_memorized: bool = False,
        save_final: bool = False,
        run_info: dict[str, Any] | None = None,
    ):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.model = model.to(self.device)
        self.primary = primary
        self.out_dir = Path(out_dir)
        self.eval_tasks = eval_tasks or {}
        self.val_masks = val_masks or {}
        self.old = old
        self.save_memorized = save_memorized
        self.save_final = save_final
        self.run_info = run_info or {}

        # Full-batch training data: concatenation of the train splits of all train tasks.
        self.x = torch.cat([t.tokens[t.train_idx] for t in train_tasks]).to(self.device)
        self.y = torch.cat([t.labels[t.train_idx] for t in train_tasks]).to(self.device)
        mod = torch.cat([t.moduli[t.train_idx] for t in train_tasks]).to(self.device)
        # The training set never changes, so the "answer is not in Z_n" mask is built once.
        classes = torch.arange(model.cfg.n_numbers, device=self.device)
        self.invalid = classes[None, :] >= mod[:, None]

        # Evaluators keep each task's tensors on the device.
        self.eval_primary = TaskEvaluator(primary, self.device)
        self.eval_extra = {name: TaskEvaluator(t, self.device) for name, t in self.eval_tasks.items()}

        # Optional torch.compile of the training forward pass. It fuses the small
        # kernels, which helps on GPUs. Evaluation uses the plain model.
        use_compile = cfg.compile is True or (cfg.compile == "auto" and self.device.type == "cuda")
        self.train_forward = torch.compile(self.model) if use_compile else self.model
        self.compiled = use_compile

        # AdamW with a short linear warm-up. A *fresh* optimiser for every run, so a
        # warm-started run differs from a scratch run only in its initial weights.
        # On CUDA the fused implementation updates all parameters in a single kernel.
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
            fused=self.device.type == "cuda",
        )
        warm = max(1, cfg.warmup_steps)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: min(1.0, (s + 1) / warm))

        # Online timing detectors (definitions in metrics.py).
        self.t_mem = SustainedCrossing(cfg.grok_threshold, patience=0)
        self.t_grok = SustainedCrossing(cfg.grok_threshold, patience=cfg.grok_patience)
        self.t_95 = SustainedCrossing(0.95, patience=cfg.grok_patience)
        self.t_forget = SustainedCrossing(0.5, patience=0, above=False)

        self.log: list[dict[str, Any]] = []
        self.repr_log: list[dict[str, Any]] = []
        self.extra_summary: dict[str, Any] = {}

    # ------------------------------------------------------------------ steps

    def _train_step(self) -> None:
        """One full-batch AdamW step.

        Nothing is copied back to the CPU here (no ``loss.item()``), so on a GPU the
        steps are queued without waiting. The loss is measured in ``_evaluate``.
        """
        self.model.train()  # no dropout / batch-norm, so train() and eval() behave the same
        logits = self.train_forward(self.x).masked_fill(self.invalid, float("-inf"))
        loss = F.cross_entropy(logits, self.y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.sched.step()

    def _evaluate(self, step: int) -> tuple[dict[str, Any], np.ndarray]:
        """Behavioural metrics of the current model. Returns (log row, MLP hidden)."""
        self.model.eval()
        t = self.primary
        loss, correct, hidden = self.eval_primary(self.model, return_hidden=True)
        row: dict[str, Any] = {
            "step": step,
            "lr": self.sched.get_last_lr()[0],
            "train_loss": float(loss[t.train_idx].mean()),
            "train_acc": float(correct[t.train_idx].mean()),
            "val_loss": float(loss[t.val_idx].mean()),
            "val_acc": float(correct[t.val_idx].mean()),
        }
        # Val accuracy restricted to subsets (agree / disagree / newtok).
        val = np.zeros(t.size, dtype=bool)
        val[t.val_idx] = True
        for name, mask in self.val_masks.items():
            row[f"val_acc_{name}"] = masked_mean(correct, val & mask)
        # Other tasks (e.g. the old group, for forgetting).
        for name, ev in self.eval_extra.items():
            _, c, _ = ev(self.model)
            row[f"{name}_train_acc"] = float(c[ev.task.train_idx].mean())
            row[f"{name}_val_acc"] = float(c[ev.task.val_idx].mean())
        row["param_norm"] = float(flat_params(self.model).norm())
        return row, hidden

    def _save(self, name: str) -> None:
        path = self.out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in self.model.state_dict().items()}, path)

    def _flush(self) -> None:
        write_csv(self.out_dir / "log.csv", self.log)
        write_csv(self.out_dir / "repr.csv", self.repr_log)

    # ------------------------------------------------------------------- main

    def fit(self) -> dict[str, Any]:
        cfg = self.cfg
        self.out_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        ref = None
        step = 0

        while True:
            row = None
            is_last = step >= cfg.max_steps
            if step % cfg.eval_every == 0 or is_last:
                row, hidden = self._evaluate(step)
                if ref is None:  # reference = the network at the start of this run
                    ref = make_reference(self.model, self.primary, hidden)
                self._update_trackers(step, row)

                stop = is_last or self._should_stop(step)
                if step % cfg.repr_every == 0 or stop:
                    self.repr_log.append({"step": step, **snapshot(self.model, self.primary, hidden, ref, self.old)})
                    if cfg.save_checkpoints:
                        self._save(f"ckpt/step_{step}.pt")
                    self._flush()
                if cfg.print_every and step % cfg.print_every == 0:
                    self._print(step, row)
                if stop:
                    self.log.append(row)
                    break

            # Optimisation step. On eval steps we also measure the effective learning
            # rate ||Δθ|| / ||θ|| of this update (Lyle et al., 2025).
            theta_before = flat_params(self.model) if row is not None else None
            self._train_step()
            if row is not None:
                delta = flat_params(self.model) - theta_before
                row["elr"] = float(delta.norm() / theta_before.norm())
                self.log.append(row)
            step += 1

        wall = time.time() - start_time
        if self.save_final:
            self._save("final.pt")
        summary = self._summary(step, wall)
        self._flush()
        write_json(self.out_dir / "summary.json", summary)
        return summary

    # ---------------------------------------------------------------- helpers

    def _update_trackers(self, step: int, row: dict[str, Any]) -> None:
        was_memorized = self.t_mem.confirmed
        self.t_mem.update(step, row["train_acc"])
        if self.t_mem.confirmed and not was_memorized:
            # Checkpoint for the "warm_memorized" condition (C2).
            self.extra_summary["val_acc_at_mem"] = row["val_acc"]
            if self.save_memorized:
                self._save("memorized.pt")
        self.t_grok.update(step, row["val_acc"])
        self.t_95.update(step, row["val_acc"])
        if "old_val_acc" in row:
            self.t_forget.update(step, row["old_val_acc"])

    def _should_stop(self, step: int) -> bool:
        if self.cfg.stop_after_grok is None or not self.t_grok.confirmed:
            return False
        return step >= self.t_grok.step + self.cfg.stop_after_grok

    def _print(self, step: int, row: dict[str, Any]) -> None:
        extra = f" old {row['old_val_acc']:.3f}" if "old_val_acc" in row else ""
        print(
            f"  step {step:>6d} | train {row['train_acc']:.3f} (loss {row['train_loss']:.3g}) "
            f"| val {row['val_acc']:.3f} (loss {row['val_loss']:.3g}){extra}",
            flush=True,
        )

    def _summary(self, steps_run: int, wall: float) -> dict[str, Any]:
        first, last = self.log[0], self.log[-1]
        last_repr = self.repr_log[-1] if self.repr_log else {}
        s: dict[str, Any] = {
            **self.run_info,
            "steps_run": steps_run,
            "budget": self.cfg.max_steps,
            "eval_every": self.cfg.eval_every,
            "T_mem": self.t_mem.step,
            "T_grok": self.t_grok.step,
            "T_95": self.t_95.step,
            "grokked": self.t_grok.confirmed,
            "censored": not self.t_grok.confirmed,
            "final_train_acc": last["train_acc"],
            "final_val_acc": last["val_acc"],
            "step0_train_acc": first["train_acc"],
            "step0_val_acc": first["val_acc"],
            "final_key_freqs": last_repr.get("key_freqs", ""),
            "final_fourier_sparsity": last_repr.get("fourier_sparsity"),
            "wall_time_s": wall,
            "ms_per_step": 1000 * wall / max(1, steps_run),
            "device": str(self.device),
            "compiled": self.compiled,
            **self.extra_summary,
        }
        for key in ("val_acc_agree", "val_acc_disagree", "val_acc_newtok"):
            if key in first:
                s[f"step0_{key}"] = first[key]
        if "old_val_acc" in first:
            s["old_val_acc_start"] = first["old_val_acc"]
            s["old_val_acc_end"] = last["old_val_acc"]
            s["forgetting"] = first["old_val_acc"] - last["old_val_acc"]
            s["T_forget"] = self.t_forget.step
        return s
