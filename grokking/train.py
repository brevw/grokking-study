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
from .metrics import SustainedCrossing, TaskEvaluator
from .model import GrokTransformer, flat_params
from .representation import OldTaskInfo, make_reference, snapshot
from .utils import resolve_device, write_csv, write_json


def masked_loss(model: GrokTransformer, x: torch.Tensor, y: torch.Tensor, invalid: torch.Tensor) -> torch.Tensor:
    """Training loss: cross-entropy over the valid answers only (``invalid`` = answer not in Z_n).

    A plain function so that ``torch.compile`` can fuse the whole forward pass, the
    mask and the cross-entropy (and their backward) into a few GPU kernels.
    """
    logits = model(x).masked_fill(invalid, float("-inf"))
    return F.cross_entropy(logits, y)


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
        self.eval_primary = TaskEvaluator(primary, self.device, val_subsets=self.val_masks)
        self.eval_extra = {name: TaskEvaluator(t, self.device) for name, t in self.eval_tasks.items()}

        # GPU speed-ups (see TrainConfig): torch.compile of the training forward pass,
        # and replaying the whole training step as one CUDA graph. Evaluation always
        # uses the plain model (it shares the same weights).
        on_cuda = self.device.type == "cuda"
        self.compiled = cfg.compile is True or (cfg.compile == "auto" and on_cuda)
        self.use_graph = cfg.cuda_graph is True or (cfg.cuda_graph == "auto" and on_cuda)
        if self.use_graph and not on_cuda:
            raise ValueError("cuda_graph=True needs a CUDA device")
        if self.compiled:
            # A worker process trains many runs with different batch shapes. Start each
            # run from a clean compile cache: otherwise torch.compile switches to slower
            # dynamic-shape kernels, and silently falls back to eager after 8 recompiles.
            # Recompiling is quick (the generated kernels are cached on disk).
            torch._dynamo.reset()
            self.train_loss = torch.compile(masked_loss, dynamic=False)
        else:
            self.train_loss = masked_loss

        # AdamW with a short linear warm-up. A *fresh* optimiser for every run, so a
        # warm-started run differs from a scratch run only in its initial weights.
        # On CUDA the fused implementation updates all parameters in a single kernel.
        # capturable=True keeps AdamW's step counter on the GPU, so it keeps counting
        # inside the CUDA graph (otherwise the bias correction would be frozen).
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
            fused=on_cuda,
            capturable=self.use_graph,
        )
        warm = max(1, cfg.warmup_steps)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: min(1.0, (s + 1) / warm))
        # The graph bakes in the learning rate, so it is captured once the warm-up is
        # over (lr constant from then on). The first steps run normally, on a side
        # stream as CUDA graph capture requires.
        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_after = max(3, warm)
        self.side_stream = torch.cuda.Stream(self.device) if self.use_graph else None

        # Online timing detectors (definitions in metrics.py).
        self.t_mem = SustainedCrossing(cfg.grok_threshold, patience=0)
        self.t_grok = SustainedCrossing(cfg.grok_threshold, patience=cfg.grok_patience)
        self.t_95 = SustainedCrossing(0.95, patience=cfg.grok_patience)
        self.t_forget = SustainedCrossing(0.5, patience=0, above=False)

        self.log: list[dict[str, Any]] = []
        self.repr_log: list[dict[str, Any]] = []
        self.extra_summary: dict[str, Any] = {}

    # ------------------------------------------------------------------ steps

    def _train_step(self, step: int) -> None:
        """One full-batch AdamW step (``step`` = number of steps done before it).

        Nothing is copied back to the CPU here (no ``loss.item()``), so on a GPU the
        steps are queued without waiting. The loss is measured in ``_evaluate``.
        """
        if self.graph is not None:
            self.graph.replay()  # the recorded step, on the same weight tensors
        elif self.use_graph:
            current = torch.cuda.current_stream(self.device)
            self.side_stream.wait_stream(current)
            with torch.cuda.stream(self.side_stream):
                self._step_ops()
            current.wait_stream(self.side_stream)
        else:
            self._step_ops()
        self.sched.step()
        if self.use_graph and self.graph is None and step + 1 >= self.capture_after:
            self._capture()

    def _step_ops(self, capturing: bool = False) -> None:
        """Forward, masked cross-entropy, backward, AdamW update."""
        loss = self.train_loss(self.model, self.x, self.y, self.invalid)
        if not capturing:  # during capture the grads are already None (see _capture)
            self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

    def _capture(self) -> None:
        """Record one training step as a CUDA graph (nothing is executed while recording).

        Grads are set to None first, so backward() allocates them inside the graph and
        every replay overwrites them (no accumulation across steps).
        """
        assert self.sched.get_last_lr()[0] == self.cfg.lr, "capture only after the warm-up"
        torch.cuda.synchronize(self.device)
        self.opt.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step_ops(capturing=True)

    def _evaluate(self, step: int) -> dict[str, Any]:
        """Behavioural metrics of the current model (one log row).

        Everything is computed on the device; the numbers of all tasks are copied to
        the CPU together, in a single transfer.
        """
        self.model.eval()
        evaluators = [(self.eval_primary, "", True)]
        evaluators += [(ev, f"{name}_", False) for name, ev in self.eval_extra.items()]
        means = torch.cat([ev(self.model) for ev, _, _ in evaluators]).tolist()
        row: dict[str, Any] = {"step": step, "lr": self.sched.get_last_lr()[0]}
        start = 0
        for ev, prefix, with_loss in evaluators:
            row.update(ev.to_row(means[start : start + len(ev.subsets)], prefix, with_loss))
            start += len(ev.subsets)
        row["param_norm"] = float(flat_params(self.model).norm())
        return row

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
                row = self._evaluate(step)
                self._update_trackers(step, row)

                stop = is_last or self._should_stop(step)
                if step % cfg.repr_every == 0 or stop:
                    hidden = self.eval_primary.hidden(self.model)  # stays on the device
                    if ref is None:  # reference = the network at the start of this run (step 0)
                        ref = make_reference(self.model, self.primary, hidden)
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
            self._train_step(step)
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
            "cuda_graph": self.use_graph,
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
