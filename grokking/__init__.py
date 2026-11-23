"""Grokking across group sizes (Z_p -> Z_q).

Modules
-------
config          dataclass configs (model, task, training, experiment grid)
data            modular-arithmetic datasets; symbols kept separate from group elements
model           1-layer transformer (Nanda et al. 2023 style)
metrics         accuracy / loss evaluation and the T_mem / T_grok / T_forget definitions
representation  Fourier, CKA, dead-neuron and weight-norm metrics
train           full-batch training loop with logging and stopping
conditions      the phase-B conditions C0..C6 of the plan
experiments     phase A (source) and phase B (transfer) runs
grid            expand a config into jobs and run them (optionally in parallel)
analysis        load results into tables (timings, transfer ratios, statistics)
plotting        figures
cli             command-line interface used by run.py
"""

__version__ = "0.1.0"
