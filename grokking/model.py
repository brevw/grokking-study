"""A one-layer decoder-only (causal) transformer, following Nanda et al. (2023).

Architecture (Nanda et al. 2023, Section 3 and Appendix A):

* token embedding + learned positional embedding
* causal multi-head self-attention, no biases
* ReLU MLP with biases
* no LayerNorm, untied embedding / unembedding
* the prediction is read at the last position (the "=" token)

The model is autoregressive in the usual sense: position t only attends to positions
<= t (causal mask). As in Nanda et al., we only train and evaluate the prediction at
the last position: a, b = -> c. Predictions at the other positions (guessing b from a)
are meaningless for this task.

Speed: only the last position's output is ever used, so the **last block computes
only that position**. Its query comes from the last token alone (keys and values from
all tokens), and its MLP runs on one position instead of n_ctx. The result is exactly
the same as running the full sequence and keeping the last position. It does about a
third of the MLP work (the MLP is most of the FLOPs). ``tests/test_core.py`` checks that
it is equal to the full computation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


class Attention(nn.Module):
    """Multi-head self-attention without biases (W_Q, W_K, W_V fused into one matrix)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)  # [W_Q; W_K; W_V] of all heads
        self.out = nn.Linear(d_model, d_model, bias=False)  # W_O

    def forward(self, x: torch.Tensor, last_only: bool = False) -> torch.Tensor:
        """``x``: (B, T, D). Returns (B, T, D), or (B, 1, D) if ``last_only``.

        last_only=False: standard causal attention for every position.
        last_only=True:  only the last position asks a query (it may see every key).
        """
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        if last_only:
            q = q[:, -1:]
        # (B, L, D) -> (B, heads, L, d_head)
        q = q.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # softmax(q kᵀ / sqrt(d_head)) v. With last_only, the single query sits at the final
        # position, so it may attend to all keys and no mask is needed. Do NOT use
        # is_causal=True there: when there are fewer queries than keys, PyTorch aligns the
        # causal mask to the top-left, and the query would only see the first key.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=not last_only)
        return self.out(y.transpose(1, 2).reshape(B, -1, D))


class Block(nn.Module):
    """Residual attention + residual ReLU MLP, no normalisation."""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.attn = Attention(d_model, n_heads)
        self.mlp_in = nn.Linear(d_model, d_mlp)  # W_in, b_in
        self.mlp_out = nn.Linear(d_mlp, d_model)  # W_out, b_out

    def forward(self, x: torch.Tensor, last_only: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (residual stream, MLP activations). With ``last_only`` both have length 1."""
        attn = self.attn(x, last_only)
        x = (x[:, -1:] if last_only else x) + attn
        hidden = F.relu(self.mlp_in(x))
        x = x + self.mlp_out(hidden)
        return x, hidden


class GrokTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)  # W_E: numbers, then "=A", "=B"
        self.pos = nn.Parameter(torch.empty(cfg.n_ctx, cfg.d_model))  # W_pos
        self.blocks = nn.ModuleList(Block(cfg.d_model, cfg.n_heads, cfg.d_mlp) for _ in range(cfg.n_layers))
        self.unembed = nn.Linear(cfg.d_model, cfg.n_numbers, bias=False)  # W_U: number tokens only
        self.reset_embeddings()

    def reset_embeddings(self, generator: torch.Generator | None = None) -> None:
        """Draw W_E, W_pos and W_U from N(0, 1/d_model) (std = 1/sqrt(d_model)).

        Called (1) in ``__init__`` to initialise these three matrices (attention and MLP
        keep PyTorch's default nn.Linear init), and (2) by the ``warm_reinit_embed``
        condition (C4), which keeps the attention/MLP of the grokked network but resets
        the token interface.
        """
        std = 1.0 / math.sqrt(self.cfg.d_model)
        with torch.no_grad():
            self.embed.weight.normal_(0.0, std, generator=generator)
            self.pos.normal_(0.0, std, generator=generator)
            self.unembed.weight.normal_(0.0, std, generator=generator)

    def forward(self, tokens: torch.Tensor, return_hidden: bool = False):
        """``tokens``: (B, n_ctx) token ids -> logits (B, n_numbers) at the last position.

        ``return_hidden=True`` also returns the last block's MLP activations at the last
        position, shape (B, d_mlp). Training never needs them. The evaluation uses them
        (every ``eval_every`` steps) for dead-neuron counts and activation CKA.
        Logits are *not* masked to Z_n here; that is done by ``metrics.mask_logits``.
        """
        x = self.embed(tokens) + self.pos
        for block in self.blocks[:-1]:  # earlier blocks (only if n_layers > 1): all positions
            x, _ = block(x)
        x, hidden = self.blocks[-1](x, last_only=True)  # last block: last position only
        logits = self.unembed(x[:, 0])
        return (logits, hidden[:, 0]) if return_hidden else logits

    @property
    def number_embeddings(self) -> torch.Tensor:
        """Rows of W_E for the number tokens, (n_numbers, d_model). Used by the Fourier metrics."""
        return self.embed.weight[: self.cfg.n_numbers]


def flat_params(model: nn.Module) -> torch.Tensor:
    """All weights as one detached 1-D vector.

    Used for the parameter norm, the effective learning rate ||Δθ|| / ||θ||, and the
    distance from the starting weights (train.py, representation.py).
    """
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])
