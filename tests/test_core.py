"""Fast unit tests for data, model, metrics and representation code (< 10 s total).

Run with:  .venv/bin/python -m pytest -q
"""

import numpy as np
import pytest
import torch

from grokking.config import ExperimentConfig, ModelConfig, apply_override
from grokking.data import ModularTask, eq_token
from grokking.metrics import SustainedCrossing, first_sustained_crossing, mask_logits
from grokking.model import GrokTransformer
from grokking.representation import (
    fourier_basis,
    fourier_power,
    fourier_sparsity,
    frequency_fit_r2,
    frequency_match_distance,
    key_frequencies,
    linear_cka,
    subgroup_fraction,
)

V = 40  # n_numbers used in the tests


# --------------------------------------------------------------------------- data


def test_task_labels_are_modular_sums():
    t = ModularTask.build(7, V, "add", 0.5, split_seed=0)
    assert t.size == 49
    assert (t.labels.numpy() == (t.a + t.b) % 7).all()
    assert (t.tokens[:, 2] == eq_token(0, V)).all()  # default slot: "=p"


def test_vocabulary_layout():
    """Numbers 0..V-1 (token id = number), then "=p" = V and "=q" = V + 1."""
    assert ModelConfig(n_numbers=V).vocab_size == V + 2
    assert eq_token(0, V) == V and eq_token(1, V) == V + 1
    with pytest.raises(ValueError):
        eq_token(2, V)
    t = ModularTask.build(5, V, eq_slot=1)
    i = 3 * 5 + 4  # the pair (a=3, b=4)
    assert t.tokens[i].tolist() == [3, 4, V + 1] and int(t.labels[i]) == (3 + 4) % 5


def test_split_is_a_partition_and_deterministic():
    t1 = ModularTask.build(11, V, train_frac=0.3, split_seed=3)
    t2 = ModularTask.build(11, V, train_frac=0.3, split_seed=3)
    assert np.array_equal(t1.train_idx, t2.train_idx)
    assert len(set(t1.train_idx) & set(t1.val_idx)) == 0
    assert len(t1.train_idx) + len(t1.val_idx) == 121
    assert len(t1.train_idx) == round(0.3 * 121)


def test_permuted_symbols_keep_the_task_consistent():
    t = ModularTask.build(13, V, perm_seed=1)
    assert not t.is_identity_map
    assert sorted(t.symbols) == list(range(13))  # permutation of the same tokens
    # inputs and labels are the images of the elements under the same map
    assert (t.tokens[:, 0].numpy() == t.symbols[t.a]).all()
    assert (t.labels.numpy() == t.symbols[t.c]).all()


def test_overlap_masks_partition_and_agreement():
    p, q = 11, 13
    t = ModularTask.build(q, V)
    m = t.overlap_masks(p)
    total = m["agree"].astype(int) + m["disagree"] + m["newtok"]
    assert (total == 1).all()
    # identity map: old and new answers agree exactly when a + b < p (both < p)
    expected = (t.a < p) & (t.b < p) & ((t.a + t.b) < p)
    assert np.array_equal(m["agree"], expected)


# -------------------------------------------------------------------------- model


def test_model_shapes_and_hidden():
    cfg = ModelConfig(n_numbers=V, d_model=32, n_heads=4, d_mlp=64)
    model = GrokTransformer(cfg)
    x = ModularTask.build(5, V).tokens
    logits, hidden = model(x, return_hidden=True)
    assert logits.shape == (25, V)
    assert hidden.shape == (25, 64)


def _full_sequence_forward(model, tokens):
    """Reference: run every block on all positions (causal), then keep the last one."""
    x = model.embed(tokens) + model.pos
    for block in model.blocks:
        x, hidden = block(x, last_only=False)
    return model.unembed(x[:, -1]), hidden[:, -1]


@pytest.mark.parametrize("n_layers", [1, 2])
def test_last_position_shortcut_is_exact(n_layers):
    """The fast forward (last block on the last position only) equals the full computation."""
    torch.manual_seed(0)
    model = GrokTransformer(ModelConfig(n_numbers=V, d_model=32, n_heads=4, d_mlp=64, n_layers=n_layers))
    tokens = ModularTask.build(11, V).tokens
    fast_logits, fast_hidden = model(tokens, return_hidden=True)
    ref_logits, ref_hidden = _full_sequence_forward(model, tokens)
    torch.testing.assert_close(fast_logits, ref_logits)
    torch.testing.assert_close(fast_hidden, ref_hidden)


def test_attention_is_causal():
    """Changing a later token never changes the outputs at earlier positions."""
    torch.manual_seed(0)
    model = GrokTransformer(ModelConfig(n_numbers=V, d_model=32, n_heads=4, d_mlp=64, n_layers=2))
    tokens = ModularTask.build(11, V).tokens[:20]
    changed = tokens.clone()
    changed[:, 1] = (changed[:, 1] + 1) % 11  # change token b

    def all_positions(t):
        x = model.embed(t) + model.pos
        for block in model.blocks:
            x, _ = block(x)
        return x

    a, b = all_positions(tokens), all_positions(changed)
    torch.testing.assert_close(a[:, 0], b[:, 0])  # position of a: unaffected by b
    assert not torch.allclose(a[:, 1], b[:, 1])  # position of b: affected
    assert not torch.allclose(a[:, 2], b[:, 2])  # position of "=": affected


def test_mask_logits_restricts_to_group():
    logits = torch.zeros(2, 10)
    out = mask_logits(logits, torch.tensor([3, 7]))
    assert torch.isinf(out[0, 3:]).all() and torch.isfinite(out[0, :3]).all()
    assert torch.isinf(out[1, 7:]).all() and torch.isfinite(out[1, :7]).all()


# ------------------------------------------------------------------------ metrics


def test_sustained_crossing_ignores_spikes():
    steps = [0, 50, 100, 150, 200, 250, 300]
    vals = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]  # spike at 50, real crossing at 150
    assert first_sustained_crossing(steps, vals, 0.99, patience=100) == 150
    assert first_sustained_crossing(steps, vals, 0.99, patience=0) == 50
    assert first_sustained_crossing(steps, vals, 0.99, patience=1000) is None


def test_sustained_crossing_below():
    tr = SustainedCrossing(0.5, patience=0, above=False)
    for s, v in [(0, 1.0), (50, 0.8), (100, 0.4)]:
        tr.update(s, v)
    assert tr.step == 100


# ----------------------------------------------------------------- representation


def _clock_embedding(n, freqs, d=16, seed=0):
    """Synthetic embedding built from a few circles of Z_n, like a grokked network."""
    rng = np.random.default_rng(seed)
    X = fourier_basis(n, freqs)[:, 1:]  # drop the constant column
    return X @ rng.normal(size=(X.shape[1], d))


def test_key_frequencies_recover_planted_circles():
    E = _clock_embedding(31, [3, 7, 12])
    power = fourier_power(E)
    assert key_frequencies(power, mass=0.99) == [3, 7, 12]
    assert fourier_sparsity(power, top=3) == pytest.approx(1.0)


def test_frequency_fit_r2():
    E = _clock_embedding(31, [4, 9])
    assert frequency_fit_r2(E, 31, [4, 9]) == pytest.approx(1.0)
    noise = np.random.default_rng(1).normal(size=(31, 16))
    assert frequency_fit_r2(noise, 31, [4, 9]) < 0.5


def test_subgroup_and_matching():
    assert subgroup_fraction([1, 7, 13, 14], 91) == 0.75  # gcd>1 for 7, 13, 14
    assert subgroup_fraction([1, 2, 3], 97) == 0.0
    assert frequency_match_distance([10], 100, [10], 100) == 0.0


def test_linear_cka_bounds():
    X = np.random.default_rng(0).normal(size=(50, 8))
    assert linear_cka(X, X) == pytest.approx(1.0)
    assert linear_cka(X, 3 * X @ np.linalg.qr(np.random.default_rng(1).normal(size=(8, 8)))[0]) == pytest.approx(1.0)


# ------------------------------------------------------------------------- config


def test_config_overrides_and_validation(tmp_path):
    cfg = ExperimentConfig()
    apply_override(cfg, "train_transfer.max_steps=123")
    apply_override(cfg, "seeds=[7]")
    assert cfg.train_transfer.max_steps == 123 and cfg.seeds == [7]
    cfg.save(tmp_path / "c.json")
    again = ExperimentConfig.load(tmp_path / "c.json", ["task.train_frac=0.5"])
    assert again.task.train_frac == 0.5 and again.targets[97][0] == 59
    with pytest.raises(KeyError):
        apply_override(cfg, "train_transfer.not_a_field=1")
    cfg.targets = {97: [200]}
    with pytest.raises(ValueError):
        cfg.validate()
