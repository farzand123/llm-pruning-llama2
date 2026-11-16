"""Verify, with a NumPy transformer, the index arithmetic that the PyTorch head-pruning code relies on.

Physically deleting heads (slicing rows of W_Q/W_K/W_V and columns of W_O by
``head_channel_indices``) must give exactly the output of the original layer with the removed
heads' contributions switched off. The PyTorch version of this check lives in test_torch_tiny.py.
"""
import numpy as np

from llmprune import selection as sel


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def attention(x, wq, wk, wv, wo, n_heads, hd):
    t = x.shape[0]
    q, k, v = ((x @ w.T).reshape(t, n_heads, hd) for w in (wq, wk, wv))
    scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd)
    scores = np.where(np.triu(np.ones((t, t), bool), 1)[None], -np.inf, scores)
    ctx = np.einsum("hts,shd->thd", softmax(scores), v).reshape(t, n_heads * hd)
    return ctx @ wo.T


def make_layer(rng, d, n_heads, hd):
    return {"wq": rng.normal(size=(n_heads * hd, d)), "wk": rng.normal(size=(n_heads * hd, d)),
            "wv": rng.normal(size=(n_heads * hd, d)), "wo": rng.normal(size=(d, n_heads * hd)),
            "w1": rng.normal(size=(2 * d, d)), "w2": rng.normal(size=(d, 2 * d))}


def forward(x, layers, heads):
    hd = layers[0]["wq"].shape[0] // heads[0]
    for L, h in zip(layers, heads):
        x = x + attention(x, L["wq"], L["wk"], L["wv"], L["wo"], h, hd)
        x = x + np.maximum(x @ L["w1"].T, 0) @ L["w2"].T
    return x


def test_physical_head_removal_equals_masking():
    rng = np.random.default_rng(0)
    d, n_heads, hd, n_layers, t = 16, 6, 4, 3, 7
    layers = [make_layer(rng, d, n_heads, hd) for _ in range(n_layers)]
    x = rng.normal(size=(t, d))

    scores = rng.random((n_layers, n_heads))
    remove, info = sel.select_heads_to_remove(scores, 0.5, min_heads_per_layer=1)
    keep = sel.keep_lists(remove)

    masked = []
    for L, row in zip(layers, remove):
        M = dict(L)
        wo = L["wo"].copy()
        for h in np.flatnonzero(row):
            wo[:, h * hd:(h + 1) * hd] = 0.0          # head h contributes nothing
        M["wo"] = wo
        masked.append(M)
    expected = forward(x, masked, [n_heads] * n_layers)

    pruned = []
    for L, k in zip(layers, keep):
        idx = sel.head_channel_indices(k, hd)
        pruned.append({**L, "wq": L["wq"][idx], "wk": L["wk"][idx], "wv": L["wv"][idx], "wo": L["wo"][:, idx]})
    got = forward(x, pruned, [len(k) for k in keep])

    assert np.allclose(expected, got, atol=1e-10)
    assert [len(k) for k in keep] == info["heads_kept_per_layer"]
    removed_params = int(remove.sum()) * 4 * d * hd
    assert sum(L["wq"].size + L["wk"].size + L["wv"].size + L["wo"].size for L in layers) - \
        sum(L["wq"].size + L["wk"].size + L["wv"].size + L["wo"].size for L in pruned) == removed_params


def test_layer_removal_equals_identity_skip():
    rng = np.random.default_rng(1)
    d, h, hd, t = 8, 2, 4, 5
    layers = [make_layer(rng, d, h, hd) for _ in range(5)]
    x = rng.normal(size=(t, d))
    bi = rng.random(5)
    removed = sel.select_layers_to_remove(bi, 0.4)
    kept = [L for i, L in enumerate(layers) if i not in removed]
    # skipping a residual block is the identity, so deleting it must equal running the others only
    assert np.allclose(forward(x, kept, [h] * len(kept)), forward(x, [layers[i] for i in range(5) if i not in removed], [h] * 3))
    assert len(kept) == 3 and len(removed) == 2
