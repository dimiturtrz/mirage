"""Unit tests for the PatchCore coreset core — the pure kNN-selection logic (no backbone, runs on cpu).

greedy_coreset is k-center greedy: it returns `n` distinct bank points and, on a clearly-clustered
set, must pick from both clusters (the farthest-point rule) rather than collapsing into one. The
ImageNet backbone / feature embedding is a network dependency, tested separately from this math.
"""
import torch

from surfscan.models.coreset import FitCfg, bank_linear, bank_nn_dist, greedy_coreset


def test_greedy_coreset_shape_and_distinct():
    feats = torch.randn(50, 8)
    bank = greedy_coreset(feats, 10, seed=0)
    assert bank.shape == (10, 8)
    rows = {tuple(r.tolist()) for r in bank}
    assert len(rows) == 10                       # farthest-point never repeats a pick


def test_greedy_coreset_spans_both_clusters():
    a = torch.zeros(20, 2)
    b = torch.full((20, 2), 100.0)
    feats = torch.cat([a, b])
    bank = greedy_coreset(feats, 4, seed=0)
    near = (bank.norm(dim=1) < 1).sum()
    far = (bank.norm(dim=1) > 50).sum()
    assert near >= 1 and far >= 1                 # covers both, not one cluster


def test_fitcfg_defaults():
    cfg = FitCfg()
    assert cfg.coreset == 0.1 and cfg.method == "greedy" and cfg.seed == 0


def test_bank_nn_dist_matches_cdist():
    # the matmul-offload form must be identical to the direct cdist().min() it replaces
    feats = torch.randn(37, 16)
    bank = torch.randn(64, 16)
    ref = torch.cdist(feats, bank).min(1).values
    got = bank_nn_dist(feats, bank)
    assert got.shape == (37,)
    assert torch.allclose(got, ref, atol=1e-4)


def test_bank_linear_is_the_distance_field():
    # Linear(W=-2B, bias=||b||^2) + ||x||^2 reproduces squared distance ||x-b||^2 exactly
    feats = torch.randn(5, 8)
    bank = torch.randn(12, 8)
    w, b = bank_linear(bank)
    assert w.shape == (12, 8) and b.shape == (12,)
    field = feats @ w.T + b + (feats * feats).sum(1, keepdim=True)   # (N,M) reconstructed sq-dist
    assert torch.allclose(field, torch.cdist(feats, bank) ** 2, atol=1e-3)
