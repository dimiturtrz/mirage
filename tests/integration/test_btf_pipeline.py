"""Integration: the geometry pipeline — FpfhBank.fit + the parallel batch score_maps -> scoring ->
harness.aggregate — on a synthetic cloud. No torchvision, no dataset, so it runs in CI and pins the
qfm speedup seam (parallel FPFH must land in a valid aggregate). Structural asserts only: toy FPFH on
a 16x16 grid is not a reliable detector, so we check the chain is wired + values are sane, not a
performance threshold (that's the e2e tier's job, on real data).
"""
import numpy as np

from surfscan.evaluation import scoring
from surfscan.evaluation.harness import Harness
from surfscan.models.fpfh_bank import FpfhBank


def _cloud(g=16, bump=0.0, seed=0):
    ys, xs = np.mgrid[0:g, 0:g].astype(np.float32)
    z = 50.0 + bump * np.exp(-((xs - g / 2) ** 2 + (ys - g / 2) ** 2) / 6.0)   # a central bump = the "defect"
    xyz = np.stack([xs - g / 2, ys - g / 2, z], -1).astype(np.float32)
    return xyz, np.ones((g, g), bool)


def test_btf_pipeline_produces_valid_aggregate():
    bank = FpfhBank(device="cpu", per_sample=200, coreset=1.0, seed=0).fit([_cloud() for _ in range(3)])

    samples = [_cloud(), _cloud(), _cloud(bump=4.0), _cloud(bump=4.0)]   # 2 good, 2 bumped
    amaps = bank.score_maps(samples)                                     # the parallel batch path
    assert amaps.shape == (4, 16, 16)

    valids = np.stack([v for _, v in samples]).astype(bool)
    masks = np.zeros_like(valids)
    masks[2:, 6:10, 6:10] = True                                        # bumped frames carry a gt region
    scores = scoring.Scoring.image_scores(amaps, valids)
    labels = np.array([0, 0, 1, 1])
    defects = np.array(["good", "good", "bump", "bump"])

    def fit(_c):
        return bank

    def score(_s, _c):
        return amaps, valids, masks, scores, labels, defects

    res = Harness.aggregate("btf_fpfh", fit, score, ["a"])
    assert set(res) == {"method", "per_category", "mean", "ece", "per_defect"}
    assert 0.0 <= res["mean"]["au_pro"] <= 1.0
    assert not np.isnan(res["mean"]["img_auroc"])                       # chain wired, values sane
    assert (amaps >= 0).all()                                          # NN distance is non-negative
