"""Integration: the bootstrap CIs run over a real ScoreArrays tuple — the harness spine's output.

A method's score_fn returns ScoreArrays(amaps, valids, masks, scores, labels, defects); the rigor
axis wraps the SAME metrics the harness reports (image_auroc over scores/labels, au_pro over
amaps/masks/valids) in Metrics.boot_ci. This proves the field wiring — the arrays index the image
axis together — end to end, no dataset or GPU needed.
"""
import numpy as np

from core.method import ScoreArrays
from core.metrics import BootCfg, Metrics


def _score_arrays(n=40, h=48, w=48, kind="good"):
    """A synthetic ScoreArrays: n/2 normals + n/2 anomalies with a central defect region."""
    rng = np.random.RandomState(0)
    half = n // 2
    masks = np.zeros((n, h, w), bool)
    masks[half:, 20:28, 20:28] = True
    valids = np.ones((n, h, w), bool)
    labels = np.array([0] * half + [1] * half)
    defects = np.array(["good"] * half + ["hole"] * half)
    amaps = masks.astype(float) + rng.rand(n, h, w) * 1e-3 if kind == "good" else rng.rand(n, h, w)
    scores = amaps.reshape(n, -1).max(1)
    return ScoreArrays(amaps, valids, masks, scores, labels, defects)


def test_boot_ci_over_scorearrays_image_auroc():
    sa = _score_arrays(kind="good")
    point, lo, hi = Metrics.boot_ci(Metrics.image_auroc, sa.scores, sa.labels, cfg=BootCfg(n_boot=200))
    assert lo <= point <= hi
    assert point > 0.9                                   # good maps -> high detection


def test_boot_ci_over_scorearrays_au_pro():
    sa = _score_arrays(kind="good")
    point, lo, hi = Metrics.boot_ci(Metrics.au_pro, sa.amaps, sa.masks, sa.valids, cfg=BootCfg(n_boot=100))
    assert 0.0 <= lo <= point <= hi <= 1.0
    assert point > 0.9                                   # good maps -> tight localization


def test_boot_delta_ci_over_scorearrays_beats_random():
    good, rand = _score_arrays(kind="good"), _score_arrays(kind="random")
    delta, lo, hi = Metrics.boot_delta_ci(
        Metrics.image_auroc, (rand.scores, rand.labels), (good.scores, good.labels),
        cfg=BootCfg(n_boot=300))
    assert delta > 0 and lo > 0                          # good genuinely beats random, CI clears 0
