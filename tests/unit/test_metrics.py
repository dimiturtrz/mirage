"""Unit tests for the eval metrics — the contribution, so they get verified.

Equivalence-class: AU-PRO on a perfect map -> ~1, random -> ~0.15 (the diagonal); image-AUROC on
perfect/reversed/single-class. No dataset needed.
"""
import numpy as np

from surfscan.evaluation.metrics import BootCfg, Metrics


def _sep_scores(n, rng, gap=0.0):
    """n/2 normals + n/2 anomalies with a mean gap; larger gap -> higher AUROC."""
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    scores = rng.random(n) + gap * labels
    return scores, labels


def _masks(n=20, h=64, w=64):
    masks = np.zeros((n, h, w), bool)
    masks[:, 28:36, 28:36] = True          # one central defect region per frame
    valids = np.ones((n, h, w), bool)
    return masks, valids


def test_au_pro_perfect():
    rng = np.random.RandomState(0)
    masks, valids = _masks()
    perfect = masks.astype(float) + rng.rand(*masks.shape) * 1e-3
    assert Metrics.au_pro(perfect, masks, valids) > 0.97


def test_au_pro_random_is_near_diagonal():
    rng = np.random.RandomState(0)
    masks, valids = _masks()
    randm = rng.rand(*masks.shape)
    assert 0.05 < Metrics.au_pro(randm, masks, valids) < 0.30      # ~0.15, the diagonal


def test_image_auroc():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert Metrics.image_auroc(scores, labels) == 1.0             # perfect separation
    assert Metrics.image_auroc(-scores, labels) == 0.0            # reversed
    assert np.isnan(Metrics.image_auroc(scores, np.array([0, 0, 0, 0])))   # single class -> undefined


def test_ece_perfect_calibration():
    # probs == binary targets -> confidence matches accuracy in every bin -> ECE ~ 0
    rng = np.random.RandomState(0)
    valids = np.ones((5, 32, 32), bool)
    targets = rng.rand(5, 32, 32) < 0.3
    assert Metrics.ece(targets.astype(float), targets, valids) < 1e-6


def test_ece_overconfident_under_shift():
    # predict 0.95 everywhere but ~1% of pixels are actually defective -> big confidence-accuracy gap
    valids = np.ones((5, 32, 32), bool)
    targets = np.zeros((5, 32, 32), bool)
    targets[:, :3, :3] = True
    assert Metrics.ece(np.full((5, 32, 32), 0.95), targets, valids) > 0.8


def test_ece_calibrated_but_uncertain():
    # predict 0.5 everywhere with ~50% positives -> confidence == accuracy -> low ECE
    rng = np.random.RandomState(1)
    valids = np.ones((6, 32, 32), bool)
    targets = rng.rand(6, 32, 32) < 0.5
    assert Metrics.ece(np.full((6, 32, 32), 0.5), targets, valids) < 0.05


def test_metrics_nan_on_no_valid_pixels():
    # no valid pixels anywhere -> both metrics are undefined, not a crash
    a = np.zeros((2, 8, 8))
    empty = np.zeros((2, 8, 8), bool)
    assert np.isnan(Metrics.ece(a, empty, empty))
    assert np.isnan(Metrics.au_pro(a, empty, empty))


# --- bootstrap CIs (the rigor axis: uncertainty from ONE run's test set, no retrain) ---

def test_boot_ci_point_is_full_sample_metric():
    # the point estimate is the metric on the un-resampled arrays, exactly
    rng = np.random.default_rng(0)
    scores, labels = _sep_scores(40, rng, gap=0.5)
    point, lo, hi = Metrics.boot_ci(Metrics.image_auroc, scores, labels, cfg=BootCfg(n_boot=200, rng=rng))
    assert point == Metrics.image_auroc(scores, labels)
    assert lo <= point <= hi                                   # the point sits inside its interval


def test_boot_ci_brackets_and_orders():
    rng = np.random.default_rng(1)
    scores, labels = _sep_scores(60, rng, gap=1.0)             # strong separation -> AUROC high
    point, lo, hi = Metrics.boot_ci(Metrics.image_auroc, scores, labels, cfg=BootCfg(n_boot=300, rng=rng))
    assert 0.0 <= lo <= hi <= 1.0
    assert point > 0.8


def test_boot_ci_width_shrinks_with_n():
    # power comes from the test-set N: a bigger test set -> a tighter interval (the whole point)
    rng = np.random.default_rng(2)
    cfg = BootCfg(n_boot=300, rng=np.random.default_rng(7))
    small = Metrics.boot_ci(Metrics.image_auroc, *_sep_scores(20, rng, gap=0.6), cfg=cfg)
    big = Metrics.boot_ci(Metrics.image_auroc, *_sep_scores(200, rng, gap=0.6),
                          cfg=BootCfg(n_boot=300, rng=np.random.default_rng(7)))
    assert (big[2] - big[1]) < (small[2] - small[1])


def test_boot_ci_reproducible_under_fixed_seed():
    # same BootCfg seed -> identical interval (a bootstrap seed, deterministic by design)
    scores, labels = _sep_scores(50, np.random.default_rng(3), gap=0.7)
    a = Metrics.boot_ci(Metrics.image_auroc, scores, labels, cfg=BootCfg(n_boot=200, rng=np.random.default_rng(9)))
    b = Metrics.boot_ci(Metrics.image_auroc, scores, labels, cfg=BootCfg(n_boot=200, rng=np.random.default_rng(9)))
    assert a == b


def test_boot_ci_nan_when_metric_undefined():
    # single-class labels -> AUROC undefined on every resample -> point + interval all NaN, no crash
    rng = np.random.default_rng(4)
    scores = rng.random(30)
    labels = np.zeros(30, int)
    point, lo, hi = Metrics.boot_ci(Metrics.image_auroc, scores, labels, cfg=BootCfg(n_boot=100, rng=rng))
    assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)


def test_boot_delta_ci_detects_real_improvement():
    # B is genuinely better -> delta > 0 and the paired CI excludes 0
    rng = np.random.default_rng(5)
    scores_a, labels = _sep_scores(80, rng, gap=0.2)
    scores_b = scores_a + 0.6 * labels                         # B lifts the anomalies further
    delta, lo, hi = Metrics.boot_delta_ci(Metrics.image_auroc, (scores_a, labels), (scores_b, labels),
                                          cfg=BootCfg(n_boot=400, rng=rng))
    assert delta > 0
    assert lo > 0                                              # CI clears zero -> honest "B beats A"


def test_boot_delta_ci_brackets_zero_when_equal():
    # A and B identical -> delta exactly 0 and the interval straddles 0 (no spurious difference)
    rng = np.random.default_rng(6)
    scores, labels = _sep_scores(80, rng, gap=0.4)
    delta, lo, hi = Metrics.boot_delta_ci(Metrics.image_auroc, (scores, labels), (scores, labels),
                                          cfg=BootCfg(n_boot=200, rng=rng))
    assert delta == 0.0
    assert lo <= 0.0 <= hi


# --- macro (mean-of-categories) bootstrap: the CI must match the reported macro point, not a pooled one ---

def test_boot_macro_ci_point_is_mean_of_categories():
    # the macro point is the plain mean of per-category AUROCs — NOT the pooled/micro AUROC
    rng = np.random.default_rng(10)
    per_cat = [_sep_scores(40, rng, gap=g) for g in (0.2, 0.8, 1.2)]
    point, lo, hi = Metrics.boot_macro_ci(Metrics.image_auroc, per_cat, cfg=BootCfg(n_boot=200, rng=rng))
    expected = np.mean([Metrics.image_auroc(s, lab) for s, lab in per_cat])
    assert abs(point - expected) < 1e-9
    assert lo <= point <= hi


def test_boot_macro_differs_from_pooled_when_sizes_uneven():
    # macro (equal category weight) != micro/pooled (image-weighted) when categories differ in size/skill
    rng = np.random.default_rng(11)
    big_easy = _sep_scores(200, rng, gap=1.5)                # many images, easy
    small_hard = _sep_scores(20, rng, gap=0.0)               # few images, chance-level
    macro, _, _ = Metrics.boot_macro_ci(Metrics.image_auroc, [big_easy, small_hard],
                                        cfg=BootCfg(n_boot=100, rng=rng))
    pooled = Metrics.image_auroc(np.concatenate([big_easy[0], small_hard[0]]),
                                 np.concatenate([big_easy[1], small_hard[1]]))
    assert abs(macro - pooled) > 0.05                        # the two aggregates genuinely diverge


def test_boot_macro_delta_ci_detects_real_macro_gap():
    # B beats A in every category -> macro delta > 0 and the paired CI clears 0
    rng = np.random.default_rng(12)
    pc_a = [_sep_scores(50, rng, gap=g) for g in (0.1, 0.3, 0.5)]
    pc_b = [(s + 0.6 * lab, lab) for s, lab in pc_a]         # B lifts anomalies in each category
    delta, lo, hi = Metrics.boot_macro_delta_ci(Metrics.image_auroc, pc_a, pc_b,
                                                cfg=BootCfg(n_boot=300, rng=rng))
    assert delta > 0 and lo > 0
