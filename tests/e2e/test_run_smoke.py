"""E2E smokes — the real methods end-to-end on ONE small category (real backbone/store/GPU).

The pre-window de-risk: proves the actual pipelines produce a sane aggregate on real data before an
expensive multi-category run — and the --amp case exercises the bf16 backbone path CI can't reach.
Kept tiny (one cat, tiny coreset) so it's a smoke, not a benchmark.

Gated on RUN_E2E so it skips CI and the normal local loop:
    RUN_E2E=1 uv run pytest tests/e2e -q
torchvision-backed imports live inside the test bodies so collection stays clean where it's absent.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_E2E"), reason="e2e: set RUN_E2E=1 (needs torchvision + data store + GPU)")

CAT = os.getenv("E2E_CAT", "bagel")


def _assert_sane(res):
    assert len(res["per_category"]) == 1
    au, img = res["mean"]["au_pro"], res["mean"]["img_auroc"]
    assert 0.0 < au <= 1.0 and not np.isnan(au)        # real signal localizes (not just non-NaN)
    assert not np.isnan(img)


def _run(method):
    from surfscan.evaluation.harness import aggregate  # noqa: PLC0415
    return aggregate(method.run_name, method.fit, method.score, [CAT])


def test_patchcore_smoke():
    from core.compute import pick_device  # noqa: PLC0415
    from surfscan.experiments.run_patchcore import PatchCoreCfg, PatchCoreMethod  # noqa: PLC0415
    _assert_sane(_run(PatchCoreMethod(PatchCoreCfg(coreset=0.05), pick_device())))


def test_patchcore_amp_smoke():
    """Closes the 'bf16 backbone path not GPU-verified' gap — runs the amp=True embed on real data."""
    from core.compute import pick_device  # noqa: PLC0415
    from surfscan.experiments.run_patchcore import PatchCoreCfg, PatchCoreMethod  # noqa: PLC0415
    _assert_sane(_run(PatchCoreMethod(PatchCoreCfg(coreset=0.05, amp=True), pick_device())))


def test_fused_smoke():
    from core.compute import pick_device  # noqa: PLC0415
    from surfscan.experiments.run_fused import FusedCfg, FusedMethod  # noqa: PLC0415
    _assert_sane(_run(FusedMethod(FusedCfg(), pick_device())))


def test_btf_smoke():
    from core.compute import pick_device  # noqa: PLC0415
    from surfscan.experiments.run_btf import BtfCfg, BtfMethod  # noqa: PLC0415
    _assert_sane(_run(BtfMethod(BtfCfg(), pick_device())))


def test_train_then_evaluate_smoke():
    """The run_all train -> evaluate path (evaluate()'s harness.run call, both image scores). A 2-epoch
    vae isn't a detector, so we assert the pipeline runs + metrics are valid — not a perf threshold."""
    from core.compute import pick_device  # noqa: PLC0415
    from surfscan.evaluation.evaluate import evaluate  # noqa: PLC0415
    from surfscan.training.hparams import HParams  # noqa: PLC0415
    from surfscan.training.train import train  # noqa: PLC0415
    dev = pick_device()
    run_id = train(HParams(cats=[CAT], epochs=2, compile=False), run_name="e2e_vae", device=dev)
    for image_score in ("residual", "mahalanobis"):
        res = evaluate(run_id, cats=[CAT], device=dev, image_score=image_score)
        assert len(res["per_category"]) == 1
        au = res["mean"]["au_pro"]
        assert 0.0 <= au <= 1.0 and not np.isnan(au)
