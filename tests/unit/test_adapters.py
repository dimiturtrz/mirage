"""Unit tests for the data adapters (mvtec + synth) — enumeration over a fake on-disk layout.

Files only need to EXIST (samples() globs rgb/*.png, checks gt presence — never reads them), so a
tmp_path tree of empty files exercises the full enumeration. Equivalence classes: good (no gt, label 0)
vs defective (gt present, label 1); missing root -> empty.
"""
from core.data import mvtec, synth


def _layout(root):
    for split, defect, has_gt in [("train", "good", False), ("test", "hole", True)]:
        d = root / "bagel" / split / defect
        (d / "rgb").mkdir(parents=True)
        (d / "rgb" / "000.png").touch()
        (d / "xyz").mkdir(parents=True)
        (d / "xyz" / "000.tiff").touch()
        if has_gt:
            (d / "gt").mkdir(parents=True)
            (d / "gt" / "000.png").touch()


def test_mvtec_categories_and_samples(tmp_path):
    _layout(tmp_path)
    assert mvtec.categories(tmp_path) == ["bagel"]
    s = mvtec.samples(tmp_path)
    assert len(s) == 2
    good = next(x for x in s if x.defect == "good")
    hole = next(x for x in s if x.defect == "hole")
    assert good.label == 0 and good.gt_path is None
    assert hole.label == 1 and hole.gt_path is not None


def test_synth_mirrors_mvtec_layout(tmp_path):
    _layout(tmp_path)
    assert synth.categories(tmp_path) == ["bagel"]
    assert len(synth.samples(tmp_path)) == 2


def test_synth_categories_empty_when_missing(tmp_path):
    assert synth.categories(tmp_path / "nope") == []
