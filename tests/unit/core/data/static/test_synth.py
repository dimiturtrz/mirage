"""The synth adapter is MVTec-over-a-different-root — its only own logic is the root, the rest delegates.

Equivalence-class: synth_root = <data root>/synth; an explicit root is kept; categories/samples delegate
to an Mvtec bound to that root (same format, only the source differs).
"""
from pathlib import Path

from core import config
from core.data.static import mvtec, synth
from core.data.static.synth import Synth


def test_synth_root_is_data_root_slash_synth(monkeypatch, tmp_path):
    monkeypatch.setattr(config.Config, "data_root", staticmethod(lambda: str(tmp_path)))
    assert Synth.synth_root() == tmp_path / "synth"


def test_explicit_root_is_kept():
    r = Path("/some/root")
    assert Synth(r).root == r


def test_categories_and_samples_delegate_to_mvtec(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(mvtec.Mvtec, "categories", lambda self: (seen.__setitem__("cat_root", self.root), ["bagel"])[1])
    monkeypatch.setattr(mvtec.Mvtec, "samples", lambda self, cats=None: (seen.__setitem__("smp", cats), ["s"])[1])
    s = Synth(tmp_path)
    assert s.categories() == ["bagel"] and seen["cat_root"] == tmp_path   # delegated to Mvtec bound to the synth root
    assert s.samples(["bagel"]) == ["s"] and seen["smp"] == ["bagel"]
