"""Unit test for TwinSynth._categories — no Isaac boot. The render loop is GPU-only and coverage-omitted;
this locks the category selection (explicit subset vs directory scan)."""
from core.data.dynamic import generate_twin_synth as g
from core.data.dynamic.generate_twin_synth import TwinSynth


class TestCategories:
    def test_explicit_cats_split(self):
        assert TwinSynth(views=1, res=64, cats="bagel,cookie")._categories() == ["bagel", "cookie"]

    def test_scans_twin_dir_for_objs(self, tmp_path, monkeypatch):
        for name in ("cookie.obj", "bagel.obj", "note.txt"):
            (tmp_path / name).write_text("")
        monkeypatch.setattr(g, "_TWIN_DIR", str(tmp_path))
        assert TwinSynth(views=1, res=64, cats=None)._categories() == ["bagel", "cookie"]   # sorted, .obj only
