"""Unit tests for the RESULTS.json -> markdown renderers (the single-source-of-numbers path).

Pure string builders over the committed RESULTS.json — assert each table renders with its header and a
known row, the inline `<!--r:key-->` refs substitute the canonical value (and reject unknown keys), and
the committed RESULTS.md is in sync (the anti-drift gate — a hand-edited number fails CI).
"""
import pytest

from surfscan.evaluation import sync_numbers as sn


def test_methods_table_renders():
    t = sn.NumberSync.methods()
    assert "| method | modality |" in t
    assert "PatchCore" in t and "SOTA" in t


def test_per_category_table_renders():
    t = sn.NumberSync.per_category()
    assert "bagel" in t
    assert "**MEAN**" in t


def test_triad_table_renders():
    t = sn.NumberSync.triad()
    assert "au_pro" in t
    assert "real" in t and "synth" in t


def test_inline_substitutes_known_ref():
    val = f"{sn.NumberSync._refs()['triad.gap']:.3f}"          # canonical value from RESULTS.json
    out = sn.NumberSync.inline("gap <!--r:triad.gap-->9.999<!--/r--> done")
    assert f"<!--r:triad.gap-->{val}<!--/r-->" in out          # stale 9.999 overwritten with canonical


def test_inline_unknown_ref_raises():
    with pytest.raises(KeyError):
        sn.NumberSync.inline("<!--r:triad.bogus-->0<!--/r-->")


def test_results_md_is_in_sync():
    # anti-drift gate: the committed RESULTS.md must equal a fresh render (tables + inline refs).
    doc = (sn.ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    assert sn.NumberSync.render(doc) == doc, "RESULTS.md out of sync — run `python -m surfscan.report sync`"
