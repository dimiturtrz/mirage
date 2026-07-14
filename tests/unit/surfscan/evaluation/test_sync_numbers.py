"""Unit tests for the RESULTS.json -> markdown renderers (the single-source-of-numbers path).

Pure string builders over the committed RESULTS.json — assert each table renders with its header and a
known row. Guards the docs-sync logic (a broken renderer would silently desync the published numbers).
"""
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
