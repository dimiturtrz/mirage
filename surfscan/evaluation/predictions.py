"""Persist a run's per-image predictions so any metric/CI/ECE recompute is OFFLINE — no GPU, no retrain.

The harness scores each method into per-category `ScoreArrays` (amaps/valids/masks/scores/labels/defects).
Those arrays ARE the run's evidence: every headline number (AU-PRO, image-AUROC, ECE, bootstrap CI) is a
pure function of them. Dump them once to a gitignored `.npz` and a new bootstrap, a macro-vs-micro fix, or
a fresh calibration metric is a two-second recompute instead of a full re-run of the GPU pipeline.

`pack` flattens the per-category list to a flat ndarray dict (npz is flat); `unpack` rebuilds it. Round-trip
is exact, so `Harness` metrics on the reloaded arrays equal the live ones.
"""
from __future__ import annotations

import numpy as np

from core.method import ScoreArrays


class Predictions:
    """Flatten/rebuild a per-category ScoreArrays list to/from an npz-friendly flat ndarray dict."""

    @staticmethod
    def pack(by_cat: list[ScoreArrays], cats: list[str]) -> dict[str, np.ndarray]:
        """(by_cat, cats) -> flat {key: ndarray} for np.savez. One key per (category, field)."""
        out: dict[str, np.ndarray] = {"_cats": np.asarray(cats)}
        for i, sa in enumerate(by_cat):
            for f in ScoreArrays._fields:
                out[f"c{i}.{f}"] = np.asarray(getattr(sa, f))
        return out

    @staticmethod
    def unpack(d) -> tuple[list[str], list[ScoreArrays]]:
        """A loaded npz dict (or np.load handle) -> (cats, by_cat), the exact input to Harness metrics."""
        cats = [str(c) for c in d["_cats"]]
        by_cat = [ScoreArrays(**{f: d[f"c{i}.{f}"] for f in ScoreArrays._fields}) for i in range(len(cats))]
        return cats, by_cat
