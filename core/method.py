"""The method contract the eval harness scores — mirage's analog of mindscape's `Decoder` protocol.

Every anomaly method, whatever its paradigm, is a `(fit_fn, score_fn)` pair the harness speaks to:

    fit_fn(category)        -> state          # build/load the detector for one category
    score_fn(state, category) -> ScoreArrays  # per-sample maps + image scores + labels

Because all three families satisfy it — **memory bank** (PatchCore, BTF), **reconstruction**
(VAE, inpaint, feature-recon), **synthesis-discriminative** (DRAEM) — one harness spine
(`harness.aggregate`) scores them identically. `ScoreArrays` names the return fields so the harness
reads them by name, not by fragile tuple position.
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple

import numpy as np


class ScoreArrays(NamedTuple):
    """A method's per-sample outputs for one category (all aligned on axis 0 = N samples)."""
    amaps: np.ndarray    # (N, H, W)  pixel anomaly maps  -> AU-PRO localization
    valids: np.ndarray   # (N, H, W)  bool object mask (background excluded)
    masks: np.ndarray    # (N, H, W)  bool ground-truth defect mask
    scores: np.ndarray   # (N,)       image-level anomaly score -> image-AUROC
    labels: np.ndarray   # (N,)       0 = good, 1 = anomaly
    defects: np.ndarray  # (N,)       defect-type string (for per-defect diagnostics)


FitFn = Callable[[str], Any]                 # fit(category) -> state
ScoreFn = Callable[[Any, str], ScoreArrays]  # score(state, category) -> ScoreArrays


class Method(NamedTuple):
    """The (fit, score) pair the harness scores — one interface object per detector."""
    fit: FitFn
    score: ScoreFn
