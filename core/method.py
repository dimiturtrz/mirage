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

from typing import Callable, Generic, NamedTuple, Protocol, TypeAlias, TypeVar, runtime_checkable

import numpy as np


class ScoreArrays(NamedTuple):
    """A method's per-sample outputs for one category (all aligned on axis 0 = N samples)."""
    amaps: np.ndarray    # (N, H, W)  pixel anomaly maps  -> AU-PRO localization
    valids: np.ndarray   # (N, H, W)  bool object mask (background excluded)
    masks: np.ndarray    # (N, H, W)  bool ground-truth defect mask
    scores: np.ndarray   # (N,)       image-level anomaly score -> image-AUROC
    labels: np.ndarray   # (N,)       0 = good, 1 = anomaly
    defects: np.ndarray  # (N,)       defect-type string (for per-defect diagnostics)


# The detector state is opaque TO THE HARNESS but concrete to each method (a PatchCore bank, an
# nn.Module, a (PatchCore, FpfhBank) pair). That is a type PARAMETER, not `Any`: StateT ties a method's
# fit return to the state its own score consumes, so a mismatched pair is a type error rather than
# silently accepted.
StateT = TypeVar("StateT")

FitFn: TypeAlias = Callable[[str], StateT]                 # fit(category) -> state
ScoreFn: TypeAlias = Callable[[StateT, str], ScoreArrays]  # score(state, category) -> ScoreArrays


class Method(NamedTuple, Generic[StateT]):
    """The (fit, score) pair the harness scores — the closure form of `AnomalyMethod` (for ad-hoc
    or orchestrated methods that don't warrant a class; e.g. the triad's per-arm methods)."""
    fit: FitFn[StateT]
    score: ScoreFn[StateT]


@runtime_checkable
class AnomalyMethod(Protocol[StateT]):
    """The minimal contract the harness scores — a configured detector object. Deliberately just
    `fit`/`score`: everything else (data loading, a Trainer, the mlflow `run_name`) is composed by
    HAS-A, never bolted on as protocol hooks. Both the `Method` closure-pair and the per-paradigm
    method classes (PatchCoreMethod, DraemMethod, …) satisfy it, so `harness.run` reads `.fit`/`.score`
    without caring which. `run_name` (below) is read via getattr when present, defaulting to the
    subcommand name."""

    def fit(self, cat: str, /) -> StateT: ...

    def score(self, state: StateT, cat: str, /) -> ScoreArrays: ...
