"""Small typed carriers for the evaluation/run boundary — the metric pair, the tracker-run params, and
the eval RESULT PAYLOAD that were repeated ad-hoc dicts across harness + run_all + the report commands.
One home for the key vocabulary, so the `{img_auroc, au_pro}` / `{method, cats}` / `{mean, ci, ece, …}`
shapes are declared once instead of re-spelled per call site."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypeAlias, TypedDict

import numpy as np

from core.method import ScoreArrays

Bracket: TypeAlias = tuple[float, float, float]   # (point, lo, hi) — one bootstrap interval

# One record of docs/RESULTS.json (a method entry, a per-category row, a triad/twin arm): a flat JSON
# object whose values are the measured scalars, the curated strings/flags, and `cost_ref`'s name list.
ResultsField: TypeAlias = float | str | bool | list[str] | None
ResultsRecord: TypeAlias = dict[str, ResultsField]


class CategoryRow(TypedDict):
    """One per-category row of the eval table (`Metrics.image_auroc` / `Metrics.au_pro` over that split)."""

    category: str
    n: int
    img_auroc: float
    au_pro: float


class DefectRow(TypedDict):
    """One per-defect-type diagnostic row (that type's anomalies scored against ALL normals)."""

    defect: str
    n: int
    img_auroc: float
    au_pro: float


class EvalResult(TypedDict):
    """The eval result payload — what `Harness.aggregate`/`compute`/`run` return and every consumer
    (invariants, the report commands, the triad's gap report) reads by key. `ece` is `None` rather than
    absent when calibration didn't apply: the harness always sets the key, so consumers can rely on it."""

    method: str
    per_category: list[CategoryRow]
    mean: dict[str, float]
    ci: dict[str, Bracket]
    ece: float | None
    per_defect: list[DefectRow]
    by_cat: list[ScoreArrays]


@dataclass(frozen=True)
class Scores:
    """The two headline metrics that always travel together (mean over categories / defects)."""

    img_auroc: float
    au_pro: float

    @classmethod
    def macro(cls, rows: list[CategoryRow]) -> Scores:
        """Macro headline (mean over categories) from per-category rows carrying `img_auroc` / `au_pro`
        — the nanmean reduction shared by the live harness and the per-category run aggregator."""
        return cls(float(np.nanmean([r["img_auroc"] for r in rows])),
                   float(np.nanmean([r["au_pro"] for r in rows])))

    def as_dict(self) -> dict[str, float]:
        """Plain-dict form for the result payload + JSON artifact (stays serializable downstream)."""
        return asdict(self)

    def tracker_means(self) -> dict[str, float]:
        """The MLflow metric names for the aggregate (mean-over-categories) point estimates."""
        return {"img_auroc_mean": self.img_auroc, "au_pro_mean": self.au_pro}


@dataclass(frozen=True)
class RunParams:
    """Identity of one tracked run — the method name and the categories it covered."""

    method: str
    cats: list[str]

    def as_dict(self) -> dict[str, str]:
        return {"method": self.method, "cats": ",".join(self.cats)}
