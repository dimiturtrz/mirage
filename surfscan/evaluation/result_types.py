"""Small typed carriers for the evaluation/run boundary — the metric pair and the tracker-run params
that were repeated ad-hoc dicts across harness + run_all. One home for the key vocabulary, so the
`{img_auroc, au_pro}` / `{method, cats}` shapes are declared once instead of re-spelled per call site."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Scores:
    """The two headline metrics that always travel together (mean over categories / defects)."""

    img_auroc: float
    au_pro: float

    @classmethod
    def macro(cls, rows: list[dict[str, Any]]) -> Scores:
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
