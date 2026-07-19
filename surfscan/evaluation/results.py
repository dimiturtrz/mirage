"""Refresh the measured numbers in docs/RESULTS.json from MLflow — the canonical source stays in one
place, and reruns flow into it automatically (carried from systole's cardioseg/evaluation/results.py).

Each method carrying an "mlflow_run" is refreshed from that run's latest `au_pro_mean` / `img_auroc_mean`;
the deployable's per-category table from its `au_pro/<cat>` + `img_auroc/<cat>` metrics. Curated fields
(paradigm, notes, the SOTA cite) are preserved; entries without an "mlflow_run" stay hand-curated.
Then render the docs:  python -m surfscan.report sync

    python -m surfscan.report results     # refresh RESULTS.json from mlflow, then run `report sync`
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

import mlflow

from core.obs import Obs
from surfscan import tracking  # noqa: F401  — importing sets the mlflow tracking uri
from surfscan.dispatch import Spec
from surfscan.evaluation.result_types import ResultsRecord

if TYPE_CHECKING:
    import pandas as pd

    # One flattened MLflow run row (`search_runs` DataFrame row -> dict): the `metrics.*` namespace is
    # float (NaN when the run never logged it), `params.*` / `tags.*` / the id+uri columns are str, and
    # `start_time`/`end_time` are pandas timestamps. Typing-only — mlflow's pandas row never crosses a
    # runtime boundary, so the alias stays behind TYPE_CHECKING with the pandas import it needs.
    RunRow: TypeAlias = dict[str, float | str | pd.Timestamp | None]

log = Obs.get()

ROOT = Path(__file__).resolve().parents[2]
RJSON = ROOT / "docs" / "RESULTS.json"


class Results:
    """Refresh the measured numbers in docs/RESULTS.json from MLflow — the canonical source in one place."""

    @staticmethod
    def latest(run_name: str) -> RunRow | None:  # pragma: no cover  mlflow run query (network/db)
        df = mlflow.search_runs(
            experiment_names=["surfscan"],
            filter_string=f"tags.`mlflow.runName` = '{run_name}'",
            order_by=["start_time DESC"], max_results=1,
        )
        # pyrefly: ignore[missing-attribute]  search_runs returns list[Run]|DataFrame keyed on the runtime
        # output_format string; the default "pandas" is always a DataFrame, but mlflow ships no Literal overloads
        return cast("RunRow", df.iloc[0].to_dict()) if len(df) else None

    @staticmethod
    def _get(row: RunRow, key: str) -> float | None:
        v = cast("float | None", row.get(f"metrics.{key}"))
        return round(float(v), 3) if v is not None and not math.isnan(v) else None   # skip None / NaN

    @staticmethod
    def _apply(
        obj: ResultsRecord, row: RunRow, pairs: Sequence[tuple[str, str]]
    ) -> None:
        for metric, field in pairs:
            v = Results._get(row, metric)
            if v is not None:
                obj[field] = v

    @staticmethod
    def _table_mean(rows: list[ResultsRecord]) -> dict[str, float]:
        """The per-category table's MEAN row, derived from its rows so it can never drift from them."""
        n = len(rows)
        return {k: round(sum(cast("float", r[k]) for r in rows) / n, 3) for k in ("img_auroc", "au_pro")}

    @staticmethod
    def refresh() -> None:  # pragma: no cover  reads RESULTS.json + mlflow; _get/_apply are the pure core
        data = json.loads(RJSON.read_text(encoding="utf-8"))
        done: list[str] = []
        skipped: list[str] = []
        for m in data["methods"]:
            rn = m.get("mlflow_run")
            if not rn:
                continue
            row = Results.latest(rn)
            if row is None:
                skipped.append(rn); continue
            Results._apply(m, row, (("au_pro_mean", "au_pro"), ("img_auroc_mean", "img_auroc"),
                                    ("au_pro_lo", "au_pro_lo"), ("au_pro_hi", "au_pro_hi"),
                                    ("img_auroc_lo", "img_auroc_lo"), ("img_auroc_hi", "img_auroc_hi")))
            if m.get("deployable"):                                   # per-category table + its derived mean
                for c in data["patchcore_per_category"]:
                    Results._apply(c, row, ((f"au_pro/{c['category']}", "au_pro"),
                                            (f"img_auroc/{c['category']}", "img_auroc")))
                data["patchcore_mean"] = Results._table_mean(data["patchcore_per_category"])
            done.append(m["name"])

        RJSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log.info(f"refreshed from mlflow: {done or '(nothing)'}")
        if skipped:
            log.info(f"  no run found (kept curated): {skipped}")
        log.info("now render the docs:  python -m surfscan.report sync")


SPEC = Spec("results", lambda _ap: None, lambda _args: Results.refresh())
