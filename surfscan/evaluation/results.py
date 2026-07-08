"""Refresh the measured numbers in docs/RESULTS.json from MLflow — the canonical source stays in one
place, and reruns flow into it automatically (carried from systole's cardioseg/evaluation/results.py).

Each method carrying an "mlflow_run" is refreshed from that run's latest `au_pro_mean` / `img_auroc_mean`;
the deployable's per-category table from its `au_pro/<cat>` + `img_auroc/<cat>` metrics. Curated fields
(paradigm, notes, the SOTA cite) are preserved; entries without an "mlflow_run" stay hand-curated.
Then render the docs:  python -m surfscan.evaluation.sync_numbers

    python -m surfscan.evaluation.results     # refresh RESULTS.json from mlflow, then run sync_numbers
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow

from core.obs import get
from surfscan import tracking  # noqa: F401  — importing sets the mlflow tracking uri

log = get()

ROOT = Path(__file__).resolve().parents[2]
RJSON = ROOT / "docs" / "RESULTS.json"


def _latest(run_name: str):
    df = mlflow.search_runs(
        experiment_names=["surfscan"],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        order_by=["start_time DESC"], max_results=1,
    )
    return df.iloc[0].to_dict() if len(df) else None


def _get(row: dict, key: str):
    v = row.get(f"metrics.{key}")
    return round(float(v), 3) if v is not None and v == v else None   # skip None / NaN


def _apply(obj: dict, row: dict, pairs) -> None:
    for metric, field in pairs:
        v = _get(row, metric)
        if v is not None:
            obj[field] = v


def refresh() -> None:
    R = json.loads(RJSON.read_text(encoding="utf-8"))
    done, skipped = [], []
    for m in R["methods"]:
        rn = m.get("mlflow_run")
        if not rn:
            continue
        row = _latest(rn)
        if row is None:
            skipped.append(rn); continue
        _apply(m, row, (("au_pro_mean", "au_pro"), ("img_auroc_mean", "img_auroc")))
        if m.get("deployable"):                                   # per-category table
            for c in R["patchcore_per_category"]:
                _apply(c, row, ((f"au_pro/{c['category']}", "au_pro"),
                                (f"img_auroc/{c['category']}", "img_auroc")))
        done.append(m["name"])

    RJSON.write_text(json.dumps(R, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info(f"refreshed from mlflow: {done or '(nothing)'}")
    if skipped:
        log.info(f"  no run found (kept curated): {skipped}")
    log.info("now render the docs:  python -m surfscan.evaluation.sync_numbers")


if __name__ == "__main__":
    refresh()
