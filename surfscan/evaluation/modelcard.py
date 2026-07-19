"""Auto-generate docs/MODEL_CARD.md from an MLflow run — so the card always matches the shipped model
(carried from systole's cardioseg/evaluation/modelcard.py). Run against the deployable's canonical run:

    python -m surfscan.report modelcard --run-name patchcore_rgb_greedy
"""
from __future__ import annotations

import math
from typing import Any

from core.obs import Obs
from surfscan.dispatch import Spec
from surfscan.evaluation.results import ROOT, Results  # DRY: reuse the mlflow lookup

log = Obs.get()

CARD = ROOT / "docs" / "MODEL_CARD.md"


class ModelCard:
    """Auto-generate docs/MODEL_CARD.md from an MLflow run — so the card always matches the shipped model."""

    @staticmethod
    def _render(run_name: str, row: dict[str, Any]) -> str:
        params = {key[len("params."):]: value for key, value in row.items() if key.startswith("params.")}
        au_pro, img_auroc = row.get("metrics.au_pro_mean"), row.get("metrics.img_auroc_mean")
        categories = sorted({key.split("/")[1] for key in row if key.startswith("metrics.au_pro/")})

        out = [
            f"# Model card — {run_name}",
            "",
            "Auto-generated from its MLflow run by `surfscan.report modelcard` — regenerate after any "
            "rerun so the card can't drift from the shipped model.",
            "",
            "## Headline (our harness, MVTec 3D-AD)",
            (f"- pixel **AU-PRO** {au_pro:.3f} · image **AUROC** {img_auroc:.3f} (per-category mean)"
             if au_pro is not None and not math.isnan(au_pro) else "- (no aggregate metrics on this run)"),
            "",
        ]
        if categories:
            out += ["## Per category", "| category | img-AUROC | AU-PRO |", "|---|---|---|"]
            for category in categories:
                cat_au, cat_img = row.get(f"metrics.au_pro/{category}"), row.get(f"metrics.img_auroc/{category}")
                out.append(f"| {category} | {cat_img:.3f} | {cat_au:.3f} |")
            out.append("")
        if params:
            out += ["## Config", "| param | value |", "|---|---|"]
            out += [f"| {key} | {value} |" for key, value in sorted(params.items())]
            out.append("")
        out += [
            "## Known limits",
            "- rgb-only; the gap to SOTA (~0.96 M3DM) is multimodal geometry fusion + greedy coreset — "
            "named and measured, not bank/resolution tuning.",
            "",
        ]
        return "\n".join(out)

    @staticmethod
    def build(run_name: str) -> None:  # pragma: no cover  mlflow query + card file write; _render is the pure core
        row = Results.latest(run_name)
        if row is None:
            raise SystemExit(f"no mlflow run '{run_name}' — train/score it first, then regenerate the card")
        CARD.write_text(ModelCard._render(run_name, row), encoding="utf-8")
        log.info(f"wrote {CARD.relative_to(ROOT)}")

    @staticmethod
    def args(ap: object) -> None:
        ap.add_argument("--run-name", default="patchcore_rgb_greedy")

    @staticmethod
    def run(args: object) -> None:  # pragma: no cover  CLI glue; _render is the pure tested core
        ModelCard.build(args.run_name)


SPEC = Spec("modelcard", ModelCard.args, ModelCard.run)
