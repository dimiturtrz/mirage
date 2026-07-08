"""Auto-generate docs/MODEL_CARD.md from an MLflow run — so the card always matches the shipped model
(carried from systole's cardioseg/evaluation/modelcard.py). Run against the deployable's canonical run:

    python -m surfscan.report modelcard --run-name patchcore_rgb_greedy
"""
from __future__ import annotations

from core.obs import get
from surfscan.dispatch import Spec
from surfscan.evaluation.results import ROOT, _latest  # DRY: reuse the mlflow lookup

log = get()

CARD = ROOT / "docs" / "MODEL_CARD.md"


def _render(run_name: str, row: dict) -> str:
    params = {k[len("params."):]: v for k, v in row.items() if k.startswith("params.") and v == v}
    au, img = row.get("metrics.au_pro_mean"), row.get("metrics.img_auroc_mean")
    cats = sorted({k.split("/")[1] for k in row if k.startswith("metrics.au_pro/")})

    out = [
        f"# Model card — {run_name}",
        "",
        "Auto-generated from its MLflow run by `surfscan.report modelcard` — regenerate after any "
        "rerun so the card can't drift from the shipped model.",
        "",
        "## Headline (our harness, MVTec 3D-AD)",
        (f"- pixel **AU-PRO** {au:.3f} · image **AUROC** {img:.3f} (per-category mean)"
         if au is not None and au == au else "- (no aggregate metrics on this run)"),
        "",
    ]
    if cats:
        out += ["## Per category", "| category | img-AUROC | AU-PRO |", "|---|---|---|"]
        for c in cats:
            a, i = row.get(f"metrics.au_pro/{c}"), row.get(f"metrics.img_auroc/{c}")
            out.append(f"| {c} | {i:.3f} | {a:.3f} |")
        out.append("")
    if params:
        out += ["## Config", "| param | value |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in sorted(params.items())]
        out.append("")
    out += [
        "## Honest limits",
        "- rgb-only; the gap to SOTA (~0.96 M3DM) is multimodal geometry fusion + greedy coreset — "
        "named and measured, not bank/resolution tuning.",
        "",
    ]
    return "\n".join(out)


def build(run_name: str) -> None:  # pragma: no cover  mlflow query + card file write; _render is the pure core
    row = _latest(run_name)
    if row is None:
        raise SystemExit(f"no mlflow run '{run_name}' — train/score it first, then regenerate the card")
    CARD.write_text(_render(run_name, row), encoding="utf-8")
    log.info(f"wrote {CARD.relative_to(ROOT)}")


def _args(ap):
    ap.add_argument("--run-name", default="patchcore_rgb_greedy")


def _run(args):  # pragma: no cover  CLI glue; _render is the pure tested core
    build(args.run_name)


SPEC = Spec("modelcard", _args, _run)
