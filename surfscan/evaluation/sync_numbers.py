"""Fill docs/RESULTS.md number tables from docs/RESULTS.json (the one canonical source) — so a metric
lives in ONE place and the markdown can't drift (carried from systole's cardioseg/evaluation/sync_numbers).
Each generated table sits between HTML markers:

    <!-- results:methods -->
    ...auto-generated, do not hand-edit...
    <!-- /results:methods -->

Headline numbers restated in prose are wired the same way — an inline `<!--r:key-->value<!--/r-->` span
is refilled from RESULTS.json on sync (`_refs` is the key map). A test asserts `render(RESULTS.md) ==
RESULTS.md`, so a hand-edited number (table or inline) fails CI. Experiment-narrative intermediates
(drill deltas, per-defect, superseded baselines) stay hand-written — they have no canonical field.

Run after updating RESULTS.json:  python -m surfscan.report sync
The root README is a curated portfolio face (CLAUDE.md) — deliberately NOT templated here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from core.obs import Obs
from surfscan.dispatch import Spec

log = Obs.get()

ROOT = Path(__file__).resolve().parents[2]
RESULTS = json.loads((ROOT / "docs" / "RESULTS.json").read_text(encoding="utf-8"))
# Deploy-cost atoms (surfscan.deploy profile) keyed by model name; a method's "cost_ref" sums over them
# (the detector = its profiled pieces; the composition also lives in surfscan.deploy.projection).
_COST = {r["name"]: r for r in json.loads((ROOT / "docs" / "DEPLOY_COST.json").read_text(encoding="utf-8"))["rows"]}


class NumberSync:
    """Fill docs/RESULTS.md number tables from docs/RESULTS.json — one canonical source, no drift."""

    @staticmethod
    def _n(x: float | None, p: int = 3) -> str:
        return "—" if x is None else f"{x:.{p}f}"

    @staticmethod
    def _cost_cols(method: dict) -> str:
        """`| params(M) | GFLOPs | int8(MB) |` summed over the method's profiled pieces, or dashes."""
        refs = method.get("cost_ref")
        if not refs:
            return "| — | — | — |"
        params = sum(_COST[r]["params_m"] for r in refs)
        gflops = sum(_COST[r]["gflops"] for r in refs)
        disk = sum(_COST[r]["disk_int8_mb"] for r in refs)
        return f"| {params:.1f} | {gflops:.1f} | {disk:.1f} |"

    @staticmethod
    def _ci(obj: dict, base: str, p: int = 3) -> str:
        """`point [lo, hi]` when a bootstrap interval is present (obj[base_lo/base_hi]), else bare point.
        One run's honest uncertainty — the bracket, not a ± seed-scatter std."""
        point = NumberSync._n(obj.get(base), p)
        lo, hi = obj.get(f"{base}_lo"), obj.get(f"{base}_hi")
        return f"{point} [{lo:.{p}f}, {hi:.{p}f}]" if lo is not None and hi is not None else point

    @staticmethod
    def methods() -> str:
        header = "| method | modality | mean img-AUROC | mean pixel AU-PRO | params (M) | GFLOPs | int8 (MB) |"
        rows = [header, "|---|---|---|---|---|---|---|"]
        for method in RESULTS["methods"]:
            name = f"**{method['name']}**" if method.get("deployable") else method["name"]
            rows.append(f"| {name} — ours | {method['modality']} | {NumberSync._ci(method, 'img_auroc')} "
                        f"| {NumberSync._ci(method, 'au_pro')} {NumberSync._cost_cols(method)}")
        sota = RESULTS["sota"]
        mark = "†" if sota.get("verified") else " ⚠"
        rows.append(f"| SOTA ({sota['name']}){mark} | {sota['modality']} "
                    f"| ~{sota['img_auroc']:.2f} | ~{sota['au_pro']:.2f} | — | — | — |")
        return "\n".join(rows)

    @staticmethod
    def per_category() -> str:
        rows = ["| category | img-AUROC | AU-PRO |", "|---|---|---|"]
        rows.extend(f"| {cat['category']} | {cat['img_auroc']:.3f} | {cat['au_pro']:.3f} |"
                    for cat in RESULTS["patchcore_per_category"])
        mean_row = RESULTS["patchcore_mean"]
        rows.append(f"| **MEAN** | **{mean_row['img_auroc']:.3f}** | **{mean_row['au_pro']:.3f}** |")
        return "\n".join(rows)

    @staticmethod
    def triad() -> str:
        triad_data = RESULTS["stage1_triad"]
        rows = ["| arm | au_pro [95% boot CI] | img-AUROC | ECE |", "|---|---|---|---|"]
        rows.extend(f"| {arm['arm']} | {NumberSync._ci(arm, 'au_pro')} | "
                    f"{arm['img_auroc']:.3f} | {arm['ece']:.3f} |" for arm in triad_data["arms"])
        return "\n".join(rows)

    @staticmethod
    def twin() -> str:
        twin_data = RESULTS["stage2_twin"]
        rows = ["| arm (label source) | rgb AU-PRO | xyz AU-PRO |", "|---|---|---|"]
        rows.extend(f"| {arm['arm']} | {arm['rgb_au_pro']:.3f} | {arm['xyz_au_pro']:.3f} |"
                    for arm in twin_data["arms"])
        return "\n".join(rows)

    @staticmethod
    def _refs() -> dict[str, float]:
        """Canonical scalars quotable inline in prose via `<!--r:key-->…<!--/r-->` — one source, no drift.
        Only headline figures restated in prose; experiment-narrative intermediates stay hand-written."""
        by_run = {method["mlflow_run"]: method for method in RESULTS["methods"] if "mlflow_run" in method}
        triad_data = RESULTS["stage1_triad"]
        arms = {arm["arm"]: arm for arm in triad_data["arms"]}
        twin_data = RESULTS["stage2_twin"]
        return {
            "twin.shape_source_rgb": twin_data["shape_source_rgb_pp"],
            "twin.shape_source_xyz": twin_data["shape_source_xyz_pp"],
            "recon.au_pro": RESULTS["methods"][0]["au_pro"],
            "patchcore.au_pro": by_run["patchcore_rgb_greedy"]["au_pro"],
            "featrecon.au_pro": by_run["feat_recon"]["au_pro"],
            "fused.au_pro": by_run["fused_rgb_fpfh"]["au_pro"],
            "triad.real.au_pro": arms["real → real (ceiling)"]["au_pro"],
            "triad.synth.au_pro": arms["synth → real (the gap)"]["au_pro"],
            "triad.synth.ece": arms["synth → real (the gap)"]["ece"],
            "triad.da.ece": arms["synth+DA → real (AdaBN)"]["ece"],
            "triad.gap": triad_data["gap_pp"], "triad.gap_lo": triad_data["gap_lo"],
            "triad.gap_hi": triad_data["gap_hi"], "triad.da_closure": triad_data["da_closure_pp"],
        }

    @staticmethod
    def inline(text: str) -> str:
        """Replace each `<!--r:key-->…<!--/r-->` span with its canonical value (.3f). Unknown key raises
        (the CI in-sync gate turns that into a build failure)."""
        refs = NumberSync._refs()

        def _sub(mt: re.Match) -> str:
            key = mt.group(2)
            if key not in refs:
                raise KeyError(f"unknown results ref '{key}' in RESULTS.md")
            return f"{mt.group(1)}{refs[key]:.3f}{mt.group(3)}"

        return _R_INLINE.sub(_sub, text)

    @staticmethod
    def render(text: str) -> str:
        """Pure core: substitute the table blocks, then the inline refs. `sync()` wraps disk IO around it,
        and the in-sync test asserts `render(RESULTS.md) == RESULTS.md` (fails on drift or an unknown ref)."""
        for key, block in _BLOCKS.items():
            pat = re.compile(rf"(<!-- results:{key} -->).*?(<!-- /results:{key} -->)", re.DOTALL)
            if pat.search(text):
                text = pat.sub(lambda mt, block=block: f"{mt.group(1)}\n{block()}\n{mt.group(2)}", text)
        return NumberSync.inline(text)

    @staticmethod
    def sync() -> None:  # pragma: no cover  reads/writes RESULTS.md (disk); render() is the pure core
        doc = ROOT / "docs" / "RESULTS.md"
        doc.write_text(NumberSync.render(doc.read_text(encoding="utf-8")), encoding="utf-8")
        log.info(f"synced -> {doc.relative_to(ROOT)}")


_BLOCKS = {"methods": NumberSync.methods, "per_category": NumberSync.per_category,
           "triad": NumberSync.triad, "twin": NumberSync.twin}
_R_INLINE = re.compile(r"(<!--r:([\w.]+)-->).*?(<!--/r-->)", re.DOTALL)

SPEC = Spec("sync", lambda _ap: None, lambda _args: NumberSync.sync())
