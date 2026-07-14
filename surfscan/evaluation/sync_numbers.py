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
R = json.loads((ROOT / "docs" / "RESULTS.json").read_text(encoding="utf-8"))


class NumberSync:
    """Fill docs/RESULTS.md number tables from docs/RESULTS.json — one canonical source, no drift."""

    @staticmethod
    def _n(x: float | None, p: int = 3) -> str:
        return "—" if x is None else f"{x:.{p}f}"

    @staticmethod
    def _ci(obj: dict, base: str, p: int = 3) -> str:
        """`point [lo, hi]` when a bootstrap interval is present (obj[base_lo/base_hi]), else bare point.
        One run's honest uncertainty — the bracket, not a ± seed-scatter std."""
        point = NumberSync._n(obj.get(base), p)
        lo, hi = obj.get(f"{base}_lo"), obj.get(f"{base}_hi")
        return f"{point} [{lo:.{p}f}, {hi:.{p}f}]" if lo is not None and hi is not None else point

    @staticmethod
    def methods() -> str:
        rows = ["| method | modality | mean img-AUROC | mean pixel AU-PRO |", "|---|---|---|---|"]
        for m in R["methods"]:
            name = f"**{m['name']}**" if m.get("deployable") else m["name"]
            rows.append(f"| {name} — ours | {m['modality']} | {NumberSync._ci(m, 'img_auroc')} "
                        f"| {NumberSync._ci(m, 'au_pro')} |")
        s = R["sota"]
        mark = "†" if s.get("verified") else " ⚠"
        rows.append(f"| SOTA ({s['name']}){mark} | {s['modality']} | ~{s['img_auroc']:.2f} | ~{s['au_pro']:.2f} |")
        return "\n".join(rows)

    @staticmethod
    def per_category() -> str:
        rows = ["| category | img-AUROC | AU-PRO |", "|---|---|---|"]
        rows.extend(f"| {c['category']} | {c['img_auroc']:.3f} | {c['au_pro']:.3f} |"
                    for c in R["patchcore_per_category"])
        m = R["patchcore_mean"]
        rows.append(f"| **MEAN** | **{m['img_auroc']:.3f}** | **{m['au_pro']:.3f}** |")
        return "\n".join(rows)

    @staticmethod
    def triad() -> str:
        t = R["stage1_triad"]
        rows = ["| arm | au_pro [95% boot CI] | img-AUROC | ECE |", "|---|---|---|---|"]
        rows.extend(f"| {a['arm']} | {NumberSync._ci(a, 'au_pro')} | "
                    f"{a['img_auroc']:.3f} | {a['ece']:.3f} |" for a in t["arms"])
        return "\n".join(rows)

    @staticmethod
    def _refs() -> dict[str, float]:
        """Canonical scalars quotable inline in prose via `<!--r:key-->…<!--/r-->` — one source, no drift.
        Only headline figures restated in prose; experiment-narrative intermediates stay hand-written."""
        by_run = {m["mlflow_run"]: m for m in R["methods"] if "mlflow_run" in m}
        t = R["stage1_triad"]
        arms = {a["arm"]: a for a in t["arms"]}
        return {
            "recon.au_pro": R["methods"][0]["au_pro"],
            "patchcore.au_pro": by_run["patchcore_rgb_greedy"]["au_pro"],
            "featrecon.au_pro": by_run["feat_recon"]["au_pro"],
            "fused.au_pro": by_run["fused_rgb_fpfh"]["au_pro"],
            "triad.real.au_pro": arms["real → real (ceiling)"]["au_pro"],
            "triad.synth.au_pro": arms["synth → real (the gap)"]["au_pro"],
            "triad.synth.ece": arms["synth → real (the gap)"]["ece"],
            "triad.da.ece": arms["synth+DA → real (AdaBN)"]["ece"],
            "triad.gap": t["gap_pp"], "triad.gap_lo": t["gap_lo"], "triad.gap_hi": t["gap_hi"],
            "triad.da_closure": t["da_closure_pp"],
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


_BLOCKS = {"methods": NumberSync.methods, "per_category": NumberSync.per_category, "triad": NumberSync.triad}
_R_INLINE = re.compile(r"(<!--r:([\w.]+)-->).*?(<!--/r-->)", re.DOTALL)

SPEC = Spec("sync", lambda _ap: None, lambda _args: NumberSync.sync())
