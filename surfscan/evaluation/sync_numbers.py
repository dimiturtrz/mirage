"""Fill docs/RESULTS.md number tables from docs/RESULTS.json (the one canonical source) — so a metric
lives in ONE place and the markdown can't drift (carried from systole's cardioseg/evaluation/sync_numbers).
Each generated table sits between HTML markers:

    <!-- results:methods -->
    ...auto-generated, do not hand-edit...
    <!-- /results:methods -->

Run after updating RESULTS.json:  python -m surfscan.report sync
The surrounding prose/footnotes stay hand-written; only the tables are generated. The root README is a
curated portfolio face (CLAUDE.md) — deliberately NOT templated here.
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
    def methods() -> str:
        rows = ["| method | modality | mean img-AUROC | mean pixel AU-PRO |", "|---|---|---|---|"]
        for m in R["methods"]:
            name = f"**{m['name']}**" if m.get("deployable") else m["name"]
            rows.append(f"| {name} — ours | {m['modality']} | {NumberSync._n(m['img_auroc'])} "
                        f"| {NumberSync._n(m['au_pro'])} |")
        s = R["sota"]
        rows.append(f"| SOTA ({s['name']}) ⚠ | {s['modality']} | ~{s['img_auroc']:.2f} | ~{s['au_pro']:.2f} |")
        return "\n".join(rows)

    @staticmethod
    def per_category() -> str:
        rows = ["| category | img-AUROC | AU-PRO |", "|---|---|---|"]
        for c in R["patchcore_per_category"]:
            rows.append(f"| {c['category']} | {c['img_auroc']:.3f} | {c['au_pro']:.3f} |")
        m = R["patchcore_mean"]
        rows.append(f"| **MEAN** | **{m['img_auroc']:.3f}** | **{m['au_pro']:.3f}** |")
        return "\n".join(rows)

    @staticmethod
    def triad() -> str:
        t = R["stage1_triad"]
        rows = ["| arm | au_pro (3-seed) | img-AUROC | ECE |", "|---|---|---|---|"]
        for a in t["arms"]:
            rows.append(f"| {a['arm']} | {a['au_pro']:.3f} ± {a['au_pro_std']:.3f} | "
                        f"{a['img_auroc']:.3f} | {a['ece']:.3f} |")
        return "\n".join(rows)

    @staticmethod
    def sync() -> None:  # pragma: no cover  reads/writes RESULTS.md (disk); the renderers are the pure core
        doc = ROOT / "docs" / "RESULTS.md"
        text = doc.read_text(encoding="utf-8")
        for key, render in _BLOCKS.items():
            pat = re.compile(rf"(<!-- results:{key} -->).*?(<!-- /results:{key} -->)", re.DOTALL)
            if not pat.search(text):
                log.info(f"  (no <!-- results:{key} --> marker in RESULTS.md — skipped)")
                continue
            text = pat.sub(lambda mt, render=render: f"{mt.group(1)}\n{render()}\n{mt.group(2)}", text)
        doc.write_text(text, encoding="utf-8")
        log.info(f"synced -> {doc.relative_to(ROOT)}")


_BLOCKS = {"methods": NumberSync.methods, "per_category": NumberSync.per_category, "triad": NumberSync.triad}

SPEC = Spec("sync", lambda _ap: None, lambda _args: NumberSync.sync())
