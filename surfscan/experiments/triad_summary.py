"""Report the sim-to-real triad from ONE deterministic run per arm — the headline numbers.

No cross-seed aggregation: rigor is a bootstrap over the shared eval set (logged as au_pro_lo/hi by
the harness), not a mean ± std over retrained seeds. Reads the latest triad_* run per arm, reports
each arm's au_pro [95% boot CI] / img_auroc / ECE, then the measured GAP (real − synth) and DA
CLOSURE (synth+DA − synth). The paired delta CIs for the gap/closure are printed by run_triad at run
time (they need the raw per-image predictions); this post-hoc view reports the per-arm brackets.

    python -m surfscan.report triad-summary
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

import mlflow

from core.obs import Obs
from surfscan.dispatch import Spec

log = Obs.get()

ARMS = ["triad_real", "triad_synth", "triad_synth_da", "triad_synth_curric"]


class TriadSummary:
    """Report the sim-to-real triad from ONE run per arm — au_pro [boot CI], GAP, DA CLOSURE."""

    @staticmethod
    def _latest(client: mlflow.MlflowClient, name: str) -> dict[str, Any] | None:
        """The most recent active run named `name` carrying au_pro_mean, or None."""
        for e in client.search_experiments():
            runs = client.search_runs(e.experiment_id, filter_string=f"tags.`mlflow.runName` = '{name}'",
                                      order_by=["start_time DESC"], max_results=1)
            if runs and "au_pro_mean" in runs[0].data.metrics:
                return runs[0].data.metrics
        return None

    @staticmethod
    def collect(uri: str = "sqlite:///mlflow.db") -> dict[str, dict[str, Any]]:
        mlflow.set_tracking_uri(uri)
        c = mlflow.MlflowClient()
        return {arm: m for arm in ARMS if (m := TriadSummary._latest(c, arm)) is not None}

    @staticmethod
    def _row(m: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
        """(au_pro, lo, hi, img_auroc, ece) for one arm — lo/hi are the bootstrap bracket (NaN-safe)."""
        return (m["au_pro_mean"], m.get("au_pro_lo"), m.get("au_pro_hi"),
                m.get("img_auroc_mean"), m.get("ece"))

    @staticmethod
    def _fmt_ci(lo: float | None, hi: float | None) -> str:
        return f"[{lo:.3f}, {hi:.3f}]" if lo is not None and hi is not None else "[—, —]"

    @staticmethod
    def summarize(uri: str = "sqlite:///mlflow.db") -> dict[str, tuple[Any, Any, Any, Any, Any]]:
        by = TriadSummary.collect(uri)
        rows = {arm: TriadSummary._row(m) for arm, m in by.items()}
        log.info(f"{'arm':22s} {'au_pro [95% boot CI]':28s} {'img_auroc':>10s} {'ECE':>8s}")
        for arm in ARMS:
            r = rows.get(arm)
            if r:
                ap, lo, hi, img, ece = r
                log.info(f"{arm:22s} {ap:.3f} {TriadSummary._fmt_ci(lo, hi):22s} "
                         f"{(img or float('nan')):10.3f} {(ece or float('nan')):8.4f}")
        TriadSummary._deltas(rows)
        return rows

    @staticmethod
    def _deltas(rows: dict[str, Any]) -> None:
        """GAP (real − synth) and DA CLOSURE (synth+DA − synth) point deltas. The paired CIs for these
        are emitted by run_triad at run time — this is the post-hoc point recap."""
        real, synth = rows.get("triad_real"), rows.get("triad_synth")
        if not (real and synth):
            return
        gap = real[0] - synth[0]
        log.info(f"\n>>> SIM-TO-REAL GAP (au_pro): {real[0]:.3f} - {synth[0]:.3f} = {gap:+.3f}")
        da = rows.get("triad_synth_da")
        if da:
            log.info(f">>> DA CLOSURE (AdaBN): {da[0]:.3f} (+{da[0] - synth[0]:.3f} vs synth; "
                     f"ECE {synth[4] or float('nan'):.4f} -> {da[4] or float('nan'):.4f})")
        cu = rows.get("triad_synth_curric")
        if cu:
            log.info(f">>> CURRICULUM: {cu[0]:.3f} (+{cu[0] - synth[0]:.3f} vs uniform synth)")

    @staticmethod
    def args(ap: ArgumentParser) -> None:
        ap.add_argument("--uri", default="sqlite:///mlflow.db")

    @staticmethod
    def run(args: Namespace) -> None:  # pragma: no cover  CLI glue; summarize is the logic
        TriadSummary.summarize(args.uri)


SPEC = Spec("triad-summary", TriadSummary.args, TriadSummary.run)
