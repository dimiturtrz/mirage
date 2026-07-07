"""Aggregate the sim-to-real triad across seeds -> the headline numbers.

Reads the clean triad_* runs from MLflow (one run per arm per seed), groups by arm, and reports
mean +/- std of au_pro / img_auroc / ECE. Then the two honest numbers: the sim-to-real GAP
(real - synth) and the DA CLOSURE (synth+DA - synth), plus the curriculum delta if present.

    python -m surfscan.experiments.triad_summary
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import mlflow
import numpy as np

ARMS = ["triad_real", "triad_synth", "triad_synth_da", "triad_synth_curric"]


def collect(uri="sqlite:///mlflow.db"):
    mlflow.set_tracking_uri(uri)
    c = mlflow.MlflowClient()
    by: dict[str, dict] = defaultdict(dict)          # name -> {seed: (au_pro, img, ece)}
    for e in c.search_experiments():
        for r in c.search_runs(e.experiment_id, max_results=5000):
            nm = r.data.tags.get("mlflow.runName", "")
            m = r.data.metrics
            if nm.startswith("triad_") and r.info.lifecycle_stage == "active" and "au_pro_mean" in m:
                seed = r.data.params.get("seed", r.info.run_id)     # dedup by seed, latest wins
                by[nm][seed] = (m["au_pro_mean"], m.get("img_auroc_mean"), m.get("ece"))
    return by


def _stat(rows):
    a = np.array([[x[0], x[1] if x[1] is not None else np.nan,
                   x[2] if x[2] is not None else np.nan] for x in rows], float)
    return {"au_pro": a[:, 0].mean(), "au_pro_std": a[:, 0].std(),
            "img_auroc": np.nanmean(a[:, 1]), "ece": np.nanmean(a[:, 2]), "n": len(rows)}


def summarize(uri="sqlite:///mlflow.db"):
    by = collect(uri)
    stats = {arm: _stat(list(by[arm].values())) for arm in ARMS if by.get(arm)}
    print(f"{'arm':22s} {'seeds':5s} {'au_pro':>16s} {'img_auroc':>10s} {'ECE':>8s}")
    for arm in ARMS:
        s = stats.get(arm)
        if s:
            print(f"{arm:22s} {s['n']:5d} {s['au_pro']:8.3f} +/-{s['au_pro_std']:.3f} "
                  f"{s['img_auroc']:10.3f} {s['ece']:8.4f}")
    r, sy = stats.get("triad_real"), stats.get("triad_synth")
    if r and sy:
        gap = r["au_pro"] - sy["au_pro"]
        print(f"\n>>> SIM-TO-REAL GAP (au_pro): {r['au_pro']:.3f} - {sy['au_pro']:.3f} = {gap:+.3f}")
        da = stats.get("triad_synth_da")
        if da:
            print(f">>> DA CLOSURE (AdaBN): {da['au_pro']:.3f} (+{da['au_pro'] - sy['au_pro']:.3f} vs synth; "
                  f"ECE {sy['ece']:.4f} -> {da['ece']:.4f})")
        cu = stats.get("triad_synth_curric")
        if cu:
            print(f">>> CURRICULUM: {cu['au_pro']:.3f} (+{cu['au_pro'] - sy['au_pro']:.3f} vs uniform synth)")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="sqlite:///mlflow.db")
    summarize(ap.parse_args().uri)
