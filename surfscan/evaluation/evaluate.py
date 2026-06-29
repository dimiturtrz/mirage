"""Score a trained reconstruction model (loaded back from its MLflow run) — image AUROC + AU-PRO,
per category + mean. Logs the test metrics + results artifact back into the same run.

Run:  python -m surfscan.evaluation.evaluate --run-id <mlflow-run-id> [--cats bagel]
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl
import torch

from surfscan import tracking
from surfscan.data.dataset import load_split
from surfscan.evaluation import metrics, scoring
from surfscan.training.hparams import HParams


def evaluate(run_id, cats=None):
    hp = HParams(**tracking.load_config(run_id))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_split(split="test", cats=cats or hp.cats, channels=hp.channels, device=dev, size=hp.size)
    model = tracking.load_model(run_id).to(dev)            # the built model + weights, from MLflow

    amaps = (scoring.inpaint_maps(model, data, grid=hp.grid) if hp.model_type == "inpaint"
             else scoring.anomaly_maps(model, data))
    valids = data.valid.squeeze(1).cpu().numpy().astype(bool)
    masks = data.gt.squeeze(1).cpu().numpy().astype(bool)
    scores = scoring.image_scores(amaps, valids)
    labels = data.df["label"].to_numpy()
    categories = np.array(data.df["category"].to_list())

    rows = []
    for c in sorted(set(categories)):
        idx = categories == c
        rows.append({"category": c, "n": int(idx.sum()),
                     "img_auroc": metrics.image_auroc(scores[idx], labels[idx]),
                     "au_pro": metrics.au_pro(amaps[idx], masks[idx], valids[idx])})
    mean = {"img_auroc": float(np.nanmean([r["img_auroc"] for r in rows])),
            "au_pro": float(np.nanmean([r["au_pro"] for r in rows]))}
    print(pl.DataFrame(rows))
    print(f"MEAN  img_auroc {mean['img_auroc']:.3f}   au_pro {mean['au_pro']:.3f}")

    res = {"run_id": run_id, "per_category": rows, "mean": mean}
    with tracking.resume(run_id):
        tracking.metrics({"test_img_auroc": mean["img_auroc"], "test_au_pro": mean["au_pro"]})
        tracking.artifact_json("results.json", res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cats", nargs="*", default=None)
    args = ap.parse_args()
    evaluate(args.run_id, cats=args.cats)


if __name__ == "__main__":
    main()
