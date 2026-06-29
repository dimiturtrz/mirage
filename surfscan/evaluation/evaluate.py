"""Score a trained reconstruction model (loaded back from its MLflow run) through the shared eval
harness — image AUROC + AU-PRO per category + mean + per-defect, logged back into the same run.

A trained model is just a (fit_fn, score_fn) pair: fit returns the loaded model, score turns its
reconstruction residual into anomaly maps. So evaluation is one spine (harness.run); this is the
recon-model adapter to it. The test metrics land next to the run's loss curves (same run_id).

Run:  python -m surfscan.evaluation.evaluate --run-id <mlflow-run-id> [--cats bagel]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from surfscan import tracking
from surfscan.data.dataset import load_split
from surfscan.evaluation import harness, scoring
from surfscan.training.hparams import HParams


def evaluate(run_id, cats=None):
    hp = HParams(**tracking.load_config(run_id))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = tracking.load_model(run_id).to(dev)            # the built model + weights, from MLflow

    def fit(_c):
        return model

    def score(m, c):
        data = load_split(split="test", cats=[c], channels=hp.channels, device=dev, size=hp.size)
        amaps = (scoring.inpaint_maps(m, data, grid=hp.grid) if hp.model_type == "inpaint"
                 else scoring.anomaly_maps(m, data))
        valids = data.valid.squeeze(1).cpu().numpy().astype(bool)
        masks = data.gt.squeeze(1).cpu().numpy().astype(bool)
        scores = scoring.image_scores(amaps, valids)
        return (amaps, valids, masks, scores,
                data.df["label"].to_numpy(), np.array(data.df["defect"].to_list()))

    return harness.run(hp.model_type, fit, score, cats=cats or hp.cats, run_id=run_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cats", nargs="*", default=None)
    args = ap.parse_args()
    evaluate(args.run_id, cats=args.cats)


if __name__ == "__main__":
    main()
