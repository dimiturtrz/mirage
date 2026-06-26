"""Score a trained reconstruction model through the eval harness — the contribution.

Image AUROC (detection) + pixel AU-PRO (localization), per category + mean. Writes
runs/<run>/results.json and logs the summary into the run's MLflow entry.

Run:  python -m mirage.evaluation.evaluate --run runs/vae [--cats bagel]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from mirage.data.dataset import load_split
from mirage.evaluation import metrics, scoring
from mirage.models.inpaint import InpaintAE
from mirage.models.vae import ConvVAE
from mirage.tracking import resume
from mirage.training.hparams import HParams


def evaluate(run: Path, cats=None):
    hp = HParams(**json.loads((run / "config.json").read_text()))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_split(split="test", cats=cats or hp.cats, channels=hp.channels, device=dev, size=hp.size)

    cls = InpaintAE if hp.model_type == "inpaint" else ConvVAE
    model = cls(in_ch=data.in_ch, base=hp.base, latent=hp.latent, size=hp.size, depth=hp.depth,
                dropout=hp.dropout).to(dev)
    model.load_state_dict(torch.load(run / "model.pt", map_location=dev))

    if hp.model_type == "inpaint":
        amaps = scoring.inpaint_maps(model, data, grid=hp.grid)
    else:
        amaps = scoring.anomaly_maps(model, data)
    valids = data.valid.squeeze(1).cpu().numpy().astype(bool)
    masks = data.gt.squeeze(1).cpu().numpy().astype(bool)
    scores = scoring.image_scores(amaps, valids)
    labels = data.df["label"].to_numpy()
    categories = np.array(data.df["category"].to_list())

    rows = []
    for c in sorted(set(categories)):
        idx = categories == c
        rows.append({
            "category": c, "n": int(idx.sum()),
            "img_auroc": metrics.image_auroc(scores[idx], labels[idx]),
            "au_pro": metrics.au_pro(amaps[idx], masks[idx], valids[idx]),
        })
    df = pl.DataFrame(rows)
    mean = {
        "img_auroc": float(np.nanmean(df["img_auroc"].to_numpy())),
        "au_pro": float(np.nanmean(df["au_pro"].to_numpy())),
    }
    print(df)
    print(f"MEAN  img_auroc {mean['img_auroc']:.3f}   au_pro {mean['au_pro']:.3f}")

    res = {"run": str(run), "per_category": rows, "mean": mean}
    (run / "results.json").write_text(json.dumps(res, indent=2))

    trk = resume("mirage", run.name, run_dir=run)
    trk.metric("test_img_auroc", mean["img_auroc"])
    trk.metric("test_au_pro", mean["au_pro"])
    trk.artifact(str(run / "results.json"))
    trk.end()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cats", nargs="*", default=None)
    args = ap.parse_args()
    evaluate(args.run, cats=args.cats)


if __name__ == "__main__":
    main()
