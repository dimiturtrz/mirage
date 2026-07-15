"""Per-category VAE train+eval over all categories -> aggregate mean. Logs to MLflow.

One model per object is the MVTec convention. Each category is its own MLflow run (vae_<cat>);
the cross-category aggregate is logged as a 'vae_all' run for the comparison board.

Run:  python -m surfscan.run vae [--cats ...] [--epochs 100]
"""
from __future__ import annotations

import numpy as np

from core.compute import Compute
from core.data import mvtec
from core.obs import Obs
from surfscan import tracking
from surfscan.dispatch import Dispatch, Spec
from surfscan.evaluation.evaluate import Evaluate
from surfscan.evaluation.result_types import RunParams, Scores
from surfscan.training.hparams import HParams
from surfscan.training.train import TrainRun

log = Obs.get()


class VaeRun:
    """Per-category VAE train+eval over all categories -> aggregate mean. Logs to MLflow."""

    @staticmethod
    def args(ap):
        Dispatch.add_cats(ap)
        ap.add_argument("--epochs", type=int, default=100)

    @staticmethod
    def run(args):
        dev = Compute.pick_device()
        cats = args.cats or mvtec.Mvtec().categories()
        rows = []
        for c in cats:
            log.info(f"\n===== {c} =====")
            hp = HParams(cats=[c], epochs=args.epochs, compile=False)
            run_id = TrainRun(hp).train(run_name=f"vae_{c}", device=dev)
            rows.append(Evaluate.evaluate(run_id, cats=[c], device=dev)["per_category"][0])

        mean = Scores(float(np.nanmean([r["img_auroc"] for r in rows])),
                      float(np.nanmean([r["au_pro"] for r in rows])))
        log.info("\n===== AGGREGATE (per-category models) =====")
        for r in rows:
            log.info(f"  {r['category']:12s}  img_auroc {r['img_auroc']:.3f}   au_pro {r['au_pro']:.3f}")
        log.info(f"  {'MEAN':12s}  img_auroc {mean.img_auroc:.3f}   au_pro {mean.au_pro:.3f}")

        with tracking.Tracker.run("surfscan", "vae_all", params=RunParams("vae_per_category", cats).as_dict()):
            tracking.Tracker.metrics(mean.tracker_means())
            tracking.Tracker.per_group("au_pro", {r["category"]: r["au_pro"] for r in rows})
            tracking.Tracker.artifact_json("aggregate.json",
                                   {"per_category": rows, "mean": mean.as_dict()})


SPEC = Spec("vae", VaeRun.args, VaeRun.run)
