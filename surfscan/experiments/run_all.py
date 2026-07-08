"""Per-category VAE train+eval over all categories -> aggregate mean. Logs to MLflow.

One model per object is the MVTec convention. Each category is its own MLflow run (vae_<cat>);
the cross-category aggregate is logged as a 'vae_all' run for the comparison board.

Run:  python -m surfscan.run vae [--cats ...] [--epochs 100]
"""
from __future__ import annotations

import numpy as np

from core.compute import pick_device
from core.data import mvtec
from core.obs import get
from surfscan import tracking
from surfscan.dispatch import Spec, add_cats
from surfscan.evaluation.evaluate import evaluate
from surfscan.training.hparams import HParams
from surfscan.training.train import train

log = get()


def _args(ap):
    add_cats(ap)
    ap.add_argument("--epochs", type=int, default=100)


def _run(args):
    dev = pick_device()
    cats = args.cats or mvtec.categories()
    rows = []
    for c in cats:
        log.info(f"\n===== {c} =====")
        hp = HParams(cats=[c], epochs=args.epochs, compile=False)
        run_id = train(hp, run_name=f"vae_{c}", device=dev)
        rows.append(evaluate(run_id, cats=[c], device=dev)["per_category"][0])

    auroc = float(np.nanmean([r["img_auroc"] for r in rows]))
    aupro = float(np.nanmean([r["au_pro"] for r in rows]))
    log.info("\n===== AGGREGATE (per-category models) =====")
    for r in rows:
        log.info(f"  {r['category']:12s}  img_auroc {r['img_auroc']:.3f}   au_pro {r['au_pro']:.3f}")
    log.info(f"  {'MEAN':12s}  img_auroc {auroc:.3f}   au_pro {aupro:.3f}")

    with tracking.run("surfscan", "vae_all", params={"method": "vae_per_category", "cats": ",".join(cats)}):
        tracking.metrics({"img_auroc_mean": auroc, "au_pro_mean": aupro})
        tracking.per_group("au_pro", {r["category"]: r["au_pro"] for r in rows})
        tracking.artifact_json("aggregate.json",
                               {"per_category": rows, "mean": {"img_auroc": auroc, "au_pro": aupro}})


SPEC = Spec("vae", _args, _run)
