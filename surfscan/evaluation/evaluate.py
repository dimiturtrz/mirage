"""Score a trained reconstruction model (loaded back from its MLflow run) through the shared eval
harness — image AUROC + AU-PRO per category + mean + per-defect, logged back into the same run.

A trained model is just a (fit_fn, score_fn) pair: fit returns the loaded model, score turns its
reconstruction residual into anomaly maps. So evaluation is one spine (harness.run); this is the
recon-model adapter to it. The test metrics land next to the run's loss curves (same run_id).

Run:  python -m surfscan.report evaluate --run-id <mlflow-run-id> [--cats bagel]
      [--image-score residual|mahalanobis]   # mahalanobis: latent-distance image score (VAE only)
"""
from __future__ import annotations

from core.compute import Compute
from core.data.dataset import GpuSplit
from core.method import Method
from surfscan import tracking
from surfscan.dispatch import Spec
from surfscan.evaluation import harness, scoring
from surfscan.training.hparams import HParams


def evaluate(run_id, cats=None, device="cuda", image_score="residual"):
    hp = HParams(**tracking.load_config(run_id))
    dev = device
    model = tracking.load_model(run_id).to(dev)            # the built model + weights, from MLflow

    def fit(_c):
        return model

    def score(m, c):
        data = GpuSplit.load_split(split="test", cats=[c], channels=hp.channels, device=dev, size=hp.size)
        amaps = (scoring.inpaint_maps(m, data, grid=hp.grid) if hp.model_type == "inpaint"
                 else scoring.anomaly_maps(m, data))
        scores = None
        if image_score == "mahalanobis":                  # latent-distance score (VAE encoder mu)
            good = GpuSplit.load_split(split="train", label=0, cats=[c], channels=hp.channels, device=dev, size=hp.size)
            scores = scoring.mahalanobis(scoring.latents(m, good), scoring.latents(m, data))
        return scoring.score_arrays(amaps, data, scores=scores)

    return harness.run(hp.model_type, Method(fit, score), cats=cats or hp.cats, run_id=run_id)


def _args(ap):
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--image-score", default="residual", choices=["residual", "mahalanobis"])


def _run(args):  # pragma: no cover  CLI glue; evaluate() is the logic (mlflow-omitted whole-file)
    evaluate(args.run_id, cats=args.cats, device=Compute.pick_device(), image_score=args.image_score)


SPEC = Spec("evaluate", _args, _run)
