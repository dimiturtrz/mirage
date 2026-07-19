"""Score a trained reconstruction model (loaded back from its MLflow run) through the shared eval
harness — image AUROC + AU-PRO per category + mean + per-defect, logged back into the same run.

A trained model is just a (fit_fn, score_fn) pair: fit returns the loaded model, score turns its
reconstruction residual into anomaly maps. So evaluation is one spine (harness.run); this is the
recon-model adapter to it. The test metrics land next to the run's loss curves (same run_id).

Run:  python -m surfscan.report evaluate --run-id <mlflow-run-id> [--cats bagel]
      [--image-score residual|mahalanobis]   # mahalanobis: latent-distance image score (VAE only)
"""
from __future__ import annotations

import argparse

import torch
from torch import nn

from core.compute import Compute
from core.data.static.dataset import GpuSplit
from core.data.static.mvtec import Split
from core.method import Method, ScoreArrays
from surfscan import tracking
from surfscan.dispatch import Spec
from surfscan.evaluation import harness, scoring
from surfscan.evaluation.result_types import EvalResult
from surfscan.training.hparams import HParams


class Evaluate:
    """Score a trained reconstruction model through the shared eval harness (recon-model adapter)."""

    @staticmethod
    def evaluate(
        run_id: str, cats: list[str] | None = None, device: str | torch.device = "cuda", image_score: str = "residual"
    ) -> EvalResult:
        hp = HParams(**tracking.Tracker.load_config(run_id))
        dev = device
        model = tracking.Tracker.load_model(run_id).to(dev)            # the built model + weights, from MLflow

        def fit(_c: str) -> nn.Module:
            return model

        def score(m: nn.Module, c: str) -> ScoreArrays:
            data = GpuSplit.load_split(split=Split.TEST, cats=[c], channels=hp.channels, device=dev, size=hp.size)
            sc = scoring.Scoring(m)
            amaps = sc.inpaint_maps(data, grid=hp.grid) if hp.model_type == "inpaint" else sc.anomaly_maps(data)
            scores = None
            if image_score == "mahalanobis":                  # latent-distance score (VAE encoder mu)
                good = GpuSplit.load_split(split=Split.TRAIN, label=0, cats=[c], channels=hp.channels,
                                           device=dev, size=hp.size)
                scores = scoring.Scoring.mahalanobis(sc.latents(good), sc.latents(data))
            return scoring.Scoring.score_arrays(amaps, data, scores=scores)

        return harness.Harness.run(hp.model_type, Method(fit, score), cats=cats or hp.cats, run_id=run_id)

    @staticmethod
    def args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--run-id", required=True)
        ap.add_argument("--cats", nargs="*", default=None)
        ap.add_argument("--image-score", default="residual", choices=["residual", "mahalanobis"])

    @staticmethod
    def run(args: argparse.Namespace) -> None:  # pragma: no cover  CLI glue; evaluate() is the logic
        Evaluate.evaluate(args.run_id, cats=args.cats, device=Compute.pick_device(), image_score=args.image_score)


SPEC = Spec("evaluate", Evaluate.args, Evaluate.run)
