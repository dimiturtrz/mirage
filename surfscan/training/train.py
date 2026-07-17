"""Train a reconstruction model on good samples — GPU-resident, bf16, on-device epochs.

One train spine; the model type (vae | inpaint) selects a (build, step) pair from the registry.
The whole train-good split lives in VRAM; each epoch shuffles indices and slices batches on the
GPU (no DataLoader, no CPU↔GPU copy). bf16 autocast + channels_last + TF32 + torch.compile.
Params, the loss curves, and the model are logged to MLflow (browse with `mlflow ui`).

Run:  python -m surfscan.training.train [--model vae|inpaint] [--cats bagel] [--epochs 100] [--no-compile]
Returns the MLflow run id (so evaluate can load the model back).
"""
from __future__ import annotations

import argparse
import time

import torch
from torch import optim

from core.compute import Compute
from core.data.static.dataset import GpuSplit
from core.data.static.mvtec import Split
from core.obs import Obs
from surfscan import tracking
from surfscan.models.inpaint import InpaintAE
from surfscan.models.vae import ConvVAE
from surfscan.training.hparams import HParams
from surfscan.training.losses import Losses
from surfscan.training.trainer import Trainer

log = Obs.get()


class TrainRun:
    """The reconstruction-model train spine + the per-model (build, step) registry helpers. `hp` (the
    run's hyperparameters) is fixed for the run's life and threaded by every method -> held in __init__;
    the (build, step) helpers read `self.hp` and are dispatched through `_REGISTRY` as unbound functions."""

    def __init__(self, hp: HParams):
        self.hp = hp

    def vae_model(self, in_ch):
        return ConvVAE(self.hp.model_cfg(in_ch))

    def inpaint_model(self, in_ch):
        return InpaintAE(self.hp.model_cfg(in_ch))

    def vae_step(self, run_model, x, m):
        recon, mu, logvar = run_model(x)
        rl = Losses.masked_recon_loss(recon, x, m)
        kl = Losses.kl_loss(mu, logvar)
        return rl + self.hp.beta * kl, {"recon": rl, "kl": kl}

    def inpaint_step(self, run_model, x, m):
        hp = self.hp
        patch = hp.size // hp.grid
        keep = InpaintAE.random_mask(x.shape[0], hp.size, patch, hp.mask_ratio, x.device)
        recon = run_model(x * keep)                 # reconstruct FULL image from masked input
        return Losses.masked_recon_loss(recon, x, m), {}

    def train(self, run_name: str | None = None, device: str = "cuda") -> str:
        hp = self.hp
        build, step = _REGISTRY[hp.model_type]
        run_name = run_name or hp.model_type
        torch.manual_seed(hp.seed)
        Compute.enable_tf32()
        dev = device

        data = GpuSplit.load_split(split=Split.TRAIN, label=0, cats=hp.cats, channels=hp.channels,
                                   device=dev, size=hp.size)
        n = len(data)
        model = build(self, data.in_ch).to(dev, memory_format=torch.channels_last)
        run_model = torch.compile(model) if (hp.compile and dev == "cuda") else model
        opt = optim.AdamW(model.parameters(), lr=hp.lr)
        amp = dev == "cuda" and hp.bf16

        # the SGD mechanics are the shared Trainer; the per-epoch metric aggregation + mlflow logging
        # (this method's observability, not the loop) live in the step/after_epoch closures.
        ep = {"i": 0, "t0": 0.0, "tot": 0.0, "nb": 0, "extra": {}}

        def step_fn(idx):
            x = data.x[idx].to(memory_format=torch.channels_last)
            with Compute.autocast(x, amp=amp):
                loss, extra = step(self, run_model, x, data.valid[idx])
            ep["tot"] += loss.item(); ep["nb"] += 1
            for k, v in extra.items():
                ep["extra"][k] = ep["extra"].get(k, 0.0) + v.item()
            return loss

        def after_epoch():
            row = {"loss": ep["tot"] / ep["nb"], **{k: t / ep["nb"] for k, t in ep["extra"].items()}}
            tracking.Tracker.metrics(row, step=ep["i"])
            if ep["i"] % max(1, hp.epochs // 20) == 0 or ep["i"] == hp.epochs - 1:
                parts = "  ".join(f"{k} {v:.4f}" for k, v in row.items())
                log.info(f"ep {ep['i']:3d}  {parts}  {time.time() - ep['t0']:.2f}s")
            ep.update(i=ep["i"] + 1, t0=time.time(), tot=0.0, nb=0, extra={})

        with tracking.Tracker.run("surfscan", run_name, params=hp.model_dump()) as run_id:
            log.info(f"train[{hp.model_type}]: {n} good | in_ch={data.in_ch} | dev={dev} | "
                     f"bf16={amp} | compile={hp.compile}")
            ep["t0"] = time.time()
            Trainer(model, opt, dev, batch=hp.batch).fit(n, hp.epochs, step_fn, after_epoch=after_epoch)

            tracking.Tracker.artifact_json("config.json", hp.model_dump())
            tracking.Tracker.log_model(model)
            log.info(f"logged -> mlflow run {run_id}")
            return run_id


# model_type -> (build_model(hp, in_ch), step(run_model, x, valid, hp) -> (loss, extra_metrics))
_REGISTRY = {
    "vae": (TrainRun.vae_model, TrainRun.vae_step),
    "inpaint": (TrainRun.inpaint_model, TrainRun.inpaint_step),
}


def main():  # pragma: no cover  CLI entry (argparse); train is the tested spine
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vae", choices=list(_REGISTRY))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--channels", nargs="*", default=None, help="e.g. xyz | xyz rgb (fused)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--grid", type=int, default=None, help="inpaint: masking grid")
    ap.add_argument("--mask-ratio", type=float, default=None, help="inpaint: fraction masked")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    hp = HParams(model_type=args.model)
    for k in ("cats", "channels", "epochs", "batch", "dropout", "grid", "mask_ratio"):
        v = getattr(args, k)
        if v is not None:
            setattr(hp, k, v)
    if args.no_compile:
        hp.compile = False
    TrainRun(hp).train(args.run_name, device=Compute.pick_device())


if __name__ == "__main__":
    main()
