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

from core.data.dataset import load_split
from core.obs import get
from surfscan import tracking
from surfscan.models.inpaint import InpaintAE, random_mask
from surfscan.models.vae import ConvVAE
from surfscan.training.hparams import HParams
from surfscan.training.losses import kl_loss, masked_recon_loss

log = get()


def _vae_model(hp, in_ch):
    return ConvVAE(in_ch=in_ch, base=hp.base, latent=hp.latent, size=hp.size, depth=hp.depth, dropout=hp.dropout)


def _inpaint_model(hp, in_ch):
    return InpaintAE(in_ch=in_ch, base=hp.base, latent=hp.latent, size=hp.size, depth=hp.depth, dropout=hp.dropout)


def _vae_step(run_model, x, m, hp):
    recon, mu, logvar = run_model(x)
    rl = masked_recon_loss(recon, x, m)
    kl = kl_loss(mu, logvar)
    return rl + hp.beta * kl, {"recon": rl, "kl": kl}


def _inpaint_step(run_model, x, m, hp):
    patch = hp.size // hp.grid
    keep = random_mask(x.shape[0], hp.size, patch, hp.mask_ratio, x.device)
    recon = run_model(x * keep)                 # reconstruct FULL image from masked input
    return masked_recon_loss(recon, x, m), {}


# model_type -> (build_model(hp, in_ch), step(run_model, x, valid, hp) -> (loss, extra_metrics))
_REGISTRY = {
    "vae": (_vae_model, _vae_step),
    "inpaint": (_inpaint_model, _inpaint_step),
}


def train(hp: HParams, run_name: str | None = None) -> str:
    build, step = _REGISTRY[hp.model_type]
    run_name = run_name or hp.model_type
    torch.manual_seed(hp.seed)
    torch.set_float32_matmul_precision("high")          # TF32 matmuls
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_split(split="train", label=0, cats=hp.cats, channels=hp.channels, device=dev, size=hp.size)
    n = len(data)
    model = build(hp, data.in_ch).to(dev, memory_format=torch.channels_last)
    run_model = torch.compile(model) if (hp.compile and dev == "cuda") else model
    opt = optim.AdamW(model.parameters(), lr=hp.lr)
    amp = dev == "cuda" and hp.bf16

    with tracking.run("surfscan", run_name, params=hp.model_dump()) as run_id:
        log.info(f"train[{hp.model_type}]: {n} good | in_ch={data.in_ch} | dev={dev} | "
                 f"bf16={amp} | compile={hp.compile}")
        for epoch in range(hp.epochs):
            model.train()
            idx = torch.randperm(n, device=dev)
            tot = 0.0
            nb = 0
            extra_tot: dict[str, float] = {}
            t0 = time.time()
            for i in range(0, n, hp.batch):
                b = idx[i:i + hp.batch]
                x = data.x[b].to(memory_format=torch.channels_last)
                m = data.valid[b]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    loss, extra = step(run_model, x, m, hp)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item(); nb += 1
                for k, v in extra.items():
                    extra_tot[k] = extra_tot.get(k, 0.0) + v.item()
            row = {"loss": tot / nb, **{k: t / nb for k, t in extra_tot.items()}}
            tracking.metrics(row, step=epoch)
            if epoch % max(1, hp.epochs // 20) == 0 or epoch == hp.epochs - 1:
                parts = "  ".join(f"{k} {v:.4f}" for k, v in row.items())
                log.info(f"ep {epoch:3d}  {parts}  {time.time()-t0:.2f}s")

        tracking.artifact_json("config.json", hp.model_dump())
        tracking.log_model(model)
        log.info(f"logged -> mlflow run {run_id}")
        return run_id


def main():
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
    train(hp, args.run_name)


if __name__ == "__main__":
    main()
