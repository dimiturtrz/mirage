"""Train a reconstruction model (VAE) on good samples — GPU-resident, bf16, on-device epochs.

The whole train-good split lives in VRAM; each epoch shuffles indices and slices batches on
the GPU (no DataLoader, no CPU↔GPU copy). bf16 autocast + channels_last + TF32 + torch.compile.
Params, the loss curves, and the model are logged to MLflow (browse with `mlflow ui`).

Run:  python -m surfscan.training.train [--run-name vae] [--cats bagel] [--epochs 100] [--no-compile]
Returns the MLflow run id (so evaluate can load the model back).
"""
from __future__ import annotations

import argparse
import time

import torch
from torch import optim

from surfscan import tracking
from surfscan.data.dataset import load_split
from surfscan.models.vae import ConvVAE
from surfscan.training.hparams import HParams
from surfscan.training.losses import kl_loss, masked_recon_loss


def train(hp: HParams, run_name: str = "vae") -> str:
    torch.manual_seed(hp.seed)
    torch.set_float32_matmul_precision("high")          # TF32 matmuls
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_split(split="train", label=0, cats=hp.cats, channels=hp.channels, device=dev, size=hp.size)
    n = len(data)
    model = ConvVAE(in_ch=data.in_ch, base=hp.base, latent=hp.latent, size=hp.size, depth=hp.depth, dropout=hp.dropout)
    model = model.to(dev, memory_format=torch.channels_last)
    run_model = torch.compile(model) if (hp.compile and dev == "cuda") else model
    opt = optim.AdamW(model.parameters(), lr=hp.lr)
    amp = dev == "cuda" and hp.bf16

    with tracking.run("surfscan", run_name, params=hp.model_dump()) as run_id:
        print(f"train: {n} good samples | in_ch={data.in_ch} | dev={dev} | bf16={amp} | compile={hp.compile}")
        for epoch in range(hp.epochs):
            model.train()
            idx = torch.randperm(n, device=dev)
            tot = rtot = ktot = 0.0
            nb = 0
            t0 = time.time()
            for i in range(0, n, hp.batch):
                b = idx[i:i + hp.batch]
                x = data.x[b].to(memory_format=torch.channels_last)
                m = data.valid[b]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    recon, mu, logvar = run_model(x)
                    rl = masked_recon_loss(recon, x, m)
                    kl = kl_loss(mu, logvar)
                    loss = rl + hp.beta * kl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item(); rtot += rl.item(); ktot += kl.item(); nb += 1
            tracking.metrics({"loss": tot / nb, "recon": rtot / nb, "kl": ktot / nb}, step=epoch)
            if epoch % max(1, hp.epochs // 20) == 0 or epoch == hp.epochs - 1:
                print(f"ep {epoch:3d}  loss {tot/nb:.4f}  recon {rtot/nb:.4f}  kl {ktot/nb:.3f}  {time.time()-t0:.2f}s")

        tracking.artifact_json("config.json", hp.model_dump())
        tracking.log_model(model)
        print(f"logged -> mlflow run {run_id}")
        return run_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="vae")
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--channels", nargs="*", default=None, help="e.g. xyz | xyz rgb (fused)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    hp = HParams()
    if args.cats is not None:
        hp.cats = args.cats
    if args.channels is not None:
        hp.channels = args.channels
    if args.epochs is not None:
        hp.epochs = args.epochs
    if args.batch is not None:
        hp.batch = args.batch
    if args.dropout is not None:
        hp.dropout = args.dropout
    if args.no_compile:
        hp.compile = False
    train(hp, args.run_name)


if __name__ == "__main__":
    main()
