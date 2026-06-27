"""Train a reconstruction model (VAE) on good samples — GPU-resident, bf16, on-device epochs.

The whole train-good split lives in VRAM; each epoch shuffles indices and slices batches on
the GPU (no DataLoader, no CPU↔GPU copy). bf16 autocast + channels_last + TF32 + torch.compile.

Run:  python -m mirage.training.train --out runs/vae [--cats bagel] [--epochs 100] [--no-compile]
A run is reproducible from runs/<run>/config.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import optim

from mirage.data.dataset import load_split
from mirage.models.vae import ConvVAE
from mirage.tracking import start
from mirage.training.hparams import HParams
from mirage.training.losses import kl_loss, masked_recon_loss


def train(hp: HParams, out: Path):
    out.mkdir(parents=True, exist_ok=True)
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
    trk = start("mirage", out.name, params=hp.model_dump(), run_dir=out)

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
        trk.metric("loss", tot / nb, step=epoch)
        trk.metric("recon", rtot / nb, step=epoch)
        trk.metric("kl", ktot / nb, step=epoch)
        if epoch % max(1, hp.epochs // 20) == 0 or epoch == hp.epochs - 1:
            print(f"ep {epoch:3d}  loss {tot/nb:.4f}  recon {rtot/nb:.4f}  kl {ktot/nb:.3f}  {time.time()-t0:.2f}s")

    torch.save(model.state_dict(), out / "model.pt")
    (out / "config.json").write_text(json.dumps(hp.model_dump(), indent=2))
    trk.artifact(str(out / "config.json"))
    trk.end()
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
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
    train(hp, args.out)


if __name__ == "__main__":
    main()
