"""Train the masked-inpainting AE on good samples — GPU-resident, bf16 (mirrors train.py).

Each step masks random patches of the input and reconstructs the FULL image from context, so
the model learns to predict normal from surroundings (it can't copy a hidden region).

Run:  python -m mirage.training.train_inpaint --out runs/inpaint --cats bagel [--epochs 200]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import optim

from mirage.data.dataset import load_split
from mirage.models.inpaint import InpaintAE, random_mask
from mirage.tracking import start
from mirage.training.hparams import HParams
from mirage.training.losses import masked_recon_loss


def train(hp: HParams, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(hp.seed)
    torch.set_float32_matmul_precision("high")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_split(split="train", label=0, cats=hp.cats, channels=hp.channels, device=dev, size=hp.size)
    n = len(data)
    patch = hp.size // hp.grid
    model = InpaintAE(in_ch=data.in_ch, base=hp.base, latent=hp.latent, size=hp.size,
                      depth=hp.depth, dropout=hp.dropout).to(dev, memory_format=torch.channels_last)
    run_model = torch.compile(model) if (hp.compile and dev == "cuda") else model
    opt = optim.AdamW(model.parameters(), lr=hp.lr)
    amp = dev == "cuda" and hp.bf16
    trk = start("mirage", out.name, params=hp.model_dump(), run_dir=out)

    print(f"inpaint: {n} good | in_ch={data.in_ch} | patch={patch} ratio={hp.mask_ratio} | dev={dev} | bf16={amp}")
    for epoch in range(hp.epochs):
        model.train()
        idx = torch.randperm(n, device=dev)
        tot = 0.0
        nb = 0
        t0 = time.time()
        for i in range(0, n, hp.batch):
            b = idx[i:i + hp.batch]
            x = data.x[b].to(memory_format=torch.channels_last)
            m = data.valid[b]
            keep = random_mask(x.shape[0], hp.size, patch, hp.mask_ratio, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                recon = run_model(x * keep)                 # reconstruct FULL from masked input
                loss = masked_recon_loss(recon, x, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        trk.metric("recon", tot / nb, step=epoch)
        if epoch % max(1, hp.epochs // 20) == 0 or epoch == hp.epochs - 1:
            print(f"ep {epoch:3d}  recon {tot/nb:.4f}  {time.time()-t0:.2f}s")

    torch.save(model.state_dict(), out / "model.pt")
    (out / "config.json").write_text(json.dumps(hp.model_dump(), indent=2))
    trk.artifact(str(out / "config.json"))
    trk.end()
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--grid", type=int, default=None)
    ap.add_argument("--mask-ratio", type=float, default=None)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    hp = HParams(model_type="inpaint", epochs=args.epochs)
    if args.cats is not None:
        hp.cats = args.cats
    if args.grid is not None:
        hp.grid = args.grid
    if args.mask_ratio is not None:
        hp.mask_ratio = args.mask_ratio
    if args.no_compile:
        hp.compile = False
    train(hp, args.out)


if __name__ == "__main__":
    main()
