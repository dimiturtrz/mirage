"""Stage-1 sim-to-real triad — the honest gap number.

ONE supervised defect-segmentation U-Net (DRAEM's disc net, no recon branch to confound), trained
from three different sources and scored on ONE shared real eval set — so the only variable is where
the *labels* came from:

    real  -> real   : train on REAL defect masks (calib half of the test split)      = the ceiling
    synth -> real   : train on OUR synthetic defects painted on good scans            = the GAP
    synth+DA -> real: synth-trained, then AdaBN (recompute BN stats on real, unlabeled) = the CLOSURE

gap (pp) = real->real AU-PRO  -  synth->real AU-PRO. "modeled it" never stands in for "measured it
transferred" — this module is that measurement. rgb channel (the 0.91 ceiling's modality).

Run:  python -m surfscan.experiments.run_triad [--cats bagel ...] [--epochs 100] [--seed 0]
      [--arms real synth synth_da] [--curriculum]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from core.data.dataset import load_split
from core.data.defects import KINDS, synthesize
from core.obs import get
from surfscan.evaluation import harness, scoring
from surfscan.models.draem import UNet
from surfscan.training import curriculum as curric

log = get()

CH = ("rgb",)      # channels; overridable via --channels (set in main)
BATCH = 16


def _split_idx(labels):
    """Deterministic per-label alternating split of the test set -> (calib_idx, eval_idx).
    Stratified by label so both halves carry good + defective samples. Same split in fit and score."""
    idx = np.arange(len(labels))
    calib, ev = [], []
    for lab in np.unique(labels):
        li = idx[labels == lab]
        calib += list(li[1::2])
        ev += list(li[::2])
    return np.array(sorted(calib)), np.array(sorted(ev))


def _new(in_ch, dev):
    return UNet(in_ch, 1).to(dev, memory_format=torch.channels_last)


def _step(model, opt, x, v, m):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x.to(memory_format=torch.channels_last))
        loss = F.binary_cross_entropy_with_logits(logits.float(), m, weight=v)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.detach())


def fit_synth(c, epochs, seed, dev, curriculum=False):
    """Train on good scans + on-the-fly synthetic defects. The synth->real arm."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    good = load_split(split="train", label=0, cats=[c], channels=CH, device=dev)
    model = _new(good.in_ch, dev)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    n = len(good)
    ctrl = curric.KindCurriculum(KINDS, seed=seed) if curriculum else None
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, device=dev)
        for i in range(0, n, BATCH):
            b = order[i:i + BATCH]
            x, v = good.x[b], good.valid[b]
            kinds = ctrl.sample(len(b)) if ctrl else KINDS
            aug, mask = synthesize(x, v, rng, channels=CH, kinds=kinds)
            loss = _step(model, opt, aug, v, mask)
            if ctrl:
                ctrl.observe(kinds, loss)
        if ctrl:
            ctrl.end_epoch()
    return model


def fit_real(c, epochs, seed, dev):
    """Train on REAL defect masks (calib half of test). The real->real ceiling — note it's a
    *scarce-label* ceiling (only ~half of an already-small defect set), not an oracle; that's the
    honest supervised bar for this detector, and why the unsupervised memory bank can beat it."""
    torch.manual_seed(seed)
    test = load_split(split="test", cats=[c], channels=CH, device=dev)
    ci, _ = _split_idx(test.df["label"].to_numpy())
    t = torch.as_tensor(ci, device=dev)
    x, v, g = test.x[t], test.valid[t], test.gt[t]
    model = _new(test.in_ch, dev)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    n = x.shape[0]
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, device=dev)
        for i in range(0, n, BATCH):
            b = order[i:i + BATCH]
            _step(model, opt, x[b], v[b], g[b])
    return model


@torch.no_grad()
def _adabn(model, x, passes=2, batch=BATCH):
    """AdaBN (Li 2017): reset BN running stats, recompute them on the target (real) images via
    forward passes in train mode (momentum=None -> cumulative average). No labels, no backward.
    Transductive — it adapts on the eval images it will then score (standard test-time adaptation;
    uses their statistics only, never their labels)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    for _ in range(passes):
        for i in range(0, x.shape[0], batch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x[i:i + batch].to(memory_format=torch.channels_last))
    model.eval()
    return model


def fit_synth_da(c, epochs, seed, dev, curriculum=False):
    """synth-trained, then AdaBN-adapted to the real eval images (unlabeled). The closure arm."""
    model = fit_synth(c, epochs, seed, dev, curriculum=curriculum)
    test = load_split(split="test", cats=[c], channels=CH, device=dev)
    _, ei = _split_idx(test.df["label"].to_numpy())
    return _adabn(model, test.x[torch.as_tensor(ei, device=dev)])


@torch.no_grad()
def score(model, c, dev, batch=BATCH):
    """Score the shared real EVAL half -> ScoreArrays fields for the harness."""
    test = load_split(split="test", cats=[c], channels=CH, device=dev)
    labels = test.df["label"].to_numpy()
    defects = np.array(test.df["defect"].to_list())
    _, ei = _split_idx(labels)
    t = torch.as_tensor(ei, device=dev)
    x, v, g = test.x[t], test.valid[t], test.gt[t]
    model.eval()
    amaps = []
    for i in range(0, x.shape[0], batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[i:i + batch].to(memory_format=torch.channels_last))
        amaps.append((torch.sigmoid(logits.float()) * v[i:i + batch]).squeeze(1).cpu())
    amaps = torch.cat(amaps).numpy()
    valids = v.squeeze(1).cpu().numpy().astype(bool)
    masks = g.squeeze(1).cpu().numpy().astype(bool)
    scores = scoring.image_scores(amaps, valids)
    return amaps, valids, masks, scores, labels[ei], defects[ei]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=["real", "synth", "synth_da"])
    ap.add_argument("--channels", nargs="*", default=["rgb"])
    ap.add_argument("--curriculum", action="store_true")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    global CH
    CH = tuple(args.channels)
    dev = "cuda"

    def score_fn(m, c):
        return score(m, c, dev)

    fits = {
        "real": lambda c: fit_real(c, args.epochs, args.seed, dev),
        "synth": lambda c: fit_synth(c, args.epochs, args.seed, dev, curriculum=args.curriculum),
        "synth_da": lambda c: fit_synth_da(c, args.epochs, args.seed, dev, curriculum=args.curriculum),
    }
    res = {}
    for arm in args.arms:
        tag = f"triad_{arm}" + ("_curric" if args.curriculum and arm != "real" else "")
        log.info(f"\n===== {tag} (seed {args.seed}) =====")
        res[arm] = harness.run(tag, fits[arm], score_fn, cats=args.cats,
                               params={"arm": arm, "seed": args.seed, "epochs": args.epochs,
                                       "curriculum": args.curriculum})
    if "real" in res and "synth" in res:
        gap = res["real"]["mean"]["au_pro"] - res["synth"]["mean"]["au_pro"]
        log.info(f"\n>>> SIM-TO-REAL GAP (au_pro): real {res['real']['mean']['au_pro']:.3f} "
              f"- synth {res['synth']['mean']['au_pro']:.3f} = {gap:+.3f} pp")
        if "synth_da" in res:
            closed = res["synth_da"]["mean"]["au_pro"] - res["synth"]["mean"]["au_pro"]
            log.info(f">>> DA CLOSURE (AdaBN): synth+DA {res['synth_da']['mean']['au_pro']:.3f} "
                  f"(+{closed:.3f} vs synth)")


if __name__ == "__main__":
    main()
