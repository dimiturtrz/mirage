"""Stage-1 sim-to-real triad — the honest gap number.

ONE supervised defect-segmentation U-Net (DRAEM's disc net, no recon branch to confound), trained
from three different sources and scored on ONE shared real eval set — so the only variable is where
the *labels* came from:

    real  -> real   : train on REAL defect masks (calib half of the test split)      = the ceiling
    synth -> real   : train on OUR synthetic defects painted on good scans            = the GAP
    synth+DA -> real: synth-trained, then AdaBN (recompute BN stats on real, unlabeled) = the CLOSURE

gap (pp) = real->real AU-PRO  -  synth->real AU-PRO. "modeled it" never stands in for "measured it
transferred" — this module is that measurement. rgb channel (the 0.91 ceiling's modality).

A META-method, deliberately NOT one `AnomalyMethod`: it runs three arms through the harness and reports
their gap. Each arm's training is the shared `Trainer`; the orchestration (arms + gap math) stays here.

Run:  python -m surfscan.run triad [--cats bagel ...] [--epochs 100] [--seed 0]
      [--arms real synth synth_da] [--curriculum]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from core.cli_config import add_config_args, build_config
from core.compute import autocast, enable_tf32, pick_device
from core.data.dataset import load_split
from core.data.defects import KINDS, synthesize
from core.method import Method
from core.obs import get
from surfscan.dispatch import Spec, add_cats
from surfscan.evaluation import harness, scoring
from surfscan.models.draem import UNet
from surfscan.training import curriculum as curric
from surfscan.training.trainer import Trainer

log = get()

BATCH = 16


@dataclass(frozen=True)
class TriadCfg:
    epochs: int = 100
    seed: int = 0
    arms: list[str] = field(default_factory=lambda: ["real", "synth", "synth_da"])
    channels: list[str] = field(default_factory=lambda: ["rgb"])
    curriculum: bool = False


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


def fit_synth(cat, cfg, dev, *, curriculum=False):
    """Train on good scans + on-the-fly synthetic defects. The synth->real arm."""
    ch = tuple(cfg.channels)
    rng = np.random.RandomState(cfg.seed)
    torch.manual_seed(cfg.seed)
    good = load_split(split="train", label=0, cats=[cat], channels=ch, device=dev)
    model = _new(good.in_ch, dev)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    ctrl = curric.KindCurriculum(KINDS, seed=cfg.seed) if curriculum else None

    def step(idx):
        x, v = good.x[idx], good.valid[idx]
        kinds = ctrl.sample(len(idx)) if ctrl else KINDS
        aug, mask = synthesize(x, v, rng, channels=ch, kinds=kinds)
        with autocast(x):
            logits = model(aug.to(memory_format=torch.channels_last))
            loss = F.binary_cross_entropy_with_logits(logits.float(), mask, weight=v)
        if ctrl:
            ctrl.observe(kinds, float(loss.detach()))
        return loss

    return Trainer(model, opt, dev, batch=BATCH).fit(
        len(good), cfg.epochs, step, after_epoch=(ctrl.end_epoch if ctrl else None))


def fit_real(cat, cfg, dev):
    """Train on REAL defect masks (calib half of test). The real->real ceiling — note it's a
    *scarce-label* ceiling (only ~half of an already-small defect set), not an oracle; that's the
    honest supervised bar for this detector, and why the unsupervised memory bank can beat it."""
    ch = tuple(cfg.channels)
    torch.manual_seed(cfg.seed)
    test = load_split(split="test", cats=[cat], channels=ch, device=dev)
    ci, _ = _split_idx(test.df["label"].to_numpy())
    t = torch.as_tensor(ci, device=dev)
    x, v, g = test.x[t], test.valid[t], test.gt[t]
    model = _new(test.in_ch, dev)
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    def step(idx):
        with autocast(x):
            logits = model(x[idx].to(memory_format=torch.channels_last))
            return F.binary_cross_entropy_with_logits(logits.float(), g[idx], weight=v[idx])

    return Trainer(model, opt, dev, batch=BATCH).fit(x.shape[0], cfg.epochs, step)


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
            with autocast(x):
                model(x[i:i + batch].to(memory_format=torch.channels_last))
    model.eval()
    return model


def fit_synth_da(cat, cfg, dev, *, curriculum=False):
    """synth-trained, then AdaBN-adapted to the real eval images (unlabeled). The closure arm."""
    model = fit_synth(cat, cfg, dev, curriculum=curriculum)
    test = load_split(split="test", cats=[cat], channels=tuple(cfg.channels), device=dev)
    _, ei = _split_idx(test.df["label"].to_numpy())
    return _adabn(model, test.x[torch.as_tensor(ei, device=dev)])


@torch.no_grad()
def score(model, cat, cfg, dev, batch=BATCH):
    """Score the shared real EVAL half -> ScoreArrays fields for the harness."""
    test = load_split(split="test", cats=[cat], channels=tuple(cfg.channels), device=dev)
    labels = test.df["label"].to_numpy()
    defects = np.array(test.df["defect"].to_list())
    _, ei = _split_idx(labels)
    t = torch.as_tensor(ei, device=dev)
    x, v, g = test.x[t], test.valid[t], test.gt[t]
    model.eval()
    amaps = []
    for i in range(0, x.shape[0], batch):
        with autocast(x):
            logits = model(x[i:i + batch].to(memory_format=torch.channels_last))
        amaps.append((torch.sigmoid(logits.float()) * v[i:i + batch]).squeeze(1).cpu())
    amaps = torch.cat(amaps).numpy()
    valids = v.squeeze(1).cpu().numpy().astype(bool)
    masks = g.squeeze(1).cpu().numpy().astype(bool)
    scores = scoring.image_scores(amaps, valids)
    return amaps, valids, masks, scores, labels[ei], defects[ei]


def _args(ap):
    add_cats(ap)
    add_config_args(ap, TriadCfg)


def _run(args):
    enable_tf32()
    cfg = build_config(TriadCfg, args)
    dev = pick_device()

    def score_fn(m, c):
        return score(m, c, cfg, dev)

    fits = {
        "real": lambda c: fit_real(c, cfg, dev),
        "synth": lambda c: fit_synth(c, cfg, dev, curriculum=cfg.curriculum),
        "synth_da": lambda c: fit_synth_da(c, cfg, dev, curriculum=cfg.curriculum),
    }
    res = {}
    for arm in cfg.arms:
        tag = f"triad_{arm}" + ("_curric" if cfg.curriculum and arm != "real" else "")
        log.info(f"\n===== {tag} (seed {cfg.seed}) =====")
        res[arm] = harness.run(tag, Method(fits[arm], score_fn), cats=args.cats,
                               params={"arm": arm, "seed": cfg.seed, "epochs": cfg.epochs,
                                       "curriculum": cfg.curriculum})
    if "real" in res and "synth" in res:
        gap = res["real"]["mean"]["au_pro"] - res["synth"]["mean"]["au_pro"]
        log.info(f"\n>>> SIM-TO-REAL GAP (au_pro): real {res['real']['mean']['au_pro']:.3f} "
              f"- synth {res['synth']['mean']['au_pro']:.3f} = {gap:+.3f} pp")
        if "synth_da" in res:
            closed = res["synth_da"]["mean"]["au_pro"] - res["synth"]["mean"]["au_pro"]
            log.info(f">>> DA CLOSURE (AdaBN): synth+DA {res['synth_da']['mean']['au_pro']:.3f} "
                  f"(+{closed:.3f} vs synth)")


SPEC = Spec("triad", _args, _run)
