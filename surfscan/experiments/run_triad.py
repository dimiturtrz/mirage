"""Stage-1 sim-to-real triad — the measured gap number.

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

from core.cli_config import CliConfig
from core.compute import Compute
from core.data.dataset import GpuSplit
from core.data.defects import KINDS, Defects
from core.method import Method
from core.obs import Obs
from surfscan.dispatch import Dispatch, Spec
from surfscan.evaluation import harness, scoring
from surfscan.evaluation.metrics import Metrics
from surfscan.models.draem import UNet
from surfscan.training import curriculum as curric
from surfscan.training.trainer import EarlyStop, Trainer

log = Obs.get()

BATCH = 16
VAL_FRAC = 5             # 1/VAL_FRAC of the good scans held out for the synth val signal
PATIENCE = 8             # epochs of no val-au_pro improvement before stopping (best weights restored)


@dataclass(frozen=True)
class TriadCfg:
    epochs: int = 100
    seed: int = 0
    arms: list[str] = field(default_factory=lambda: ["real", "synth", "synth_da"])
    channels: list[str] = field(default_factory=lambda: ["rgb"])
    curriculum: bool = False


class TriadRun:
    """Stage-1 sim-to-real triad — the measured gap number. A meta-method: three arms
    (real / synth / synth+DA) through the shared harness, reporting their gap.

    `cfg` (arm/run config) and `dev` (device) are fixed for the whole triad run — held as state;
    `cat` (category) varies per harness call — stays a method arg."""

    def __init__(self, cfg, dev):
        self.cfg = cfg
        self.dev = dev

    @staticmethod
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

    def _new(self, in_ch):
        return UNet(in_ch, 1).to(self.dev, memory_format=torch.channels_last)

    @staticmethod
    @torch.no_grad()
    def _synth_val_au_pro(model, x, v, ch, seed):
        """Val signal for early stopping: au_pro on a FIXED held-out synth set (same seed each call ->
        identical val defects -> a stable plateau to detect). Synthetic, from the arm's own train
        distribution — never the real eval half, so the synth->real arm stays honestly synth-only."""
        aug, mask = Defects(np.random.RandomState(seed + 9973)).synthesize(x, v, channels=ch)
        model.eval()
        with Compute.autocast(x):
            logits = model(aug.to(memory_format=torch.channels_last))
        amaps = (torch.sigmoid(logits.float()) * v).squeeze(1).cpu().numpy()
        return Metrics.au_pro(amaps, mask.squeeze(1).cpu().numpy().astype(bool),
                              v.squeeze(1).cpu().numpy().astype(bool))

    def fit_synth(self, cat):
        """Train on good scans + on-the-fly synthetic defects, EARLY-STOPPED on a held-out synth val
        (one optimized run, best weights restored). The synth->real arm."""
        cfg, dev = self.cfg, self.dev
        ch = tuple(cfg.channels)
        rng = np.random.RandomState(cfg.seed)
        torch.manual_seed(cfg.seed)
        good = GpuSplit.load_split(split="train", label=0, cats=[cat], channels=ch, device=dev)
        perm = np.random.RandomState(cfg.seed).permutation(good.x.shape[0])
        nval = max(BATCH, good.x.shape[0] // VAL_FRAC)
        val_t = torch.as_tensor(perm[:nval], device=dev)
        tr_t = torch.as_tensor(perm[nval:], device=dev)                     # disjoint train / val good scans
        model = self._new(good.in_ch)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        ctrl = curric.KindCurriculum(KINDS, seed=cfg.seed) if cfg.curriculum else None

        def step(idx):
            sel = tr_t[idx]
            x, v = good.x[sel], good.valid[sel]
            kinds = ctrl.sample(len(idx)) if ctrl else KINDS
            aug, mask = Defects(rng).synthesize(x, v, channels=ch, kinds=kinds)
            with Compute.autocast(x):
                logits = model(aug.to(memory_format=torch.channels_last))
                loss = F.binary_cross_entropy_with_logits(logits.float(), mask, weight=v)
            if ctrl:
                ctrl.observe(kinds, float(loss.detach()))
            return loss

        stop = EarlyStop(lambda: TriadRun._synth_val_au_pro(model, good.x[val_t], good.valid[val_t], ch,
                                                            cfg.seed), PATIENCE)
        return Trainer(model, opt, dev, batch=BATCH).fit(
            tr_t.shape[0], cfg.epochs, step, after_epoch=(ctrl.end_epoch if ctrl else None), stop=stop)

    def fit_real(self, cat):
        """Train on REAL defect masks (calib half of test). The real->real ceiling — note it's a
        *scarce-label* ceiling (only ~half of an already-small defect set), not an oracle; that's the
        measured supervised bar for this detector, and why the unsupervised memory bank can beat it."""
        cfg, dev = self.cfg, self.dev
        ch = tuple(cfg.channels)
        torch.manual_seed(cfg.seed)
        test = GpuSplit.load_split(split="test", cats=[cat], channels=ch, device=dev)
        ci, _ = TriadRun._split_idx(test.df["label"].to_numpy())
        t = torch.as_tensor(ci, device=dev)
        x, v, g = test.x[t], test.valid[t], test.gt[t]
        model = self._new(test.in_ch)
        opt = optim.AdamW(model.parameters(), lr=1e-3)

        def step(idx):
            with Compute.autocast(x):
                logits = model(x[idx].to(memory_format=torch.channels_last))
                return F.binary_cross_entropy_with_logits(logits.float(), g[idx], weight=v[idx])

        return Trainer(model, opt, dev, batch=BATCH).fit(x.shape[0], cfg.epochs, step)

    @staticmethod
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
                with Compute.autocast(x):
                    model(x[i:i + batch].to(memory_format=torch.channels_last))
        model.eval()
        return model

    def fit_synth_da(self, cat):
        """synth-trained, then AdaBN-adapted to the real eval images (unlabeled). The closure arm."""
        dev = self.dev
        model = self.fit_synth(cat)
        test = GpuSplit.load_split(split="test", cats=[cat], channels=tuple(self.cfg.channels), device=dev)
        _, ei = TriadRun._split_idx(test.df["label"].to_numpy())
        return TriadRun._adabn(model, test.x[torch.as_tensor(ei, device=dev)])

    @torch.no_grad()
    def score(self, model, cat, batch=BATCH):
        """Score the shared real EVAL half -> ScoreArrays fields for the harness."""
        dev = self.dev
        test = GpuSplit.load_split(split="test", cats=[cat], channels=tuple(self.cfg.channels), device=dev)
        labels = test.df["label"].to_numpy()
        defects = np.array(test.df["defect"].to_list())
        _, ei = TriadRun._split_idx(labels)
        t = torch.as_tensor(ei, device=dev)
        x, v, g = test.x[t], test.valid[t], test.gt[t]
        model.eval()
        amaps = []
        for i in range(0, x.shape[0], batch):
            with Compute.autocast(x):
                logits = model(x[i:i + batch].to(memory_format=torch.channels_last))
            amaps.append((torch.sigmoid(logits.float()) * v[i:i + batch]).squeeze(1).cpu())
        amaps = torch.cat(amaps).numpy()
        valids = v.squeeze(1).cpu().numpy().astype(bool)
        masks = g.squeeze(1).cpu().numpy().astype(bool)
        scores = scoring.Scoring.image_scores(amaps, valids)
        return amaps, valids, masks, scores, labels[ei], defects[ei]

    @staticmethod
    def _args(ap):
        Dispatch.add_cats(ap)
        CliConfig.add_config_args(ap, TriadCfg)

    @staticmethod
    def _run(args):
        Compute.enable_tf32()
        cfg = CliConfig.build_config(TriadCfg, args)
        run = TriadRun(cfg, Compute.pick_device())

        fits = {"real": run.fit_real, "synth": run.fit_synth, "synth_da": run.fit_synth_da}
        res = {}
        for arm in cfg.arms:
            tag = f"triad_{arm}" + ("_curric" if cfg.curriculum and arm != "real" else "")
            log.info(f"\n===== {tag} (seed {cfg.seed}) =====")
            res[arm] = harness.Harness.run(tag, Method(fits[arm], run.score), cats=args.cats,
                                   params={"arm": arm, "seed": cfg.seed, "epochs": cfg.epochs,
                                           "curriculum": cfg.curriculum})
        TriadRun._report_gap(res)

    @staticmethod
    def _pro_delta_ci(res_a, res_b):
        """Paired MACRO bootstrap CI for au_pro(B) − au_pro(A) — mean-of-categories, matching the headline
        arm numbers. Arms score the same deterministic eval half per category in the same order, so the
        within-category paired resample cancels shared image noise (a CI clearing 0 = an honest gap)."""
        pc_a = [(sa.amaps, sa.masks, sa.valids) for sa in res_a["by_cat"]]
        pc_b = [(sa.amaps, sa.masks, sa.valids) for sa in res_b["by_cat"]]
        return Metrics.boot_macro_delta_ci(Metrics.au_pro, pc_a, pc_b)

    @staticmethod
    def _report_gap(res):
        """The headline: sim-to-real GAP and DA CLOSURE, each as a point with a PAIRED bootstrap CI —
        one deterministic run, uncertainty from the shared eval set (no seed scatter)."""
        if "real" not in res or "synth" not in res:
            return
        gap, glo, ghi = TriadRun._pro_delta_ci(res["synth"], res["real"])   # real − synth (the gap)
        log.info(f"\n>>> SIM-TO-REAL GAP (au_pro): real {res['real']['mean']['au_pro']:.3f} "
                 f"- synth {res['synth']['mean']['au_pro']:.3f} = {gap:+.3f} pp  [{glo:+.3f}, {ghi:+.3f}]")
        if "synth_da" in res:
            clo_pt, clo_lo, clo_hi = TriadRun._pro_delta_ci(res["synth"], res["synth_da"])  # DA − synth
            log.info(f">>> DA CLOSURE (AdaBN): synth+DA {res['synth_da']['mean']['au_pro']:.3f} "
                     f"(+{clo_pt:.3f} vs synth)  [{clo_lo:+.3f}, {clo_hi:+.3f}]")


SPEC = Spec("triad", TriadRun._args, TriadRun._run)
