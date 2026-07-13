"""Run-end invariant checks — catch a plausible-but-wrong number AT run time, not after a re-run.

This project once shipped a gap CI computed on the wrong aggregate (pooled/micro instead of macro);
the number looked reasonable and the mistake surfaced only after a full GPU re-run. These are the cheap
asserts that would have caught it the instant the run finished: the reported CI point must equal the
reported mean, every bracket must contain its point, and ECE must only exist for bounded [0,1] maps.
Loud by default (WARNING per violation); `strict=True` raises for tests / CI.
"""
from __future__ import annotations

import numpy as np

from core.obs import Obs

log = Obs.get()

_TOL = 1e-6


class Invariants:
    """Assert a run's reported numbers are internally consistent (the guardrail against silent-wrong)."""

    @staticmethod
    def check(res, *, strict=False):
        """-> list of violation messages (empty = clean). Logs each loudly; raises if `strict`."""
        violations = Invariants._brackets(res) + Invariants._ci_matches_mean(res) + Invariants._ece_bounded(res)
        for m in violations:
            log.warning(f"INVARIANT VIOLATION: {m}")
        if strict and violations:
            raise AssertionError("; ".join(violations))
        return violations

    @staticmethod
    def _finite(*xs):
        return all(x == x for x in xs)                     # NaN != NaN — skip undefined metrics

    @staticmethod
    def _brackets(res):
        """Every bootstrap bracket must contain its own point estimate."""
        out = []
        for m, (p, lo, hi) in res.get("ci", {}).items():
            if Invariants._finite(p, lo, hi) and not lo <= p <= hi:
                out.append(f"{m} bracket [{lo:.4f}, {hi:.4f}] excludes point {p:.4f}")
        return out

    @staticmethod
    def _ci_matches_mean(res):
        """The CI point and the reported mean are the SAME statistic (macro) — they must agree. This is
        the check that catches a pooled-vs-macro mix-up (the historical bug)."""
        out = []
        mean = res.get("mean", {})
        for m, (p, _lo, _hi) in res.get("ci", {}).items():
            if m in mean and Invariants._finite(p, mean[m]) and abs(p - mean[m]) > _TOL:
                out.append(f"{m} ci-point {p:.4f} != reported mean {mean[m]:.4f} (aggregate mismatch)")
        return out

    @staticmethod
    def _ece_bounded(res):
        """ECE is only meaningful for probability maps; if it was computed, the amaps must lie in [0,1]."""
        if res.get("ece") is None or not res.get("by_cat"):
            return []
        amaps = np.concatenate([np.asarray(sa.amaps).ravel() for sa in res["by_cat"]])
        if amaps.size and (np.nanmin(amaps) < 0.0 or np.nanmax(amaps) > 1.0):
            return [f"ECE computed but amaps out of [0,1] ({np.nanmin(amaps):.3f}..{np.nanmax(amaps):.3f})"]
        return []

    @staticmethod
    def reconciles(delta, point_a, point_b, label="delta"):
        """A paired delta must equal point_b − point_a (same macro statistic). Loud + returns ok/bad —
        the triad's guard: gap == real_mean − synth_mean, so a micro/macro slip fails immediately."""
        ok = Invariants._finite(delta, point_a, point_b) and abs(delta - (point_b - point_a)) <= _TOL
        if not ok:
            log.warning(f"INVARIANT VIOLATION: {label} {delta:.4f} != {point_b:.4f} - {point_a:.4f}")
        return ok
