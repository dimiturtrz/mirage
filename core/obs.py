"""Logging for the CLI runners — the T201 rule bans bare print().

A plain `%(message)s` formatter keeps result tables (harness board, triad_summary) readable while
routing through logging: an optional file handler survives subprocess/CI stdout buffering (tail the
file to watch a long run live). `get()` auto-initializes a stdout handler; `setup(logfile)` adds a file.
"""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

_NAME = "surfscan"


class Obs:
    """Logging for the CLI runners (the free helpers folded in as staticmethods, public names kept)."""

    @staticmethod
    def setup(logfile: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
        log = logging.getLogger(_NAME)
        log.setLevel(level)
        log.propagate = False
        log.handlers.clear()
        fmt = logging.Formatter("%(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)
        if logfile is not None:
            Path(logfile).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
            fh.setFormatter(fmt)
            log.addHandler(fh)
        return log

    @staticmethod
    def get() -> logging.Logger:
        log = logging.getLogger(_NAME)
        if not log.handlers:
            return Obs.setup()
        return log


class Progress:
    """Per-item ETA + per-stage wall-time — so a slow run reveals its cost live, not after the fact.
    `tick(label)` after each of `total` items logs done/total + elapsed + a projected ETA (mean-rate
    extrapolation); `stage(name)` times a named block. The remedy for the 24-min blind bootstrap."""

    def __init__(self, total: int, tag: str = ""):
        self.total = total
        self.tag = tag
        self.t0 = perf_counter()
        self.done = 0

    @staticmethod
    def _eta(elapsed: float, done: int, total: int) -> float:
        """Mean-rate projection of remaining seconds (0 before any item / when complete)."""
        return elapsed / done * (total - done) if done else 0.0

    def tick(self, label: str = "") -> None:
        self.done += 1
        elapsed = perf_counter() - self.t0
        eta = Progress._eta(elapsed, self.done, self.total)
        head = f"[{self.tag}] " if self.tag else ""
        Obs.get().info(f"  {head}{self.done}/{self.total} {label}  {elapsed:.0f}s elapsed  ~{eta:.0f}s left")

    @staticmethod
    @contextmanager
    def stage(name: str):
        """Time a named stage (e.g. bank-fit, bootstrap) and log its wall-time on exit."""
        t = perf_counter()
        yield
        Obs.get().info(f"  [{name}] {perf_counter() - t:.1f}s")
