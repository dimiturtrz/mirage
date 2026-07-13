"""Logging for the CLI runners — the T201 rule bans bare print().

A plain `%(message)s` formatter keeps result tables (harness board, triad_summary) readable while
routing through logging: an optional file handler survives subprocess/CI stdout buffering (tail the
file to watch a long run live). `get()` auto-initializes a stdout handler; `setup(logfile)` adds a file.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

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
