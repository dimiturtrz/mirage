"""Unit test for obs logging setup — the optional file-handler branch (survives CI stdout buffering).

setup(logfile) adds a FileHandler that mirrors log records to disk; get() lazily initializes a logger.
"""
import logging

from core import obs
from core.obs import Progress


def test_eta_projects_remaining_time():
    assert Progress._eta(10.0, 0, 5) == 0.0            # before any item -> no estimate
    assert Progress._eta(10.0, 5, 10) == 10.0          # half done -> ~elapsed remaining
    assert Progress._eta(10.0, 10, 10) == 0.0          # complete -> nothing left


def test_progress_tick_counts_and_stage_runs(tmp_path):
    f = tmp_path / "run.log"
    obs.Obs.setup(f)                                   # route through a file handler we can read
    p = Progress(3, tag="unit")
    for c in ("a", "b", "c"):
        p.tick(c)
    assert p.done == 3
    with Progress.stage("bootstrap"):
        pass
    for h in obs.Obs.get().handlers:
        h.flush()
    text = f.read_text(encoding="utf-8")
    assert "3/3" in text and "[bootstrap]" in text     # ETA line + stage timing both emitted
    f = tmp_path / "sub" / "run.log"
    log = obs.Obs.setup(f)
    log.info("hello-line")
    for h in log.handlers:                       # flush the file handler
        h.flush()
    assert f.exists() and "hello-line" in f.read_text(encoding="utf-8")


def test_get_returns_configured_logger():
    log = obs.Obs.get()
    assert isinstance(log, logging.Logger) and log.handlers
