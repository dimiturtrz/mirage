"""Unit test for obs logging setup — the optional file-handler branch (survives CI stdout buffering).

setup(logfile) adds a FileHandler that mirrors log records to disk; get() lazily initializes a logger.
"""
import logging

from core import obs


def test_setup_writes_to_logfile(tmp_path):
    f = tmp_path / "sub" / "run.log"
    log = obs.setup(f)
    log.info("hello-line")
    for h in log.handlers:                       # flush the file handler
        h.flush()
    assert f.exists() and "hello-line" in f.read_text(encoding="utf-8")


def test_get_returns_configured_logger():
    log = obs.get()
    assert isinstance(log, logging.Logger) and log.handlers
