"""Unit tests for the data-root config — the SURFSCAN_DATA env override + derived paths."""
from core import config


def test_data_root_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SURFSCAN_DATA", str(tmp_path))
    assert config.Config.data_root() == tmp_path
    assert config.Config.raw_dir() == tmp_path / "raw"
    assert config.Config.processed_dir() == tmp_path / "processed"
    assert config.Config.mvtec_root() == tmp_path / "raw" / "mvtec_3d_anomaly_detection"
