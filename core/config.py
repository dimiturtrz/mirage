"""One data root, everything derived — mirrors systole's cardioseg/config.py.

Source of truth: `paths.yaml` at the repo root (gitignored). Copy
`paths.example.yaml` -> `paths.yaml` and set the single `data:` line.
Override with the env var SURFSCAN_DATA (e.g. in CI).

Under <data> the layout is:
    <data>/raw/        register-gated downloads (you drop them here)
    <data>/processed/  preprocess cache (created on first run)
"""
import os
from pathlib import Path

from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    env = os.environ.get("SURFSCAN_DATA")
    if env:
        return Path(env)
    cfg = _REPO / "paths.yaml"
    if not cfg.exists():
        raise FileNotFoundError(
            f"{cfg} not found — copy paths.example.yaml -> paths.yaml and set `data:` "
            f"(or set the SURFSCAN_DATA env var)."
        )
    return Path(OmegaConf.load(cfg).data)


def raw_dir() -> Path:
    return data_root() / "raw"


def processed_dir() -> Path:
    return data_root() / "processed"


def mvtec_root() -> Path:
    """The MVTec 3D-AD download root: <data>/raw/mvtec_3d_anomaly_detection."""
    return raw_dir() / "mvtec_3d_anomaly_detection"
