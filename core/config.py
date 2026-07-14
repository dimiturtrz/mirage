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


class Config:
    """One data root, everything derived (the free helpers folded in as staticmethods, public names kept)."""

    @staticmethod
    def _from_yaml() -> Path:  # pragma: no cover  reads paths.yaml (disk config)
        cfg = _REPO / "paths.yaml"
        if not cfg.exists():
            raise FileNotFoundError(
                f"{cfg} not found — copy paths.example.yaml -> paths.yaml and set `data:` "
                f"(or set the SURFSCAN_DATA env var)."
            )
        return Path(OmegaConf.load(cfg).data)

    @staticmethod
    def data_root() -> Path:
        env = os.environ.get("SURFSCAN_DATA")
        return Path(env) if env else Config._from_yaml()

    @staticmethod
    def raw_dir() -> Path:
        return Config.data_root() / "raw"

    @staticmethod
    def processed_dir() -> Path:
        return Config.data_root() / "processed"

    @staticmethod
    def mvtec_root() -> Path:
        """The MVTec 3D-AD download root: <data>/raw/mvtec_3d_anomaly_detection."""
        return Config.raw_dir() / "mvtec_3d_anomaly_detection"

    @staticmethod
    def twin_dir() -> Path:
        """Reconstructed digital-twin assets (per-category meshes): <data>/twin."""
        return Config.data_root() / "twin"
