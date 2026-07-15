"""Deploy fit model — analytical deployability: measure each detector's compute/memory footprint, then
cross it against typed accelerator specs to get a prescriptive architecture->accelerator fit verdict.

The data lives in a browsable, top-level `deployment/` piece (like `pointcloud-viewer/`):
  * deployment/models_params.json          — measured per-component footprint + detector op-classes (generated)
  * deployment/accelerators/<type>_params.json — cited, hand-authored specs, one file per accelerator TYPE
  * deployment/fit_matrix.json             — the (detector x accelerator x options) verdict matrix (generated)

Honesty boundary: latency is a PROJECTION (FLOPs / peak-TOPS x a stated efficiency band), never a measured
FPS; the load-bearing outputs are the op-fit and memory-fit verdicts, which hold without any TOPS number.
"""
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deployment"
MODELS_DOC = DEPLOY_DIR / "models_params.json"
ACCEL_DIR = DEPLOY_DIR / "accelerators"
FIT_DOC = DEPLOY_DIR / "fit_matrix.json"
