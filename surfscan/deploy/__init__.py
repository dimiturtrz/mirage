"""Deploy fit model — analytical deployability: measure each detector's compute/memory footprint, then
cross it against typed accelerator specs to get a prescriptive architecture->accelerator fit verdict.

The data lives in a browsable, top-level `deploy/` piece (like `pointcloud-viewer/`):
  * deploy/models_params.json          — measured per-component footprint + detector op-classes (generated)
  * deploy/accelerators/<type>_params.json — cited, hand-authored specs, one file per accelerator TYPE
  * deploy/fit_matrix.json             — the (detector x accelerator x options) verdict matrix (generated)

Honesty boundary: latency is a PROJECTION (FLOPs / peak-TOPS x a stated efficiency band), never a measured
FPS; the load-bearing outputs are the op-fit and memory-fit verdicts, which hold without any TOPS number.

`OpClass` lives here, with the paths, because it is the leg's shared vocabulary rather than any one step's
output: it is the JOIN KEY between the two halves of the fit — `profile` assigns an op-class to a detector,
`accelerators` declare op-support per op-class, and `fit` matches them. Every other payload type sits beside
the step that produces it.
"""
from enum import StrEnum
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"
MODELS_DOC = DEPLOY_DIR / "models_params.json"
ACCEL_DIR = DEPLOY_DIR / "accelerators"
FIT_DOC = DEPLOY_DIR / "fit_matrix.json"


class OpClass(StrEnum):
    """What kind of compute graph a detector is — the axis that decides on-device fit, not FLOP count."""
    CONV_NATIVE = "conv_native"                    # pure conv/deconv (ConvVAE, Draem, feat-recon)
    BANK_LOOKUP = "bank_lookup"                     # conv backbone + kNN distance/argmin/top-k tail (PatchCore)
    TRANSFORMER_ATTENTION = "transformer_attention"  # attention blocks (none of ours yet; the axis rejects them)
