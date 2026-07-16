"""Fit matrix in-sync guard — the committed deploy/fit_matrix.json must equal a fresh recompute.

`fit` is a pure CPU join (committed deploy/models_params.json x the accelerator specs), so unlike the
GPU-measured footprints it can be reproduced anywhere — including CI. This asserts the checked-in matrix
matches `Fit.matrix` of the checked-in models doc, so editing an accelerator spec (or the fit logic)
without regenerating the matrix fails the build. Mirrors the RESULTS.md render==file sync test.
"""
import json
from dataclasses import asdict

from surfscan.deploy import FIT_DOC, MODELS_DOC
from surfscan.deploy.fit import Fit


class TestFitMatrixInSync:
    def test_committed_matrix_matches_recompute(self):
        models = json.loads(MODELS_DOC.read_text(encoding="utf-8"))
        committed = json.loads(FIT_DOC.read_text(encoding="utf-8"))["cells"]
        fresh = json.loads(json.dumps([asdict(c) for c in Fit.matrix(models)]))  # normalize StrEnum->str
        assert fresh == committed, "deploy/fit_matrix.json is stale — run `python -m surfscan.deploy fit`"
