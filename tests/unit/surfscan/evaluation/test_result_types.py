"""Unit tests for the eval result carriers — the metric pair + tracker-run params.

Equivalence-class: Scores round-trips to the result-payload dict and to the MLflow `_mean` metric
names; RunParams joins one / many categories into the tracked params dict.
"""
from surfscan.evaluation.result_types import RunParams, Scores


def test_scores_as_dict():
    assert Scores(0.9, 0.8).as_dict() == {"img_auroc": 0.9, "au_pro": 0.8}


def test_scores_tracker_means():
    assert Scores(0.9, 0.8).tracker_means() == {"img_auroc_mean": 0.9, "au_pro_mean": 0.8}


def test_scores_reconstruct_from_payload():
    s = Scores(0.42, 0.37)
    assert Scores(**s.as_dict()) == s


def test_run_params_single_category():
    assert RunParams("patchcore", ["bagel"]).as_dict() == {"method": "patchcore", "cats": "bagel"}


def test_run_params_many_categories():
    assert RunParams("vae_per_category", ["bagel", "cable_gland", "carrot"]).as_dict() == {
        "method": "vae_per_category", "cats": "bagel,cable_gland,carrot"}
