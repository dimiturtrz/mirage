"""Unit test for the sim-to-real triad's split logic — no leakage is the whole point.

The calib (real-train) and eval (shared test) halves must be disjoint, stratified (both carry every
label), and deterministic (fit and score recompute the same split independently). No dataset, no GPU.
"""
import numpy as np

from surfscan.experiments.run_triad import _split_idx


def test_split_disjoint_stratified_deterministic():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    calib, ev = _split_idx(labels)

    assert set(calib).isdisjoint(set(ev))                 # no sample in both -> no leakage
    assert set(calib) | set(ev) == set(range(len(labels)))   # partition covers everything
    for half in (calib, ev):
        assert set(labels[half]) == {0, 1}                # each half has good + defective

    calib2, ev2 = _split_idx(labels)
    assert np.array_equal(calib, calib2) and np.array_equal(ev, ev2)   # deterministic
