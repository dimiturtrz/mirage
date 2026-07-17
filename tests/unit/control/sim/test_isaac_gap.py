"""Unit test for IsaacGap's pure aggregation — no Isaac boot, so it runs in any numpy env. The rest of
isaac_gap (env rollout + kit boot) is GPU-only and coverage-omitted; this locks the mean/std reducer."""
import numpy as np

from control.sim.isaac_gap import IsaacGap


class TestAggregate:
    def test_mean_and_std_over_seeds(self):
        rows = [{"gap": 10.0, "s": 1.0}, {"gap": 20.0, "s": 3.0}]
        agg = IsaacGap._aggregate(rows)
        assert agg["gap_mean"] == 15.0
        assert agg["s_mean"] == 2.0
        np.testing.assert_allclose(agg["gap_std"], np.std([10.0, 20.0], ddof=1))   # sample sd (ddof=1)

    def test_single_seed_zero_std(self):
        agg = IsaacGap._aggregate([{"gap": 5.0}])
        assert agg["gap_mean"] == 5.0
        assert agg["gap_std"] == 0.0                                               # ddof=1 undefined at n=1 -> 0
