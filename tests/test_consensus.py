"""Correctness tests for core/consensus.py."""
from __future__ import annotations

import numpy as np
import pytest

from core.consensus import (
    geometric_median_ilr,
    aitchison_distances_to_consensus,
    mahalanobis_distances_to_consensus,
    krippendorff_alpha_aitchison,
    krippendorff_alpha_bootstrap_ci,
)
from core.compositional import cmult_repl, ilr, aitchison_distance
from tests.conftest import make_ilr_panel


# ── geometric_median_ilr ──────────────────────────────────────────────────────

class TestGeometricMedian:
    def test_single_point_returns_itself(self):
        x = np.array([[1.0, 2.0, -1.0]])
        result = geometric_median_ilr(x)
        np.testing.assert_allclose(result, x[0], atol=1e-10)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            geometric_median_ilr(np.empty((0, 3)))

    def test_symmetric_arrangement_returns_centre(self):
        """4 points symmetrically placed → median at origin."""
        pts = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        result = geometric_median_ilr(pts)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_two_equal_points_returns_midpoint(self):
        pts = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, -1.0]])
        result = geometric_median_ilr(pts)
        expected = np.array([1.0, 1.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_minimises_sum_of_distances(self):
        """Geometric median minimises Σ ‖x − y_i‖."""
        rng = np.random.default_rng(7)
        pts = rng.standard_normal((20, 3))
        median = geometric_median_ilr(pts)

        total_at_median = np.linalg.norm(pts - median, axis=1).sum()
        # Perturbation should not decrease total distance
        for _ in range(100):
            perturb = rng.standard_normal(3) * 0.01
            total_perturbed = np.linalg.norm(pts - (median + perturb), axis=1).sum()
            assert total_perturbed >= total_at_median - 1e-6, \
                f"Median did not minimise sum of distances (Δ={total_perturbed - total_at_median:.2e})"

    def test_robust_to_outlier(self):
        """Geometric median should not be pulled far by a single outlier."""
        pts = np.vstack([
            np.random.default_rng(0).standard_normal((27, 3)) * 0.1,
            [[100.0, 100.0, 100.0]],  # severe outlier
        ])
        median = geometric_median_ilr(pts)
        # Median should remain close to the cluster centre (0,0,0)
        assert np.linalg.norm(median) < 5.0, \
            f"Median pulled too far by outlier: {np.linalg.norm(median):.2f}"

    def test_convergence_tolerance(self):
        """Result should not change meaningfully with tighter tolerance."""
        pts = np.random.default_rng(3).standard_normal((10, 3))
        m1 = geometric_median_ilr(pts, tol=1e-5)
        m2 = geometric_median_ilr(pts, tol=1e-10)
        np.testing.assert_allclose(m1, m2, atol=1e-4)


# ── distances to consensus ────────────────────────────────────────────────────

class TestDistancesToConsensus:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.M = 5
        self.D1 = 3
        self.method_ilrs = rng.standard_normal((self.M, self.D1))
        self.consensus = geometric_median_ilr(self.method_ilrs)

    def test_aitchison_distances_shape(self):
        dists = aitchison_distances_to_consensus(self.method_ilrs, self.consensus)
        assert dists.shape == (self.M,)

    def test_aitchison_distances_non_negative(self):
        dists = aitchison_distances_to_consensus(self.method_ilrs, self.consensus)
        assert (dists >= 0).all()

    def test_aitchison_distance_of_consensus_to_itself_zero(self):
        one_point = self.consensus[None, :]
        dists = aitchison_distances_to_consensus(one_point, self.consensus)
        np.testing.assert_allclose(dists[0], 0.0, atol=1e-10)

    def test_mahalanobis_distances_shape(self):
        dists = mahalanobis_distances_to_consensus(self.method_ilrs, self.consensus)
        assert dists.shape == (self.M,)

    def test_mahalanobis_distances_non_negative(self):
        dists = mahalanobis_distances_to_consensus(self.method_ilrs, self.consensus)
        assert (dists >= 0).all()

    def test_mahalanobis_falls_back_when_underdetermined(self):
        """M-1 <= D-1: falls back to Euclidean (no error)."""
        method_ilrs = np.random.default_rng(0).standard_normal((4, 3))  # M-1=3 == d
        consensus = method_ilrs.mean(axis=0)
        dists = mahalanobis_distances_to_consensus(method_ilrs, consensus)
        assert dists.shape == (4,)
        assert (dists >= 0).all()

    def test_mahalanobis_leave_one_out_unmasks_outlier(self):
        """LOO covariance: a planted outlier is NOT masked by its own inflation.

        With a pooled covariance that includes the outlier, the outlier inflates
        the variance along its own direction and shrinks its Mahalanobis
        distance. The leave-one-out covariance (the other M-1 methods) must give
        the outlier the largest distance of all methods.
        """
        rng = np.random.default_rng(123)
        inliers = rng.standard_normal((27, 3)) * 0.1   # tight cluster
        outlier = np.array([[8.0, 0.0, 0.0]])          # far along axis 0
        ilrs = np.vstack([inliers, outlier])
        consensus = np.zeros(3)
        dists = mahalanobis_distances_to_consensus(ilrs, consensus)
        assert dists.shape == (28,)
        assert int(np.argmax(dists)) == 27, "Outlier should have the max LOO distance"

    def test_mahalanobis_bounded_when_methods_near_collinear(self):
        """Near-identical methods → near-singular LOO covariance. Without the ridge
        the inverse explodes (real data reached ~1e23) yet never raises LinAlgError,
        so distances must be kept finite/bounded by the regularisation — while the
        planted outlier still ranks highest (the outlier flags are rank-based)."""
        rng = np.random.default_rng(7)
        # 27 inliers spread along ONE direction → covariance near-singular in the
        # other two coords (mirrors methods agreeing to ~1e-7 there).
        t = rng.standard_normal((27, 1))
        inliers = t * np.array([1.0, 1.0, 1.0]) + rng.standard_normal((27, 3)) * 1e-7
        outlier = np.array([[0.0, 5.0, 0.0]])   # deviates in a near-zero-variance dir
        ilrs = np.vstack([inliers, outlier])
        consensus = inliers.mean(axis=0)
        dists = mahalanobis_distances_to_consensus(ilrs, consensus)
        assert np.isfinite(dists).all()
        assert dists.max() < 1e6, f"ridge should bound the explosion, got {dists.max():.2e}"
        assert int(np.argmax(dists)) == 27, "planted outlier must still rank highest"

    def test_mahalanobis_bounded_when_covariance_exactly_zero(self):
        """Fully degenerate: all 27 other methods identical → zero covariance, where
        a purely *relative* ridge would still vanish. The absolute floor must keep
        the held-out outlier's distance finite and bounded."""
        inliers = np.zeros((27, 3))
        outlier = np.array([[0.0, 5.0, 0.0]])
        ilrs = np.vstack([inliers, outlier])
        dists = mahalanobis_distances_to_consensus(ilrs, np.zeros(3))
        assert np.isfinite(dists).all()
        assert dists.max() < 1e6, f"absolute floor should bound it, got {dists.max():.2e}"
        assert int(np.argmax(dists)) == 27


# ── krippendorff_alpha_aitchison ─────────────────────────────────────────────

class TestKrippendorffAlpha:
    def test_perfect_agreement_returns_one(self):
        """All methods give the same ILR value per block → α = 1."""
        M, N, D1 = 5, 20, 3
        rng = np.random.default_rng(0)
        block_values = rng.standard_normal((N, D1))
        # All methods identical to the block value
        panel = np.broadcast_to(block_values[None, :, :], (M, N, D1)).copy()
        alpha = krippendorff_alpha_aitchison(panel)
        np.testing.assert_allclose(alpha, 1.0, atol=1e-6)

    def test_insufficient_data_returns_nan(self):
        assert np.isnan(krippendorff_alpha_aitchison(np.zeros((1, 10, 3))))
        assert np.isnan(krippendorff_alpha_aitchison(np.zeros((5, 1, 3))))

    def test_random_panel_between_neg1_and_1(self):
        """α should be in a reasonable range for random data."""
        rng = np.random.default_rng(7)
        panel = rng.standard_normal((10, 50, 3))
        alpha = krippendorff_alpha_aitchison(panel)
        assert np.isfinite(alpha)
        assert alpha <= 1.0

    def test_high_noise_gives_low_alpha(self):
        """High within-block noise → low α (near 0 or negative)."""
        rng = np.random.default_rng(1)
        # 10 blocks, 8 methods — methods are pure noise
        panel = rng.standard_normal((8, 10, 3))
        alpha = krippendorff_alpha_aitchison(panel)
        assert alpha < 0.5, f"Expected low α for pure-noise panel, got {alpha:.3f}"

    def test_low_noise_gives_high_alpha(self):
        """Low within-block noise → high α."""
        rng = np.random.default_rng(2)
        M, N, D1 = 6, 30, 3
        true_blocks = rng.standard_normal((N, D1)) * 2
        # Methods deviate very little from the block truth
        panel = true_blocks[None, :, :] + rng.standard_normal((M, N, D1)) * 0.01
        alpha = krippendorff_alpha_aitchison(panel)
        assert alpha > 0.9, f"Expected high α for low-noise panel, got {alpha:.3f}"

    def test_alpha_formula_d_o_over_d_e(self):
        """Verify α = 1 - D_o/D_e against a brute-force CANONICAL reference.

        D_e is the textbook Krippendorff expected disagreement: the mean squared
        distance over *all* ordered pairs of values in the panel (same-unit pairs
        included), not just cross-unit pairs.
        """
        rng = np.random.default_rng(5)
        M, N, D1 = 4, 8, 3
        panel = rng.standard_normal((M, N, D1))

        # Brute-force D_o (within-unit ordered method pairs)
        d_o_total = 0.0
        n_o = 0
        for u in range(N):
            for m in range(M):
                for m2 in range(m + 1, M):
                    d_o_total += np.linalg.norm(panel[m, u] - panel[m2, u]) ** 2
                    n_o += 1
        D_o = d_o_total / n_o

        # Brute-force canonical D_e: ALL ordered pairs of values (a != b)
        vals = panel.reshape(M * N, D1)
        d_e_total = 0.0
        n_e = 0
        for a in range(len(vals)):
            for b in range(len(vals)):
                if a == b:
                    continue
                d_e_total += np.linalg.norm(vals[a] - vals[b]) ** 2
                n_e += 1
        D_e = d_e_total / n_e

        alpha_expected = 1.0 - D_o / D_e
        alpha_computed = krippendorff_alpha_aitchison(panel)
        np.testing.assert_allclose(alpha_computed, alpha_expected, atol=1e-8)

    def test_alpha_matches_reference_krippendorff_interval(self):
        """1-D panel must match the reference `krippendorff` package (interval)."""
        krip = pytest.importorskip("krippendorff")
        rng = np.random.default_rng(11)
        M, N = 6, 25
        panel = rng.standard_normal((M, N, 1))
        ours = krippendorff_alpha_aitchison(panel)
        ref = krip.alpha(reliability_data=panel[:, :, 0],
                         level_of_measurement="interval")
        np.testing.assert_allclose(ours, ref, atol=1e-8)


class TestKrippendorffPerCoordinate:
    def test_shape_and_decomposition(self):
        from core.consensus import krippendorff_alpha_per_coordinate
        rng = np.random.default_rng(3)
        panel = rng.standard_normal((7, 30, 3))
        per = krippendorff_alpha_per_coordinate(panel)
        assert per.shape == (3,)
        assert np.isfinite(per).all()

    def test_isolates_noisy_coordinate(self):
        """A coordinate with extra within-block noise gets a lower α."""
        from core.consensus import krippendorff_alpha_per_coordinate
        rng = np.random.default_rng(8)
        M, N = 6, 40
        truth = rng.standard_normal((N, 3)) * 2
        panel = truth[None, :, :] + rng.standard_normal((M, N, 3)) * 0.02
        panel[:, :, 2] += rng.standard_normal((M, N)) * 2.0  # noise into coord 3
        per = krippendorff_alpha_per_coordinate(panel)
        assert per[2] < per[0] and per[2] < per[1]

    def test_alpha_consistent_with_different_scales(self):
        """α is scale-invariant: rescaling the ILR panel by a constant doesn't change α."""
        rng = np.random.default_rng(9)
        panel = rng.standard_normal((5, 20, 3))
        alpha1 = krippendorff_alpha_aitchison(panel)
        alpha2 = krippendorff_alpha_aitchison(panel * 3.14)
        np.testing.assert_allclose(alpha1, alpha2, atol=1e-8)


# ── bootstrap CI ─────────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_returns_three_floats(self):
        panel = make_ilr_panel(n_methods=5, n_blocks=20, noise=0.05)
        alpha, lo, hi = krippendorff_alpha_bootstrap_ci(panel, n_boot=100)
        assert all(np.isfinite(v) for v in (alpha, lo, hi))

    def test_lo_le_alpha_le_hi(self):
        panel = make_ilr_panel(n_methods=5, n_blocks=20, noise=0.05)
        alpha, lo, hi = krippendorff_alpha_bootstrap_ci(panel, n_boot=200)
        assert lo <= alpha + 1e-6
        assert alpha <= hi + 1e-6

    def test_ci_narrower_with_larger_n(self):
        """More blocks → narrower CI (on average)."""
        rng = np.random.default_rng(0)
        panel_small = make_ilr_panel(n_methods=5, n_blocks=10, noise=0.1, seed=0)
        panel_large = make_ilr_panel(n_methods=5, n_blocks=100, noise=0.1, seed=1)
        _, lo_s, hi_s = krippendorff_alpha_bootstrap_ci(panel_small, n_boot=500, rng=rng)
        _, lo_l, hi_l = krippendorff_alpha_bootstrap_ci(panel_large, n_boot=500, rng=rng)
        width_small = hi_s - lo_s
        width_large = hi_l - lo_l
        assert width_large < width_small, \
            f"Expected narrower CI with more blocks: {width_large:.3f} vs {width_small:.3f}"

    def test_reproducible_with_same_rng(self):
        panel = make_ilr_panel(n_methods=4, n_blocks=15, noise=0.1)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        r1 = krippendorff_alpha_bootstrap_ci(panel, n_boot=100, rng=rng1)
        r2 = krippendorff_alpha_bootstrap_ci(panel, n_boot=100, rng=rng2)
        np.testing.assert_allclose(r1, r2, atol=1e-12)
