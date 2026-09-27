"""Correctness tests for core/stats.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from core.stats import (
    friedman_test,
    nemenyi_post_hoc,
    bland_altman_stats,
    inverse_variance_weights,
    weighted_ilr_mean,
)


# ── friedman_test ─────────────────────────────────────────────────────────────

class TestFriedmanTest:
    def test_known_output(self):
        """Verify against scipy.stats.friedmanchisquare directly."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal(20)
        b = rng.standard_normal(20) + 0.5
        c = rng.standard_normal(20) + 1.0
        result = friedman_test({"a": a, "b": b, "c": c})
        stat_ref, p_ref = scipy_stats.friedmanchisquare(a, b, c)
        np.testing.assert_allclose(result["statistic"], stat_ref, rtol=1e-6)
        np.testing.assert_allclose(result["p_value"], p_ref, rtol=1e-6)

    def test_identical_groups_high_p(self):
        """Identical values across groups → high p-value (no difference)."""
        data = np.ones(30)
        result = friedman_test({"a": data, "b": data, "c": data})
        assert result["p_value"] > 0.05

    def test_tied_input_silent_and_p_one(self):
        """Tied input must NOT leak scipy's divide RuntimeWarning to the console
        (it floods `p1 all`); the handled nan becomes p=1.0."""
        import warnings
        data = np.ones(30)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = friedman_test({"a": data, "b": data, "c": data})
        assert result["p_value"] == 1.0

    def test_kendall_w_bounds(self):
        """Kendall's W ∈ [0, 1]."""
        rng = np.random.default_rng(1)
        vals = {str(i): rng.standard_normal(15) for i in range(5)}
        result = friedman_test(vals)
        assert 0.0 <= result["kendall_w"] <= 1.0 + 1e-9

    def test_returns_required_keys(self):
        a = np.random.standard_normal(10)
        b = np.random.standard_normal(10)
        c = np.random.standard_normal(10)
        result = friedman_test({"a": a, "b": b, "c": c})
        for key in ["statistic", "df", "p_value", "kendall_w", "n_subjects", "n_methods"]:
            assert key in result

    def test_df_equals_k_minus_1(self):
        vals = {f"m{i}": np.random.standard_normal(12) for i in range(6)}
        result = friedman_test(vals)
        assert result["df"] == 5

    def test_requires_at_least_3_methods(self):
        with pytest.raises(ValueError):
            friedman_test({"only_one": np.ones(5)})
        with pytest.raises(ValueError):
            friedman_test({"a": np.ones(5), "b": np.ones(5)})


# ── nemenyi_post_hoc ──────────────────────────────────────────────────────────

class TestNemenyiPostHoc:
    def test_shape_is_k_by_k(self):
        rng = np.random.default_rng(0)
        vals = {f"m{i}": rng.standard_normal(20) for i in range(4)}
        result = nemenyi_post_hoc(vals)
        assert result.shape == (4, 4)

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(1)
        vals = {f"m{i}": rng.standard_normal(20) for i in range(3)}
        result = nemenyi_post_hoc(vals)
        np.testing.assert_allclose(np.diag(result.values), 1.0, atol=1e-6)

    def test_p_values_between_0_and_1(self):
        rng = np.random.default_rng(2)
        vals = {f"m{i}": rng.standard_normal(15) for i in range(4)}
        result = nemenyi_post_hoc(vals)
        assert (result.values >= 0).all() and (result.values <= 1.0 + 1e-9).all()


# ── bland_altman_stats ────────────────────────────────────────────────────────

class TestBlandAltman:
    def test_zero_bias_zero_mean_diff(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        result = bland_altman_stats(a, b)
        np.testing.assert_allclose(result["mean_diff"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["std_diff"], 0.0, atol=1e-12)

    def test_known_bias(self):
        a = np.array([0.0, 1.0, 2.0, 3.0]) + 0.5
        b = np.array([0.0, 1.0, 2.0, 3.0])
        result = bland_altman_stats(a, b)
        np.testing.assert_allclose(result["mean_diff"], 0.5, atol=1e-10)

    def test_loa_formula(self):
        """LOA = mean_diff ± 1.96 * std_diff."""
        rng = np.random.default_rng(3)
        a = rng.standard_normal(50)
        b = rng.standard_normal(50)
        result = bland_altman_stats(a, b)
        np.testing.assert_allclose(
            result["loa_upper"],
            result["mean_diff"] + 1.96 * result["std_diff"],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            result["loa_lower"],
            result["mean_diff"] - 1.96 * result["std_diff"],
            atol=1e-10,
        )


# ── inverse_variance_weights ──────────────────────────────────────────────────

class TestInverseVarianceWeights:
    def test_no_moran_returns_n(self):
        n = np.array([100.0, 200.0, 50.0])
        w = inverse_variance_weights(n)
        np.testing.assert_allclose(w, n)

    def test_zero_moran_same_as_no_moran(self):
        n = np.array([100.0, 200.0, 50.0])
        rho = np.zeros(3)
        w_with = inverse_variance_weights(n, rho)
        w_without = inverse_variance_weights(n)
        np.testing.assert_allclose(w_with, w_without, atol=1e-10)

    def test_high_moran_reduces_weight(self):
        """High autocorrelation → smaller effective sample size → smaller weight."""
        n = np.array([1000.0])
        w_no_corr = inverse_variance_weights(n, np.array([0.0]))
        w_high_corr = inverse_variance_weights(n, np.array([0.9]))
        assert w_high_corr[0] < w_no_corr[0]

    def test_formula_manual(self):
        """Verify w = N / (1 + (N-1)*rho) for a known case."""
        n = np.array([100.0])
        rho = np.array([0.5])
        expected = 100.0 / (1 + 99 * 0.5)
        w = inverse_variance_weights(n, rho)
        np.testing.assert_allclose(w[0], expected, atol=1e-10)

    def test_negative_moran_clipped_to_zero(self):
        """Negative Moran's I should not increase effective weight beyond N."""
        n = np.array([100.0])
        w_negative = inverse_variance_weights(n, np.array([-0.5]))
        w_zero = inverse_variance_weights(n, np.array([0.0]))
        np.testing.assert_allclose(w_negative, w_zero, atol=1e-10)

    def test_weights_all_positive(self):
        rng = np.random.default_rng(4)
        n = rng.integers(50, 500, 20).astype(float)
        rho = rng.uniform(0, 1, 20)
        w = inverse_variance_weights(n, rho)
        assert (w > 0).all()

    def test_cluster_size_design_effect_formula(self):
        """With cluster size m̄, DEFF = 1 + (m̄-1)*rho (not (N-1)*rho)."""
        n = np.array([5000.0])          # cells
        m_bar = np.array([50.0])        # cells per tile
        rho = np.array([0.3])
        expected = 5000.0 / (1 + 49 * 0.3)
        w = inverse_variance_weights(n, rho, cluster_size_per_slide=m_bar)
        np.testing.assert_allclose(w[0], expected, atol=1e-9)

    def test_cluster_size_does_not_collapse_across_slides(self):
        """The corrected weight tracks slide size; the cells-as-cluster bug did not.

        Two slides with equal cell density (same m̄) but different totals should
        get clearly different weights under the cluster-sampling design effect,
        whereas plugging N as the cluster size makes them nearly identical.
        """
        n = np.array([2000.0, 8000.0])
        m_bar = np.array([40.0, 40.0])      # same density
        rho = np.array([0.3, 0.3])
        w_correct = inverse_variance_weights(n, rho, cluster_size_per_slide=m_bar)
        w_buggy = inverse_variance_weights(n, rho)   # m̄ = N (old behaviour)
        # Correct: weight ratio tracks the 4x size difference.
        assert w_correct[1] / w_correct[0] > 3.0
        # Buggy: weights collapse to ~1/rho regardless of size.
        assert w_buggy[1] / w_buggy[0] < 1.2


# ── weighted_ilr_mean ─────────────────────────────────────────────────────────

class TestWeightedIlrMean:
    def test_equal_weights_equal_to_arithmetic_mean(self):
        ilr_arr = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [2.0, 3.0, 4.0]])
        w = np.ones(3)
        result = weighted_ilr_mean(ilr_arr, w)
        np.testing.assert_allclose(result, ilr_arr.mean(axis=0), atol=1e-12)

    def test_zero_weight_slide_ignored(self):
        """Slide with weight 0 contributes nothing to the mean."""
        ilr_arr = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        w = np.array([0.0, 1.0])
        result = weighted_ilr_mean(ilr_arr, w)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-12)

    def test_all_weight_on_one_slide(self):
        ilr_arr = np.array([[5.0, 1.0, -1.0], [0.0, 0.0, 0.0]])
        w = np.array([100.0, 0.0])
        result = weighted_ilr_mean(ilr_arr, w)
        np.testing.assert_allclose(result, [5.0, 1.0, -1.0], atol=1e-12)

    def test_shape(self):
        ilr_arr = np.random.standard_normal((10, 3))
        w = np.random.uniform(1, 10, 10)
        result = weighted_ilr_mean(ilr_arr, w)
        assert result.shape == (3,)

    def test_two_slide_synthetic(self):
        """Verify against manual two-slide calculation."""
        ilr_arr = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        w = np.array([1.0, 3.0])
        # Expected: (1*1 + 3*3) / (1+3) = 10/4 = 2.5
        result = weighted_ilr_mean(ilr_arr, w)
        np.testing.assert_allclose(result[0], 2.5, atol=1e-12)
