"""Correctness tests for core/stats.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from core.stats import (
    friedman_test,
    nemenyi_post_hoc,
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
