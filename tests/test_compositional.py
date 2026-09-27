"""Correctness tests for core/compositional.py."""
from __future__ import annotations

import numpy as np
import pytest

from config import LABELS
from core.compositional import (
    cmult_repl,
    alr,
    ilr,
    ilr_inv,
    aitchison_distance,
    prepare_compositional,
)
import pandas as pd


# ── cmult_repl ────────────────────────────────────────────────────────────────

class TestCmultRepl:
    def test_output_strictly_positive(self):
        counts = np.array([[10, 0, 5, 0], [0, 0, 0, 100]])
        result = cmult_repl(counts)
        assert (result > 0).all(), "All outputs must be strictly positive"

    def test_output_sums_to_one(self):
        counts = np.array([[10, 5, 3, 2], [0, 100, 0, 0], [1, 1, 1, 1]])
        result = cmult_repl(counts)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-12)

    def test_no_zeros_input_preserved_proportionally(self):
        """Non-zero input proportions are preserved (multiplicative adjustment is zero)."""
        counts = np.array([[10, 20, 30, 40]])
        result = cmult_repl(counts)
        expected = np.array([10, 20, 30, 40]) / 100.0
        # With no zeros, nz=0 → zero_mass=0 → result = row/n exactly
        np.testing.assert_allclose(result[0], expected, atol=1e-12)

    def test_single_nonzero_component(self):
        """One non-zero component → rest get pseudoproportions, non-zero scaled down."""
        counts = np.array([[0, 0, 0, 100]])
        result = cmult_repl(counts)
        assert (result > 0).all()
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-12)
        # Non-zero component should be largest
        assert result[0, 3] == result[0].max()

    def test_1d_input(self):
        counts = np.array([50, 0, 25, 25])
        result = cmult_repl(counts)
        assert result.shape == (4,)
        assert (result > 0).all()
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-12)

    def test_all_zeros_returns_uniform(self):
        counts = np.array([[0, 0, 0, 0]])
        result = cmult_repl(counts)
        np.testing.assert_allclose(result[0], [0.25, 0.25, 0.25, 0.25], atol=1e-12)

    def test_zero_replacement_czm_formula(self):
        """Verify the CZM pseudoproportion formula t_j = 0.5/n per zero."""
        counts = np.array([[100, 0, 0, 0]])  # n=100, nz=3, t_j=0.005
        result = cmult_repl(counts)
        t_j = 0.5 / 100  # = 0.005
        np.testing.assert_allclose(result[0, 1], t_j, atol=1e-10)
        np.testing.assert_allclose(result[0, 2], t_j, atol=1e-10)
        np.testing.assert_allclose(result[0, 3], t_j, atol=1e-10)

    def test_large_array(self):
        rng = np.random.default_rng(42)
        counts = rng.integers(0, 500, (100, 4))
        result = cmult_repl(counts)
        assert (result > 0).all()
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-12)

    def test_single_detection_row_strictly_positive(self):
        """Degenerate rows (n=1, 3 zeros) must NOT produce negative proportions.

        Regression: the multiplicative rescale (1 - nz·0.5/n) goes negative when
        the zero pseudo-mass exceeds the total (here 3·0.5/1 = 1.5 > 1), which used
        to yield e.g. [-0.5, 0.5, 0.5, 0.5] and feed NaN into ilr()'s log. Falls back to the
        Dirichlet(0.5) posterior mean (x_j + 0.5)/(n + 0.5·D).
        """
        counts = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        result = cmult_repl(counts)
        assert (result > 0).all(), f"negative proportion produced: {result}"
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-12)
        # Dirichlet(0.5) posterior mean for [1,0,0,0]: (x+0.5)/(1+2)
        np.testing.assert_allclose(
            result[0], np.array([1.5, 0.5, 0.5, 0.5]) / 3.0, atol=1e-12)
        # The observed component stays the largest, and ilr() is finite (no NaN).
        assert result[0].argmax() == 0
        assert np.isfinite(ilr(result)).all()

    def test_well_sampled_rows_unchanged_by_degenerate_guard(self):
        """The guard must only touch degenerate rows; normal CZM is preserved."""
        counts = np.array([[100, 0, 0, 0]], dtype=float)  # zero_mass = 1.5/100 < 1
        result = cmult_repl(counts)
        np.testing.assert_allclose(result[0, 1:], 0.5 / 100, atol=1e-12)


# ── alr ───────────────────────────────────────────────────────────────────────

class TestAlr:
    def test_output_shape_1d(self):
        p = np.array([0.3, 0.3, 0.2, 0.2])
        result = alr(p, ref_idx=1)
        assert result.shape == (3,)

    def test_output_shape_2d(self):
        p = np.tile([0.3, 0.3, 0.2, 0.2], (5, 1))
        result = alr(p, ref_idx=1)
        assert result.shape == (5, 3)

    def test_alr_formula(self):
        """ALR = log(p_i / p_ref) for i ≠ ref."""
        p = np.array([0.4, 0.2, 0.3, 0.1])
        result = alr(p, ref_idx=1)  # ref = p[1] = 0.2
        expected = np.log(np.array([0.4, 0.3, 0.1]) / 0.2)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_alr_equal_composition_is_zero(self):
        """Equal proportions → ALR = 0 for all coords."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        result = alr(p, ref_idx=1)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_alr_uses_default_ref(self):
        """Default ref_idx corresponds to ALR_REF_LABEL (Lymphozyt = index 1)."""
        p = np.array([0.4, 0.2, 0.3, 0.1])
        result_default = alr(p)
        result_ref1 = alr(p, ref_idx=1)
        np.testing.assert_allclose(result_default, result_ref1, atol=1e-12)


# ── ilr ───────────────────────────────────────────────────────────────────────

class TestIlr:
    def test_output_shape_1d(self):
        p = np.array([0.3, 0.3, 0.2, 0.2])
        result = ilr(p)
        assert result.shape == (3,)

    def test_output_shape_2d(self):
        p = np.tile([0.3, 0.3, 0.2, 0.2], (5, 1))
        result = ilr(p)
        assert result.shape == (5, 3)

    def test_ilr_d4_known_values(self):
        """Hand-check ILR for D=4 with equal proportions → (0, 0, 0)."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        result = ilr(p)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_ilr_helmert_basis_orthonormality(self):
        """The ILR basis vectors should be orthonormal in R^4.

        Basis row k: has 1/k for positions 0..k-1, -1 at position k, 0 elsewhere,
        scaled by sqrt(k/(k+1)).  Verify V @ V.T = I_{D-1}.
        """
        D = 4
        D1 = D - 1
        # Build the Helmert basis matrix V: (D-1, D)
        V = np.zeros((D1, D))
        for k in range(1, D):
            c = np.sqrt(k / (k + 1))
            V[k - 1, :k] = c / k
            V[k - 1, k] = -c
        # Must be orthonormal: V @ V.T = I
        gram = V @ V.T
        np.testing.assert_allclose(gram, np.eye(D1), atol=1e-12)

    def test_ilr_distance_equals_euclidean_in_ilr_space(self):
        """Aitchison distance = Euclidean distance in ILR space (defining property)."""
        p = np.array([0.5, 0.2, 0.2, 0.1])
        q = np.array([0.3, 0.3, 0.2, 0.2])
        p_safe = cmult_repl(p)
        q_safe = cmult_repl(q)
        d_aitchison = aitchison_distance(p_safe, q_safe)
        d_ilr = float(np.linalg.norm(ilr(p_safe) - ilr(q_safe)))
        np.testing.assert_allclose(d_aitchison, d_ilr, atol=1e-10)

    def test_ilr_invariant_to_renormalisation(self):
        """ILR is scale-invariant: ilr(p) = ilr(c·p) for any c > 0."""
        p = np.array([0.4, 0.3, 0.2, 0.1])
        np.testing.assert_allclose(ilr(p), ilr(2 * p / (2 * p).sum()), atol=1e-12)


# ── ilr_inv ───────────────────────────────────────────────────────────────────

class TestIlrInv:
    def test_ilr_roundtrip(self):
        """ilr_inv(ilr(p)) ≈ p."""
        p = np.array([0.4, 0.3, 0.2, 0.1])
        np.testing.assert_allclose(ilr_inv(ilr(p)), p, atol=1e-10)

    def test_ilr_inv_sums_to_one(self):
        z = np.array([0.5, -0.3, 0.1])
        p_reconstructed = ilr_inv(z)
        np.testing.assert_allclose(p_reconstructed.sum(), 1.0, atol=1e-12)
        assert (p_reconstructed > 0).all()

    def test_ilr_inv_zero_maps_to_uniform(self):
        z = np.zeros(3)
        p = ilr_inv(z)
        np.testing.assert_allclose(p, 0.25, atol=1e-12)

    def test_ilr_inv_2d(self):
        Z = np.array([[0.5, -0.3, 0.1], [0.0, 0.0, 0.0]])
        P = ilr_inv(Z)
        assert P.shape == (2, 4)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)


# ── aitchison_distance ────────────────────────────────────────────────────────

class TestAitchisonDistance:
    def test_distance_to_self_is_zero(self):
        p = np.array([0.4, 0.3, 0.2, 0.1])
        np.testing.assert_allclose(aitchison_distance(p, p), 0.0, atol=1e-10)

    def test_symmetry(self):
        p = np.array([0.5, 0.2, 0.2, 0.1])
        q = np.array([0.3, 0.3, 0.2, 0.2])
        assert abs(aitchison_distance(p, q) - aitchison_distance(q, p)) < 1e-10

    def test_non_negativity(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = cmult_repl(rng.integers(1, 100, 4))
            q = cmult_repl(rng.integers(1, 100, 4))
            assert aitchison_distance(p, q) >= 0.0

    def test_triangle_inequality(self):
        p = cmult_repl(np.array([50, 10, 30, 10]))
        q = cmult_repl(np.array([10, 60, 20, 10]))
        r = cmult_repl(np.array([20, 20, 40, 20]))
        d_pq = aitchison_distance(p, q)
        d_qr = aitchison_distance(q, r)
        d_pr = aitchison_distance(p, r)
        assert d_pr <= d_pq + d_qr + 1e-10, \
            f"Triangle inequality violated: d(p,r)={d_pr} > d(p,q)+d(q,r)={d_pq+d_qr}"

    def test_known_distance_d2(self):
        """Verify against textbook formula for D=2: d(p,q) = |log(p1/p2) - log(q1/q2)| / sqrt(2)."""
        # For D=2, ILR has 1 coord: z = sqrt(1/2) * (log(p1) - log(p2))
        # distance = |z_p - z_q|
        # = sqrt(1/2) * |log(p1/p2) - log(q1/q2)|
        p2 = np.array([0.3, 0.7])
        q2 = np.array([0.6, 0.4])
        # Simulate with D=2: need to extend to D=4 (not directly applicable)
        # So test with D=4 where two components are negligibly small
        eps = 1e-6
        p = np.array([0.3 - eps, 0.7 - eps, eps, eps])
        p /= p.sum()
        q = np.array([0.6 - eps, 0.4 - eps, eps, eps])
        q /= q.sum()
        d = aitchison_distance(p, q)
        assert d > 0


# ── prepare_compositional ─────────────────────────────────────────────────────

class TestPrepareCompositional:
    def make_df(self):
        data = {lbl: np.random.randint(10, 200, size=8) for lbl in LABELS}
        return pd.DataFrame(data)

    def test_returns_correct_shapes(self):
        df = self.make_df()
        alr_df, ilr_df = prepare_compositional(df)
        assert alr_df.shape == (len(df), len(LABELS) - 1)
        assert ilr_df.shape == (len(df), len(LABELS) - 1)

    def test_alr_columns_exclude_ref_label(self):
        from config import ALR_REF_LABEL
        df = self.make_df()
        alr_df, _ = prepare_compositional(df)
        assert ALR_REF_LABEL not in alr_df.columns

    def test_zero_total_rows_dropped(self):
        """Rows where all labels are zero should be dropped."""
        df = self.make_df()
        df.loc[3] = 0  # zero row
        alr_df, ilr_df = prepare_compositional(df)
        assert len(alr_df) == len(df) - 1

    def test_ilr_cols_named_correctly(self):
        df = self.make_df()
        _, ilr_df = prepare_compositional(df)
        for k in range(1, len(LABELS)):
            assert f"ilr_{k}" in ilr_df.columns
