"""Tests for the V2 regression-based causal effect evaluator."""

import numpy as np
import pandas as pd
import pytest

from causal_discovery.idp_and_cidp.idp import idp
from causal_discovery.idp_and_cidp.cidp import cidp
from causal_discovery.idp_and_cidp.evaluate import (
    evaluate_causal_effect,
    _parse_query,
)


def make_pag(var_names, edges):
    n = len(var_names)
    amat = np.zeros((n, n), dtype=int)
    for frm, to, ep_from, ep_to in edges:
        i = var_names.index(frm)
        j = var_names.index(to)
        amat[i, j] = ep_to
        amat[j, i] = ep_from
    return amat


# ---------------------------------------------------------------------------
# Query parsing
# ---------------------------------------------------------------------------

class TestParseQuery:
    def test_simple(self):
        t, o = _parse_query("P_{x}(y)")
        assert t == ["x"]
        assert o == ["y"]

    def test_multi_treatment(self):
        t, o = _parse_query("P_{x1,x2}(y)")
        assert t == ["x1", "x2"]
        assert o == ["y"]

    def test_multi_outcome(self):
        t, o = _parse_query("P_{x}(y,z)")
        assert t == ["x"]
        assert o == ["y", "z"]

    def test_conditional(self):
        t, o = _parse_query("P_{w,x1,x2}(y | z)")
        assert t == ["w", "x1", "x2"]
        assert o == ["y"]


# ---------------------------------------------------------------------------
# Hyttinen-like chain with visible edges: x->z->y, x<->h, h->y
# P(y|do(x)) is identifiable.
# ---------------------------------------------------------------------------

class TestVisibleChainEvaluator:
    @pytest.fixture
    def chain_data(self):
        rng = np.random.default_rng(42)
        n = 10_000
        U = rng.choice([0, 1], size=n)
        h = (U + rng.choice([0, 1], size=n, p=[0.6, 0.4])) % 2
        x = (U + rng.choice([0, 1], size=n, p=[0.5, 0.5])) % 2
        z = (x + rng.choice([0, 1], size=n, p=[0.8, 0.2])) % 2
        y = (z + h + rng.choice([0, 1], size=n, p=[0.7, 0.3])) % 3 % 2
        return pd.DataFrame({"x": x, "z": z, "y": y, "h": h})

    @pytest.fixture
    def chain_pag(self):
        names = ["x", "z", "y", "h"]
        edges = [
            ("x", "z", 3, 2),
            ("z", "y", 3, 2),
            ("h", "y", 3, 2),
            ("x", "h", 2, 2),
        ]
        return make_pag(names, edges), names

    def test_identifiable(self, chain_pag):
        amat, names = chain_pag
        result = idp(amat, ["x"], ["y"], names)
        assert result["id"] is True

    def test_evaluation_sums_to_one(self, chain_data, chain_pag):
        amat, names = chain_pag
        result = idp(amat, ["x"], ["y"], names)
        effect = evaluate_causal_effect(
            result["Qop"], result["query"], chain_data,
            treatment_vars=["x"], outcome_vars=["y"],
        )

        assert "x" in effect.columns
        assert "y" in effect.columns
        assert "prob" in effect.columns

        for x_val in effect["x"].unique():
            subset = effect[effect["x"] == x_val]
            assert abs(subset["prob"].sum() - 1.0) < 0.05


# ---------------------------------------------------------------------------
# Bidirected: X <-> Y (latent confounder, no direct effect)
# P(Y|do(X)) = P(Y)
# ---------------------------------------------------------------------------

class TestBidirectedEvaluator:
    @pytest.fixture
    def bidirected_data(self):
        rng = np.random.default_rng(42)
        n = 10_000
        U = rng.choice([0, 1], size=n, p=[0.5, 0.5])
        X = np.where(U == 0,
                     rng.choice([0, 1], size=n, p=[0.8, 0.2]),
                     rng.choice([0, 1], size=n, p=[0.3, 0.7]))
        Y = np.where(U == 0,
                     rng.choice([0, 1], size=n, p=[0.7, 0.3]),
                     rng.choice([0, 1], size=n, p=[0.4, 0.6]))
        return pd.DataFrame({"X": X, "Y": Y})

    @pytest.fixture
    def bidirected_pag(self):
        names = ["X", "Y"]
        edges = [("X", "Y", 2, 2)]
        return make_pag(names, edges), names

    def test_bidirected_evaluation(self, bidirected_data, bidirected_pag):
        """X <-> Y: P(Y|do(X)) = P(Y), constant across X values."""
        amat, names = bidirected_pag
        result = idp(amat, ["X"], ["Y"], names)
        assert result["id"] is True

        effect = evaluate_causal_effect(
            result["Qop"], result["query"], bidirected_data,
            treatment_vars=["X"], outcome_vars=["Y"],
        )

        assert "X" in effect.columns
        assert "Y" in effect.columns
        assert "prob" in effect.columns

        # P(Y|do(X)) should equal marginal P(Y) for all X values
        for x_val in effect["X"].unique():
            for y_val in [0, 1]:
                row = effect[(effect["X"] == x_val) & (effect["Y"] == y_val)]
                p_marginal = (bidirected_data["Y"] == y_val).mean()
                assert abs(row["prob"].iloc[0] - p_marginal) < 1e-6


# ---------------------------------------------------------------------------
# Hyttinen example: x->z->y, h->y, x<->h, x<->w
# ---------------------------------------------------------------------------

class TestHyttinenEvaluator:
    @pytest.fixture
    def hyttinen_data(self):
        rng = np.random.default_rng(42)
        n = 10_000
        u_xh = rng.choice([0, 1], size=n, p=[0.5, 0.5])
        u_xw = rng.choice([0, 1], size=n, p=[0.5, 0.5])
        x = (u_xh + u_xw + rng.choice([0, 1], size=n, p=[0.7, 0.3])) % 2
        h = (u_xh + rng.choice([0, 1], size=n, p=[0.6, 0.4])) % 2
        w = (u_xw + rng.choice([0, 1], size=n, p=[0.5, 0.5])) % 2
        z = (x + rng.choice([0, 1], size=n, p=[0.8, 0.2])) % 2
        y = (z + h + rng.choice([0, 1], size=n, p=[0.7, 0.3])) % 3 % 2
        return pd.DataFrame({"x": x, "y": y, "z": z, "h": h, "w": w})

    @pytest.fixture
    def hyttinen_pag(self):
        names = ["x", "y", "z", "h", "w"]
        edges = [
            ("x", "z", 3, 2),
            ("z", "y", 3, 2),
            ("h", "y", 3, 2),
            ("x", "h", 2, 2),
            ("x", "w", 2, 2),
        ]
        return make_pag(names, edges), names

    def test_hyttinen_runs(self, hyttinen_data, hyttinen_pag):
        amat, names = hyttinen_pag
        result = idp(amat, ["x"], ["y"], names)
        assert result["id"] is True

        effect = evaluate_causal_effect(
            result["Qop"], result["query"], hyttinen_data,
            treatment_vars=["x"], outcome_vars=["y"],
        )

        assert "prob" in effect.columns
        for x_val in effect["x"].unique():
            subset = effect[effect["x"] == x_val]
            assert abs(subset["prob"].sum() - 1.0) < 0.05


# ---------------------------------------------------------------------------
# CIDP: conditional causal effect
# ---------------------------------------------------------------------------

class TestCIDPEvaluator:
    @pytest.fixture
    def pag1_data(self):
        rng = np.random.default_rng(42)
        n = 10_000
        a = rng.choice([0, 1], size=n)
        x2 = rng.choice([0, 1], size=n)
        x1 = (a + x2 + rng.choice([0, 1], size=n)) % 2
        w = (a + x2 + x1 + rng.choice([0, 1], size=n)) % 2
        z = (w + rng.choice([0, 1], size=n)) % 2
        y = (z + a + rng.choice([0, 1], size=n)) % 2
        return pd.DataFrame({"x1": x1, "x2": x2, "y": y, "a": a, "w": w, "z": z})

    @pytest.fixture
    def pag1(self):
        names = ["x1", "x2", "y", "a", "w", "z"]
        amat = np.zeros((6, 6), dtype=int)
        idx = {n: i for i, n in enumerate(names)}
        amat[idx["w"], idx["z"]] = 2; amat[idx["z"], idx["w"]] = 3
        amat[idx["z"], idx["y"]] = 2; amat[idx["y"], idx["z"]] = 3
        amat[idx["a"], idx["y"]] = 2; amat[idx["y"], idx["a"]] = 3
        amat[idx["a"], idx["w"]] = 2; amat[idx["w"], idx["a"]] = 1
        amat[idx["a"], idx["x1"]] = 2; amat[idx["x1"], idx["a"]] = 1
        amat[idx["x2"], idx["x1"]] = 2; amat[idx["x1"], idx["x2"]] = 1
        amat[idx["x2"], idx["w"]] = 2; amat[idx["w"], idx["x2"]] = 1
        amat[idx["x1"], idx["w"]] = 1; amat[idx["w"], idx["x1"]] = 1
        return amat, names

    def test_cidp_evaluation(self, pag1_data, pag1):
        amat, names = pag1
        result = cidp(amat, ["w", "x1", "x2"], ["y"], ["z"], names)
        assert result["id"] is True

        effect = evaluate_causal_effect(
            result["Qop"], result["query"], pag1_data,
            treatment_vars=["w", "x1", "x2"], outcome_vars=["y"],
        )
        assert "prob" in effect.columns
        assert "y" in effect.columns
        assert effect["prob"].sum() > 0


# ---------------------------------------------------------------------------
# Continuous data
# ---------------------------------------------------------------------------

class TestContinuousEvaluator:
    def test_continuous_hyttinen(self):
        """Test that continuous data can be evaluated without intermediate binning."""
        rng = np.random.default_rng(42)
        n = 5_000
        u_xh = rng.normal(0, 1, n)
        u_xw = rng.normal(0, 1, n)
        x = u_xh + u_xw + rng.normal(0, 0.5, n)
        h = u_xh + rng.normal(0, 0.5, n)
        w = u_xw + rng.normal(0, 0.5, n)
        z = x + rng.normal(0, 0.5, n)
        y = z + h + rng.normal(0, 0.5, n)
        data = pd.DataFrame({"x": x, "y": y, "z": z, "h": h, "w": w})

        names = ["x", "y", "z", "h", "w"]
        amat = make_pag(names, [
            ("x", "z", 3, 2), ("z", "y", 3, 2), ("h", "y", 3, 2),
            ("x", "h", 2, 2), ("x", "w", 2, 2),
        ])
        result = idp(amat, ["x"], ["y"], names)
        assert result["id"] is True

        effect = evaluate_causal_effect(
            result["Qop"], result["query"], data,
            treatment_vars=["x"], outcome_vars=["y"],
            treatment_bins=5, outcome_bins=5,
        )
        assert "prob" in effect.columns
        assert len(effect) > 0
        assert (effect["prob"] >= -1e-10).all()


class TestPerTreatmentOutcomeBinning:
    """Test that outcome bins are computed per treatment group via qcut."""

    @pytest.fixture
    def chain_scm(self):
        rng = np.random.default_rng(42)
        n = 5_000
        X1 = rng.normal(-5.56, 1.0, n)
        X2 = -2.11 * X1 + rng.normal(-7.26, 1.0, n)
        X3 = 7.82 * X1 + rng.normal(4.01, 1.0, n)
        X4 = 8.08 * X1 + rng.normal(2.55, 1.0, n)
        X5 = -2.13 * X4 + rng.normal(-1.14, 1.0, n)
        data = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5})

        names = ["X1", "X2", "X3", "X4", "X5"]
        amat = make_pag(names, [
            ("X1", "X2", 3, 2), ("X1", "X3", 3, 2),
            ("X1", "X4", 3, 2), ("X4", "X5", 3, 2),
        ])
        result = idp(amat, ["X4"], ["X5"], names)
        assert result["id"] is True
        return result, data

    def test_every_treatment_bin_has_outcome_bins_and_sums_to_one(self, chain_scm):
        result, data = chain_scm
        effect = evaluate_causal_effect(
            result["Qop"], result["query"], data,
            treatment_vars=["X4"], outcome_vars=["X5"],
            treatment_bins=5, outcome_bins=5,
        )
        assert "X4" in effect.columns
        assert "X5" in effect.columns
        for t_val in effect["X4"].unique():
            subset = effect[effect["X4"] == t_val]
            assert len(subset) >= 2, (
                f"Treatment bin {t_val} has {len(subset)} outcome bins, expected >= 2"
            )
            assert abs(subset["prob"].sum() - 1.0) < 0.05

    def test_outcome_bins_differ_per_treatment(self, chain_scm):
        result, data = chain_scm
        effect = evaluate_causal_effect(
            result["Qop"], result["query"], data,
            treatment_vars=["X4"], outcome_vars=["X5"],
            treatment_bins=5, outcome_bins=5,
        )
        treat_values = effect["X4"].unique()
        outcome_bins_per_treat = []
        for t_val in treat_values:
            subset = effect[effect["X4"] == t_val]
            bins = sorted(subset["X5"].unique(), key=str)
            outcome_bins_per_treat.append(bins)
        assert len(set(str(b) for b in outcome_bins_per_treat)) > 1

    def test_no_zero_probability_bins(self, chain_scm):
        result, data = chain_scm
        effect = evaluate_causal_effect(
            result["Qop"], result["query"], data,
            treatment_vars=["X4"], outcome_vars=["X5"],
            treatment_bins=5, outcome_bins=5,
        )
        for t_val in effect["X4"].unique():
            subset = effect[effect["X4"] == t_val]
            assert (subset["prob"] > 0).all(), (
                f"Treatment bin {t_val} has zero-probability outcome bins"
            )


# ---------------------------------------------------------------------------
# Auto-detection: evaluate without explicit treatment/outcome vars
# ---------------------------------------------------------------------------

class TestAutoDetection:
    def test_auto_detect_from_query(self):
        """evaluate_causal_effect auto-detects treatment/outcome from query."""
        rng = np.random.default_rng(42)
        n = 10_000
        U = rng.choice([0, 1], size=n)
        h = (U + rng.choice([0, 1], size=n, p=[0.6, 0.4])) % 2
        x = (U + rng.choice([0, 1], size=n, p=[0.5, 0.5])) % 2
        z = (x + rng.choice([0, 1], size=n, p=[0.8, 0.2])) % 2
        y = (z + h + rng.choice([0, 1], size=n, p=[0.7, 0.3])) % 3 % 2
        data = pd.DataFrame({"x": x, "z": z, "y": y, "h": h})

        names = ["x", "z", "y", "h"]
        amat = make_pag(names, [
            ("x", "z", 3, 2), ("z", "y", 3, 2),
            ("h", "y", 3, 2), ("x", "h", 2, 2),
        ])
        result = idp(amat, ["x"], ["y"], names)
        assert result["id"] is True

        # Call without treatment_vars/outcome_vars — auto-detect from query
        effect = evaluate_causal_effect(result["Qop"], result["query"], data)
        assert "x" in effect.columns
        assert "y" in effect.columns
        assert "prob" in effect.columns

        for x_val in effect["x"].unique():
            subset = effect[effect["x"] == x_val]
            assert abs(subset["prob"].sum() - 1.0) < 0.05
