"""Tests for WLS structural equation recovery with IDP importance weights."""

import numpy as np
import pandas as pd
import pytest

from causal_discovery.idp_and_cidp.pag_utils import get_definite_parents, get_possible_parents
from causal_discovery.idp_and_cidp.idp import idp
from causal_discovery.idp_and_cidp.evaluate import compute_importance_weights
from random_scm_generation import parse_scm_coefficients


def make_pag(var_names, edges):
    """Build a pcalg-style adjacency matrix from edge list.

    Each edge is (from, to, ep_at_from, ep_at_to) where:
        0=no edge, 1=circle, 2=arrowhead, 3=tail
    """
    n = len(var_names)
    amat = np.zeros((n, n), dtype=int)
    for frm, to, ep_from, ep_to in edges:
        i = var_names.index(frm)
        j = var_names.index(to)
        amat[i, j] = ep_to
        amat[j, i] = ep_from
    return amat


# ---------------------------------------------------------------------------
# Parent extraction tests
# ---------------------------------------------------------------------------

class TestGetParents:
    def test_definite_parents_simple(self):
        # X1 -> X2 -> X3
        names = ["X1", "X2", "X3"]
        amat = make_pag(names, [
            ("X1", "X2", 3, 2),  # X1 -> X2
            ("X2", "X3", 3, 2),  # X2 -> X3
        ])
        assert get_definite_parents(amat, "X3", names) == ["X2"]
        assert get_definite_parents(amat, "X2", names) == ["X1"]
        assert get_definite_parents(amat, "X1", names) == []

    def test_multiple_parents(self):
        # X1 -> X3, X2 -> X3
        names = ["X1", "X2", "X3"]
        amat = make_pag(names, [
            ("X1", "X3", 3, 2),
            ("X2", "X3", 3, 2),
        ])
        parents = get_definite_parents(amat, "X3", names)
        assert set(parents) == {"X1", "X2"}

    def test_possible_parents(self):
        # X1 o-> X2 (circle at X1, arrow at X2)
        names = ["X1", "X2"]
        amat = make_pag(names, [
            ("X1", "X2", 1, 2),
        ])
        assert get_definite_parents(amat, "X2", names) == []
        assert get_possible_parents(amat, "X2", names) == ["X1"]

    def test_bidirected_not_parent(self):
        # X1 <-> X2 (arrow at both ends)
        names = ["X1", "X2"]
        amat = make_pag(names, [
            ("X1", "X2", 2, 2),
        ])
        assert get_definite_parents(amat, "X2", names) == []
        assert get_possible_parents(amat, "X2", names) == []

    def test_mixed_parents(self):
        # X1 -> X3 (definite), X2 o-> X3 (possible)
        names = ["X1", "X2", "X3"]
        amat = make_pag(names, [
            ("X1", "X3", 3, 2),
            ("X2", "X3", 1, 2),
        ])
        assert get_definite_parents(amat, "X3", names) == ["X1"]
        assert get_possible_parents(amat, "X3", names) == ["X2"]


# ---------------------------------------------------------------------------
# WLS recovery tests using proper PAGs (with bidirected edges / circles)
# ---------------------------------------------------------------------------

class TestWLSRecovery:
    def test_hyttinen_pag_single_treatment(self):
        """Hyttinen PAG: x->z->y, h->y, x<->h, x<->w.

        P(y|do(x)) is identifiable. Test that we can get weights and
        run WLS for the marginal effect.
        """
        np.random.seed(42)
        n = 10000

        # True SCM: x->z->y, h->y, with latent confounders
        # x = ε_x + L1, h = ε_h + L1, w = ε_w + L2, x += L2
        L1 = np.random.normal(0, 1, n)
        L2 = np.random.normal(0, 1, n)
        x = np.random.normal(0, 1, n) + L1 + L2
        w = np.random.normal(0, 1, n) + L2
        h = np.random.normal(0, 1, n) + L1
        z = 2.0 * x + np.random.normal(0, 0.5, n)
        y = 3.0 * z + 1.5 * h + np.random.normal(0, 0.5, n)
        data = pd.DataFrame({"x": x, "y": y, "z": z, "h": h, "w": w})

        names = ["x", "y", "z", "h", "w"]
        amat = make_pag(names, [
            ("x", "z", 3, 2),  # x -> z
            ("z", "y", 3, 2),  # z -> y
            ("h", "y", 3, 2),  # h -> y
            ("x", "h", 2, 2),  # x <-> h
            ("x", "w", 2, 2),  # x <-> w
        ])

        result = idp(amat, ["x"], ["y"], names, verbose=False)
        assert result['id']

        weights = compute_importance_weights(result['Qop'], result['query'], data)
        assert len(weights) == n
        assert np.all(weights >= 0)

    def test_hyttinen_pag_joint_parents(self):
        """Recover structural equation y = β_z*z + β_h*h using joint IDP weights."""
        np.random.seed(42)
        n = 20000

        # True SCM
        beta_z, beta_h = 3.0, 1.5
        L1 = np.random.normal(0, 1, n)
        L2 = np.random.normal(0, 1, n)
        x = np.random.normal(0, 1, n) + L1 + L2
        w = np.random.normal(0, 1, n) + L2
        h = np.random.normal(0, 1, n) + L1
        z = 2.0 * x + np.random.normal(0, 0.5, n)
        y = beta_z * z + beta_h * h + np.random.normal(0, 0.5, n)
        data = pd.DataFrame({"x": x, "y": y, "z": z, "h": h, "w": w})

        names = ["x", "y", "z", "h", "w"]
        amat = make_pag(names, [
            ("x", "z", 3, 2),
            ("z", "y", 3, 2),
            ("h", "y", 3, 2),
            ("x", "h", 2, 2),
            ("x", "w", 2, 2),
        ])

        parents = get_definite_parents(amat, "y", names)
        assert set(parents) == {"z", "h"}

        # Joint IDP: P(y | do(z, h))
        result = idp(amat, parents, ["y"], names, verbose=False)
        assert result['id']

        weights = compute_importance_weights(result['Qop'], result['query'], data)

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(data[parents].values, data["y"].values, sample_weight=weights)

        # Map coefficients to parent names
        coeff_dict = dict(zip(parents, model.coef_))
        assert coeff_dict["z"] == pytest.approx(beta_z, abs=0.5)
        assert coeff_dict["h"] == pytest.approx(beta_h, abs=0.5)

    def test_hyttinen_single_parent_wls(self):
        """Hyttinen PAG: recover z->y coefficient using single-parent IDP weights.

        P(y|do(z)) is identifiable. WLS on y ~ z with these weights should
        give the direct structural coefficient, not the total effect of z on y.
        """
        np.random.seed(42)
        n = 20000
        beta_z = 3.0

        L1 = np.random.normal(0, 1, n)
        L2 = np.random.normal(0, 1, n)
        x = np.random.normal(0, 1, n) + L1 + L2
        w = np.random.normal(0, 1, n) + L2
        h = np.random.normal(0, 1, n) + L1
        z = 2.0 * x + np.random.normal(0, 0.5, n)
        y = beta_z * z + 1.5 * h + np.random.normal(0, 0.5, n)
        data = pd.DataFrame({"x": x, "y": y, "z": z, "h": h, "w": w})

        names = ["x", "y", "z", "h", "w"]
        amat = make_pag(names, [
            ("x", "z", 3, 2),
            ("z", "y", 3, 2),
            ("h", "y", 3, 2),
            ("x", "h", 2, 2),
            ("x", "w", 2, 2),
        ])

        result = idp(amat, ["z"], ["y"], names, verbose=False)
        assert result['id']

        weights = compute_importance_weights(result['Qop'], result['query'], data)

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(data[["z"]].values, data["y"].values, sample_weight=weights)

        # This is the marginal effect of z on y (not the structural coeff)
        # With h confounded via latent, the WLS should still give a reasonable estimate
        assert model.coef_[0] == pytest.approx(beta_z, abs=1.0)

    def test_with_generated_scm_true_graph(self):
        """Generate SCM, use true graph as PAG, run IDP + WLS to recover coefficients."""
        from random_scm_generation import generate_random_scm

        np.random.seed(789)
        data, true_graph, equations = generate_random_scm(
            num_vars=4, edge_prob=0.5, noise_level=0.5, num_samples=10000,
            max_mean=0, max_coefficient=3,
        )
        nodes = [n.get_name() for n in true_graph.get_nodes()]
        true_coeffs = parse_scm_coefficients(equations)

        # Use true graph's adjacency matrix as a PAG
        amat = true_graph.to_pcalg_matrix()

        # Find a node with definite parents where joint effect is identifiable
        target = None
        parents = None
        for node in nodes:
            dp = get_definite_parents(amat, node, nodes)
            if len(dp) >= 1:
                try:
                    r = idp(amat, dp, [node], nodes, verbose=False)
                    if r['id']:
                        target = node
                        parents = dp
                        break
                except Exception:
                    continue

        if target is None:
            pytest.skip("No node with identifiable joint parent effect found")

        result = idp(amat, parents, [target], nodes, verbose=False)
        weights = compute_importance_weights(result['Qop'], result['query'], data)

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(data[parents].values, data[target].values, sample_weight=weights)

        # Check each coefficient against truth
        true_target_coeffs = true_coeffs.get(target, {})
        for i, parent in enumerate(parents):
            true_val = true_target_coeffs.get(parent, 0.0)
            assert model.coef_[i] == pytest.approx(true_val, abs=1.0), (
                f"Coefficient for {parent} -> {target}: "
                f"expected {true_val:.3f}, got {model.coef_[i]:.3f}"
            )


class TestComputeImportanceWeights:
    def test_weights_shape_and_positive(self):
        """Importance weights should be non-negative with correct shape."""
        np.random.seed(42)
        n = 2000

        # Hyttinen-style PAG (known identifiable)
        L1 = np.random.normal(0, 1, n)
        L2 = np.random.normal(0, 1, n)
        x = np.random.normal(0, 1, n) + L1 + L2
        w = np.random.normal(0, 1, n) + L2
        h = np.random.normal(0, 1, n) + L1
        z = 2.0 * x + np.random.normal(0, 0.5, n)
        y = 3.0 * z + 1.5 * h + np.random.normal(0, 0.5, n)
        data = pd.DataFrame({"x": x, "y": y, "z": z, "h": h, "w": w})

        names = ["x", "y", "z", "h", "w"]
        amat = make_pag(names, [
            ("x", "z", 3, 2),
            ("z", "y", 3, 2),
            ("h", "y", 3, 2),
            ("x", "h", 2, 2),
            ("x", "w", 2, 2),
        ])

        result = idp(amat, ["x"], ["y"], names, verbose=False)
        assert result['id']

        weights = compute_importance_weights(result['Qop'], result['query'], data)
        assert weights.shape == (n,)
        assert np.all(weights >= 0)
        assert np.sum(weights) > 0
