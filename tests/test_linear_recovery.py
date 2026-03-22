"""Tests for linear equation recovery from SCM equations."""

import numpy as np
import pytest

from random_scm_generation import (
    generate_random_scm,
    parse_scm_coefficients,
    total_causal_effect,
)


class TestParseScmCoefficients:
    def test_single_parent(self):
        equations = {"X2": "X2 = 3.50*X1 + ε ~ N(0.00, 1.00)"}
        coeffs = parse_scm_coefficients(equations)
        assert coeffs["X2"] == {"X1": 3.5}

    def test_multiple_parents(self):
        equations = {"X3": "X3 = 2.00*X1 + -1.50*X2 + ε ~ N(0.00, 1.00)"}
        coeffs = parse_scm_coefficients(equations)
        assert coeffs["X3"]["X1"] == pytest.approx(2.0)
        assert coeffs["X3"]["X2"] == pytest.approx(-1.5)

    def test_no_parents(self):
        equations = {"X1": "X1 = ε ~ N(5.00, 1.00)"}
        coeffs = parse_scm_coefficients(equations)
        assert coeffs["X1"] == {}


class TestTotalCausalEffect:
    def test_direct_effect(self):
        # X1 -> X2 with coefficient 3.0
        equations = {
            "X1": "X1 = ε ~ N(0.00, 1.00)",
            "X2": "X2 = 3.00*X1 + ε ~ N(0.00, 1.00)",
        }
        assert total_causal_effect(equations, "X1", "X2") == pytest.approx(3.0)

    def test_no_effect(self):
        # X1 and X2 are independent
        equations = {
            "X1": "X1 = ε ~ N(0.00, 1.00)",
            "X2": "X2 = ε ~ N(0.00, 1.00)",
        }
        assert total_causal_effect(equations, "X1", "X2") == pytest.approx(0.0)

    def test_mediated_effect(self):
        # X1 -> X2 -> X3, coefficients 2.0 and 3.0
        # Total effect of X1 on X3 = 2.0 * 3.0 = 6.0
        equations = {
            "X1": "X1 = ε ~ N(0.00, 1.00)",
            "X2": "X2 = 2.00*X1 + ε ~ N(0.00, 1.00)",
            "X3": "X3 = 3.00*X2 + ε ~ N(0.00, 1.00)",
        }
        assert total_causal_effect(equations, "X1", "X3") == pytest.approx(6.0)

    def test_direct_plus_mediated(self):
        # X1 -> X2 -> X3, plus X1 -> X3
        # Path 1: X1 -> X3 = 1.0
        # Path 2: X1 -> X2 -> X3 = 2.0 * 3.0 = 6.0
        # Total = 7.0
        equations = {
            "X1": "X1 = ε ~ N(0.00, 1.00)",
            "X2": "X2 = 2.00*X1 + ε ~ N(0.00, 1.00)",
            "X3": "X3 = 3.00*X2 + 1.00*X1 + ε ~ N(0.00, 1.00)",
        }
        assert total_causal_effect(equations, "X1", "X3") == pytest.approx(7.0)

    def test_diamond_graph(self):
        # X1 -> X2 (a=2), X1 -> X3 (b=3), X2 -> X4 (c=4), X3 -> X4 (d=5)
        # Total X1->X4 = a*c + b*d = 2*4 + 3*5 = 23
        equations = {
            "X1": "X1 = ε ~ N(0.00, 1.00)",
            "X2": "X2 = 2.00*X1 + ε ~ N(0.00, 1.00)",
            "X3": "X3 = 3.00*X1 + ε ~ N(0.00, 1.00)",
            "X4": "X4 = 4.00*X2 + 5.00*X3 + ε ~ N(0.00, 1.00)",
        }
        assert total_causal_effect(equations, "X1", "X4") == pytest.approx(23.0)

    def test_negative_coefficients(self):
        equations = {
            "X1": "X1 = ε ~ N(0.00, 1.00)",
            "X2": "X2 = -2.50*X1 + ε ~ N(0.00, 1.00)",
        }
        assert total_causal_effect(equations, "X1", "X2") == pytest.approx(-2.5)

    def test_with_generated_scm(self):
        """Verify total_causal_effect against brute-force interventional simulation."""
        np.random.seed(42)
        data, true_graph, equations = generate_random_scm(
            num_vars=5, edge_prob=0.4, noise_level=0.01, num_samples=50000,
            max_mean=0, max_coefficient=5,
        )
        nodes = [n.get_name() for n in true_graph.get_nodes()]
        treatment, outcome = nodes[0], nodes[-1]

        analytic = total_causal_effect(equations, treatment, outcome)

        # Empirical: E[outcome | do(treatment=1)] - E[outcome | do(treatment=0)]
        # For linear SCM this equals the total causal effect coefficient
        coeffs = parse_scm_coefficients(equations)

        # Simulate intervention do(treatment=1) vs do(treatment=0)
        from collections import defaultdict
        import networkx as nx

        n_sim = 100000
        num_vars = len(nodes)
        adj_matrix = np.zeros((num_vars, num_vars), dtype=int)
        for child, parents in coeffs.items():
            ci = nodes.index(child)
            for parent in parents:
                pi = nodes.index(parent)
                adj_matrix[pi, ci] = 1

        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        ti = nodes.index(treatment)
        oi = nodes.index(outcome)

        def simulate_intervention(treat_val):
            sim = np.zeros((n_sim, num_vars))
            for i in nx.topological_sort(G):
                if i == ti:
                    sim[:, i] = treat_val
                else:
                    parents = list(G.predecessors(i))
                    var_name = nodes[i]
                    linear_comb = sum(
                        sim[:, p] * coeffs[var_name].get(nodes[p], 0)
                        for p in parents
                    )
                    sim[:, i] = linear_comb + np.random.normal(0, 0.01, n_sim)
                    # Add offset parsed from equation
                    import re
                    m = re.search(r'N\(([+-]?\d+\.?\d*)', equations[var_name])
                    if m:
                        sim[:, i] += float(m.group(1))
            return sim[:, oi].mean()

        empirical = simulate_intervention(1.0) - simulate_intervention(0.0)
        assert analytic == pytest.approx(empirical, abs=0.1)
