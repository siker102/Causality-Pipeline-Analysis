"""Tests for IDP on directed-only PAGs (no circles/bidirected).

Background
----------
Zhang 2008 Definition 3.1 defines a directed edge A -> B as "visible" if
there exists a third vertex C, not adjacent to B, such that C *-> A. This
definition was designed for PAGs with genuine ambiguity (circle endpoints)
or latent confounders (bidirected edges).

The problem: in a PAG with ONLY directed edges (no circles, no bidirected),
there is no equivalence-class ambiguity and no latent confounding — the graph
is effectively a known DAG. But Zhang's definition can still classify edges
as "invisible" simply because no qualifying third vertex C exists (e.g., in
a 2-node graph, or when all neighbors of A are also adjacent to B).

This caused IDP to incorrectly report "not identifiable" for trivially
identifiable DAG effects like P(X2|do(X1)) in X1 -> X2.

Cross-validation with R (pcalg::visibleEdge + PAGId::IDP) confirmed that R
has the same limitation — this is a gap in Zhang's formal definition, not a
bug in any implementation.

Fix: _is_dag_like() in pag_utils.py detects when the PAG has no circles and
no bidirected edges, and visible_edge() returns True for all directed edges
in that case. See theory/idp_invisible_edge_investigation.md for full details.
"""

import numpy as np

from causal_discovery.idp_and_cidp.idp import idp


def make_pag(var_names, edges):
    n = len(var_names)
    amat = np.zeros((n, n), dtype=int)
    for frm, to, ep_from, ep_to in edges:
        i = var_names.index(frm)
        j = var_names.index(to)
        amat[i, j] = ep_to
        amat[j, i] = ep_from
    return amat


class TestIDPDirectedOnly:
    def test_two_node_directed(self):
        """X1 -> X2: P(X2|do(X1)) should be identifiable.

        Simplest case — only 2 nodes, no third vertex C can exist for
        Zhang's visibility check. The _is_dag_like short-circuit handles this.
        """
        names = ["X1", "X2"]
        amat = make_pag(names, [("X1", "X2", 3, 2)])
        result = idp(amat, ["X1"], ["X2"], names, verbose=False)
        assert result['id']

    def test_three_node_chain(self):
        """X -> Z -> Y: P(Y|do(X)) should be identifiable.

        Edge X -> Z has no qualifying C (Y is adjacent to Z). Without the
        DAG fix, this edge is invisible, inflating the PC-component and
        blocking Prop 6. With the fix, all edges are visible in this DAG.
        """
        names = ["X", "Z", "Y"]
        amat = make_pag(names, [
            ("X", "Z", 3, 2),
            ("Z", "Y", 3, 2),
        ])
        result = idp(amat, ["X"], ["Y"], names, verbose=False)
        assert result['id']
