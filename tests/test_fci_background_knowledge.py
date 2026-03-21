"""
Tests for FCI background knowledge orientation logic.

Covers:
- fci_orient_bk: single-direction forbidden, both-directions forbidden,
  required edges, priority of required over forbidden
- is_arrow_point_allowed: guard function used by orientation rules
- reorientAllWith: preservation of required edges during circle reset
- Integration: fci_remake with background knowledge constraints
"""
import numpy as np
import pytest

from causal_discovery.graph.edge import Edge
from causal_discovery.graph.endpoint import Endpoint
from causal_discovery.graph.general_graph import GeneralGraph
from causal_discovery.graph.graph_node import GraphNode
from causal_discovery.knowledge.background_knowledge import BackgroundKnowledge
from causal_discovery.fci.fci_algorithm import (
    fci_orient_bk,
    is_arrow_point_allowed,
    reorientAllWith,
    fci_remake,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_graph(n: int):
    """Create n nodes (X1..Xn) and return (graph, nodes_list)."""
    nodes = [GraphNode(f"X{i + 1}") for i in range(n)]
    return GeneralGraph(nodes), nodes


def make_circle_circle_graph(n: int):
    """Create a complete graph with all circle-circle edges."""
    graph, nodes = make_graph(n)
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(Edge(nodes[i], nodes[j], Endpoint.CIRCLE, Endpoint.CIRCLE))
    return graph, nodes


def get_endpoints(graph: GeneralGraph, node_a, node_b):
    """Return (endpoint_at_a, endpoint_at_b) for the edge between a and b."""
    edge = graph.get_edge(node_a, node_b)
    if edge is None:
        return None, None
    return (
        edge.get_proximal_endpoint(node_a),
        edge.get_proximal_endpoint(node_b),
    )


# ===========================================================================
# fci_orient_bk
# ===========================================================================

class TestFciOrientBk:
    """Tests for the fci_orient_bk function."""

    def test_no_knowledge_is_noop(self):
        graph, nodes = make_circle_circle_graph(3)
        original_graph = graph.graph.copy()
        fci_orient_bk(None, graph)
        np.testing.assert_array_equal(graph.graph, original_graph)

    def test_single_direction_forbidden_orients_circle_arrow(self):
        """Forbidden X1->X2 should orient as X1 <-o X2 (arrow at X1, circle at X2)."""
        graph, nodes = make_circle_circle_graph(3)
        x1, x2, x3 = nodes

        bk = BackgroundKnowledge()
        bk.add_forbidden_by_node(x1, x2)

        fci_orient_bk(bk, graph)

        ep_at_x1, ep_at_x2 = get_endpoints(graph, x1, x2)
        assert ep_at_x1 == Endpoint.ARROW, "X1 endpoint should be ARROW (forbidden to cause X2)"
        assert ep_at_x2 == Endpoint.CIRCLE, "X2 endpoint should be CIRCLE (could be tail or arrow)"

        # X1-X3 and X2-X3 should be unchanged (still circle-circle)
        assert get_endpoints(graph, x1, x3) == (Endpoint.CIRCLE, Endpoint.CIRCLE)
        assert get_endpoints(graph, x2, x3) == (Endpoint.CIRCLE, Endpoint.CIRCLE)

    def test_single_direction_forbidden_reverse(self):
        """Forbidden X2->X1 should orient as X1 o-> X2 (circle at X1, arrow at X2)."""
        graph, nodes = make_circle_circle_graph(2)
        x1, x2 = nodes

        bk = BackgroundKnowledge()
        bk.add_forbidden_by_node(x2, x1)

        fci_orient_bk(bk, graph)

        ep_at_x1, ep_at_x2 = get_endpoints(graph, x1, x2)
        assert ep_at_x1 == Endpoint.CIRCLE, "X1 endpoint should be CIRCLE"
        assert ep_at_x2 == Endpoint.ARROW, "X2 endpoint should be ARROW (forbidden to cause X1)"

    def test_both_directions_forbidden_orients_bidirected(self):
        """Forbidden in both directions should orient as X1 <-> X2 (bidirected)."""
        graph, nodes = make_circle_circle_graph(2)
        x1, x2 = nodes

        bk = BackgroundKnowledge()
        bk.add_forbidden_by_node(x1, x2)
        bk.add_forbidden_by_node(x2, x1)

        fci_orient_bk(bk, graph)

        assert graph.is_adjacent_to(x1, x2), "Edge should still exist (not removed)"
        ep_at_x1, ep_at_x2 = get_endpoints(graph, x1, x2)
        assert ep_at_x1 == Endpoint.ARROW, "X1 endpoint should be ARROW (bidirected)"
        assert ep_at_x2 == Endpoint.ARROW, "X2 endpoint should be ARROW (bidirected)"

    def test_both_directions_forbidden_does_not_remove_edge(self):
        """Both-directions-forbidden must keep the edge (latent common cause possible)."""
        graph, nodes = make_circle_circle_graph(3)
        x1, x2, x3 = nodes

        bk = BackgroundKnowledge()
        bk.add_forbidden_by_node(x1, x2)
        bk.add_forbidden_by_node(x2, x1)

        fci_orient_bk(bk, graph)

        assert graph.is_adjacent_to(x1, x2), "Edge must not be removed"
        assert graph.get_num_edges() == 3, "All 3 edges should still exist"

    def test_required_edge_orients_directed(self):
        """Required X1->X2 should orient as X1 --> X2 (tail at X1, arrow at X2)."""
        graph, nodes = make_circle_circle_graph(2)
        x1, x2 = nodes

        bk = BackgroundKnowledge()
        bk.add_required_by_node(x1, x2)

        fci_orient_bk(bk, graph)

        ep_at_x1, ep_at_x2 = get_endpoints(graph, x1, x2)
        assert ep_at_x1 == Endpoint.TAIL, "X1 endpoint should be TAIL (required cause)"
        assert ep_at_x2 == Endpoint.ARROW, "X2 endpoint should be ARROW (required effect)"

    def test_required_takes_priority_over_forbidden(self):
        """If X1->X2 is both required and forbidden, required wins."""
        graph, nodes = make_circle_circle_graph(2)
        x1, x2 = nodes

        bk = BackgroundKnowledge()
        bk.add_required_by_node(x1, x2)
        bk.add_forbidden_by_node(x1, x2)

        fci_orient_bk(bk, graph)

        ep_at_x1, ep_at_x2 = get_endpoints(graph, x1, x2)
        assert ep_at_x1 == Endpoint.TAIL, "Required should override forbidden"
        assert ep_at_x2 == Endpoint.ARROW

    def test_both_forbidden_but_one_required_keeps_required(self):
        """Both forbidden + one required: required wins, edge is not made bidirected."""
        graph, nodes = make_circle_circle_graph(2)
        x1, x2 = nodes

        bk = BackgroundKnowledge()
        bk.add_forbidden_by_node(x1, x2)
        bk.add_forbidden_by_node(x2, x1)
        bk.add_required_by_node(x1, x2)

        fci_orient_bk(bk, graph)

        ep_at_x1, ep_at_x2 = get_endpoints(graph, x1, x2)
        assert ep_at_x1 == Endpoint.TAIL, "Required X1->X2 should win"
        assert ep_at_x2 == Endpoint.ARROW

    def test_tier_based_forbidden_single_direction(self):
        """Higher-tier node cannot cause lower-tier: should orient as circle-arrow."""
        graph, nodes = make_circle_circle_graph(2)
        x1, x2 = nodes

        bk = BackgroundKnowledge()
        bk.add_node_to_tier(x1, 0)
        bk.add_node_to_tier(x2, 1)
        # tier(X2)=1 > tier(X1)=0, so is_forbidden(X2, X1) is True

        fci_orient_bk(bk, graph)

        ep_at_x1, ep_at_x2 = get_endpoints(graph, x1, x2)
        assert ep_at_x1 == Endpoint.CIRCLE, "X1 (lower tier) endpoint should be CIRCLE"
        assert ep_at_x2 == Endpoint.ARROW, "X2 (higher tier, forbidden to cause X1) should be ARROW"

    def test_same_tier_no_forbidden(self):
        """Same-tier nodes should NOT be forbidden (uses > not >=)."""
        graph, nodes = make_circle_circle_graph(2)
        x1, x2 = nodes

        bk = BackgroundKnowledge()
        bk.add_node_to_tier(x1, 1)
        bk.add_node_to_tier(x2, 1)

        fci_orient_bk(bk, graph)

        # Should remain circle-circle (same tier, no constraint)
        ep_at_x1, ep_at_x2 = get_endpoints(graph, x1, x2)
        assert ep_at_x1 == Endpoint.CIRCLE
        assert ep_at_x2 == Endpoint.CIRCLE

    def test_multiple_edges_oriented_independently(self):
        """Each edge should be oriented based on its own forbidden/required constraints."""
        graph, nodes = make_circle_circle_graph(3)
        x1, x2, x3 = nodes

        bk = BackgroundKnowledge()
        bk.add_forbidden_by_node(x1, x2)  # X1->X2 forbidden
        bk.add_required_by_node(x2, x3)   # X2->X3 required

        fci_orient_bk(bk, graph)

        # X1-X2: arrow at X1, circle at X2
        ep_at_x1, ep_at_x2 = get_endpoints(graph, x1, x2)
        assert ep_at_x1 == Endpoint.ARROW
        assert ep_at_x2 == Endpoint.CIRCLE

        # X2-X3: tail at X2, arrow at X3
        ep_at_x2, ep_at_x3 = get_endpoints(graph, x2, x3)
        assert ep_at_x2 == Endpoint.TAIL
        assert ep_at_x3 == Endpoint.ARROW

        # X1-X3: unchanged (circle-circle)
        assert get_endpoints(graph, x1, x3) == (Endpoint.CIRCLE, Endpoint.CIRCLE)

    def test_pattern_based_forbidden(self):
        """Pattern-based forbidden rules should work the same as node-based."""
        graph, nodes = make_circle_circle_graph(2)
        x1, x2 = nodes

        bk = BackgroundKnowledge()
        bk.add_forbidden_by_pattern("X1", "X2")

        fci_orient_bk(bk, graph)

        ep_at_x1, ep_at_x2 = get_endpoints(graph, x1, x2)
        assert ep_at_x1 == Endpoint.ARROW
        assert ep_at_x2 == Endpoint.CIRCLE


# ===========================================================================
# is_arrow_point_allowed
# ===========================================================================

class TestIsArrowPointAllowed:
    """Tests for the is_arrow_point_allowed guard function."""

    def test_already_arrow_returns_true(self):
        """If endpoint is already ARROW, always allowed."""
        graph, nodes = make_graph(2)
        x1, x2 = nodes
        graph.add_edge(Edge(x1, x2, Endpoint.CIRCLE, Endpoint.ARROW))

        assert is_arrow_point_allowed(x1, x2, graph, None) is True

    def test_already_tail_returns_false(self):
        """If endpoint is already TAIL, arrow is not allowed."""
        graph, nodes = make_graph(2)
        x1, x2 = nodes
        graph.add_directed_edge(x1, x2)  # tail at X1, arrow at X2

        # Endpoint at X1 is TAIL — can't change to arrow
        assert is_arrow_point_allowed(x2, x1, graph, None) is False

    def test_circle_with_no_knowledge_returns_true(self):
        """CIRCLE endpoint with no knowledge should allow arrow."""
        graph, nodes = make_circle_circle_graph(2)
        x1, x2 = nodes

        assert is_arrow_point_allowed(x1, x2, graph, None) is True

    def test_forbidden_returns_false(self):
        """If X->Y is forbidden, arrow at Y is not allowed."""
        graph, nodes = make_circle_circle_graph(2)
        x1, x2 = nodes

        bk = BackgroundKnowledge()
        bk.add_forbidden_by_node(x1, x2)

        assert is_arrow_point_allowed(x1, x2, graph, bk) is False

    def test_forbidden_reverse_allows_forward(self):
        """If X2->X1 is forbidden, X1->X2 (arrow at X2) should still be allowed."""
        graph, nodes = make_circle_circle_graph(2)
        x1, x2 = nodes

        bk = BackgroundKnowledge()
        bk.add_forbidden_by_node(x2, x1)

        assert is_arrow_point_allowed(x1, x2, graph, bk) is True

    def test_required_reverse_returns_false(self):
        """If Y->X is required, arrow at Y from X is not allowed (would conflict with tail at Y)."""
        graph, nodes = make_circle_circle_graph(2)
        x1, x2 = nodes

        bk = BackgroundKnowledge()
        bk.add_required_by_node(x2, x1)
        # Required X2->X1 means tail at X2. So arrow at X2 (from X1) is not allowed.

        assert is_arrow_point_allowed(x1, x2, graph, bk) is False

    def test_tier_forbidden_returns_false(self):
        """Higher-tier node pointing to lower-tier should be forbidden."""
        graph, nodes = make_circle_circle_graph(2)
        x1, x2 = nodes

        bk = BackgroundKnowledge()
        bk.add_node_to_tier(x1, 0)
        bk.add_node_to_tier(x2, 1)
        # is_forbidden(X2, X1) = True because tier(X2) > tier(X1)

        # Arrow at X1 from X2 should be forbidden
        assert is_arrow_point_allowed(x2, x1, graph, bk) is False
        # Arrow at X2 from X1 should be allowed
        assert is_arrow_point_allowed(x1, x2, graph, bk) is True


# ===========================================================================
# reorientAllWith
# ===========================================================================

class TestReorientAllWith:
    """Tests for the reorientAllWith function."""

    def test_reorient_to_circles(self):
        """All edges should become circle-circle."""
        graph, nodes = make_graph(3)
        x1, x2, x3 = nodes
        graph.add_directed_edge(x1, x2)
        graph.add_edge(Edge(x1, x3, Endpoint.ARROW, Endpoint.ARROW))

        reorientAllWith(graph, Endpoint.CIRCLE, knowledge=None)

        assert get_endpoints(graph, x1, x2) == (Endpoint.CIRCLE, Endpoint.CIRCLE)
        assert get_endpoints(graph, x1, x3) == (Endpoint.CIRCLE, Endpoint.CIRCLE)

    def test_reorient_preserves_required_edges(self):
        """Required edges should stay directed after reorientAllWith."""
        graph, nodes = make_graph(3)
        x1, x2, x3 = nodes
        graph.add_directed_edge(x1, x2)
        graph.add_directed_edge(x2, x3)

        bk = BackgroundKnowledge()
        bk.add_required_by_node(x1, x2)

        reorientAllWith(graph, Endpoint.CIRCLE, knowledge=bk)

        # X1->X2 is required: should remain directed
        ep_at_x1, ep_at_x2 = get_endpoints(graph, x1, x2)
        assert ep_at_x1 == Endpoint.TAIL
        assert ep_at_x2 == Endpoint.ARROW

        # X2->X3 is not required: should become circle-circle
        assert get_endpoints(graph, x2, x3) == (Endpoint.CIRCLE, Endpoint.CIRCLE)

    def test_reorient_preserves_required_reverse(self):
        """Required edge in reverse direction should also be preserved."""
        graph, nodes = make_graph(2)
        x1, x2 = nodes
        graph.add_directed_edge(x1, x2)

        bk = BackgroundKnowledge()
        bk.add_required_by_node(x2, x1)  # X2->X1 required

        reorientAllWith(graph, Endpoint.CIRCLE, knowledge=bk)

        ep_at_x1, ep_at_x2 = get_endpoints(graph, x1, x2)
        # Should be oriented as X2->X1 (tail at X2, arrow at X1)
        assert ep_at_x2 == Endpoint.TAIL
        assert ep_at_x1 == Endpoint.ARROW


# ===========================================================================
# Integration: fci_remake with background knowledge
# ===========================================================================

class TestFciWithBackgroundKnowledge:
    """Integration tests for the full FCI algorithm with background knowledge."""

    @pytest.fixture
    def correlated_data(self):
        """Generate data from a known 3-variable chain: X1 -> X2 -> X3."""
        np.random.seed(42)
        n = 500
        x1 = np.random.normal(0, 1, n)
        x2 = 0.8 * x1 + np.random.normal(0, 0.5, n)
        x3 = 0.8 * x2 + np.random.normal(0, 0.5, n)
        return np.column_stack([x1, x2, x3])

    def test_forbidden_edge_not_fully_directed(self, correlated_data):
        """A forbidden edge should never appear as fully directed in the forbidden direction."""
        nodes = [GraphNode(f"X{i + 1}") for i in range(3)]

        bk = BackgroundKnowledge()
        bk.add_forbidden_by_node(nodes[2], nodes[0])  # X3->X1 forbidden

        graph, _ = fci_remake(
            dataset=correlated_data,
            independence_test_method="fisherz",
            alpha=0.05,
            verbose=False,
            show_progress=False,
            background_knowledge=bk,
        )

        # X3->X1 must not be a directed edge (tail at X3, arrow at X1)
        assert not graph.is_directed_from_to(
            graph.get_node("X3"), graph.get_node("X1")
        ), "Forbidden edge X3->X1 should not appear as directed"

    def test_required_edge_is_directed(self, correlated_data):
        """A required edge should appear as directed in the output."""
        nodes = [GraphNode(f"X{i + 1}") for i in range(3)]

        bk = BackgroundKnowledge()
        bk.add_required_by_node(nodes[0], nodes[1])  # X1->X2 required

        graph, _ = fci_remake(
            dataset=correlated_data,
            independence_test_method="fisherz",
            alpha=0.05,
            verbose=False,
            show_progress=False,
            background_knowledge=bk,
        )

        x1 = graph.get_node("X1")
        x2 = graph.get_node("X2")
        if graph.is_adjacent_to(x1, x2):
            assert graph.is_directed_from_to(x1, x2), (
                "Required edge X1->X2 should be directed"
            )

    def test_tier_ordering_constrains_orientation(self, correlated_data):
        """Tier ordering should prevent higher-tier nodes from causing lower-tier nodes."""
        nodes = [GraphNode(f"X{i + 1}") for i in range(3)]

        bk = BackgroundKnowledge()
        bk.add_node_to_tier(nodes[0], 0)  # X1 in tier 0 (earliest)
        bk.add_node_to_tier(nodes[1], 1)  # X2 in tier 1
        bk.add_node_to_tier(nodes[2], 2)  # X3 in tier 2 (latest)

        graph, _ = fci_remake(
            dataset=correlated_data,
            independence_test_method="fisherz",
            alpha=0.05,
            verbose=False,
            show_progress=False,
            background_knowledge=bk,
        )

        x1 = graph.get_node("X1")
        x2 = graph.get_node("X2")
        x3 = graph.get_node("X3")

        # Higher-tier nodes should never have a directed edge to lower-tier nodes
        assert not graph.is_directed_from_to(x2, x1), "X2 (tier 1) should not cause X1 (tier 0)"
        assert not graph.is_directed_from_to(x3, x1), "X3 (tier 2) should not cause X1 (tier 0)"
        assert not graph.is_directed_from_to(x3, x2), "X3 (tier 2) should not cause X2 (tier 1)"

    def test_no_knowledge_matches_unconstrained(self, correlated_data):
        """FCI with empty BackgroundKnowledge should match FCI with None."""
        graph_none, _ = fci_remake(
            dataset=correlated_data,
            independence_test_method="fisherz",
            alpha=0.05,
            verbose=False,
            show_progress=False,
            background_knowledge=None,
        )
        graph_empty, _ = fci_remake(
            dataset=correlated_data,
            independence_test_method="fisherz",
            alpha=0.05,
            verbose=False,
            show_progress=False,
            background_knowledge=BackgroundKnowledge(),
        )

        np.testing.assert_array_equal(
            graph_none.graph,
            graph_empty.graph,
            err_msg="Empty BK should produce identical results to no BK",
        )

    def test_both_forbidden_preserves_edge_in_full_pipeline(self):
        """Both-directions-forbidden should keep the edge through the full FCI pipeline."""
        np.random.seed(123)
        n = 500
        # Create strongly correlated variables (shared latent cause)
        latent = np.random.normal(0, 1, n)
        x1 = 0.9 * latent + np.random.normal(0, 0.3, n)
        x2 = 0.9 * latent + np.random.normal(0, 0.3, n)
        x3 = np.random.normal(0, 1, n)
        data = np.column_stack([x1, x2, x3])

        nodes = [GraphNode(f"X{i + 1}") for i in range(3)]
        bk = BackgroundKnowledge()
        bk.add_forbidden_by_node(nodes[0], nodes[1])
        bk.add_forbidden_by_node(nodes[1], nodes[0])

        graph, _ = fci_remake(
            dataset=data,
            independence_test_method="fisherz",
            alpha=0.05,
            verbose=False,
            show_progress=False,
            background_knowledge=bk,
        )

        x1 = graph.get_node("X1")
        x2 = graph.get_node("X2")

        if graph.is_adjacent_to(x1, x2):
            # If edge exists, it should not be directed in either direction
            assert not graph.is_directed_from_to(x1, x2), "X1->X2 should not be directed (both forbidden)"
            assert not graph.is_directed_from_to(x2, x1), "X2->X1 should not be directed (both forbidden)"
