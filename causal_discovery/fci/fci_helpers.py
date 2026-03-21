from __future__ import annotations

from typing import List, Tuple

from causal_discovery.graph.edge import Edge
from causal_discovery.graph.endpoint import Endpoint
from causal_discovery.graph.general_graph import GeneralGraph
from causal_discovery.knowledge.background_knowledge import BackgroundKnowledge
from causal_discovery.fci.fci_algorithm import existsSemiDirectedPath, fci_remake


def visibleEdgeHelperVisit(graph: GeneralGraph, node_c, node_a, node_b, path) -> bool:
    if node_a in path:
        return False

    path.append(node_a)

    if node_a == node_b:
        return True

    for node_D in graph.get_nodes_into(node_a, Endpoint.ARROW):
        if graph.is_parent_of(node_D, node_c):
            return True

        if not graph.is_def_collider(node_D, node_c, node_a):
            continue
        elif not graph.is_parent_of(node_c, node_b):
            continue

        if visibleEdgeHelperVisit(graph, node_D, node_c, node_b, path):
            return True

    path.pop()
    return False


def visibleEdgeHelper(node_A, node_B, graph: GeneralGraph) -> bool:
    path = [node_A]

    for node_C in graph.get_nodes_into(node_A, Endpoint.ARROW):
        if graph.is_parent_of(node_C, node_A):
            return True

        if visibleEdgeHelperVisit(graph, node_C, node_A, node_B, path):
            return True

    return False


def defVisible(edge: Edge, graph: GeneralGraph) -> bool:
    if graph.contains_edge(edge):
        if edge.get_endpoint1() == Endpoint.TAIL:
            node_A = edge.get_node1()
            node_B = edge.get_node2()
        else:
            node_A = edge.get_node2()
            node_B = edge.get_node1()

        for node_C in graph.get_adjacent_nodes(node_A):
            if node_C != node_B and not graph.is_adjacent_to(node_C, node_B):
                e = graph.get_edge(node_C, node_A)

                if e.get_proximal_endpoint(node_A) == Endpoint.ARROW:
                    return True

        return visibleEdgeHelper(node_A, node_B, graph)
    else:
        raise Exception("Given edge is not in the graph.")


def get_color_edges(graph: GeneralGraph) -> List[Edge]:
    edges = graph.get_graph_edges()
    for edge in edges:
        if (edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW) or \
                (edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.TAIL):
            if edge.get_endpoint1() == Endpoint.TAIL:
                node_x = edge.get_node1()
                node_y = edge.get_node2()
            else:
                node_x = edge.get_node2()
                node_y = edge.get_node1()

            graph.remove_edge(edge)

            if not existsSemiDirectedPath(node_x, node_y, graph):
                edge.properties.append(Edge.Property.dd)  # green
            else:
                edge.properties.append(Edge.Property.pd)

            graph.add_edge(edge)

            if defVisible(edge, graph):
                edge.properties.append(Edge.Property.nl)  # bold
                print(edge)
            else:
                edge.properties.append(Edge.Property.pl)
    return edges


def calculate_accuracy_of_graphs(g: GeneralGraph, true_graph: GeneralGraph) -> dict:
    """Compare output graph (PAG or CPDAG) to true DAG and return accuracy counts.

    Returns a dict with keys:
        correct       - edge found with exactly matching endpoints
        partial       - edge found but at least one endpoint is uncertain (circle) or undirected (TAIL-TAIL in CPDAG)
        wrong_orient  - edge found but orientation is specifically wrong
        missing       - edge in true graph not found in output
        spurious      - edge in output not in true graph
        total_true    - total edges in true graph
        total_output  - total edges in output graph
    """
    true_edges: list[Edge] = true_graph.get_graph_edges()
    g_edges: list[Edge] = g.get_graph_edges()

    # Build lookup: frozenset of node pair → output edge
    g_edge_by_pair: dict = {}
    for edge in g_edges:
        key = frozenset([edge.get_node1(), edge.get_node2()])
        g_edge_by_pair[key] = edge

    true_pairs: set = set()
    correct = 0
    partial = 0
    wrong_orient = 0
    missing = 0

    for true_edge in true_edges:
        n1, n2 = true_edge.get_node1(), true_edge.get_node2()
        true_pairs.add(frozenset([n1, n2]))
        key = frozenset([n1, n2])

        if key not in g_edge_by_pair:
            missing += 1
            continue

        g_edge = g_edge_by_pair[key]
        # Align g_edge endpoints to true_edge node ordering
        if g_edge.get_node1() == n1:
            g_ep1, g_ep2 = g_edge.get_endpoint1(), g_edge.get_endpoint2()
        else:
            g_ep1, g_ep2 = g_edge.get_endpoint2(), g_edge.get_endpoint1()

        true_ep1, true_ep2 = true_edge.get_endpoint1(), true_edge.get_endpoint2()

        if g_ep1 == true_ep1 and g_ep2 == true_ep2:
            correct += 1
        elif (g_ep1 == Endpoint.CIRCLE or g_ep2 == Endpoint.CIRCLE or
              (g_ep1 == Endpoint.TAIL and g_ep2 == Endpoint.TAIL)):
            # Uncertain (FCI circles) or undirected (CPDAG TAIL-TAIL)
            partial += 1
        else:
            wrong_orient += 1

    spurious = sum(
        1 for edge in g_edges
        if frozenset([edge.get_node1(), edge.get_node2()]) not in true_pairs
    )

    return {
        'correct': correct,
        'partial': partial,
        'wrong_orient': wrong_orient,
        'missing': missing,
        'spurious': spurious,
        'total_true': len(true_edges),
        'total_output': len(g_edges),
    }


def hash_background_knowledge(bk: BackgroundKnowledge) -> int:
    def _convert_sets_to_tuples(obj):
        if isinstance(obj, set):
            return tuple(sorted(obj))
        elif isinstance(obj, list):
            return tuple(_convert_sets_to_tuples(x) for x in obj)
        return obj

    forbidden_rules = tuple(
        tuple(_convert_sets_to_tuples(rule) for rule in bk.forbidden_rules_specs)
    )
    required_rules = tuple(
        tuple(_convert_sets_to_tuples(rule) for rule in bk.required_rules_specs)
    )
    forbidden_patterns = tuple(
        tuple(_convert_sets_to_tuples(pattern) for pattern in bk.forbidden_pattern_rules_specs)
    )
    required_patterns = tuple(
        tuple(_convert_sets_to_tuples(pattern) for pattern in bk.required_pattern_rules_specs)
    )
    tier_map = tuple(sorted(bk.tier_map.items()))
    tier_values = tuple(sorted(bk.tier_value_map.items()))

    hashable = (
        forbidden_rules,
        required_rules,
        forbidden_patterns,
        required_patterns,
        tier_map,
        tier_values
    )
    return hash(hashable)


def run_FCI_analysis(dataframe, test_type, alpha, bk: BackgroundKnowledge) -> Tuple[GeneralGraph, List[Edge]]:
    """FCI + Caching"""
    data = dataframe.to_numpy()
    column_names = dataframe.columns.tolist()

    print(bk.forbidden_pattern_rules_specs)
    print(bk.required_pattern_rules_specs)
    print(bk.forbidden_rules_specs)
    print(bk.required_rules_specs)
    print(bk.tier_map)
    print(bk.tier_value_map)
    g, edges = fci_remake(data, test_type, alpha=alpha, background_knowledge=bk)

    for rule in bk.required_rules_specs:
        print("Required rules specs:", rule[0].get_name(), "->", rule[1].get_name())
    print("FCI Analysis completed.")

    for i, node in enumerate(g.get_nodes()):
        if node is not None:
            node.set_name(column_names[i])
    return g, edges
