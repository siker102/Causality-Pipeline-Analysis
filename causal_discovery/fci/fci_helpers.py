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


def calculate_accuracy_of_graphs(g: GeneralGraph, true_graph: GeneralGraph) -> Tuple[float, float, float]:
    """Calculates the accuracy of the transformation from DAG to PAG"""
    true_edges: list[Edge] = true_graph.get_graph_edges()
    g_edges: list[Edge] = g.get_graph_edges()

    total_true = len(true_edges)
    total_g = len(g_edges)

    correct_count = 0
    partial_count = 0
    false_count_in_true = 0
    false_count_extra = 0
    missing_edges_count = 0

    true_node_pairs = set((edge.get_node1(), edge.get_node2()) for edge in true_edges)

    for true_edge in true_edges:
        x_true, y_true = true_edge.get_node1(), true_edge.get_node2()
        found = False
        for g_edge in g_edges:
            x_g, y_g = g_edge.get_node1(), g_edge.get_node2()
            if x_g == x_true and y_g == y_true:
                true_ep = (true_edge.get_endpoint1(), true_edge.get_endpoint2())
                g_ep = (g_edge.get_endpoint1(), g_edge.get_endpoint2())

                if true_ep == g_ep:
                    correct_count += 1
                else:
                    if g_ep[0].value == 2 and g_ep[1].value == 2:
                        partial_count += 1
                    else:
                        correct = 0
                        circle = 0
                        for i in range(2):
                            if g_ep[i] == true_ep[i]:
                                correct += 1
                            elif g_ep[i].value == 2:
                                circle += 1
                        if correct == 1 and circle == 1:
                            partial_count += 1
                        else:
                            false_count_in_true += 1
                found = True
                break
        if not found:
            missing_edges_count += 1

    for g_edge in g_edges:
        x_g, y_g = g_edge.get_node1(), g_edge.get_node2()
        if (x_g, y_g) not in true_node_pairs:
            false_count_extra += 1

    correct_percentage = (correct_count / total_true * 100) if total_true != 0 else 1.0
    falsely_percentage = ((false_count_in_true + false_count_extra + missing_edges_count) / total_g * 100) if total_g != 0 else 0.0
    partial_percentage = (partial_count / total_true * 100) if total_true != 0 else 0.0

    return (correct_percentage, falsely_percentage, partial_percentage)


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
