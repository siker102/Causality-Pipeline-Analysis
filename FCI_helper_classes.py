import streamlit as st
from causallearn.graph.Edge import Edge
from causallearn.graph import Endpoint
from typing import List, Tuple
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from FCI_causallearn_remake_with_background_controls import fci_remake
from causallearn.search.ConstraintBased.PC import pc
from causallearn.graph.GraphClass import CausalGraph

def calculate_accuracy_of_graphs(g: GeneralGraph, true_graph: GeneralGraph) -> Tuple[float, float, float]:
    """Calculates the accuracy of the transformation from DAG to PAG"""
    # Extract edges from both graphs
    true_edges: list[Edge] = true_graph.get_graph_edges()
    g_edges: list[Edge] = g.get_graph_edges()
    
    total_true = len(true_edges)
    total_g = len(g_edges)
    
    correct_count = 0
    partial_count = 0
    false_count_in_true = 0
    false_count_extra = 0
    missing_edges_count = 0  # New counter for edges in true_graph not found in g
    
    # Create a set of node pairs from true_edges for quick lookup
    true_node_pairs = set((edge.get_node1(), edge.get_node2()) for edge in true_edges)
    
    # Check each edge in the true graph against the inferred graph 'g'
    for true_edge in true_edges:
        x_true, y_true = true_edge.get_node1(), true_edge.get_node2()
        found = False
        for g_edge in g_edges:
            x_g, y_g = g_edge.get_node1(), g_edge.get_node2()
            if x_g == x_true and y_g == y_true:
                # Compare endpoints
                true_ep = (true_edge.get_endpoint1(), true_edge.get_endpoint2())
                g_ep = (g_edge.get_endpoint1(), g_edge.get_endpoint2())
                
                #if equal its true
                if true_ep == g_ep:
                    correct_count += 1
                else:
                    # Check if both endpoints in the inferred edge are circles
                    if g_ep[0].value == 2 and g_ep[1].value == 2:
                        partial_count += 1
                    else:
                        # Check for partial correctness (one correct, one circle)
                        correct = 0
                        circle = 0
                        for i in range(2):
                            #check for full correctness
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
        # Count missing edges from true_graph not present in g
        if not found:
            missing_edges_count += 1
    
    # Count edges in 'g' that are not in the true graph
    for g_edge in g_edges:
        x_g, y_g = g_edge.get_node1(), g_edge.get_node2()
        if (x_g, y_g) not in true_node_pairs:
            false_count_extra += 1
    
    # Calculating the  percentages
    correct_percentage = (correct_count / total_true * 100) if total_true != 0 else 1.0
    falsely_percentage = ((false_count_in_true + false_count_extra + missing_edges_count) / total_g * 100) if total_g != 0 else 0.0
    partial_percentage = (partial_count / total_true * 100) if total_true != 0 else 0.0
    
    return (correct_percentage, falsely_percentage, partial_percentage)

def hash_background_knowledge(bk: BackgroundKnowledge) -> int:
    """Hash BackgroundKnowledge by converting all nested sets to sorted tuples."""
    # Convert rules with sets to tuples
    def _convert_sets_to_tuples(obj):
        if isinstance(obj, set):
            return tuple(sorted(obj))  # Convert sets to sorted tuples
        elif isinstance(obj, list):
            return tuple(_convert_sets_to_tuples(x) for x in obj)  # Recurse
        return obj

    # Extract rules and patterns, handling nested sets
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

    # Tier mappings (already index-based)
    tier_map = tuple(sorted(bk.tier_map.items()))  # Keys are indices, values are tiers
    tier_values = tuple(sorted(bk.tier_value_map.items()))

    # Combine into a hashable structure
    hashable = (
        forbidden_rules,
        required_rules,
        forbidden_patterns,
        required_patterns,
        tier_map,
        tier_values
    )
    return hash(hashable)

#@st.cache_resource(show_spinner=False, hash_funcs={BackgroundKnowledge: hash_background_knowledge})
def run_FCI_analysis(dataframe, test_type, alpha, bk: BackgroundKnowledge) -> Tuple[GeneralGraph, List[Edge]]:
    """FCI + Caching"""
    data = dataframe.to_numpy()
    column_names = dataframe.columns.tolist()
    #print("FCI Analysis started... with background knowledge:", _bk.forbidden_rules_specs, _bk.required_rules_specs, _bk.tier_map, _bk.tier_value_map, _bk.forbidden_pattern_rules_specs ,_bk.required_pattern_rules_specs)
    
    g: GeneralGraph
    edges: List[Edge]
    
    print(bk.forbidden_pattern_rules_specs)
    print(bk.required_pattern_rules_specs)
    print(bk.forbidden_rules_specs)
    print(bk.required_rules_specs)
    print(bk.tier_map)
    print(bk.tier_value_map)
    # g, edges = fci(data, test_type, alpha=alpha, background_knowledge=_bk)
    g, edges = fci_remake(data, test_type, alpha=alpha, background_knowledge=bk)

    for rule in bk.required_rules_specs:
        print("Required rules specs:", rule[0].get_name(), "->", rule[1].get_name())
    print("FCI Analysis completed.")

    # Assign column names to graph nodes only if FCI
   
    for i, node in enumerate(g.get_nodes()):
        if node is not None:
            node.set_name(column_names[i])
            # st.write(f"Assigned {column_names[i]} to node {node}")
    return g, edges