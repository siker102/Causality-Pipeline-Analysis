import streamlit as st
from causallearn.graph.Edge import Edge
from typing import List, Tuple
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from FCI_causallearn_remake_with_background_controls import fci_remake

@st.cache_resource(show_spinner=False)
def run_fci_analysis(dataframe, test_type, alpha, _bk: BackgroundKnowledge) -> Tuple[GeneralGraph, List[Edge]]:
    """FCI + Caching"""
    data = dataframe.to_numpy()
    column_names = dataframe.columns.tolist()
    
    #print("FCI Analysis started... with background knowledge:", _bk.forbidden_rules_specs, _bk.required_rules_specs, _bk.tier_map, _bk.tier_value_map, _bk.forbidden_pattern_rules_specs ,_bk.required_pattern_rules_specs)
    
    g: GeneralGraph
    edges: List[Edge]
    
    print(_bk.forbidden_pattern_rules_specs)
    print(_bk.required_pattern_rules_specs)
    print(_bk.forbidden_rules_specs)
    print(_bk.required_rules_specs)
    print(_bk.tier_map)
    print(_bk.tier_value_map)
    # g, edges = fci(data, test_type, alpha=alpha, background_knowledge=_bk)
    g, edges = fci_remake(data, test_type, alpha=alpha, background_knowledge=_bk)
    edges1 = g.get_graph_edges()

    # Debugging
    #for edge in edges1:
    #    if _bk.is_forbidden(edge.get_node1(), edge.get_node2()):
    #        #graph.remove_edge(edge)
    #        #graph.add_directed_edge(edge.get_node2(), edge.get_node1())
    #        print("Orienting edge (Knowledge): " + str(g.get_edge(edge.get_node2(), edge.get_node1())))
    #    elif _bk.is_forbidden(edge.get_node2(), edge.get_node1()):
    #        #graph.remove_edge(edge)
    #        #graph.add_directed_edge(edge.get_node1(), edge.get_node2())
    #        print("Orienting edge (Knowledge): " + str(g.get_edge(edge.get_node2(), edge.get_node1())))
    #    elif _bk.is_required(edge.get_node1(), edge.get_node2()):
    #        #graph.remove_edge(edge)
    #        #graph.add_directed_edge(edge.get_node1(), edge.get_node2())
    #        print("Orienting edge (Knowledge): " + str(g.get_edge(edge.get_node2(), edge.get_node1())))
    #    elif _bk.is_required(edge.get_node2(), edge.get_node1()):
    #        #graph.remove_edge(edge)
    #        #graph.add_directed_edge(edge.get_node2(), edge.get_node1())
    #        print("Orienting edge (Knowledge): " + str(g.get_edge(edge.get_node2(), edge.get_node1())))
    #    else:
    #        print("Not orienting edge (Knowledge): " + str(g.get_edge(edge.get_node2(), edge.get_node1())))
    for rule in _bk.required_rules_specs:
        print("Required rules specs:", rule[0].get_name(), "->", rule[1].get_name())
    print("FCI Analysis completed.")

    # Assign column names to graph nodes
    for i, node in enumerate(g.get_nodes()):
        if node is not None:
            node.set_name(column_names[i])
            # st.write(f"Assigned {column_names[i]} to node {node}")
    return g, edges
