from causal_discovery.graph.causal_graph import CausalGraph
from causal_discovery.knowledge.background_knowledge import BackgroundKnowledge


def orient_by_background_knowledge(cg: CausalGraph, background_knowledge: BackgroundKnowledge):
    for edge in cg.G.get_graph_edges():
        if cg.G.is_undirected_from_to(edge.get_node1(), edge.get_node2()):
            if background_knowledge.is_forbidden(edge.get_node2(), edge.get_node1()):
                cg.G.remove_edge(edge)
                cg.G.add_directed_edge(edge.get_node1(), edge.get_node2())
            elif background_knowledge.is_forbidden(edge.get_node1(), edge.get_node2()):
                cg.G.remove_edge(edge)
                cg.G.add_directed_edge(edge.get_node2(), edge.get_node1())
            elif background_knowledge.is_required(edge.get_node2(), edge.get_node1()):
                cg.G.remove_edge(edge)
                cg.G.add_directed_edge(edge.get_node2(), edge.get_node1())
            elif background_knowledge.is_required(edge.get_node1(), edge.get_node2()):
                cg.G.remove_edge(edge)
                cg.G.add_directed_edge(edge.get_node1(), edge.get_node2())
