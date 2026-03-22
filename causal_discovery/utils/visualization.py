from __future__ import annotations

from typing import List

import pydot

from causal_discovery.graph.edge import Edge
from causal_discovery.graph.endpoint import Endpoint
from causal_discovery.graph.general_graph import GeneralGraph
from causal_discovery.graph.node import NodeType


def to_pydot(G: GeneralGraph, edges: List[Edge] | None = None, labels: List[str] | None = None,
             title: str = "", dpi: float = 200):
    """Convert a GeneralGraph to a pydot Dot object for visualization."""

    nodes = G.get_nodes()
    if labels is not None:
        assert len(labels) == len(nodes)

    pydot_g = pydot.Dot(title, graph_type="digraph", fontsize=18)
    pydot_g.obj_dict["attributes"]["dpi"] = dpi
    for i, node in enumerate(nodes):
        node_name = labels[i] if labels is not None else node.get_name()
        if node.get_node_type() == NodeType.LATENT:
            pydot_g.add_node(pydot.Node(i, label=node_name, shape='square'))
        else:
            pydot_g.add_node(pydot.Node(i, label=node_name))

    def get_g_arrow_type(endpoint):
        if endpoint == Endpoint.TAIL:
            return 'none'
        elif endpoint == Endpoint.ARROW:
            return 'normal'
        elif endpoint == Endpoint.CIRCLE:
            return 'odot'
        elif endpoint == Endpoint.NULL:
            return 'none'
        elif endpoint == Endpoint.STAR:
            return 'diamond'
        elif endpoint == Endpoint.TAIL_AND_ARROW:
            return 'normal'
        elif endpoint == Endpoint.ARROW_AND_ARROW:
            return 'normal'
        else:
            raise NotImplementedError(f"Unknown endpoint type: {endpoint}")

    if edges is None:
        edges = G.get_graph_edges()

    for edge in edges:
        node1 = edge.get_node1()
        node2 = edge.get_node2()
        node1_id = nodes.index(node1)
        node2_id = nodes.index(node2)
        dot_edge = pydot.Edge(node1_id, node2_id, dir='both',
                              arrowtail=get_g_arrow_type(edge.get_endpoint1()),
                              arrowhead=get_g_arrow_type(edge.get_endpoint2()))

        if Edge.Property.dd in edge.properties:
            dot_edge.obj_dict["attributes"]["color"] = "green3"

        if Edge.Property.nl in edge.properties:
            dot_edge.obj_dict["attributes"]["penwidth"] = 2.0

        pydot_g.add_edge(dot_edge)

    return pydot_g
