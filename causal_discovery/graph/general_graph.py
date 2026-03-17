#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Dict, Tuple

import numpy as np
from numpy import ndarray
from causal_discovery.graph.edge import Edge
from causal_discovery.graph.endpoint import Endpoint
from causal_discovery.graph.node import Node


class GeneralGraph:

    def __init__(self, nodes: List[Node]):
        self.nodes: List[Node] = nodes
        self.num_vars: int = len(nodes)

        node_map: Dict[Node, int] = {}

        for i in range(self.num_vars):
            node = nodes[i]
            node_map[node] = i

        self.node_map: Dict[Node, int] = node_map

        self.graph: ndarray = np.zeros((self.num_vars, self.num_vars), np.dtype(int))
        self.dpath: ndarray = np.zeros((self.num_vars, self.num_vars), np.dtype(int))

        self.reconstitute_dpath([])

        self.ambiguous_triples: List[Tuple[Node, Node, Node]] = []
        self.underline_triples: List[Tuple[Node, Node, Node]] = []
        self.dotted_underline_triples: List[Tuple[Node, Node, Node]] = []

        self.attributes = {}
        self.pattern = False
        self.pag = False

    ### Helper Functions ###

    def adjust_dpath(self, i: int, j: int):
        dpath = self.dpath
        dpath[j, i] = 1

        for k in range(self.num_vars):
            if dpath[i, k] == 1:
                dpath[j, k] = 1

            if dpath[k, j] == 1:
                dpath[k, i] = 1

        self.dpath = dpath

    def reconstitute_dpath(self, edges: List[Edge]):
        self.dpath = np.zeros((self.num_vars, self.num_vars), np.dtype(int))
        for i in range(self.num_vars):
            self.adjust_dpath(i, i)

        while len(edges) > 0:
            edge = edges.pop()
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            i = self.node_map[node1]
            j = self.node_map[node2]
            if self.is_parent_of(node1, node2):
                self.adjust_dpath(i, j)
            elif self.is_parent_of(node2, node1):
                self.adjust_dpath(j, i)


    def collect_ancestors(self, node: Node, ancestors: List[Node]):
        if node in ancestors:
            return

        ancestors.append(node)
        parents = self.get_parents(node)

        if parents:
            for parent in parents:
                self.collect_ancestors(parent, ancestors)

    ### Public Functions ###

    # Adds a directed edge --> to the graph.
    def add_directed_edge(self, node1: Node, node2: Node):
        i = self.node_map[node1]
        j = self.node_map[node2]
        self.graph[j, i] = 1
        self.graph[i, j] = -1

        self.adjust_dpath(i, j)

    def add_edge(self, edge: Edge):
        node1 = edge.get_node1()
        node2 = edge.get_node2()
        endpoint1 = str(edge.get_endpoint1())
        endpoint2 = str(edge.get_endpoint2())

        i = self.node_map[node1]
        j = self.node_map[node2]

        e1 = self.graph[i, j]
        e2 = self.graph[j, i]

        bidirected = e2 == 1 and e1 == 1
        existing_edge = not bidirected and (e2 != 0 or e1 != 0)

        if endpoint1 == "TAIL":
            if existing_edge:
                return False
            if endpoint2 == "TAIL":
                if bidirected:
                    self.graph[j, i] = Endpoint.TAIL_AND_ARROW.value
                    self.graph[i, j] = Endpoint.TAIL_AND_ARROW.value
                else:
                    self.graph[j, i] = -1
                    self.graph[i, j] = -1
            else:
                if endpoint2 == "ARROW":
                    if bidirected:
                        self.graph[j, i] = Endpoint.ARROW_AND_ARROW.value
                        self.graph[i, j] = Endpoint.TAIL_AND_ARROW.value
                    else:
                        self.graph[j, i] = 1
                        self.graph[i, j] = -1
                    self.adjust_dpath(i, j)
                else:
                    if endpoint2 == "CIRCLE":
                        if bidirected:
                            return False
                        else:
                            self.graph[j, i] = 2
                            self.graph[i, j] = -1
                    else:
                        return False
        else:
            if endpoint1 == "ARROW":
                if endpoint2 == "ARROW":
                    if existing_edge:

                        if e1 == 2 or e2 == 2:
                            return False
                        if self.graph[j, i] == Endpoint.ARROW.value:
                            self.graph[j, i] = Endpoint.ARROW_AND_ARROW.value
                        else:
                            self.graph[j, i] = Endpoint.TAIL_AND_ARROW.value
                        if self.graph[i, j] == Endpoint.ARROW.value:
                            self.graph[i, j] = Endpoint.ARROW_AND_ARROW.value
                        else:
                            self.graph[i, j] = Endpoint.TAIL_AND_ARROW.value
                    else:
                        self.graph[j, i] = Endpoint.ARROW.value
                        self.graph[i, j] = Endpoint.ARROW.value
                else:
                    return False
            else:
                if endpoint1 == "CIRCLE":
                    if existing_edge:
                        return False
                    if endpoint2 == "ARROW":
                        if bidirected:
                            return False
                        else:
                            self.graph[j, i] = 1
                            self.graph[i, j] = 2
                    else:
                        if endpoint2 == "CIRCLE":
                            if bidirected:
                                return False
                            else:
                                self.graph[j, i] = 2
                                self.graph[i, j] = 2
                        else:
                            return False
                else:
                    return False

            return True

    def add_node(self, node: Node) -> bool:
        if node in self.nodes:
            return False

        nodes = self.nodes
        nodes.append(node)
        self.nodes = nodes

        self.num_vars = self.num_vars + 1

        self.node_map[node] = self.num_vars - 1

        row = np.zeros(self.num_vars - 1)
        graph = np.vstack((self.graph, row))
        dpath = np.vstack((self.dpath, row))

        col = np.zeros(self.num_vars)
        graph = np.column_stack((graph, col))
        dpath = np.column_stack((dpath, col))

        self.graph = graph
        self.dpath = dpath

        self.adjust_dpath(self.num_vars - 1, self.num_vars - 1)

        return True

    def clear(self):
        self.nodes = []
        self.num_vars = 0
        self.node_map = {}
        self.graph = np.zeros((self.num_vars, self.num_vars), np.dtype(int))
        self.dpath = np.zeros((self.num_vars, self.num_vars), np.dtype(int))

    def contains_edge(self, edge: Edge) -> bool:
        endpoint1 = str(edge.get_endpoint1())
        endpoint2 = str(edge.get_endpoint2())

        node1 = edge.get_node1()
        node2 = edge.get_node2()

        i = self.node_map[node1]
        j = self.node_map[node2]

        e1 = self.graph[i, j]
        e2 = self.graph[j, i]

        if endpoint1 == "TAIL":
            if endpoint2 == "TAIL":
                if (e2 == -1 and e1 == -1) \
                        or (e2 == Endpoint.TAIL_AND_ARROW.value and e1 == Endpoint.TAIL_AND_ARROW.value):
                    return True
                else:
                    return False
            else:
                if endpoint2 == "ARROW":
                    if (e1 == -1 and e2 == 1) \
                            or (e1 == Endpoint.TAIL_AND_ARROW.value and e2 == Endpoint.ARROW_AND_ARROW.value):
                        return True
                    else:
                        return False
                else:
                    if endpoint2 == "CIRCLE":
                        if e1 == -1 and e2 == 2:
                            return True
                        else:
                            return False
                    else:
                        return False
        else:
            if endpoint1 == "ARROW":
                if endpoint2 == "ARROW":
                    if (e1 == Endpoint.ARROW.value and e2 == Endpoint.ARROW.value) \
                            or (e1 == Endpoint.TAIL_AND_ARROW.value and e2 == Endpoint.TAIL_AND_ARROW.value) \
                            or (e1 == Endpoint.ARROW_AND_ARROW.value or e2 == Endpoint.ARROW_AND_ARROW.value):
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                if endpoint1 == "CIRCLE":
                    if endpoint2 == "ARROW":
                        if e1 == 2 and e2 == 1:
                            return True
                        else:
                            return False
                    else:
                        if endpoint2 == "CIRCLE":
                            if e1 == 2 and e2 == 2:
                                return True
                            else:
                                return False
                        else:
                            return False
                else:
                    return False

    def contains_node(self, node: Node) -> bool:
        node_list = self.nodes
        return node in node_list

    def exists_directed_cycle(self) -> bool:
        for node in self.nodes:
            if self._exists_directed_path_from_to(node, node):
                return True
        return False

    def _exists_directed_path_from_to(self, node1: Node, node2: Node) -> bool:
        Q = [node1]
        V = set()
        while Q:
            node_t = Q.pop(0)
            for node_c in self.get_children(node_t):
                if node_c == node2:
                    return True
                if node_c not in V:
                    V.add(node_c)
                    Q.append(node_c)
        return False

    def exists_trek(self, node1: Node, node2: Node) -> bool:
        for node in self.nodes:
            if self.is_ancestor_of(node, node1) and self.is_ancestor_of(node, node2):
                return True
        return False

    def __eq__(self, other):
        if isinstance(other, GeneralGraph):
            sorted_list = self.nodes.sort()
            if sorted_list == other.nodes.sort() and np.array_equal(self.graph, other.graph):
                return True
            else:
                return False
        else:
            return False

    def get_adjacent_nodes(self, node: Node) -> List[Node]:
        j = self.node_map[node]
        adj_list: List[Node] = []

        for i in range(self.num_vars):
            if (not self.graph[j, i] == 0) and (not self.graph[i, j] == 0):
                node2 = self.nodes[i]
                adj_list.append(node2)

        return adj_list

    def get_parents(self, node) -> List[Node]:
        j = self.node_map[node]
        parents: List[Node] = []

        for i in range(self.num_vars):
            if (self.graph[i, j] == -1 and self.graph[j, i] == 1) \
                    or (self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value
                        and self.graph[j, i] == Endpoint.ARROW_AND_ARROW.value):
                node2 = self.nodes[i]
                parents.append(node2)

        return parents

    def get_ancestors(self, nodes: List[Node]) -> List[Node]:
        if not isinstance(nodes, list):
            raise TypeError("Must be a list of nodes")

        ancestors: List[Node] = []

        for node in nodes:
            self.collect_ancestors(node, ancestors)

        return ancestors

    def get_children(self, node: Node) -> List[Node]:
        i = self.node_map[node]
        children: List[Node] = []

        for j in range(self.num_vars):
            if (self.graph[j, i] == 1 and self.graph[i, j] == -1) \
                    or (self.graph[j, i] == Endpoint.ARROW_AND_ARROW.value
                        and self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value):
                node2 = self.nodes[j]
                children.append(node2)

        return children

    def get_indegree(self, node: Node) -> int:
        i = self.node_map[node]
        indegree = 0

        for j in range(self.num_vars):
            if self.graph[i, j] == 1:
                indegree = indegree + 1
            else:
                if self.graph[i, j] == Endpoint.ARROW_AND_ARROW.value:
                    indegree = indegree + 2

        return indegree

    def get_outdegree(self, node: Node) -> int:
        i = self.node_map[node]
        outdegree = 0

        for j in range(self.num_vars):
            if self.graph[i, j] == -1 or self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value:
                outdegree = outdegree + 1

        return outdegree

    def get_degree(self, node: Node) -> int:
        i = self.node_map[node]
        degree = 0

        for j in range(self.num_vars):
            if self.graph[i, j] == 1 or self.graph[i, j] == -1 or self.graph[i, j] == 2:
                degree = degree + 1
            else:
                if self.graph[i, j] != 0:
                    degree = degree + 2

        return degree

    def get_max_degree(self) -> int:
        nodes = self.nodes
        max_degree = -1

        for node in nodes:
            deg = self.get_degree(node)
            if deg > max_degree:
                max_degree = deg

        return max_degree

    def get_node(self, name: str) -> Node | None:
        for node in self.nodes:
            if node.get_name() == name:
                return node
        return None

    def get_nodes(self) -> List[Node]:
        return self.nodes

    def get_node_names(self) -> List[str]:
        node_names: List[str] = []
        for node in self.nodes:
            node_names.append(node.get_name())
        return node_names

    def get_num_edges(self) -> int:
        edges = 0
        for i in range(self.num_vars):
            for j in range(i + 1, self.num_vars):
                if self.graph[i, j] == 1 or self.graph[i, j] == -1 or self.graph[i, j] == 2:
                    edges = edges + 1
                else:
                    if self.graph[i, j] != 0:
                        edges = edges + 2
        return edges

    def get_num_connected_edges(self, node: Node) -> int:
        i = self.node_map[node]
        edges = 0
        for j in range(self.num_vars):
            if self.graph[j, i] == 1 or self.graph[j, i] == -1 or self.graph[j, i] == 2:
                edges = edges + 1
            else:
                if self.graph[j, i] != 0:
                    edges = edges + 2
        return edges

    def get_num_nodes(self) -> int:
        return self.num_vars

    def is_adjacent_to(self, node1: Node, node2: Node) -> bool:
        i = self.node_map[node1]
        j = self.node_map[node2]
        return self.graph[j, i] != 0

    def is_ancestor_of(self, node1: Node, node2: Node) -> bool:
        i = self.node_map[node1]
        j = self.node_map[node2]
        return self.dpath[j, i] == 1

    def is_child_of(self, node1: Node, node2: Node) -> bool:
        i = self.node_map[node1]
        j = self.node_map[node2]
        return (self.graph[j, i] == Endpoint.TAIL.value and self.graph[i, j] == Endpoint.ARROW.value) \
               or self.graph[j, i] == Endpoint.TAIL_AND_ARROW.value

    def is_parent_of(self, node1: Node, node2: Node) -> bool:
        i = self.node_map[node1]
        j = self.node_map[node2]
        return (self.graph[j, i] == Endpoint.ARROW.value and self.graph[i, j] == Endpoint.TAIL.value) \
               or self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value

    def is_proper_ancestor_of(self, node1: Node, node2: Node) -> bool:
        return self.is_ancestor_of(node1, node2) and not (node1 == node2)

    def is_proper_descendant_of(self, node1: Node, node2: Node) -> bool:
        return self.is_descendant_of(node1, node2) and not (node1 == node2)

    def is_descendant_of(self, node1: Node, node2: Node) -> bool:
        return self.is_ancestor_of(node2, node1)

    def get_edge(self, node1: Node, node2: Node) -> Edge | None:
        i = self.node_map[node1]
        j = self.node_map[node2]

        end_1 = self.graph[i, j]
        end_2 = self.graph[j, i]

        if end_1 == 0:
            return None

        edge = Edge(node1, node2, Endpoint(end_1), Endpoint(end_2))
        return edge

    def get_directed_edge(self, node1: Node, node2: Node) -> Edge | None:
        i = self.node_map[node1]
        j = self.node_map[node2]

        end_1 = self.graph[i, j]
        end_2 = self.graph[j, i]

        if end_1 > 1 or end_1 == 0 or (end_1 == -1 and end_2 == -1):
            return None

        edge = Edge(node1, node2, Endpoint(end_1), Endpoint(end_2))
        return edge

    def get_node_edges(self, node: Node) -> List[Edge]:
        i = self.node_map[node]
        edges: List[Edge] = []

        for j in range(self.num_vars):
            node2 = self.nodes[j]
            if self.graph[j, i] == 1 or self.graph[j, i] == -1 or self.graph[j, i] == 2:
                edges.append(self.get_edge(node, node2))
            else:
                if self.graph[j, i] == Endpoint.TAIL_AND_ARROW.value \
                        and self.graph[i, j] == Endpoint.ARROW_AND_ARROW.value:
                    edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.TAIL))
                    edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.ARROW))
                else:
                    if self.graph[j, i] == Endpoint.ARROW_AND_ARROW.value \
                            and self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value:
                        edges.append(Edge(node, node2, Endpoint.TAIL, Endpoint.ARROW))
                        edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.ARROW))
                    else:
                        if self.graph[j, i] == Endpoint.TAIL_AND_ARROW.value \
                                and self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value:
                            edges.append(Edge(node, node2, Endpoint.TAIL, Endpoint.TAIL))
                            edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.ARROW))

        return edges

    def get_graph_edges(self) -> List[Edge]:
        edges: List[Edge] = []
        for i in range(self.num_vars):
            node = self.nodes[i]
            for j in range(i + 1, self.num_vars):
                node2 = self.nodes[j]
                if self.graph[j, i] == 1 or self.graph[j, i] == -1 or self.graph[j, i] == 2:
                    edges.append(self.get_edge(node, node2))
                else:
                    if self.graph[j, i] == Endpoint.TAIL_AND_ARROW.value \
                            and self.graph[i, j] == Endpoint.ARROW_AND_ARROW.value:
                        edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.TAIL))
                        edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.ARROW))
                    else:
                        if self.graph[j, i] == Endpoint.ARROW_AND_ARROW.value \
                                and self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value:
                            edges.append(Edge(node, node2, Endpoint.TAIL, Endpoint.ARROW))
                            edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.ARROW))
                        else:
                            if self.graph[j, i] == Endpoint.TAIL_AND_ARROW.value \
                                    and self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value:
                                edges.append(Edge(node, node2, Endpoint.TAIL, Endpoint.TAIL))
                                edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.ARROW))

        return edges

    def get_endpoint(self, node1: Node, node2: Node) -> Endpoint | None:
        edge = self.get_edge(node1, node2)
        if edge:
            return edge.get_proximal_endpoint(node2)
        else:
            return None

    def is_def_noncollider(self, node1: Node, node2: Node, node3: Node) -> bool:
        edges = self.get_node_edges(node2)
        circle12 = False
        circle23 = False

        for edge in edges:
            _node1 = edge.get_distal_node(node2) == node1
            _node3 = edge.get_distal_node(node2) == node3

            if _node1 and edge.points_toward(node1):
                return True
            if _node3 and edge.points_toward(node3):
                return True

            if _node1 and edge.get_proximal_endpoint(node2) == Endpoint.CIRCLE:
                circle12 = True
            if _node3 and edge.get_proximal_endpoint(node2) == Endpoint.CIRCLE:
                circle23 = True
            if circle12 and circle23 and not self.is_adjacent_to(node1, node2):
                return True

        return False

    def is_def_collider(self, node1: Node, node2: Node, node3: Node) -> bool:
        edge1 = self.get_edge(node1, node2)
        edge2 = self.get_edge(node2, node3)

        if edge1 is None or edge2 is None:
            return False

        return str(edge1.get_proximal_endpoint(node2)) == "ARROW" and str(edge2.get_proximal_endpoint(node2)) == "ARROW"

    def is_def_unshielded_collider(self, node1: Node, node2: Node, node3: Node) -> bool:
        return self.is_def_collider(node1, node2, node3) and not self.is_directly_connected_to(node1, node3)

    def is_dconnected_to(self, node1: Node, node2: Node, z: List[Node]) -> bool:
        # Inline d-connection check (BFS-based)
        return self._is_dconnected_to(node1, node2, z)

    def _is_dconnected_to(self, node1: Node, node2: Node, z: List[Node]) -> bool:
        # Simple BFS d-connection test
        z_set = set(z)
        z_ancestors = set()
        for z_node in z:
            self.collect_ancestors(z_node, list(z_ancestors))

        Q = []
        V = set()
        # Start with all edges from node1
        for adj in self.get_adjacent_nodes(node1):
            edge = self.get_edge(node1, adj)
            if edge is not None:
                # Track (node, direction_into_node)
                Q.append((adj, 'into' if edge.get_proximal_endpoint(adj) == Endpoint.ARROW else 'out'))

        while Q:
            current, direction = Q.pop(0)
            if current == node2:
                return True
            if (current, direction) in V:
                continue
            V.add((current, direction))

            for adj in self.get_adjacent_nodes(current):
                edge = self.get_edge(current, adj)
                if edge is None:
                    continue
                # Check d-connection rules
                into_current = edge.get_proximal_endpoint(current) == Endpoint.ARROW
                if direction == 'into' and self.is_def_collider(
                    self.nodes[0] if len(self.nodes) > 0 else current, current, adj
                ):
                    if current in z_set or current in z_ancestors:
                        new_dir = 'into' if edge.get_proximal_endpoint(adj) == Endpoint.ARROW else 'out'
                        Q.append((adj, new_dir))
                elif direction == 'out' and current not in z_set:
                    new_dir = 'into' if edge.get_proximal_endpoint(adj) == Endpoint.ARROW else 'out'
                    Q.append((adj, new_dir))

        return False

    def is_dseparated_from(self, node1: Node, node2: Node, z: List[Node]) -> bool:
        return not self.is_dconnected_to(node1, node2, z)

    def is_pattern(self) -> bool:
        return self.pattern

    def set_pattern(self, pat: bool):
        self.pattern = pat

    def is_pag(self) -> bool:
        return self.pag

    def set_pag(self, pag: bool):
        self.pag = pag

    def is_directed_from_to(self, node1: Node, node2: Node) -> bool:
        i = self.node_map[node1]
        j = self.node_map[node2]
        return self.graph[j, i] == 1 and self.graph[i, j] == -1

    def is_undirected_from_to(self, node1: Node, node2: Node) -> bool:
        i = self.node_map[node1]
        j = self.node_map[node2]
        return self.graph[j, i] == -1 and self.graph[i, j] == -1

    def is_directly_connected_to(self, node1: Node, node2: Node) -> bool:
        i = self.node_map[node1]
        j = self.node_map[node2]
        return not (self.graph[j, i] == 0 and self.graph[i, j] == 0)

    def is_exogenous(self, node: Node) -> bool:
        return self.get_indegree(node) == 0

    def get_nodes_into(self, node: Node, endpoint: Endpoint) -> List[Node]:
        i = self.node_map[node]
        nodes: List[Node] = []

        if str(endpoint) == "ARROW":
            for j in range(self.num_vars):
                if self.graph[i, j] == 1 or self.graph[i, j] == Endpoint.ARROW_AND_ARROW.value:
                    node2 = self.nodes[j]
                    nodes.append(node2)
        else:
            if str(endpoint) == "TAIL":
                for j in range(self.num_vars):
                    if self.graph[i, j] == -1 or self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value:
                        node2 = self.nodes[j]
                        nodes.append(node2)
            else:
                if str(endpoint) == "CIRCLE":
                    for j in range(self.num_vars):
                        if self.graph[i, j] == 2:
                            node2 = self.nodes[j]
                            nodes.append(node2)

        return nodes

    def get_nodes_out_of(self, node: Node, endpoint: Endpoint) -> List[Node]:
        i = self.node_map[node]
        nodes: List[Node] = []

        if str(endpoint) == "ARROW":
            for j in range(self.num_vars):
                if self.graph[j, i] == 1 or self.graph[j, i] == Endpoint.ARROW_AND_ARROW.value:
                    node2 = self.nodes[j]
                    nodes.append(node2)
        else:
            if str(endpoint) == "TAIL":
                for j in range(self.num_vars):
                    if self.graph[j, i] == -1 or self.graph[j, i] == Endpoint.TAIL_AND_ARROW.value:
                        node2 = self.nodes[j]
                        nodes.append(node2)
            else:
                if str(endpoint) == "CIRCLE":
                    for j in range(self.num_vars):
                        if self.graph[j, i] == 2:
                            node2 = self.nodes[j]
                            nodes.append(node2)

        return nodes

    def remove_edge(self, edge: Edge):
        node1 = edge.get_node1()
        node2 = edge.get_node2()

        i = self.node_map[node1]
        j = self.node_map[node2]

        out_of = self.graph[j, i]
        in_to = self.graph[i, j]

        end1 = edge.get_numerical_endpoint1()
        end2 = edge.get_numerical_endpoint2()

        is_fully_directed = self.is_parent_of(node1, node2) or self.is_parent_of(node2, node1)

        if out_of == Endpoint.TAIL_AND_ARROW.value and in_to == Endpoint.TAIL_AND_ARROW.value:
            if end1 == Endpoint.ARROW.value:
                self.graph[j, i] = -1
                self.graph[i, j] = -1
            else:
                if end1 == -1:
                    self.graph[i, j] = Endpoint.ARROW.value
                    self.graph[j, i] = Endpoint.ARROW.value
        else:
            if out_of == Endpoint.ARROW_AND_ARROW.value and in_to == Endpoint.TAIL_AND_ARROW.value:
                if end1 == Endpoint.ARROW.value:
                    self.graph[j, i] = 1
                    self.graph[i, j] = -1
                else:
                    if end1 == -1:
                        self.graph[j, i] = Endpoint.ARROW.value
                        self.graph[i, j] = Endpoint.ARROW.value
            else:
                if out_of == Endpoint.TAIL_AND_ARROW.value and in_to == Endpoint.ARROW_AND_ARROW.value:
                    if end1 == Endpoint.ARROW.value:
                        self.graph[j, i] = -1
                        self.graph[i, j] = 1
                    else:
                        if end1 == -1:
                            self.graph[j, i] = Endpoint.ARROW.value
                            self.graph[i, j] = Endpoint.ARROW.value
                else:
                    if end1 == in_to and end2 == out_of:
                        self.graph[j, i] = 0
                        self.graph[i, j] = 0

        if is_fully_directed:
            self.reconstitute_dpath(self.get_graph_edges())

    def remove_connecting_edge(self, node1: Node, node2: Node):
        i = self.node_map[node1]
        j = self.node_map[node2]
        self.graph[j, i] = 0
        self.graph[i, j] = 0

    def remove_connecting_edges(self, node1: Node, node2: Node):
        i = self.node_map[node1]
        j = self.node_map[node2]
        self.graph[j, i] = 0
        self.graph[i, j] = 0

    def remove_edges(self, edges: List[Edge]):
        for edge in edges:
            self.remove_edge(edge)

    def remove_node(self, node: Node):
        i = self.node_map[node]

        graph = self.graph
        graph = np.delete(graph, i, axis=0)
        graph = np.delete(graph, i, axis=1)
        self.graph = graph

        nodes = self.nodes
        nodes.remove(node)
        self.nodes = nodes

        node_map = {}
        for i, node in enumerate(self.nodes):
            node_map[node] = i
        self.node_map = node_map

        self.num_vars -= 1
        self.reconstitute_dpath(self.get_graph_edges())

    def remove_nodes(self, nodes: List[Node]):
        for node in nodes:
            self.remove_node(node)

    def subgraph(self, nodes: List[Node]):
        subgraph = GeneralGraph(nodes)

        graph = self.graph

        nodes_to_delete = []

        for i in range(self.num_vars):
            if not (self.nodes[i] in nodes):
                nodes_to_delete.append(i)

        graph = np.delete(graph, nodes_to_delete, axis=0)
        graph = np.delete(graph, nodes_to_delete, axis=1)

        subgraph.graph = graph
        subgraph.reconstitute_dpath(subgraph.get_graph_edges())

        return subgraph

    def __str__(self):
        s = "Graph Nodes:\n"
        for node in self.nodes:
            s += "  " + node.get_name() + "\n"
        s += "\nGraph Edges:\n"
        for edge in self.get_graph_edges():
            s += "  " + str(edge) + "\n"
        return s

    def transfer_nodes_and_edges(self, graph):
        for node in graph.nodes:
            self.add_node(node)
        for edge in graph.get_graph_edges():
            self.add_edge(edge)

    def transfer_attributes(self, graph):
        graph.attributes = self.attributes

    def get_ambiguous_triples(self) -> List[Tuple[Node, Node, Node]]:
        return self.ambiguous_triples

    def get_underlines(self) -> List[Tuple[Node, Node, Node]]:
        return self.underline_triples

    def get_dotted_underlines(self) -> List[Tuple[Node, Node, Node]]:
        return self.dotted_underline_triples

    def is_ambiguous_triple(self, node1: Node, node2: Node, node3: Node) -> bool:
        return (node1, node2, node3) in self.ambiguous_triples

    def is_underline_triple(self, node1: Node, node2: Node, node3: Node) -> bool:
        return (node1, node2, node3) in self.underline_triples

    def is_dotted_underline_triple(self, node1: Node, node2: Node, node3: Node) -> bool:
        return (node1, node2, node3) in self.dotted_underline_triples

    def add_ambiguous_triple(self, node1: Node, node2: Node, node3: Node):
        self.ambiguous_triples.append((node1, node2, node3))

    def add_underline_triple(self, node1: Node, node2: Node, node3: Node):
        self.underline_triples.append((node1, node2, node3))

    def add_dotted_underline_triple(self, node1: Node, node2: Node, node3: Node):
        self.dotted_underline_triples.append((node1, node2, node3))

    def remove_ambiguous_triple(self, node1: Node, node2: Node, node3: Node):
        self.ambiguous_triples.remove((node1, node2, node3))

    def remove_underline_triple(self, node1: Node, node2: Node, node3: Node):
        self.underline_triples.remove((node1, node2, node3))

    def remove_dotted_underline_triple(self, node1: Node, node2: Node, node3: Node):
        self.dotted_underline_triples.remove((node1, node2, node3))

    def set_ambiguous_triples(self, triples: List[Tuple[Node, Node, Node]]):
        self.ambiguous_triples = triples

    def set_underline_triples(self, triples: List[Tuple[Node, Node, Node]]):
        self.underline_triples = triples

    def set_dotted_underline_triples(self, triples: List[Tuple[Node, Node, Node]]):
        self.dotted_underline_triples = triples

    def get_causal_ordering(self) -> List[Node]:
        # Simple topological sort based on parent relationships
        ordering = []
        visited = set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for parent in self.get_parents(node):
                visit(parent)
            ordering.append(node)

        for node in self.nodes:
            visit(node)
        return ordering

    def is_parameterizable(self, node: Node) -> bool:
        return True

    def is_time_lag_model(self) -> bool:
        return False

    def get_sepset(self, node1: Node, node2: Node) -> List[Node]:
        # Simplified sepset finder
        return []

    def set_nodes(self, nodes: List[Node]):
        if len(nodes) != self.num_vars:
            raise ValueError("Sorry, there is a mismatch in the number of variables you are trying to set.")
        self.nodes = nodes

    def get_all_attributes(self):
        return self.attributes

    def get_attribute(self, key):
        return self.attributes[key]

    def remove_attribute(self, key):
        self.attributes.pop(key)

    def add_attribute(self, key, value):
        self.attributes[key] = value

    def get_node_map(self) -> Dict[Node, int]:
        return self.node_map

    def to_pcalg_matrix(self) -> ndarray:
        """Convert to pcalg-compatible adjacency matrix.

        Returns an n x n integer matrix where amat[i,j] encodes the endpoint
        type at j on the i-j edge:
            0 = no edge
            1 = circle (o)
            2 = arrowhead (>)
            3 = tail (-)
        """
        n = self.num_vars
        amat = np.zeros((n, n), dtype=int)

        for edge in self.get_graph_edges():
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            i = self.node_map[node1]
            j = self.node_map[node2]
            ep1 = edge.get_endpoint1()
            ep2 = edge.get_endpoint2()

            def _ep_to_pcalg(ep: Endpoint) -> int:
                if ep == Endpoint.CIRCLE:
                    return 1
                elif ep == Endpoint.ARROW:
                    return 2
                elif ep == Endpoint.TAIL:
                    return 3
                return 0

            # amat[i,j] = endpoint at j (on the i->j side)
            amat[i, j] = _ep_to_pcalg(ep1)
            amat[j, i] = _ep_to_pcalg(ep2)

        return amat
