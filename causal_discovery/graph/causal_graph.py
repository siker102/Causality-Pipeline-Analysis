from __future__ import annotations

import warnings
from itertools import permutations
from typing import List, Tuple

import networkx as nx
import numpy as np

from causal_discovery.graph.edge import Edge
from causal_discovery.graph.endpoint import Endpoint
from causal_discovery.graph.general_graph import GeneralGraph
from causal_discovery.graph.graph_node import GraphNode
from causal_discovery.graph.node import Node
from causal_discovery.utils.helpers import append_value, powerset, list_union, sort_dict_ascending
from causallearn.utils.cit import CIT


class CausalGraph:
    def __init__(self, no_of_var: int, node_names: List[str] | None = None):
        if node_names is None:
            node_names = [("X%d" % (i + 1)) for i in range(no_of_var)]
        assert len(node_names) == no_of_var, "number of node_names must match number of variables"
        assert len(node_names) == len(set(node_names)), "node_names must be unique"
        nodes: List[Node] = []
        for name in node_names:
            node = GraphNode(name)
            nodes.append(node)
        self.G: GeneralGraph = GeneralGraph(nodes)
        for i in range(no_of_var):
            for j in range(i + 1, no_of_var):
                self.G.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
        self.test: CIT | None = None
        self.sepset = np.empty((no_of_var, no_of_var), object)  # store the collection of sepsets
        self.definite_UC = []  # store the list of definite unshielded colliders
        self.definite_non_UC = []  # store the list of definite unshielded non-colliders
        self.PC_elapsed = -1  # store the elapsed time of running PC
        self.redundant_nodes = []  # store the list of redundant nodes (for subgraphs)
        self.nx_graph = nx.DiGraph()  # store the directed graph
        self.nx_skel = nx.Graph()  # store the undirected graph
        self.labels = {}
        self.prt_m = {}  # store the parents of missingness indicators

    def set_ind_test(self, indep_test):
        self.test = indep_test

    def ci_test(self, i: int, j: int, S) -> float:
        if self.test.method == 'mc_fisherz':
            return self.test(i, j, S, self.nx_skel, self.prt_m)
        return self.test(i, j, S)

    def neighbors(self, i: int):
        return np.where(self.G.graph[i, :] != 0)[0]

    def max_degree(self) -> int:
        return max(np.sum(self.G.graph != 0, axis=1))

    def find_arrow_heads(self) -> List[Tuple[int, int]]:
        L = np.where(self.G.graph == 1)
        return list(zip(L[1], L[0]))

    def find_tails(self) -> List[Tuple[int, int]]:
        L = np.where(self.G.graph == -1)
        return list(zip(L[1], L[0]))

    def find_undirected(self) -> List[Tuple[int, int]]:
        return [(edge[0], edge[1]) for edge in self.find_tails() if self.G.graph[edge[0], edge[1]] == -1]

    def find_fully_directed(self) -> List[Tuple[int, int]]:
        return [(edge[0], edge[1]) for edge in self.find_arrow_heads() if self.G.graph[edge[0], edge[1]] == -1]

    def find_bi_directed(self) -> List[Tuple[int, int]]:
        return [(edge[1], edge[0]) for edge in self.find_arrow_heads() if (
                self.G.graph[edge[1], edge[0]] == Endpoint.ARROW.value and self.G.graph[
            edge[0], edge[1]] == Endpoint.ARROW.value)]

    def find_adj(self):
        return list(self.find_tails() + self.find_arrow_heads())

    def is_undirected(self, i, j) -> bool:
        return self.G.graph[i, j] == -1 and self.G.graph[j, i] == -1

    def is_fully_directed(self, i, j) -> bool:
        return self.G.graph[i, j] == -1 and self.G.graph[j, i] == 1

    def find_unshielded_triples(self) -> List[Tuple[int, int, int]]:
        return [(pair[0][0], pair[0][1], pair[1][1]) for pair in permutations(self.find_adj(), 2)
                if pair[0][1] == pair[1][0] and pair[0][0] != pair[1][1] and self.G.graph[pair[0][0], pair[1][1]] == 0]

    def find_triangles(self) -> List[Tuple[int, int, int]]:
        Adj = self.find_adj()
        return [(pair[0][0], pair[0][1], pair[1][1]) for pair in permutations(Adj, 2)
                if pair[0][1] == pair[1][0] and pair[0][0] != pair[1][1] and (pair[0][0], pair[1][1]) in Adj]

    def find_kites(self) -> List[Tuple[int, int, int, int]]:
        return [(pair[0][0], pair[0][1], pair[1][1], pair[0][2]) for pair in permutations(self.find_triangles(), 2)
                if pair[0][0] == pair[1][0] and pair[0][2] == pair[1][2]
                and pair[0][1] < pair[1][1] and self.G.graph[pair[0][1], pair[1][1]] == 0]

    def find_cond_sets(self, i: int, j: int) -> List[Tuple[int]]:
        neigh_x = self.neighbors(i)
        neigh_y = self.neighbors(j)
        pow_neigh_x = powerset(neigh_x)
        pow_neigh_y = powerset(neigh_y)
        return list_union(pow_neigh_x, pow_neigh_y)

    def find_cond_sets_with_mid(self, i: int, j: int, k: int) -> List[Tuple[int]]:
        return [S for S in self.find_cond_sets(i, j) if k in S]

    def find_cond_sets_without_mid(self, i: int, j: int, k: int) -> List[Tuple[int]]:
        return [S for S in self.find_cond_sets(i, j) if k not in S]

    def to_nx_graph(self):
        nodes = range(len(self.G.graph))
        self.labels = {i: self.G.nodes[i].get_name() for i in nodes}
        self.nx_graph.add_nodes_from(nodes)
        undirected = self.find_undirected()
        directed = self.find_fully_directed()
        bidirected = self.find_bi_directed()
        for (i, j) in undirected:
            self.nx_graph.add_edge(i, j, color='g')
        for (i, j) in directed:
            self.nx_graph.add_edge(i, j, color='b')
        for (i, j) in bidirected:
            self.nx_graph.add_edge(i, j, color='r')

    def to_nx_skeleton(self):
        nodes = range(len(self.G.graph))
        self.nx_skel.add_nodes_from(nodes)
        adj = [(i, j) for (i, j) in self.find_adj() if i < j]
        for (i, j) in adj:
            self.nx_skel.add_edge(i, j, color='g')
