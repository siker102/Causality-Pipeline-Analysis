from __future__ import annotations

from itertools import combinations
from typing import List, Dict, Tuple, Set

import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm

from causal_discovery.graph.general_graph import GeneralGraph
from causal_discovery.graph.causal_graph import CausalGraph
from causal_discovery.graph.node import Node
from causal_discovery.utils.helpers import append_value
from causal_discovery.knowledge.background_knowledge import BackgroundKnowledge
from causallearn.utils.cit import *


def fas_remake(data: ndarray, nodes: List[Node], independence_test_method: CIT_Base, alpha: float = 0.05,
        knowledge: BackgroundKnowledge | None = None, depth: int = -1,
        verbose: bool = False, stable: bool = True, show_progress: bool = True) -> Tuple[
    GeneralGraph, Dict[Tuple[int, int], Set[int]], Dict[Tuple[int, int, Set[int]], float]]:
    """
    Implements the "fast adjacency search" used in several causal algorithms.
    """
    ## ------- check parameters ------------
    if type(data) != np.ndarray:
        raise TypeError("'data' must be 'np.ndarray' type!")
    if not all(isinstance(node, Node) for node in nodes):
        raise TypeError("'nodes' must be 'List[Node]' type!")
    if not isinstance(independence_test_method, CIT_Base):
        raise TypeError("'independence_test_method' must be 'CIT_Base' type!")
    if type(alpha) != float or alpha <= 0 or alpha >= 1:
        raise TypeError("'alpha' must be 'float' type and between 0 and 1!")
    if knowledge is not None and type(knowledge) != BackgroundKnowledge:
        raise TypeError("'knowledge' must be 'BackgroundKnowledge' type!")
    if type(depth) != int or depth < -1:
        raise TypeError("'depth' must be 'int' type >= -1!")
    ## ------- end check parameters ------------

    if depth == -1:
        depth = float('inf')

    no_of_var = data.shape[1]
    node_names = [node.get_name() for node in nodes]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(independence_test_method)
    sep_sets: Dict[Tuple[int, int], Set[int]] = {}
    test_results: Dict[Tuple[int, int, Set[int]], float] = {}

    def remove_if_exists(x: int, y: int) -> None:
        edge = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
        if edge is not None:
            cg.G.remove_edge(edge)

    var_range = tqdm(range(no_of_var), leave=True) if show_progress \
        else range(no_of_var)
    current_depth: int = -1
    while cg.max_degree() - 1 > current_depth and current_depth < depth:
        current_depth += 1
        edge_removal = set()
        for x in var_range:
            if show_progress:
                var_range.set_description(f'Depth={current_depth}, working on node {x}')
                var_range.update()
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < current_depth - 1:
                continue
            for y in Neigh_x:
                sepsets = set()

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, current_depth):
                    p = cg.ci_test(x, y, S)
                    test_results[(x, y, S)] = p
                    if p > alpha:
                        if verbose:
                            print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            remove_if_exists(x, y)
                            remove_if_exists(y, x)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            sep_sets[(x, y)] = set(S)
                            sep_sets[(y, x)] = set(S)
                            break
                        else:
                            edge_removal.add((x, y))
                            edge_removal.add((y, x))
                            for s in S:
                                sepsets.add(s)
                    else:
                        if verbose:
                            print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                append_value(cg.sepset, x, y, tuple(sepsets))
                append_value(cg.sepset, y, x, tuple(sepsets))

        for (x, y) in edge_removal:
            remove_if_exists(x, y)
            if cg.sepset[x, y] is not None:
                origin_set = set(l_in for l_out in cg.sepset[x, y]
                                 for l_in in l_out)
                sep_sets[(x, y)] = origin_set
                sep_sets[(y, x)] = origin_set

    return cg.G, sep_sets, test_results
