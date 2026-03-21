from __future__ import annotations

import warnings
from queue import Queue
from typing import List, Set, Tuple, Dict
from numpy import ndarray

from causal_discovery.graph.edge import Edge
from causal_discovery.graph.endpoint import Endpoint
from causal_discovery.graph.general_graph import GeneralGraph
from causal_discovery.graph.graph_node import GraphNode
from causal_discovery.graph.node import Node
from causal_discovery.utils.choice_generator import ChoiceGenerator, DepthChoiceGenerator
from causal_discovery.knowledge.background_knowledge import BackgroundKnowledge
from causal_discovery.fci.fas import fas_remake
from causallearn.utils.cit import *


def traverseSemiDirected(node: Node, edge: Edge) -> Node | None:
    if node == edge.get_node1():
        if edge.get_endpoint1() == Endpoint.TAIL or edge.get_endpoint1() == Endpoint.CIRCLE:
            return edge.get_node2()
    elif node == edge.get_node2():
        if edge.get_endpoint2() == Endpoint.TAIL or edge.get_endpoint2() == Endpoint.CIRCLE:
            return edge.get_node1()
    return None


def existsSemiDirectedPath(node_from: Node, node_to: Node, G: GeneralGraph) -> bool:
    Q = Queue()
    V = set()

    for node_u in G.get_adjacent_nodes(node_from):
        edge = G.get_edge(node_from, node_u)
        node_c = traverseSemiDirected(node_from, edge)

        if node_c is None:
            continue

        if node_c not in V:
            V.add(node_c)
            Q.put(node_c)

    while not Q.empty():
        node_t = Q.get_nowait()
        if node_t == node_to:
            return True

        for node_u in G.get_adjacent_nodes(node_t):
            edge = G.get_edge(node_t, node_u)
            node_c = traverseSemiDirected(node_t, edge)

            if node_c is None:
                continue

            if node_c not in V:
                V.add(node_c)
                Q.put(node_c)

    return False


def existOnePathWithPossibleParents(previous, node_w: Node, node_x: Node, node_b: Node, graph: GeneralGraph) -> bool:
    if node_w == node_x:
        return True

    p = previous.get(node_w)
    if p is None:
        return False

    for node_r in p:
        if node_r == node_b or node_r == node_x:
            continue

        if existsSemiDirectedPath(node_r, node_x, graph) or existsSemiDirectedPath(node_r, node_b, graph):
            return True

    return False


def getPossibleDsep(node_x: Node, node_y: Node, graph: GeneralGraph, maxPathLength: int) -> List[Node]:
    dsep = set()

    Q = Queue()
    V = set()

    previous = {node_x: None}

    e = None
    distance = 0

    adjacentNodes = set(graph.get_adjacent_nodes(node_x))

    for node_b in adjacentNodes:
        if node_b == node_y:
            continue
        edge = (node_x, node_b)
        if e is None:
            e = edge
        Q.put(edge)
        V.add(edge)

        node_list = previous.get(node_x)
        if node_list is None:
            previous[node_x] = set()
            node_list = previous.get(node_x)
        node_list.add(node_b)
        previous[node_x] = node_list

        dsep.add(node_b)

    while not Q.empty():
        t = Q.get_nowait()
        if e == t:
            e = None
            distance += 1
            if distance > 0 and distance > (1000 if maxPathLength == -1 else maxPathLength):
                break
        node_a, node_b = t

        if existOnePathWithPossibleParents(previous, node_b, node_x, node_b, graph):
            dsep.add(node_b)

        for node_c in graph.get_adjacent_nodes(node_b):
            if node_c == node_a:
                continue
            if node_c == node_x:
                continue
            if node_c == node_y:
                continue

            node_list = previous.get(node_c)
            if node_list is None:
                previous[node_c] = set()
                node_list = previous.get(node_c)
            node_list.add(node_b)
            previous[node_c] = node_list

            if graph.is_def_collider(node_a, node_b, node_c) or graph.is_adjacent_to(node_a, node_c):
                u = (node_a, node_c)
                if u in V:
                    continue

                V.add(u)
                Q.put(u)

                if e is None:
                    e = u

    if node_x in dsep:
        dsep.remove(node_x)
    if node_y in dsep:
        dsep.remove(node_y)

    _dsep = list(dsep)
    _dsep.sort(reverse=True)
    return _dsep


def fci_orient_bk(bk: BackgroundKnowledge | None, graph: GeneralGraph):
    if bk is None:
        return

    edges = graph.get_graph_edges()
    edges_to_orient_bidirected = []
    for edge in edges:
        node1, node2 = edge.get_node1(), edge.get_node2()
        if bk.is_forbidden(node1, node2) and bk.is_forbidden(node2, node1):
            if not (bk.is_required(node1, node2) or bk.is_required(node2, node1)):
                edges_to_orient_bidirected.append(edge)
    for edge in edges_to_orient_bidirected:
        node1, node2 = edge.get_node1(), edge.get_node2()
        graph.remove_edge(edge)
        graph.add_edge(Edge(node1, node2, Endpoint.ARROW, Endpoint.ARROW))

    edges = graph.get_graph_edges()
    for edge in edges:
        node1, node2 = edge.get_node1(), edge.get_node2()
        if bk.is_required(node1, node2):
            graph.remove_edge(edge)
            graph.add_directed_edge(node1, node2)
        elif bk.is_required(node2, node1):
            graph.remove_edge(edge)
            graph.add_directed_edge(node2, node1)
        elif bk.is_forbidden(node1, node2) and bk.is_forbidden(node2, node1):
            continue  # already handled as bidirected above
        elif bk.is_forbidden(node1, node2):
            graph.remove_edge(edge)
            graph.add_edge(Edge(node1, node2, Endpoint.ARROW, Endpoint.CIRCLE))
        elif bk.is_forbidden(node2, node1):
            graph.remove_edge(edge)
            graph.add_edge(Edge(node1, node2, Endpoint.CIRCLE, Endpoint.ARROW))


def is_arrow_point_allowed(node_x: Node, node_y: Node, graph: GeneralGraph, knowledge: BackgroundKnowledge | None) -> bool:
    if graph.get_endpoint(node_x, node_y) == Endpoint.ARROW:
        return True
    if graph.get_endpoint(node_x, node_y) == Endpoint.TAIL:
        return False
    if knowledge is not None and knowledge.is_forbidden(node_x, node_y):
        return False
    if knowledge is not None and knowledge.is_required(node_y, node_x):
        return False
    return graph.get_endpoint(node_x, node_y) == Endpoint.CIRCLE


def rule0(graph: GeneralGraph, nodes: List[Node], sep_sets: Dict[Tuple[int, int], Set[int]],
          knowledge: BackgroundKnowledge | None,
          verbose: bool):
    reorientAllWith(graph, Endpoint.CIRCLE, knowledge=knowledge)
    fci_orient_bk(knowledge, graph)

    for node_b in nodes:
        adjacent_nodes = graph.get_adjacent_nodes(node_b)
        if len(adjacent_nodes) < 2:
            continue

        cg = ChoiceGenerator(len(adjacent_nodes), 2)
        combination = cg.next()

        while combination is not None:
            node_a = adjacent_nodes[combination[0]]
            node_c = adjacent_nodes[combination[1]]
            combination = cg.next()

            if graph.is_adjacent_to(node_a, node_c):
                continue

            if graph.is_def_collider(node_a, node_b, node_c):
                continue

            sep_set = sep_sets.get((graph.get_node_map()[node_a], graph.get_node_map()[node_c]))

            if sep_set is not None and graph.get_node_map()[node_b] not in sep_set:
                if not is_arrow_point_allowed(node_a, node_b, graph, knowledge):
                    continue

                if not is_arrow_point_allowed(node_c, node_b, graph, knowledge):
                    continue

                if knowledge is None or not knowledge.is_required(node_b, node_a):
                    edge1 = graph.get_edge(node_a, node_b)
                    graph.remove_edge(edge1)
                    graph.add_edge(Edge(node_a, node_b, edge1.get_proximal_endpoint(node_a), Endpoint.ARROW))

                if knowledge is None or not knowledge.is_required(node_b, node_c):
                    edge2 = graph.get_edge(node_c, node_b)
                    graph.remove_edge(edge2)
                    graph.add_edge(Edge(node_c, node_b, edge2.get_proximal_endpoint(node_c), Endpoint.ARROW))

                if verbose:
                    print("Orienting collider: " + node_a.get_name() + " *-> " + node_b.get_name() +
                          " <-* " + node_c.get_name())


def reorientAllWith(graph: GeneralGraph, endpoint: Endpoint, knowledge: BackgroundKnowledge | None):
    ori_edges = graph.get_graph_edges()
    bk = knowledge

    for ori_edge in ori_edges:
        node1, node2 = ori_edge.get_node1(), ori_edge.get_node2()

        if bk is not None:
            if bk.is_required(node1, node2):
                graph.remove_edge(ori_edge)
                graph.add_directed_edge(node1, node2)
                continue

            if bk.is_required(node2, node1):
                graph.remove_edge(ori_edge)
                graph.add_directed_edge(node2, node1)
                continue

        graph.remove_edge(ori_edge)
        graph.add_edge(Edge(node1, node2, endpoint, endpoint))


def ruleR1(node_a: Node, node_b: Node, node_c: Node, graph: GeneralGraph, bk: BackgroundKnowledge | None,
           changeFlag: bool, verbose: bool = False) -> bool:
    if graph.is_adjacent_to(node_a, node_c):
        return changeFlag

    if graph.get_endpoint(node_a, node_b) == Endpoint.ARROW and graph.get_endpoint(node_c, node_b) == Endpoint.CIRCLE:
        if not is_arrow_point_allowed(node_b, node_c, graph, bk):
            return changeFlag

        if bk is not None and bk.is_required(node_c, node_b):
            return changeFlag

        edge1 = graph.get_edge(node_c, node_b)
        graph.remove_edge(edge1)
        graph.add_edge(Edge(node_c, node_b, Endpoint.ARROW, Endpoint.TAIL))
        changeFlag = True

        if verbose:
            print("Orienting edge (Away from collider):" + graph.get_edge(node_b, node_c).__str__())

    return changeFlag


def ruleR2(node_a: Node, node_b: Node, node_c: Node, graph: GeneralGraph, bk: BackgroundKnowledge | None, changeFlag: bool,
           verbose=False) -> bool:
    if graph.is_adjacent_to(node_a, node_c) and graph.get_endpoint(node_a, node_c) == Endpoint.CIRCLE:
        if graph.get_endpoint(node_a, node_b) == Endpoint.ARROW and \
                graph.get_endpoint(node_b, node_c) == Endpoint.ARROW and \
                (graph.get_endpoint(node_b, node_a) == Endpoint.TAIL or
                 graph.get_endpoint(node_c, node_b) == Endpoint.TAIL):
            if not is_arrow_point_allowed(node_a, node_c, graph, bk):
                return changeFlag

            edge1 = graph.get_edge(node_a, node_c)
            graph.remove_edge(edge1)
            graph.add_edge(Edge(node_a, node_c, edge1.get_proximal_endpoint(node_a), Endpoint.ARROW))

            if verbose:
                print("Orienting edge (Away from ancestor): " + graph.get_edge(node_a, node_c).__str__())

            changeFlag = True

    return changeFlag


def rulesR1R2cycle(graph: GeneralGraph, bk: BackgroundKnowledge | None, changeFlag: bool, verbose: bool = False) -> bool:
    nodes = graph.get_nodes()
    for node_B in nodes:
        adj = graph.get_adjacent_nodes(node_B)

        if len(adj) < 2:
            continue

        cg = ChoiceGenerator(len(adj), 2)
        combination = cg.next()

        while combination is not None:
            node_A = adj[combination[0]]
            node_C = adj[combination[1]]
            combination = cg.next()

            changeFlag = ruleR1(node_A, node_B, node_C, graph, bk, changeFlag, verbose)
            changeFlag = ruleR1(node_C, node_B, node_A, graph, bk, changeFlag, verbose)
            changeFlag = ruleR2(node_A, node_B, node_C, graph, bk, changeFlag, verbose)
            changeFlag = ruleR2(node_C, node_B, node_A, graph, bk, changeFlag, verbose)

    return changeFlag


def isNoncollider(graph: GeneralGraph, sep_sets: Dict[Tuple[int, int], Set[int]], node_i: Node, node_j: Node,
                  node_k: Node) -> bool:
    sep_set = sep_sets[(graph.get_node_map()[node_i], graph.get_node_map()[node_k])]
    return sep_set is not None and graph.get_node_map()[node_j] in sep_set


def ruleR3(graph: GeneralGraph, sep_sets: Dict[Tuple[int, int], Set[int]], bk: BackgroundKnowledge | None, changeFlag: bool,
           verbose: bool = False) -> bool:
    nodes = graph.get_nodes()
    for node_B in nodes:
        intoBArrows = graph.get_nodes_into(node_B, Endpoint.ARROW)
        intoBCircles = graph.get_nodes_into(node_B, Endpoint.CIRCLE)

        for node_D in intoBCircles:
            if len(intoBArrows) < 2:
                continue
            gen = ChoiceGenerator(len(intoBArrows), 2)
            choice = gen.next()

            while choice is not None:
                node_A = intoBArrows[choice[0]]
                node_C = intoBArrows[choice[1]]
                choice = gen.next()

                if graph.is_adjacent_to(node_A, node_C):
                    continue

                if (not graph.is_adjacent_to(node_A, node_D)) or (not graph.is_adjacent_to(node_C, node_D)):
                    continue

                if not isNoncollider(graph, sep_sets, node_A, node_D, node_C):
                    continue

                if graph.get_endpoint(node_A, node_D) != Endpoint.CIRCLE:
                    continue

                if graph.get_endpoint(node_C, node_D) != Endpoint.CIRCLE:
                    continue

                if not is_arrow_point_allowed(node_D, node_B, graph, bk):
                    continue

                edge1 = graph.get_edge(node_D, node_B)
                graph.remove_edge(edge1)
                graph.add_edge(Edge(node_D, node_B, edge1.get_proximal_endpoint(node_D), Endpoint.ARROW))

                if verbose:
                    print("Orienting edge (Double triangle): " + graph.get_edge(node_D, node_B).__str__())

                changeFlag = True
    return changeFlag


def getPath(node_c: Node, previous) -> List[Node]:
    l = []
    node_p = previous[node_c]
    if node_p is not None:
        l.append(node_p)
    while node_p is not None:
        node_p = previous.get(node_p)
        if node_p is not None:
            l.append(node_p)
    return l


def doDdpOrientation(node_d: Node, node_a: Node, node_b: Node, node_c: Node, previous, graph: GeneralGraph,
                    data, independence_test_method, alpha: float,
                    sep_sets: Dict[Tuple[int, int], Set[int]], change_flag: bool,
                    bk: BackgroundKnowledge | None, verbose: bool = False) -> (bool, bool):
    if graph.is_adjacent_to(node_d, node_c):
        raise Exception("illegal argument!")
    path = getPath(node_d, previous)

    X, Y = graph.get_node_map()[node_d], graph.get_node_map()[node_c]
    condSet = tuple([graph.get_node_map()[nn] for nn in path])
    p_value = independence_test_method(X, Y, condSet)
    ind = p_value > alpha

    path2 = list(path)
    path2.remove(node_b)

    X, Y = graph.get_node_map()[node_d], graph.get_node_map()[node_c]
    condSet = tuple([graph.get_node_map()[nn2] for nn2 in path2])
    p_value2 = independence_test_method(X, Y, condSet)
    ind2 = p_value2 > alpha

    if not ind and not ind2:
        sep_set = sep_sets.get((graph.get_node_map()[node_d], graph.get_node_map()[node_c]))
        if verbose:
            message = "Sepset for d = " + node_d.get_name() + " and c = " + node_c.get_name() + " = [ "
            if sep_set is not None:
                for ss in sep_set:
                    message += graph.get_nodes()[ss].get_name() + " "
            message += "]"
            print(message)

        if sep_set is None:
            if verbose:
                print(
                    "Must be a sepset: " + node_d.get_name() + " and " + node_c.get_name() + "; they're non-adjacent.")
            return False, change_flag

        ind = graph.get_node_map()[node_b] in sep_set

    if ind:
        if bk is not None and bk.is_required(node_c, node_b):
            return True, change_flag

        if not is_arrow_point_allowed(node_b, node_c, graph, bk):
            return True, change_flag

        edge = graph.get_edge(node_c, node_b)
        graph.remove_edge(edge)
        graph.add_edge(Edge(node_c, node_b, edge.get_proximal_endpoint(node_c), Endpoint.TAIL))

        if verbose:
            print("Orienting edge (Definite discriminating path): " +
                  graph.get_edge(node_b, node_c).__str__())

        change_flag = True
        return True, change_flag
    else:
        if not is_arrow_point_allowed(node_a, node_b, graph, bk) or not is_arrow_point_allowed(node_c, node_b, graph, bk):
            return False, change_flag

        edge1 = graph.get_edge(node_a, node_b)
        graph.remove_edge(edge1)
        graph.add_edge(Edge(node_a, node_b, edge1.get_proximal_endpoint(node_a), Endpoint.ARROW))

        edge2 = graph.get_edge(node_c, node_b)
        graph.remove_edge(edge2)
        graph.add_edge(Edge(node_c, node_b, edge2.get_proximal_endpoint(node_c), Endpoint.ARROW))

        if verbose:
            print(
                "Orienting collider (Definite discriminating path.. d = " + node_d.get_name() + "): " + node_a.get_name() + " *-> " + node_b.get_name() + " <-* " + node_c.get_name())

        change_flag = True
        return True, change_flag


def ddpOrient(node_a: Node, node_b: Node, node_c: Node, graph: GeneralGraph, maxPathLength: int, data: ndarray,
              independence_test_method, alpha: float, sep_sets: Dict[Tuple[int, int], Set[int]], change_flag: bool,
              bk: BackgroundKnowledge | None, verbose: bool = False) -> bool:
    Q = Queue()
    V = set()
    e = None
    distance = 0
    previous = {}

    cParents = graph.get_parents(node_c)

    Q.put(node_a)
    V.add(node_a)
    V.add(node_b)
    previous[node_a] = node_b

    while not Q.empty():
        node_t = Q.get_nowait()

        if e is None or e == node_t:
            e = node_t
            distance += 1
            if distance > 0 and distance > (1000 if maxPathLength == -1 else maxPathLength):
                return change_flag

        nodesInTo = graph.get_nodes_into(node_t, Endpoint.ARROW)

        for node_d in nodesInTo:
            if node_d in V:
                continue

            previous[node_d] = node_t
            node_p = previous[node_t]

            if not graph.is_def_collider(node_d, node_t, node_p):
                continue

            previous[node_d] = node_t

            if not graph.is_adjacent_to(node_d, node_c) and node_d != node_c:
                res, change_flag = \
                    doDdpOrientation(node_d, node_a, node_b, node_c, previous, graph, data,
                                     independence_test_method, alpha, sep_sets, change_flag, bk, verbose)

                if res:
                    return change_flag

            if node_d in cParents:
                Q.put(node_d)
                V.add(node_d)
    return change_flag


def ruleR4B(graph: GeneralGraph, maxPathLength: int, data: ndarray, independence_test_method, alpha: float,
            sep_sets: Dict[Tuple[int, int], Set[int]],
            change_flag: bool, bk: BackgroundKnowledge | None,
            verbose: bool = False) -> bool:
    nodes = graph.get_nodes()

    for node_b in nodes:
        possA = graph.get_nodes_out_of(node_b, Endpoint.ARROW)
        possC = graph.get_nodes_into(node_b, Endpoint.CIRCLE)

        for node_a in possA:
            for node_c in possC:
                if not graph.is_parent_of(node_a, node_c):
                    continue

                if graph.get_endpoint(node_b, node_c) != Endpoint.ARROW:
                    continue

                change_flag = ddpOrient(node_a, node_b, node_c, graph, maxPathLength, data, independence_test_method,
                                        alpha, sep_sets, change_flag, bk, verbose)
    return change_flag


def removeByPossibleDsep(graph: GeneralGraph, independence_test_method: CIT, alpha: float,
                         sep_sets: Dict[Tuple[int, int], Set[int]], background_knowledge: BackgroundKnowledge | None):
    def _contains_all(set_a: Set[Node], set_b: Set[Node]):
        for node_b in set_b:
            if node_b not in set_a:
                return False
        return True

    edges = graph.get_graph_edges()
    for edge in edges:
        node_a = edge.get_node1()
        node_b = edge.get_node2()

        if background_knowledge is not None:
            if background_knowledge.is_required(node_a, node_b) or background_knowledge.is_required(node_b, node_a):
                continue

        possibleDsep = getPossibleDsep(node_a, node_b, graph, -1)
        gen = DepthChoiceGenerator(len(possibleDsep), len(possibleDsep))

        choice = gen.next()
        while choice is not None:
            origin_choice = choice
            choice = gen.next()
            if len(origin_choice) < 2:
                continue
            sepset = tuple([possibleDsep[index] for index in origin_choice])
            if _contains_all(set(graph.get_adjacent_nodes(node_a)), set(sepset)):
                continue
            if _contains_all(set(graph.get_adjacent_nodes(node_b)), set(sepset)):
                continue
            X, Y = graph.get_node_map()[node_a], graph.get_node_map()[node_b]
            condSet_index = tuple([graph.get_node_map()[possibleDsep[index]] for index in origin_choice])
            p_value = independence_test_method(X, Y, condSet_index)
            independent = p_value > alpha
            if independent:
                graph.remove_edge(edge)
                sep_sets[(X, Y)] = set(condSet_index)
                break

        if graph.contains_edge(edge):
            possibleDsep = getPossibleDsep(node_b, node_a, graph, -1)
            gen = DepthChoiceGenerator(len(possibleDsep), len(possibleDsep))

            choice = gen.next()
            while choice is not None:
                origin_choice = choice
                choice = gen.next()
                if len(origin_choice) < 2:
                    continue
                sepset = tuple([possibleDsep[index] for index in origin_choice])
                if _contains_all(set(graph.get_adjacent_nodes(node_a)), set(sepset)):
                    continue
                if _contains_all(set(graph.get_adjacent_nodes(node_b)), set(sepset)):
                    continue
                X, Y = graph.get_node_map()[node_a], graph.get_node_map()[node_b]
                condSet_index = tuple([graph.get_node_map()[possibleDsep[index]] for index in origin_choice])
                p_value = independence_test_method(X, Y, condSet_index)
                independent = p_value > alpha
                if independent:
                    graph.remove_edge(edge)
                    sep_sets[(X, Y)] = set(condSet_index)
                    break


def fci_remake(dataset: ndarray, independence_test_method: str=fisherz, alpha: float = 0.05, depth: int = -1,
        max_path_length: int = -1, verbose: bool = False, background_knowledge: BackgroundKnowledge | None = None,
        show_progress: bool = True, **kwargs) -> Tuple[GeneralGraph, List[Edge]]:
    """Perform Fast Causal Inference (FCI) algorithm for causal discovery."""

    if dataset.shape[0] < dataset.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    independence_test_method = CIT(dataset, method=independence_test_method, **kwargs)

    if (depth is None) or type(depth) != int:
        raise TypeError("'depth' must be 'int' type!")
    if (background_knowledge is not None) and type(background_knowledge) != BackgroundKnowledge:
        raise TypeError("'background_knowledge' must be 'BackgroundKnowledge' type!")
    if type(max_path_length) != int:
        raise TypeError("'max_path_length' must be 'int' type!")

    nodes = []
    for i in range(dataset.shape[1]):
        node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        nodes.append(node)

    graph, sep_sets, test_results = fas_remake(dataset, nodes, independence_test_method=independence_test_method, alpha=alpha,
                                        knowledge=background_knowledge, depth=depth, verbose=verbose, show_progress=show_progress)
    reorientAllWith(graph, Endpoint.CIRCLE, knowledge=background_knowledge)

    rule0(graph, nodes, sep_sets, background_knowledge, verbose)

    removeByPossibleDsep(graph, independence_test_method, alpha, sep_sets, background_knowledge)

    reorientAllWith(graph, Endpoint.CIRCLE, knowledge=background_knowledge)
    rule0(graph, nodes, sep_sets, background_knowledge, verbose)

    change_flag = True
    first_time = True

    while change_flag:
        change_flag = False
        change_flag = rulesR1R2cycle(graph, background_knowledge, change_flag, verbose)
        change_flag = ruleR3(graph, sep_sets, background_knowledge, change_flag, verbose)

        if change_flag or (first_time and background_knowledge is not None and
                           len(background_knowledge.forbidden_rules_specs) > 0 and
                           len(background_knowledge.required_rules_specs) > 0 and
                           len(background_knowledge.tier_map.keys()) > 0):
            change_flag = ruleR4B(graph, max_path_length, dataset, independence_test_method, alpha, sep_sets,
                                  change_flag,
                                  background_knowledge, verbose)

            first_time = False

            if verbose:
                print("Epoch")

    graph.set_pag(True)

    from causal_discovery.fci.fci_helpers import get_color_edges
    edges = get_color_edges(graph)

    return graph, edges
