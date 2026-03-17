from __future__ import annotations

import time
import warnings
from itertools import combinations, permutations
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from numpy import ndarray

from causal_discovery.graph.causal_graph import CausalGraph
from causal_discovery.knowledge.background_knowledge import BackgroundKnowledge
from causal_discovery.utils.helpers import append_value
from causal_discovery.pc.meek import meek, definite_meek
from causal_discovery.pc.uc_sepset import uc_sepset, maxp, definite_maxp
from causal_discovery.pc.orient_utils import orient_by_background_knowledge

from causallearn.utils.cit import *
from causallearn.utils.cit import CIT

from tqdm.auto import tqdm


def pc_remake(
    data: ndarray,
    alpha=0.05,
    indep_test=fisherz,
    stable: bool = True,
    uc_rule: int = 0,
    uc_priority: int = 2,
    mvpc: bool = False,
    correction_name: str = 'MV_Crtn_Fisher_Z',
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    node_names: List[str] | None = None,
    **kwargs
):
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    if mvpc:  # missing value PC
        if indep_test == fisherz:
            indep_test = mv_fisherz
        return mvpc_alg_remake(data=data, node_names=node_names, alpha=alpha, indep_test=indep_test,
                               correction_name=correction_name, stable=stable,
                               uc_rule=uc_rule, uc_priority=uc_priority,
                               background_knowledge=background_knowledge,
                               verbose=verbose,
                               show_progress=show_progress, **kwargs)
    else:
        return pc_alg_remake(data=data, node_names=node_names, alpha=alpha, indep_test=indep_test, stable=stable,
                             uc_rule=uc_rule, uc_priority=uc_priority,
                             background_knowledge=background_knowledge, verbose=verbose,
                             show_progress=show_progress, **kwargs)


def pc_alg_remake(
    data: ndarray,
    node_names: List[str] | None,
    alpha: float,
    indep_test: str,
    stable: bool,
    uc_rule: int,
    uc_priority: int,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    **kwargs
) -> CausalGraph:
    """
    Perform Peter-Clark (PC) algorithm for causal discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features).
    node_names: Shape [n_features]. The name for each feature.
    alpha : float, desired significance level of independence tests (p_value) in (0, 1)
    indep_test : str, the name of the independence test being used
    stable : run stabilized skeleton discovery if True (default = True)
    uc_rule : how unshielded colliders are oriented (0: uc_sepset, 1: maxP, 2: definiteMaxP)
    uc_priority : rule of resolving conflicts between unshielded colliders
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object
    """

    start = time.time()
    indep_test = CIT(data, indep_test, **kwargs)
    cg_1 = remake_skeleton_discovery(data, alpha, indep_test, stable,
                                     background_knowledge=background_knowledge, verbose=verbose,
                                     show_progress=show_progress, node_names=node_names)

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = uc_sepset(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = uc_sepset(cg_1, background_knowledge=background_knowledge)
        cg = meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = maxp(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = maxp(cg_1, background_knowledge=background_knowledge)
        cg = meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = definite_maxp(cg_1, alpha, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = definite_maxp(cg_1, alpha, background_knowledge=background_knowledge)
        cg_before = definite_meek(cg_2, background_knowledge=background_knowledge)
        cg = meek(cg_before, background_knowledge=background_knowledge)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")
    end = time.time()

    cg.PC_elapsed = end - start

    return cg


def mvpc_alg_remake(
    data: ndarray,
    node_names: List[str] | None,
    alpha: float,
    indep_test: str,
    correction_name: str,
    stable: bool,
    uc_rule: int,
    uc_priority: int,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    **kwargs,
) -> CausalGraph:
    """
    Perform missing value Peter-Clark (PC) algorithm for causal discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features).
    node_names: Shape [n_features]. The name for each feature.
    alpha : float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : str, name of the test-wise deletion independence test being used
    correction_name : name of the missingness correction
    stable : run stabilized skeleton discovery if True (default = True)
    uc_rule : how unshielded colliders are oriented
    uc_priority : rule of resolving conflicts between unshielded colliders
    background_knowledge: background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object
    """

    start = time.time()
    indep_test = CIT(data, indep_test, **kwargs)
    ## Step 1: detect the direct causes of missingness indicators
    prt_m = get_parent_missingness_pairs(data, alpha, indep_test, stable)

    ## Step 2:
    ## a) Run PC algorithm with the 1st step skeleton;
    cg_pre = remake_skeleton_discovery(data, alpha, indep_test, stable,
                                       background_knowledge=background_knowledge,
                                       verbose=verbose, show_progress=show_progress, node_names=node_names)
    if background_knowledge is not None:
        orient_by_background_knowledge(cg_pre, background_knowledge)

    cg_pre.to_nx_skeleton()

    ## b) Correction of the extra edges
    cg_corr = skeleton_correction(data, alpha, correction_name, cg_pre, prt_m, stable)

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_corr, background_knowledge)

    ## Step 3: Orient the edges
    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = uc_sepset(cg_corr, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = uc_sepset(cg_corr, background_knowledge=background_knowledge)
        cg = meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = maxp(cg_corr, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = maxp(cg_corr, background_knowledge=background_knowledge)
        cg = meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = definite_maxp(cg_corr, alpha, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = definite_maxp(cg_corr, alpha, background_knowledge=background_knowledge)
        cg_before = definite_meek(cg_2, background_knowledge=background_knowledge)
        cg = meek(cg_before, background_knowledge=background_knowledge)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")
    end = time.time()

    cg.PC_elapsed = end - start

    return cg


#######################################################################################################################
## *********** Functions for Step 1 ***********
def get_parent_missingness_pairs(data: ndarray, alpha: float, indep_test, stable: bool = True) -> Dict[str, list]:
    """
    Detect the parents of missingness indicators
    If a missingness indicator has no parent, it will not be included in the result
    """
    parent_missingness_pairs = {'prt': [], 'm': []}

    missingness_index = get_missingness_index(data)

    for missingness_i in missingness_index:
        parent_of_missingness_i = detect_parent(missingness_i, data, alpha, indep_test, stable)
        if not isempty(parent_of_missingness_i):
            parent_missingness_pairs['prt'].append(parent_of_missingness_i)
            parent_missingness_pairs['m'].append(missingness_i)
    return parent_missingness_pairs


def isempty(prt_r) -> bool:
    """Test whether the parent of a missingness indicator is empty"""
    return len(prt_r) == 0


def get_missingness_index(data: ndarray) -> List[int]:
    """Detect the missingness indicators (columns with NaN values)"""
    missingness_index = []
    _, ncol = np.shape(data)
    for i in range(ncol):
        if np.isnan(data[:, i]).any():
            missingness_index.append(i)
    return missingness_index


def detect_parent(r: int, data_: ndarray, alpha: float, indep_test, stable: bool = True) -> ndarray:
    """Detect the parents of a missingness indicator"""
    data = data_.copy()

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    data[:, r] = np.isnan(data[:, r]).astype(float)
    if sum(data[:, r]) == 0 or sum(data[:, r]) == len(data[:, r]):
        return np.empty(0)

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var)
    cg.set_ind_test(CIT(data, indep_test.method))

    node_ids = range(no_of_var)
    pair_of_variables = list(permutations(node_ids, 2))

    depth = -1
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        for (x, y) in pair_of_variables:
            if x != r:
                continue

            Neigh_x = cg.neighbors(x)
            if y not in Neigh_x:
                continue
            else:
                Neigh_x = np.delete(Neigh_x, np.where(Neigh_x == y))

            if len(Neigh_x) >= depth:
                for S in combinations(Neigh_x, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                        else:
                            edge_removal.append((x, y))
                            edge_removal.append((y, x))
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                        break

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    cg.to_nx_skeleton()
    cg_skel_adj = nx.to_numpy_array(cg.nx_skel).astype(int)
    prt = get_parent(r, cg_skel_adj)

    return prt


def get_parent(r: int, cg_skel_adj: ndarray) -> ndarray:
    """Get the neighbors of missingness indicators which are the parents"""
    num_var = len(cg_skel_adj[0, :])
    indx = np.array([i for i in range(num_var)])
    prt = indx[cg_skel_adj[r, :] == 1]
    return prt


#######################################################################################################################

def skeleton_correction(data: ndarray, alpha: float, test_with_correction_name: str, init_cg: CausalGraph, prt_m: dict,
                        stable: bool = True) -> CausalGraph:
    """Perform skeleton correction for missing value PC"""

    assert type(data) == np.ndarray
    assert 0 < alpha < 1
    assert test_with_correction_name in ["MV_Crtn_Fisher_Z", "MV_Crtn_G_sq"]

    no_of_var = data.shape[1]
    cg = init_cg

    if test_with_correction_name in ["MV_Crtn_Fisher_Z", "MV_Crtn_G_sq"]:
        cg.set_ind_test(CIT(data, "mc_fisherz"))
    cg.prt_m = prt_m

    node_ids = range(no_of_var)
    pair_of_variables = list(permutations(node_ids, 2))

    depth = -1
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        for (x, y) in pair_of_variables:
            Neigh_x = cg.neighbors(x)
            if y not in Neigh_x:
                continue
            else:
                Neigh_x = np.delete(Neigh_x, np.where(Neigh_x == y))

            if len(Neigh_x) >= depth:
                for S in combinations(Neigh_x, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                        else:
                            edge_removal.append((x, y))
                            edge_removal.append((y, x))
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                        break

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    return cg


#######################################################################################################################

# *********** Evaluation util ***********

def get_adjacancy_matrix(g: CausalGraph) -> ndarray:
    return nx.to_numpy_array(g.nx_graph).astype(int)


def matrix_diff(cg1: CausalGraph, cg2: CausalGraph) -> (float, List[Tuple[int, int]]):
    adj1 = get_adjacancy_matrix(cg1)
    adj2 = get_adjacancy_matrix(cg2)
    count = 0
    diff_ls = []
    for i in range(len(adj1[:, ])):
        for j in range(len(adj2[:, ])):
            if adj1[i, j] != adj2[i, j]:
                diff_ls.append((i, j))
                count += 1
    return count / 2, diff_ls


def remake_skeleton_discovery(
    data: ndarray,
    alpha: float,
    indep_test: CIT,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    node_names: List[str] | None = None,
) -> CausalGraph:
    """
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features).
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : class CIT, the independence test being used
    stable : run stabilized skeleton discovery if True (default = True)
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.
    node_names: Shape [n_features]. The name for each feature.

    Returns
    -------
    cg : a CausalGraph object
    """

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(indep_test)

    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()
        for x in range(no_of_var):
            if show_progress:
                pbar.update()
            if show_progress:
                pbar.set_description(f'Depth={depth}, working on node {x}')
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue
            for y in Neigh_x:
                knowledge_ban_edge = False
                sepsets = set()
                if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    knowledge_ban_edge = True
                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        break
                    else:
                        edge_removal.append((x, y))
                        edge_removal.append((y, x))

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if verbose:
                            print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            break
                        else:
                            edge_removal.append((x, y))
                            edge_removal.append((y, x))
                            for s in S:
                                sepsets.add(s)
                    else:
                        if verbose:
                            print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                if (x, y) in edge_removal or not cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y]):
                    append_value(cg.sepset, x, y, tuple(sepsets))
                    append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if show_progress:
        pbar.close()

    return cg
