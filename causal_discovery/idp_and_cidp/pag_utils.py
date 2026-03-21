"""PAG utility functions for IDP/CIDP algorithms.

Ported from PAGUtils.R in the PAGId R package.
All functions operate on pcalg-style adjacency matrices (numpy arrays)
where amat[i,j] encodes the endpoint at j on the i-j edge:
    0 = no edge
    1 = circle (o)
    2 = arrowhead (>)
    3 = tail (-)
Column/row names are tracked via a separate list of variable names.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray
from typing import Optional


# ---------------------------------------------------------------------------
# Adjacency-matrix helpers
# ---------------------------------------------------------------------------

def induced_pag(amat: ndarray, cnames: list[str], all_names: list[str]) -> ndarray:
    """Return the induced sub-PAG over the variables in *cnames*."""
    idx = [all_names.index(c) for c in cnames]
    return amat[np.ix_(idx, idx)].copy()


def get_adj_nodes(amat: ndarray, x: str, names: list[str]) -> list[str]:
    """Return names of all nodes adjacent to *x*."""
    xi = names.index(x)
    return [names[j] for j in range(len(names)) if amat[xi, j] != 0]


# ---------------------------------------------------------------------------
# Possible ancestors / descendants  (replaces pcalg::possAn / possDe)
# ---------------------------------------------------------------------------

def _poss_reach(amat: ndarray, start_ids: list[int], direction: str) -> set[int]:
    """BFS over edges that are *possibly* in the given direction.

    For ancestors:  follow edges j -> i where amat[j,i] could be an arrow
                    i.e. amat[j,i] in {1,2} (circle or arrowhead at i)
                    AND amat[i,j] in {1,3} (circle or tail at j — could be a tail)
    For descendants: mirror of the above.

    This is a conservative (superset) reachability — any edge with a circle
    endpoint *might* go either way in the true DAG.
    """
    n = amat.shape[0]
    visited: set[int] = set(start_ids)
    queue = list(start_ids)

    while queue:
        cur = queue.pop(0)
        for nb in range(n):
            if nb in visited or amat[cur, nb] == 0:
                continue

            if direction == "ancestor":
                # We want nodes that could be ancestors of cur.
                # nb -> cur is possible if:
                #   endpoint at cur on nb-cur edge could be arrowhead: amat[nb, cur] in {1, 2}
                #   endpoint at nb on nb-cur edge could be tail: amat[cur, nb] in {1, 3}
                # But we traverse backwards: from cur, follow to nb if nb could be ancestor.
                # nb is possibly an ancestor of cur if there's a possibly directed path nb ~~> cur.
                # From cur's perspective, we look for edges where cur could be a descendant of nb:
                #   amat[nb, cur] in {1, 2} (arrowhead or circle at cur) AND
                #   amat[cur, nb] in {1, 3} (tail or circle at nb)
                if amat[nb, cur] in (1, 2) and amat[cur, nb] in (1, 3):
                    visited.add(nb)
                    queue.append(nb)
            else:  # descendant
                # nb is possibly a descendant of cur if:
                #   amat[cur, nb] in {1, 2} (arrowhead or circle at nb) AND
                #   amat[nb, cur] in {1, 3} (tail or circle at cur)
                if amat[cur, nb] in (1, 2) and amat[nb, cur] in (1, 3):
                    visited.add(nb)
                    queue.append(nb)

    return visited


def get_poss_ancestors(amat: ndarray, ynames: list[str], names: list[str]) -> list[str]:
    """Return names of all possible ancestors of Y (including Y itself)."""
    yids = [names.index(y) for y in ynames]
    anc = _poss_reach(amat, yids, "ancestor")
    return [names[i] for i in sorted(anc)]


def get_poss_descendants(amat: ndarray, ynames: list[str], names: list[str]) -> list[str]:
    """Return names of all possible descendants of Y (including Y itself)."""
    yids = [names.index(y) for y in ynames]
    desc = _poss_reach(amat, yids, "descendant")
    return [names[i] for i in sorted(desc)]


# ---------------------------------------------------------------------------
# Descendants via searchAM (used by m-separation for collider paths)
# ---------------------------------------------------------------------------

def search_am_descendants(amat: ndarray, x_id: int) -> list[int]:
    """Return indices of all descendants of node x_id (including itself).

    Follows definitely directed edges: amat[x_id, j] == 2 (arrowhead at j)
    and amat[j, x_id] == 3 (tail at x_id), i.e. x_id -> j.
    Also follows possible edges (circles) conservatively.

    This mirrors pcalg::searchAM(amat, x, type="de") which follows
    any edge where amat[from, to] != 0 and the edge *could* be directed
    from -> to.
    """
    n = amat.shape[0]
    visited = {x_id}
    queue = [x_id]
    while queue:
        cur = queue.pop(0)
        for nb in range(n):
            if nb not in visited and amat[cur, nb] != 0:
                # Follow any edge out of cur — this matches pcalg::searchAM "de"
                # which does a simple reachability on the adjacency structure
                visited.add(nb)
                queue.append(nb)
    return sorted(visited)


# ---------------------------------------------------------------------------
# Visible edges  (replaces pcalg::visibleEdge)
# ---------------------------------------------------------------------------

def visible_edge(amat: ndarray, a_id: int, b_id: int) -> bool:
    """Check if the directed edge A -> B is visible in the PAG.

    From Zhang 2008, Definition 3.1: A directed edge A -> B in a PAG is
    **visible** if there is a vertex C not adjacent to B, such that either:
      (a) there is an edge between C and A that is into A (C *-> A), or
      (b) there is a collider path between C and A that is into A and every
          non-endpoint vertex on the path is a parent of B.

    Uses pcalg-style adjacency matrix where amat[i,j] = endpoint at j.
    """
    n = amat.shape[0]

    # Check: A -> B must exist (arrowhead at B, tail at A)
    if not (amat[a_id, b_id] == 2 and amat[b_id, a_id] == 3):
        return False

    # Condition (a): Find C not adjacent to B such that C *-> A
    for c_id in range(n):
        if c_id == a_id or c_id == b_id:
            continue
        if amat[c_id, a_id] == 0:  # C not adjacent to A
            continue
        if amat[c_id, b_id] != 0:  # C adjacent to B — skip
            continue
        # C is adjacent to A, not adjacent to B
        # Check if edge C-A is into A: endpoint at A from C is arrowhead
        if amat[c_id, a_id] == 2:
            return True

    # Condition (b): Find C not adjacent to B such that there is a collider
    # path C = V1 *-> V2 <-> ... <-> Vk *-> A where every Vi (i=2..k) is
    # a parent of B.
    # We search for such paths by starting from A and working backwards.
    def _has_collider_path_to_non_adj_of_b(start: int) -> bool:
        """BFS backwards from A along collider paths where intermediates are parents of B."""
        # visited tracks nodes we've checked as intermediate collider-path nodes
        visited = {a_id, b_id}
        # Start with nodes that have an edge into A and are parents of B
        queue: list[int] = []
        for v in range(n):
            if v == a_id or v == b_id:
                continue
            if amat[v, a_id] == 0:  # not adjacent to A
                continue
            # v *-> A: endpoint at A from v is arrowhead
            if amat[v, a_id] != 2:
                continue
            # v must be a parent of B: v -> B (tail at v, arrowhead at B)
            if not (amat[v, b_id] == 2 and amat[b_id, v] == 3):
                continue
            # v is a valid intermediate: v *-> A and v -> B
            if amat[v, b_id] != 0 and amat[b_id, v] == 3:
                # Check if v is not adjacent to B... wait, v IS a parent of B so it IS adjacent
                # The intermediate nodes must be parents of B (hence adjacent).
                # Only the START of the collider path (C) must not be adjacent to B.
                # We need to continue the path backwards from v.
                # v <-> ... <-> C where C is not adjacent to B
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        # BFS: extend path backwards via bidirected edges (v <-> w)
        # looking for a node C not adjacent to B
        while queue:
            cur = queue.pop(0)
            # Check all neighbors of cur via bidirected edges
            for w in range(n):
                if w in visited:
                    continue
                # w <-> cur: arrowhead at both ends
                if amat[w, cur] == 2 and amat[cur, w] == 2:
                    # w is not adjacent to B?
                    if amat[w, b_id] == 0:
                        return True  # Found C = w
                    # w is adjacent to B — w must be a parent of B to continue
                    if amat[w, b_id] == 2 and amat[b_id, w] == 3:
                        visited.add(w)
                        queue.append(w)

        return False

    return _has_collider_path_to_non_adj_of_b(a_id)


def get_visible_nodes_from_x(amat: ndarray, xname: str, names: list[str]) -> list[str]:
    """Return nodes connected to X by a visible edge out of X (X -> V visible)."""
    ix = names.index(xname)
    vis_nodes = []
    for j in range(len(names)):
        if amat[j, ix] == 3:  # tail at X, so X -> j candidate
            if visible_edge(amat, ix, j):
                vis_nodes.append(names[j])
    return vis_nodes


def visible_edge_by_names(amat: ndarray, v1name: str, v2name: str, names: list[str]) -> bool:
    """Check if v1 -> v2 is a visible edge."""
    iv1 = names.index(v1name)
    iv2 = names.index(v2name)
    return visible_edge(amat, iv1, iv2)


# ---------------------------------------------------------------------------
# Invisible possible parents / children
# ---------------------------------------------------------------------------

def get_inv_poss_parents_a(amat_v: ndarray, amat: ndarray,
                           node_a: str, names_v: list[str], names: list[str]) -> list[str]:
    """Get 'invisible' possible parents: {V : V o-> A or V -> A with invisible edge}."""
    inv_poss_par = []
    a_idx = names.index(node_a)

    # V o-> A: amat[V, A] == 2 and amat[A, V] == 1
    for j in range(len(names)):
        if amat[j, a_idx] == 2 and amat[a_idx, j] == 1:
            inv_poss_par.append(names[j])

    # V -> A with invisible edge: amat[V, A] == 2 and amat[A, V] == 3
    for j in range(len(names)):
        if amat[j, a_idx] == 2 and amat[a_idx, j] == 3:
            vname = names[j]
            if vname in names_v and node_a in names_v:
                if not visible_edge_by_names(amat_v, vname, node_a, names_v):
                    inv_poss_par.append(vname)

    return inv_poss_par


def get_inv_poss_children_a(amat_v: ndarray, amat: ndarray,
                            node_a: str, names_v: list[str], names: list[str]) -> list[str]:
    """Get 'invisible' possible children: {V : A o-> V or A -> V with invisible edge}."""
    inv_poss_child = []
    a_idx = names.index(node_a)

    # A o-> V: amat[A, V] == 2 and amat[V, A] == 1
    for j in range(len(names)):
        if amat[a_idx, j] == 2 and amat[j, a_idx] == 1:
            inv_poss_child.append(names[j])

    # A -> V with invisible edge: amat[A, V] == 2 and amat[V, A] == 3
    for j in range(len(names)):
        if amat[a_idx, j] == 2 and amat[j, a_idx] == 3:
            vname = names[j]
            if vname in names_v and node_a in names_v:
                if not visible_edge_by_names(amat_v, node_a, vname, names_v):
                    inv_poss_child.append(vname)

    return inv_poss_child


# ---------------------------------------------------------------------------
# PC-component (possible c-component)
# ---------------------------------------------------------------------------

def get_pc_component_a(amat_v: ndarray, amat: ndarray,
                       nodes_a: list[str], names_v: list[str], names: list[str]) -> list[str]:
    """Get the possible C-component of node set A."""
    pcc_nodes_a: list[str] = []

    for node_a in nodes_a:
        a_idx = names.index(node_a)

        # Invisible possible children of A
        inv_poss_ch = get_inv_poss_children_a(amat_v, amat, node_a, names_v, names)
        new_nodes = [node_a] + inv_poss_ch
        pcc_a = list(new_nodes)

        # Expand via bidirected edges (A <-> V)
        hh_nodes: list[str] = []
        while len(new_nodes) > 0:
            for node in new_nodes:
                n_idx = names.index(node)
                for j in range(len(names)):
                    if amat[j, n_idx] == 2 and amat[n_idx, j] == 2:
                        if names[j] not in pcc_a:
                            hh_nodes.append(names[j])
                hh_nodes = [h for h in hh_nodes if h not in pcc_a]
            new_nodes = list(hh_nodes)
            pcc_a.extend(new_nodes)
            hh_nodes = []

        # Invisible possible parents of pcc_a nodes
        poss_par: list[str] = []
        for pcc_node in pcc_a:
            inv_pp = get_inv_poss_parents_a(amat_v, amat, pcc_node, names_v, names)
            poss_par.extend(inv_pp)

        # Circle-circle neighbors of A
        cc_nodes = []
        for j in range(len(names)):
            if amat[a_idx, j] == 1 and amat[j, a_idx] == 1:
                cc_nodes.append(names[j])

        for n in set(cc_nodes + poss_par):
            if n not in pcc_a:
                pcc_a.append(n)
        pcc_nodes_a.extend(pcc_a)

    return list(dict.fromkeys(pcc_nodes_a))  # unique, preserving order


# ---------------------------------------------------------------------------
# Buckets (groups of nodes connected by circle-circle edges)
# ---------------------------------------------------------------------------

def get_cc_nodes(amat: ndarray, nodes: list[str], exclude_nodes: list[str],
                 names: list[str]) -> list[str]:
    """Get nodes connected to *nodes* by circle-circle edges, excluding *exclude_nodes*."""
    cc_nodes = []
    for node in nodes:
        n_idx = names.index(node)
        for j in range(len(names)):
            if amat[j, n_idx] == 1 and amat[n_idx, j] == 1:
                if names[j] not in exclude_nodes:
                    cc_nodes.append(names[j])
    return cc_nodes


def get_bucket(amat: ndarray, node: str, names: list[str]) -> list[str]:
    """Get the bucket containing *node* (all nodes reachable via o-o edges)."""
    cur_nodes = [node]
    bucket = list(cur_nodes)
    exclude_nodes: list[str] = []
    while True:
        cc = get_cc_nodes(amat, cur_nodes, exclude_nodes, names)
        if len(cc) == 0:
            break
        exclude_nodes.extend(bucket)
        bucket.extend(cc)
        cur_nodes = cc
    return bucket


def get_bucket_list(amat: ndarray, names: list[str]) -> list[list[str]]:
    """Get all buckets in the PAG."""
    buckets: list[list[str]] = []
    assigned: set[str] = set()
    for name in names:
        if name not in assigned:
            b = get_bucket(amat, name, names)
            buckets.append(b)
            assigned.update(b)
    return buckets


# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------

def get_region(amat_v: ndarray, amat: ndarray,
               nodes_a: list[str], names_v: list[str], names: list[str]) -> list[str]:
    """Get the region of node set A."""
    region_a: list[str] = []
    vpcc_a = get_pc_component_a(amat_v, amat, nodes_a, names_v, names)
    for vpcc_node in vpcc_a:
        for bnode in get_bucket(amat, vpcc_node, names):
            if bnode not in region_a:
                region_a.append(bnode)
    return region_a


# ---------------------------------------------------------------------------
# M-separation
# ---------------------------------------------------------------------------

def is_collider(amat: ndarray, vi: str, vm: str, vj: str, names: list[str]) -> bool:
    """Check if vi *-> vm <-* vj (collider at vm)."""
    return amat[names.index(vi), names.index(vm)] == 2 and amat[names.index(vj), names.index(vm)] == 2


def is_definite_noncollider(amat: ndarray, vi: str, vm: str, vj: str, names: list[str]) -> bool:
    """Check if the triplet is a definite non-collider."""
    vi_i, vm_i, vj_i = names.index(vi), names.index(vm), names.index(vj)
    # vm has a tail from vi or vj
    if amat[vi_i, vm_i] == 3 or amat[vj_i, vm_i] == 3:
        return True
    # vi o- vm -o vj and vi not adjacent to vj
    if amat[vi_i, vm_i] == 1 and amat[vj_i, vm_i] == 1 and amat[vi_i, vj_i] == 0:
        return True
    return False


def is_def_m_con_triplet(amat: ndarray, vi_name: str, vm_name: str, vj_name: str,
                         z: list[str], names: list[str]) -> bool:
    """Check if a triplet is definitely m-connecting given Z."""
    if is_collider(amat, vi_name, vm_name, vj_name, names):
        vm_id = names.index(vm_name)
        de_vm = search_am_descendants(amat, vm_id)
        de_names = [names[i] for i in de_vm]
        if not any(d in z for d in de_names):
            return False  # no descendant of collider in Z
        return True
    elif is_definite_noncollider(amat, vi_name, vm_name, vj_name, names):
        if vm_name in z:
            return False  # non-collider in Z blocks
        return True
    else:
        return False  # non-definite status


def get_def_con_path(amat: ndarray, xnames: list[str], ynames: list[str],
                     snames: list[str], names: list[str]) -> Optional[list[str]]:
    """Find a definite connecting path between X and Y given S, or None if m-separated.

    Mirrors the R implementation which always processes pathList[[1]] (index 0 in Python).
    """
    for x in xnames:
        for y in ynames:
            path_list: list[list[str]] = [[x]]

            while len(path_list) > 0:
                curpath = path_list[0]
                len_curpath = len(curpath)
                lastnode = curpath[-1]

                if lastnode == y and len_curpath == 2:
                    return curpath  # direct edge X - Y

                if len_curpath > 2:
                    vi_name = curpath[-3]
                    vm_name = curpath[-2]
                    vj_name = lastnode
                    if not is_def_m_con_triplet(amat, vi_name, vm_name, vj_name, snames, names):
                        path_list.pop(0)
                        continue
                    else:
                        if lastnode == y:
                            return curpath

                # Extend path with adjacent nodes
                adj_nodes = get_adj_nodes(amat, lastnode, names)
                adj_nodes = [a for a in adj_nodes if a not in curpath]
                n_adj = len(adj_nodes)

                if n_adj == 0:
                    path_list.pop(0)
                else:
                    if y in adj_nodes:
                        # Replace current path with path ending at y
                        path_list[0] = curpath + [y]
                        adj_nodes = [a for a in adj_nodes if a != y]
                        n_adj -= 1
                    else:
                        path_list.pop(0)

                    # Add remaining branches
                    for a in adj_nodes:
                        path_list.append(curpath + [a])

    return None


def is_m_separated(amat: ndarray, xnames: list[str], ynames: list[str],
                   snames: list[str], names: list[str]) -> bool:
    """Check if X is m-separated from Y given S in the PAG."""
    return get_def_con_path(amat, xnames, ynames, snames, names) is None
