"""CIDP algorithm for identifying conditional causal effects from PAGs.

Ported from CIDP.R in the PAGId R package by Adele Ribeiro.

Decides whether the interventional distribution P(y|do(x),z) is
identifiable or not given a PAG and the observational distribution P(V).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import ndarray

from causal_discovery.idp_and_cidp.idp import idp
from causal_discovery.idp_and_cidp.pag_calculus import rule2
from causal_discovery.idp_and_cidp.pag_utils import (
    induced_pag,
    get_poss_ancestors,
    get_bucket_list,
)


def cidp(amat: ndarray, x: list[str], y: list[str], z: list[str] | None,
         names: list[str], verbose: bool = False) -> dict[str, Any]:
    """Run the CIDP algorithm.

    Parameters
    ----------
    amat : ndarray
        pcalg-style adjacency matrix for the PAG.
    x : list[str]
        Treatment variable names.
    y : list[str]
        Outcome variable names.
    z : list[str] or None
        Conditioning variable names. If None, delegates to IDP.
    names : list[str]
        All variable names (matching amat rows/columns).
    verbose : bool
        Print log messages.

    Returns
    -------
    dict with keys:
        'id'    : bool
        'query' : str
        'Qexpr' : dict (if identifiable)
        'Qop'   : dict (if identifiable)
    """
    if z is None or len(z) == 0:
        return idp(amat, x, y, names, verbose=verbose)

    # Make mutable copies
    x = list(x)
    z = list(z)

    def _check_cond_line2(buckets: list[list[str]], d: list[str]):
        """Find a bucket bi where bi intersects d but bi is not contained in d."""
        for bi in buckets:
            if any(b in d for b in bi) and not all(b in d for b in bi):
                return bi
        return None

    def _check_cond_line9(x_: list[str], y_: list[str], z_: list[str],
                          zpartition: list[list[str]]):
        """Find index of zi in zpartition where Rule2 applies."""
        for i, zi in enumerate(zpartition):
            z_minus_zi = [zv for zv in z_ if zv not in zi]
            if rule2(amat, zi, y_, z_minus_zi, x_, names):
                return i
        return None

    v = list(names)

    # Build query string
    intervset_str = "obs"
    p_interv = "P"
    if len(x) > 0:
        intervset_str = ",".join(x)
        p_interv = f"P_{{{intervset_str}}}"
    query = f"{p_interv}({','.join(y)} | {','.join(z)})"

    v_minus_x = [n for n in v if n not in x]
    amat_vmx = induced_pag(amat, v_minus_x, v)
    d = get_poss_ancestors(amat_vmx, y + z, v_minus_x)
    buckets = get_bucket_list(amat, names)

    # Line 2-8: Check buckets that intersect D but are not contained in D
    while True:
        bi = _check_cond_line2(buckets, d)
        if bi is None:
            break

        if verbose:
            print(f"bi={{{','.join(bi)}}} satisfies Cond Line 2")

        xprime = [b for b in bi if b in x]
        w = [xi for xi in x if xi not in xprime]

        if rule2(amat, xprime, y, z, w, names):
            x = w
            z = list(set(z + xprime))
            v_minus_x = [n for n in v if n not in x]
            amat_vmx = induced_pag(amat, v_minus_x, v)
            d = get_poss_ancestors(amat_vmx, y + z, v_minus_x)
        else:
            if verbose:
                print(f"FAIL in Line 8 for B={{{','.join(bi)}}} and D={{{','.join(d)}}}")
            return {"id": False, "query": query}

    # Build Z-partition from buckets
    zpartition: list[list[str]] = []
    for bi in buckets:
        inter = [b for b in bi if b in z]
        if len(inter) > 0:
            zpartition.append(inter)

    # Line 9: Move zi from Z to X if Rule2 applies
    while True:
        zpid = _check_cond_line9(x, y, z, zpartition)
        if zpid is None:
            break
        zi = zpartition[zpid]
        if verbose:
            print(f"zi={{{','.join(zi)}}} satisfies Cond Line 9")
        x = list(set(x + zi))
        z = [zv for zv in z if zv not in zi]
        zpartition.pop(zpid)

    # Call IDP for P_x(y, z)
    ret = idp(amat, x, y + z, names, verbose=verbose)

    if ret["id"]:
        # P_x(y|z) = P_x(y,z) / sum_y P_x(y,z)
        den = f"\\sum_{{{','.join(y)}}}{ret['query']}"
        ret["Qexpr"][query] = f"\\frac{{{ret['query']}}}{{{den}}}"
        ret["Qop"][query] = {"type": "frac_cond", "param": {"den.sumset": y, "prob": ret["query"]}}
        return {"id": True, "query": query, "Qop": ret["Qop"], "Qexpr": ret["Qexpr"]}
    else:
        return {"id": False, "query": query}
