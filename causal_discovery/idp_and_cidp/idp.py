"""IDP algorithm for identifying marginal causal effects from PAGs.

Ported from IDP.R in the PAGId R package by Adele Ribeiro.

Decides whether the interventional distribution P(y|do(x)) is
identifiable or not given a PAG and the observational distribution P(V).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy import ndarray

from causal_discovery.idp_and_cidp.pag_utils import (
    induced_pag,
    get_poss_ancestors,
    get_poss_descendants,
    get_bucket_list,
    get_pc_component_a,
    get_region,
)


class IDPNotIdentifiable(Exception):
    """Raised when a Q-factor is not identifiable."""
    pass


def idp(amat: ndarray, x: list[str], y: list[str], names: list[str],
        verbose: bool = False) -> dict[str, Any]:
    """Run the IDP algorithm.

    Parameters
    ----------
    amat : ndarray
        pcalg-style adjacency matrix for the PAG.
    x : list[str]
        Treatment variable names.
    y : list[str]
        Outcome variable names.
    names : list[str]
        All variable names (matching amat rows/columns).
    verbose : bool
        Print log messages.

    Returns
    -------
    dict with keys:
        'id'    : bool — whether P(y|do(x)) is identifiable
        'query' : str  — string representation of the query
        'Qexpr' : dict — LaTeX identification formula steps (if identifiable)
        'Qop'   : dict — structured operation steps (if identifiable)
    """
    q_expr_list: dict[str, Any] = {}
    q_op_list: dict[str, Any] = {}

    def _interv_str(intervset: list[str]) -> str:
        if len(intervset) == 0:
            return "obs"
        return ",".join(intervset)

    def _p_interv(intervset: list[str]) -> str:
        if len(intervset) == 0:
            return "P"
        return f"P_{{{','.join(intervset)}}}"

    def identify(c: list[str], t: list[str]) -> None:
        """Recursive identification: compute Q[C] from Q[T]."""
        if verbose:
            print(f"#### Computing Q[{','.join(c)}], from Q[{','.join(t)}]")

        vnames = names
        intervset = [v for v in vnames if v not in c]
        intervset_str = _interv_str(intervset)
        p_interv = _p_interv(intervset)

        # Line 1: C is empty
        if len(c) == 0:
            if verbose:
                print("Returning 1")
            q_expr_list[intervset_str] = 1
            q_op_list[intervset_str] = 1
            return

        # Line 2: T == C
        if len([v for v in t if v not in c]) == 0:
            if verbose:
                print("Returning Q[C] = Q[T]")
            if intervset_str not in q_expr_list:
                q_expr_list[intervset_str] = f"{p_interv}({','.join(t)})"
                q_op_list[intervset_str] = {"type": "none", "param": {"interv": intervset, "var": list(t)}}
            return

        # Induced PAGs
        amat_t = induced_pag(amat, t, vnames)
        amat_c = induced_pag(amat, c, vnames)
        t_minus_c = [v for v in t if v not in c]
        buckets_t = get_bucket_list(amat_t, t)

        # Line 5-6: Look for bucket B in T\C with interB contained in B (Prop. 6)
        for b_t in buckets_t:
            if all(b in t_minus_c for b in b_t):
                pcb = get_pc_component_a(amat, amat_t, b_t, vnames, t)
                poss_de_b = get_poss_descendants(amat_t, b_t, t)
                inter_b = [v for v in pcb if v in poss_de_b]

                if all(v in b_t for v in inter_b):
                    if verbose:
                        print(f"Cond. in line 6 is satisfied for bucket {{{','.join(b_t)}}}")

                    t_minus_b = [v for v in t if v not in b_t]
                    intervset2 = [v for v in vnames if v not in t_minus_b]
                    intervset2_str = _interv_str(intervset2)
                    p_interv2 = _p_interv(intervset2)

                    if verbose:
                        print(f"Q[T\\B] = {p_interv2}")

                    v_minus_t = [v for v in vnames if v not in t]
                    p_vminus_t = "P" if len(v_minus_t) == 0 else f"P_{{{','.join(v_minus_t)}}}"

                    if verbose:
                        print(f"Computing Q[T\\B]=Q[{','.join(t_minus_b)}]={p_interv2}"
                              f" from Q=Q[T]=Q[{','.join(t)}]={p_vminus_t} via Prop. 6")

                    # Q[T\B] = P_{v\t}(t) / P_{v\t}(b_t | t\possDeB)
                    t_minus_possde = [v for v in t if v not in poss_de_b]
                    q_op_list[intervset2_str] = {
                        "type": "prop6",
                        "param": {
                            "interv": v_minus_t,
                            "num.var": list(t),
                            "den.var": list(b_t),
                            "den.cond": t_minus_possde,
                        }
                    }

                    qtmb_str = (
                        f"\\frac{{{p_vminus_t}({','.join(t)})}}"
                        f"{{{p_vminus_t}({','.join(b_t)} | {','.join(t_minus_possde)})}}"
                    )
                    q_expr_list[intervset2_str] = qtmb_str

                    identify(c, t_minus_b)
                    return

        # Line 8-9: Look for bucket B in C with Region(B) != C (Prop. 7)
        for b_t in buckets_t:
            if all(b in c for b in b_t):
                region_bt = get_region(amat, amat_c, b_t, vnames, c)
                if len([v for v in c if v not in region_bt]) != 0:
                    if verbose:
                        print(f"Cond. in line 9 is satisfied for bucket {{{','.join(b_t)}}}")

                    c_minus_rb = [v for v in c if v not in region_bt]
                    region_cmrb = get_region(amat, amat_c, c_minus_rb, vnames, c)
                    region_inter = [v for v in region_bt if v in region_cmrb]

                    if verbose:
                        print(f"Computing Q[C]=Q[{','.join(c)}]={p_interv} via Prop. 7")
                        print(f"R_b: {','.join(region_bt)}")
                        print(f"R_cminusrb: {','.join(region_cmrb)}")
                        print(f"R_inter: {','.join(region_inter)}")

                    identify(region_bt, t)
                    identify(region_cmrb, t)
                    identify(region_inter, t)

                    rb_interv = [v for v in vnames if v not in region_bt]
                    rcmb_interv = [v for v in vnames if v not in region_cmrb]
                    rinter_interv = [v for v in vnames if v not in region_inter]

                    q_op_list[intervset_str] = {
                        "type": "prop7",
                        "param": {
                            "num.prod1": rb_interv,
                            "num.prod2": rcmb_interv,
                            "den": rinter_interv,
                        }
                    }

                    q_rb_str = _p_interv(rb_interv)
                    q_cmrb_str = _p_interv(rcmb_interv)
                    q_inter_str = _p_interv(rinter_interv)

                    q_expr_list[intervset_str] = f"\\frac{{{q_rb_str} . {q_cmrb_str}}}{{{q_inter_str}}}"
                    return

        raise IDPNotIdentifiable(
            f"Q[{','.join(c)}] is not identifiable from Q[{','.join(t)}]"
        )

    # --- Main IDP logic ---
    v = list(names)
    v_minus_x = [n for n in v if n not in x]
    amat_vmx = induced_pag(amat, v_minus_x, v)

    d = get_poss_ancestors(amat_vmx, y, v_minus_x)
    d_minus_y = [n for n in d if n not in y]

    try:
        identify(d, v)
        is_id = True
    except IDPNotIdentifiable as e:
        if verbose:
            print(f"FAIL: {e}")
        is_id = False

    # Build query string
    intervset_str = "obs"
    p_interv = "P"
    if len(x) > 0:
        intervset_str = ",".join(x)
        p_interv = f"P_{{{intervset_str}}}"
    query = f"{p_interv}({','.join(y)})"

    if is_id:
        qd_intervset = [v_i for v_i in v if v_i not in d]
        qd_intervset_str = _interv_str(qd_intervset)

        if len(d_minus_y) > 0:
            sumset_exp = f"\\sum_{{{','.join(d_minus_y)}}}"
            q_expr_list[query] = f"{sumset_exp}{q_expr_list[qd_intervset_str]}"
            q_op_list[query] = {"type": "sumset", "param": {"sumset": d_minus_y, "interv": qd_intervset}}
        else:
            q_expr_list[query] = q_expr_list[qd_intervset_str]
            del q_expr_list[qd_intervset_str]
            q_op_list[query] = q_op_list[qd_intervset_str]
            del q_op_list[qd_intervset_str]

        return {"id": True, "query": query, "Qop": q_op_list, "Qexpr": q_expr_list}
    else:
        return {"id": False, "query": query}
