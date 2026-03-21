"""Do-calculus rules for PAGs (Rules 1-3) and PAG manipulations.

Ported from PAGCalculus.R in the PAGId R package.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from causal_discovery.idp_and_cidp.pag_utils import (
    get_visible_nodes_from_x,
    induced_pag,
    get_poss_ancestors,
    is_m_separated,
)


def get_x_upper_manipulated_pag(amat: ndarray, xnames: list[str],
                                 names: list[str]) -> ndarray:
    """Apply X-upper manipulation: remove all edges incoming to X.

    Returns P_{overline{X}} — the PAG with all edges *into* X removed.
    """
    amat_xupp = amat.copy()
    for xi_name in xnames:
        xi = names.index(xi_name)
        for j in range(len(names)):
            if amat_xupp[j, xi] == 2:  # arrowhead at xi from j
                amat_xupp[j, xi] = 0
                amat_xupp[xi, j] = 0
    return amat_xupp


def get_x_lower_manipulated_pag(amat: ndarray, xnames: list[str],
                                 names: list[str]) -> ndarray:
    """Apply X-lower manipulation: remove all visible edges outgoing from X.

    Returns P_{underline{X}}.
    """
    amat_xlow = amat.copy()
    for xi_name in xnames:
        vis_nodes = get_visible_nodes_from_x(amat, xi_name, names)
        xi = names.index(xi_name)
        for vnode in vis_nodes:
            vi = names.index(vnode)
            amat_xlow[vi, xi] = 0
            amat_xlow[xi, vi] = 0
    return amat_xlow


def rule1(amat: ndarray, x: list[str], y: list[str], z: list[str],
          w: list[str], names: list[str]) -> bool:
    """Do-calculus Rule 1: check if X and Y are m-separated by W U Z in P_{overline{W}}."""
    amat_upp = get_x_upper_manipulated_pag(amat, w, names)
    return is_m_separated(amat_upp, x, y, list(set(w + z)), names)


def rule2(amat: ndarray, x: list[str], y: list[str], z: list[str],
          w: list[str], names: list[str]) -> bool:
    """Do-calculus Rule 2: check if X and Y are m-separated by W U Z in P_{overline{W}, underline{X}}."""
    amat_upp = get_x_upper_manipulated_pag(amat, w, names)
    amat_lower = get_x_lower_manipulated_pag(amat_upp, x, names)
    return is_m_separated(amat_lower, x, y, list(set(w + z)), names)


def rule3(amat: ndarray, x: list[str], y: list[str], z: list[str],
          w: list[str], names: list[str]) -> bool:
    """Do-calculus Rule 3: check if X and Y are m-separated by W U Z in P_{overline{W}, overline{X(Z)}}.

    X(Z) = X \\ PossAn(Z) in P_{V \\ W}
    """
    v_minus_w = [n for n in names if n not in w]
    amat_vmw = induced_pag(amat, v_minus_w, names)
    poss_an_z = get_poss_ancestors(amat_vmw, z, v_minus_w)
    x_of_z = [xi for xi in x if xi not in poss_an_z]
    amat_upp = get_x_upper_manipulated_pag(amat, list(set(w + x_of_z)), names)
    return is_m_separated(amat_upp, x, y, list(set(w + z)), names)
