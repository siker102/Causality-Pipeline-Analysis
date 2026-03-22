"""Cross-validation: R PAGId vs Python IDP on directed-only PAGs.

Confirms that R's IDP fails on fully-directed PAGs (DAGs) due to
pcalg::visibleEdge returning False when no third vertex C exists.
Python fixes this with the _is_dag_like check in visible_edge.

Requires rpy2 + pcalg + PAGId R source files.
"""

import numpy as np
import pytest

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()

    # Source the PAGId R package
    ro.r('source("R code/PAGId/R/PAGUtils.R")')
    ro.r('source("R code/PAGId/R/IDP.R")')
    HAS_RPY2 = True
except Exception:
    HAS_RPY2 = False

from causal_discovery.idp_and_cidp.idp import idp


def make_pag(var_names, edges):
    n = len(var_names)
    amat = np.zeros((n, n), dtype=int)
    for frm, to, ep_from, ep_to in edges:
        i = var_names.index(frm)
        j = var_names.index(to)
        amat[i, j] = ep_to
        amat[j, i] = ep_from
    return amat


def run_r_idp(amat, x_vars, y_vars, names):
    """Run the R IDP implementation on the given PAG."""
    nr, nc = amat.shape
    ro.r.assign("amat_py", amat)
    ro.r(f'''
        amat <- matrix(as.integer(amat_py), nrow={nr}, ncol={nc})
        colnames(amat) <- rownames(amat) <- c({", ".join(f'"{n}"' for n in names)})
    ''')
    ro.r.assign("x_vars", ro.StrVector(x_vars))
    ro.r.assign("y_vars", ro.StrVector(y_vars))
    ro.r('result <- IDP(amat, x=x_vars, y=y_vars, verbose=TRUE)')
    r_id = ro.r('result$id')[0]
    return bool(r_id)


@pytest.mark.skipif(not HAS_RPY2, reason="rpy2 or R PAGId not available")
class TestRCrossValidationDirectedOnly:
    """Confirm Python correctly identifies DAG effects that R misses."""

    def test_two_node_directed(self):
        """X1 -> X2: Python identifies, R does not (R's known limitation)."""
        names = ["X1", "X2"]
        amat = make_pag(names, [("X1", "X2", 3, 2)])

        py_result = idp(amat, ["X1"], ["X2"], names, verbose=False)
        r_result = run_r_idp(amat, ["X1"], ["X2"], names)

        assert py_result['id'] is True
        assert r_result is False  # R's known limitation

    def test_three_node_chain(self):
        """X -> Z -> Y: Python identifies, R does not."""
        names = ["X", "Z", "Y"]
        amat = make_pag(names, [
            ("X", "Z", 3, 2),
            ("Z", "Y", 3, 2),
        ])

        py_result = idp(amat, ["X"], ["Y"], names, verbose=False)
        r_result = run_r_idp(amat, ["X"], ["Y"], names)

        assert py_result['id'] is True
        assert r_result is False  # R's known limitation

    def test_collider_directed_only(self):
        """X1 -> X3, X2 -> X3: Python identifies, R does not."""
        names = ["X1", "X2", "X3"]
        amat = make_pag(names, [
            ("X1", "X3", 3, 2),
            ("X2", "X3", 3, 2),
        ])

        py_result = idp(amat, ["X1"], ["X3"], names, verbose=False)
        r_result = run_r_idp(amat, ["X1"], ["X3"], names)

        assert py_result['id'] is True
        assert r_result is False  # R's known limitation


@pytest.mark.skipif(not HAS_RPY2, reason="rpy2 or R PAGId not available")
class TestVisibleEdgeRCrossValidation:
    """Confirm Python's visible_edge intentionally diverges from R on DAGs."""

    def test_visible_edge_two_node(self):
        """In a 2-node DAG, Python returns True (correct), R returns False."""
        names = ["X1", "X2"]
        amat = make_pag(names, [("X1", "X2", 3, 2)])

        ro.r.assign("amat_py", amat)
        ro.r(f'''
            amat <- matrix(as.integer(amat_py), nrow=2, ncol=2)
            colnames(amat) <- rownames(amat) <- c("X1", "X2")
            r_visible <- pcalg::visibleEdge(amat, 1, 2)
        ''')
        r_visible = bool(ro.r('r_visible')[0])

        from causal_discovery.idp_and_cidp.pag_utils import visible_edge
        py_visible = visible_edge(amat, 0, 1)

        assert py_visible is True
        assert r_visible is False  # R's known limitation

    def test_visible_edge_three_node_chain(self):
        """In a 3-node DAG chain, Python returns True for all, R misses X->Z."""
        names = ["X", "Z", "Y"]
        amat = make_pag(names, [
            ("X", "Z", 3, 2),
            ("Z", "Y", 3, 2),
        ])

        ro.r.assign("amat_py", amat)
        ro.r(f'''
            amat <- matrix(as.integer(amat_py), nrow=3, ncol=3)
            colnames(amat) <- rownames(amat) <- c("X", "Z", "Y")
            r_visible_xz <- pcalg::visibleEdge(amat, 1, 2)
            r_visible_zy <- pcalg::visibleEdge(amat, 2, 3)
        ''')
        r_visible_xz = bool(ro.r('r_visible_xz')[0])
        r_visible_zy = bool(ro.r('r_visible_zy')[0])

        from causal_discovery.idp_and_cidp.pag_utils import visible_edge
        py_visible_xz = visible_edge(amat, 0, 1)
        py_visible_zy = visible_edge(amat, 1, 2)

        # Python: all edges visible in a DAG
        assert py_visible_xz is True
        assert py_visible_zy is True
        # R: X->Z invisible (no C not-adj-to-Z with C*->X), Z->Y visible (X qualifies as C)
        assert r_visible_xz is False
        assert r_visible_zy is True
