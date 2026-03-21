"""Cross-validation tests for the Python IDP/CIDP port against the R PAGId package.

Tests build PAG adjacency matrices directly and compare Python vs R results.
R tests require rpy2 + PAGId installed; they are skipped if unavailable.
"""

import numpy as np
import pytest

from causal_discovery.idp_and_cidp.idp import idp
from causal_discovery.idp_and_cidp.cidp import cidp
from causal_discovery.idp_and_cidp.pag_utils import (
    induced_pag,
    get_poss_ancestors,
    get_poss_descendants,
    get_bucket_list,
    get_region,
    get_pc_component_a,
    is_m_separated,
    visible_edge,
)

# ---------------------------------------------------------------------------
# Helper: build PAG matrices from edge specs
# ---------------------------------------------------------------------------

def make_pag(var_names: list[str], edges: list[tuple[str, str, int, int]]) -> np.ndarray:
    """Build a pcalg-style adjacency matrix.

    edges: list of (from, to, endpoint_at_from, endpoint_at_to)
    pcalg convention: amat[i,j] = endpoint at j on the i-j edge.
    So amat[from, to] = endpoint_at_to, amat[to, from] = endpoint_at_from.
    Endpoints: 0=none, 1=circle, 2=arrowhead, 3=tail
    """
    n = len(var_names)
    amat = np.zeros((n, n), dtype=int)
    for frm, to, ep_from, ep_to in edges:
        i = var_names.index(frm)
        j = var_names.index(to)
        amat[i, j] = ep_to     # endpoint at 'to' stored at [from, to]
        amat[j, i] = ep_from   # endpoint at 'from' stored at [to, from]
    return amat


# ---------------------------------------------------------------------------
# PAG Fixtures from PAGInstances.R
# ---------------------------------------------------------------------------

def get_pag_simple_chain():
    """Simple chain: X -> Z -> Y (all directed, no latents).

    P(y|do(x)) = sum_z P(y|z)P(z|x) — identifiable.
    """
    names = ["X", "Z", "Y"]
    edges = [
        ("X", "Z", 3, 2),  # X -> Z
        ("Z", "Y", 3, 2),  # Z -> Y
    ]
    return make_pag(names, edges), names


def get_pag_bow_arc():
    """X -> Y with latent confounder: X <-> Y, X -> Y.

    PAG: X <-> Y (bidirected) plus X -> Y.
    Actually the PAG for a bow-arc is just X o-> Y.
    Let's use: X -> Y and X <-> Y merged = X -> Y with confounder.
    In PAG form: both edges collapse. Let's just use X o-> Y.
    """
    names = ["X", "Y"]
    edges = [
        ("X", "Y", 1, 2),  # X o-> Y
    ]
    return make_pag(names, edges), names


def get_pag_hyttinen():
    """PAG from Hyttinen example (getDAG2aHyttinen in PAGInstances.R).

    Observed: x, y, z, h, w (latents: uwx, uxh removed).
    After FCI: the PAG over {x, y, z, h, w}.
    From the DAG: x->z->y, h->y, uwx->x, uwx->w, uxh->h, uxh->x
    PAG should have: z->y, h->y, x->z, x<->h (due to uxh), x<->w (due to uwx)
    """
    names = ["x", "y", "z", "h", "w"]
    edges = [
        ("x", "z", 3, 2),  # x -> z
        ("z", "y", 3, 2),  # z -> y
        ("h", "y", 3, 2),  # h -> y
        ("x", "h", 2, 2),  # x <-> h (due to uxh)
        ("x", "w", 2, 2),  # x <-> w (due to uwx)
    ]
    return make_pag(names, edges), names


def get_pag1():
    """PAG1 from PAGInstances.R — getPAG1().

    6 variables: x1, x2, y, a, w, z
    """
    names = ["x1", "x2", "y", "a", "w", "z"]
    amat = np.zeros((6, 6), dtype=int)
    idx = {n: i for i, n in enumerate(names)}

    # w -> z
    amat[idx["w"], idx["z"]] = 2; amat[idx["z"], idx["w"]] = 3
    # z -> y
    amat[idx["z"], idx["y"]] = 2; amat[idx["y"], idx["z"]] = 3
    # a -> y
    amat[idx["a"], idx["y"]] = 2; amat[idx["y"], idx["a"]] = 3
    # a o-> w
    amat[idx["a"], idx["w"]] = 2; amat[idx["w"], idx["a"]] = 1
    # a o-> x1
    amat[idx["a"], idx["x1"]] = 2; amat[idx["x1"], idx["a"]] = 1
    # x2 o-> x1
    amat[idx["x2"], idx["x1"]] = 2; amat[idx["x1"], idx["x2"]] = 1
    # x2 o-> w
    amat[idx["x2"], idx["w"]] = 2; amat[idx["w"], idx["x2"]] = 1
    # x1 o-o w
    amat[idx["x1"], idx["w"]] = 1; amat[idx["w"], idx["x1"]] = 1

    return amat, names


# ---------------------------------------------------------------------------
# Pure Python unit tests
# ---------------------------------------------------------------------------

class TestPagUtils:
    def test_induced_pag(self):
        amat, names = get_pag_simple_chain()
        sub = induced_pag(amat, ["X", "Y"], names)
        assert sub.shape == (2, 2)
        # X and Y are not directly connected in X->Z->Y
        assert sub[0, 1] == 0

    def test_poss_ancestors(self):
        amat, names = get_pag_simple_chain()
        anc = get_poss_ancestors(amat, ["Y"], names)
        assert "X" in anc
        assert "Z" in anc
        assert "Y" in anc

    def test_poss_descendants(self):
        amat, names = get_pag_simple_chain()
        desc = get_poss_descendants(amat, ["X"], names)
        assert "Z" in desc
        assert "Y" in desc

    def test_bucket_list_no_circles(self):
        amat, names = get_pag_simple_chain()
        buckets = get_bucket_list(amat, names)
        # No circle-circle edges, so each node is its own bucket
        assert len(buckets) == 3

    def test_bucket_list_with_circles(self):
        amat, names = get_pag1()
        buckets = get_bucket_list(amat, names)
        # x1 o-o w should be in the same bucket
        for b in buckets:
            if "x1" in b:
                assert "w" in b
                break

    def test_m_separation_chain(self):
        amat, names = get_pag_simple_chain()
        # X and Y should be m-separated by Z
        assert is_m_separated(amat, ["X"], ["Y"], ["Z"], names)
        # X and Y should NOT be m-separated by empty set
        assert not is_m_separated(amat, ["X"], ["Y"], [], names)

    def test_visible_edge(self):
        amat, names = get_pag_simple_chain()
        x_idx, z_idx, y_idx = 0, 1, 2
        # X->Z: no C not-adj-to-Z with C*->X exists (only Y, which IS adj to Z)
        assert visible_edge(amat, x_idx, z_idx) is False
        # Z->Y: X is adj to Z, NOT adj to Y, and edge X-Z has arrowhead at Z (X*->Z)
        assert visible_edge(amat, z_idx, y_idx) is True

    def test_visible_edge_hyttinen(self):
        amat, names = get_pag_hyttinen()
        x_i, z_i = names.index("x"), names.index("z")
        y_i, h_i = names.index("y"), names.index("h")
        # x->z: h is adj to x (x<->h), not adj to z, and h*->x (arrowhead at x). Visible!
        assert visible_edge(amat, x_i, z_i) is True
        # z->y: x is adj to z (x->z), not adj to y, and x*->z (arrowhead at z). Visible!
        assert visible_edge(amat, z_i, y_i) is True


class TestIDP:
    def test_hyttinen_identifiable(self):
        """Hyttinen example: P(y|do(x)) should be identifiable."""
        amat, names = get_pag_hyttinen()
        result = idp(amat, ["x"], ["y"], names)
        assert result["id"] is True
        assert "Qexpr" in result

    def test_bidirected_identifiable(self):
        """X <-> Y (pure bidirected): P(y|do(x)) = P(y), which IS identifiable."""
        names = ["X", "Y"]
        amat = make_pag(names, [("X", "Y", 2, 2)])  # X <-> Y
        result = idp(amat, ["X"], ["Y"], names)
        assert result["id"] is True

    def test_bow_arc_not_identifiable(self):
        """X o-> Y: circle at X means X might cause Y or just confounding.
        P(y|do(x)) is NOT identifiable from this PAG."""
        amat, names = get_pag_bow_arc()
        result = idp(amat, ["X"], ["Y"], names)
        assert result["id"] is False

    def test_pag1_idp(self):
        """PAG1: IDP with x=["w","x1","x2"], y=["y"]."""
        amat, names = get_pag1()
        result = idp(amat, ["w", "x1", "x2"], ["y"], names)
        assert result["id"] is True

    def test_result_has_qexpr_and_qop(self):
        """When identifiable, result should contain Qexpr and Qop dicts."""
        amat, names = get_pag_hyttinen()
        result = idp(amat, ["x"], ["y"], names)
        assert result["id"] is True
        assert isinstance(result["Qexpr"], dict)
        assert isinstance(result["Qop"], dict)
        assert result["query"] in result["Qexpr"]


class TestCIDP:
    def test_cidp_no_conditioning(self):
        """CIDP with z=None should delegate to IDP."""
        amat, names = get_pag_hyttinen()
        result = cidp(amat, ["x"], ["y"], None, names)
        assert result["id"] is True

    def test_cidp_with_conditioning(self):
        """PAG1: CIDP with z=["z"] conditioning."""
        amat, names = get_pag1()
        result = cidp(amat, ["w", "x1", "x2"], ["y"], ["z"], names)
        assert result["id"] is True


# ---------------------------------------------------------------------------
# Cross-validation against R (skipped if rpy2/PAGId not available)
# ---------------------------------------------------------------------------

def _r_available():
    try:
        import rpy2.robjects as ro
        ro.r('library(PAGId)')
        return True
    except Exception:
        return False


r_available = pytest.mark.skipif(not _r_available(), reason="rpy2 or PAGId R package not available")


def _run_r_idp(amat: np.ndarray, names: list[str], x: list[str], y: list[str]) -> dict:
    """Run IDP via R and return result dict."""
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    with ro.conversion.localconverter(ro.default_converter + numpy2ri.converter):
        n = len(names)
        flattened = np.ravel(amat, order='F').tolist()
        r_matrix = ro.r.matrix(ro.IntVector(flattened), nrow=n, ncol=n)
        ro.r.assign("amat", r_matrix)
        ro.r(f'colnames(amat) <- c({",".join(repr(n) for n in names)})')
        ro.r(f'rownames(amat) <- c({",".join(repr(n) for n in names)})')
        ro.r.assign("x", ro.StrVector(x))
        ro.r.assign("y", ro.StrVector(y))
        ro.r('result <- IDP(amat, x, y, verbose=FALSE)')
        r_result = ro.r('result')
        # Access R list — may be OrdDict or ListVector depending on rpy2 version
        if hasattr(r_result, 'rx2'):
            return {"id": bool(r_result.rx2("id")[0])}
        else:
            return {"id": bool(r_result["id"][0])}


def _run_r_cidp(amat: np.ndarray, names: list[str], x: list[str],
                y: list[str], z: list[str]) -> dict:
    """Run CIDP via R and return result dict."""
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    with ro.conversion.localconverter(ro.default_converter + numpy2ri.converter):
        n = len(names)
        flattened = np.ravel(amat, order='F').tolist()
        r_matrix = ro.r.matrix(ro.IntVector(flattened), nrow=n, ncol=n)
        ro.r.assign("amat", r_matrix)
        ro.r(f'colnames(amat) <- c({",".join(repr(n) for n in names)})')
        ro.r(f'rownames(amat) <- c({",".join(repr(n) for n in names)})')
        ro.r.assign("x", ro.StrVector(x))
        ro.r.assign("y", ro.StrVector(y))
        ro.r.assign("z", ro.StrVector(z))
        ro.r('result <- CIDP(amat, x, y, z, verbose=FALSE)')
        r_result = ro.r('result')
        if hasattr(r_result, 'rx2'):
            return {"id": bool(r_result.rx2("id")[0])}
        else:
            return {"id": bool(r_result["id"][0])}


@r_available
class TestCrossValidationIDP:
    def test_simple_chain(self):
        amat, names = get_pag_simple_chain()
        py = idp(amat, ["X"], ["Y"], names)
        r = _run_r_idp(amat, names, ["X"], ["Y"])
        assert py["id"] == r["id"]

    def test_bow_arc(self):
        amat, names = get_pag_bow_arc()
        py = idp(amat, ["X"], ["Y"], names)
        r = _run_r_idp(amat, names, ["X"], ["Y"])
        assert py["id"] == r["id"]

    def test_hyttinen(self):
        amat, names = get_pag_hyttinen()
        py = idp(amat, ["x"], ["y"], names)
        r = _run_r_idp(amat, names, ["x"], ["y"])
        assert py["id"] == r["id"]

    def test_bidirected(self):
        names = ["X", "Y"]
        amat = make_pag(names, [("X", "Y", 2, 2)])
        py = idp(amat, ["X"], ["Y"], names)
        r = _run_r_idp(amat, names, ["X"], ["Y"])
        assert py["id"] == r["id"]

    def test_bidirected_triangle(self):
        names = ["X", "Y", "Z"]
        amat = make_pag(names, [
            ("X", "Y", 2, 2), ("X", "Z", 2, 2), ("Y", "Z", 2, 2),
        ])
        py = idp(amat, ["X"], ["Y"], names)
        r = _run_r_idp(amat, names, ["X"], ["Y"])
        assert py["id"] == r["id"]

    def test_pag1(self):
        amat, names = get_pag1()
        py = idp(amat, ["w", "x1", "x2"], ["y"], names)
        r = _run_r_idp(amat, names, ["w", "x1", "x2"], ["y"])
        assert py["id"] == r["id"]


@r_available
class TestCrossValidationCIDP:
    def test_pag1_cidp(self):
        amat, names = get_pag1()
        py = cidp(amat, ["w", "x1", "x2"], ["y"], ["z"], names)
        r = _run_r_cidp(amat, names, ["w", "x1", "x2"], ["y"], ["z"])
        assert py["id"] == r["id"]
