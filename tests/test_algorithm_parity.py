"""
Parity tests: verify our causal_discovery FCI and PC implementations produce
identical graphs to causallearn's reference implementations when no background
knowledge is provided.

Data is kept small (4 variables, 300 samples) to keep runtimes short in CI.
"""
import numpy as np
import pytest
from causallearn.search.ConstraintBased.FCI import fci as cl_fci
from causallearn.search.ConstraintBased.PC import pc as cl_pc

from causal_discovery.fci.fci_algorithm import fci_remake
from causal_discovery.pc.pc_algorithm import pc_remake
from random_scm_generation import generate_random_scm

# Random seeds generated fresh each run — pytest prints the seed value on failure so it's reproducible
rng = np.random.default_rng()
SEEDS = rng.integers(0, 100_000, size=5).tolist()
N_VARS = 4
N_SAMPLES = 300
ALPHA = 0.05


def make_data(seed: int) -> np.ndarray:
    """Generate reproducible observational data from a random SCM."""
    np.random.seed(seed)
    df, _, _ = generate_random_scm(
        num_vars=N_VARS,
        edge_prob=0.3,
        noise_level=1.0,
        num_samples=N_SAMPLES,
        max_mean=5,
        max_coefficient=5,
    )
    return df.to_numpy()


# ---------------------------------------------------------------------------
# FCI parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_fci_adjacency_matrix_matches_causallearn(seed):
    """Our fci_remake must produce the same PAG adjacency matrix as causallearn's fci."""
    data = make_data(seed)

    our_g, _ = fci_remake(
        dataset=data,
        independence_test_method="fisherz",
        alpha=ALPHA,
        verbose=False,
        show_progress=False,
    )
    cl_g, _ = cl_fci(
        data,
        "fisherz",
        ALPHA,
        verbose=False,
        show_progress=False,
    )

    np.testing.assert_array_equal(
        our_g.graph,
        cl_g.graph,
        err_msg=(
            f"FCI graph matrices differ for seed={seed}.\n"
            f"Ours:\n{our_g.graph}\nCausallearn:\n{cl_g.graph}"
        ),
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_fci_edge_count_matches_causallearn(seed):
    """Our FCI must return the same number of edges as causallearn's FCI."""
    data = make_data(seed)

    _, our_edges = fci_remake(
        dataset=data,
        independence_test_method="fisherz",
        alpha=ALPHA,
        verbose=False,
        show_progress=False,
    )
    _, cl_edges = cl_fci(
        data,
        "fisherz",
        ALPHA,
        verbose=False,
        show_progress=False,
    )

    assert len(our_edges) == len(cl_edges), (
        f"Edge count mismatch for seed={seed}: ours={len(our_edges)}, "
        f"causallearn={len(cl_edges)}"
    )


# ---------------------------------------------------------------------------
# PC parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_pc_adjacency_matrix_matches_causallearn(seed):
    """Our pc_remake must produce the same CPDAG adjacency matrix as causallearn's pc."""
    data = make_data(seed)

    our_cg = pc_remake(
        data=data,
        alpha=ALPHA,
        indep_test="fisherz",
        verbose=False,
        show_progress=False,
    )
    cl_cg = cl_pc(
        data,
        ALPHA,
        "fisherz",
        verbose=False,
        show_progress=False,
    )

    np.testing.assert_array_equal(
        our_cg.G.graph,
        cl_cg.G.graph,
        err_msg=(
            f"PC graph matrices differ for seed={seed}.\n"
            f"Ours:\n{our_cg.G.graph}\nCausallearn:\n{cl_cg.G.graph}"
        ),
    )


# ---------------------------------------------------------------------------
# Smoke tests (basic sanity regardless of causallearn)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_fci_returns_correct_node_count(seed):
    data = make_data(seed)
    g, _ = fci_remake(
        dataset=data,
        independence_test_method="fisherz",
        alpha=ALPHA,
        verbose=False,
        show_progress=False,
    )
    assert len(g.get_nodes()) == N_VARS


@pytest.mark.parametrize("seed", SEEDS)
def test_pc_returns_correct_node_count(seed):
    data = make_data(seed)
    cg = pc_remake(
        data=data,
        alpha=ALPHA,
        indep_test="fisherz",
        verbose=False,
        show_progress=False,
    )
    assert len(cg.G.get_nodes()) == N_VARS


@pytest.mark.parametrize("seed", SEEDS)
def test_fci_graph_matrix_is_square(seed):
    data = make_data(seed)
    g, _ = fci_remake(
        dataset=data,
        independence_test_method="fisherz",
        alpha=ALPHA,
        verbose=False,
        show_progress=False,
    )
    assert g.graph.shape == (N_VARS, N_VARS)


@pytest.mark.parametrize("seed", SEEDS)
def test_pc_graph_matrix_is_square(seed):
    data = make_data(seed)
    cg = pc_remake(
        data=data,
        alpha=ALPHA,
        indep_test="fisherz",
        verbose=False,
        show_progress=False,
    )
    assert cg.G.graph.shape == (N_VARS, N_VARS)
