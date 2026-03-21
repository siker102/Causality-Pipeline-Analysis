# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bachelor thesis project: a Streamlit-based causal discovery pipeline ("Data To Discovery") that takes observational data, runs causal discovery algorithms (FCI/PC), and identifies causal effects using IDP/CIDP algorithms.

## Commands

```bash
# Run the app locally
python -m streamlit run causality_pipeline.py

# Docker
docker build -t causality-pipeline .
docker run -p 8501:8501 causality-pipeline

# Install dependencies
pip install -r requirements.txt
```

Tests: `python -m pytest tests/` (84 tests including IDP/CIDP cross-validation against R).

## Architecture

**Entry point:** `causality_pipeline.py` — Streamlit app with sidebar for data source selection (CSV upload or random SCM generation), algorithm config, and background knowledge controls.

**Pipeline flow:** Input Data → Background Knowledge (optional) → FCI or PC algorithm → Graph visualization → Optional IDP/CIDP causal effect identification

**Core package: `causal_discovery/`** — Standalone causal discovery library decoupled from causallearn (only `causallearn.utils.cit` remains for CI tests).

```
causal_discovery/
├── __init__.py                    # Re-exports core classes
├── types.py                       # ProgressCallback protocol
├── graph/
│   ├── endpoint.py                # Endpoint enum (TAIL, ARROW, CIRCLE, etc.)
│   ├── node.py                    # Node ABC + NodeType enum
│   ├── graph_node.py              # GraphNode concrete class
│   ├── edge.py                    # Edge with pointing_left normalization + Property enum
│   ├── general_graph.py           # GeneralGraph (numpy adjacency matrix)
│   └── causal_graph.py            # CausalGraph wrapper (uses causallearn CIT)
├── knowledge/
│   └── background_knowledge.py    # BackgroundKnowledge (fixed tier check: > not >=)
├── utils/
│   ├── choice_generator.py        # ChoiceGenerator, DepthChoiceGenerator
│   ├── visualization.py           # to_pydot() standalone function
│   └── helpers.py                 # append_value, powerset, sort_dict_ascending
├── fci/
│   ├── fas.py                     # fas_remake (skeleton discovery)
│   ├── fci_algorithm.py           # fci_remake + rules R0-R4B
│   └── fci_helpers.py             # get_color_edges, defVisible, accuracy, run_FCI_analysis
├── idp_and_cidp/
│   ├── __init__.py                # Exports idp, cidp
│   ├── idp.py                     # IDP algorithm (marginal causal effect identification)
│   ├── cidp.py                    # CIDP algorithm (conditional causal effect identification)
│   ├── pag_utils.py               # PAG utility functions (induced_pag, m-separation, buckets, etc.)
│   └── pag_calculus.py            # Do-calculus rules 1-3 for PAGs
└── pc/
    ├── pc_algorithm.py            # pc_remake + skeleton discovery + MVPC
    ├── meek.py                    # Meek orientation rules
    ├── uc_sepset.py               # UCSepset, maxp, definite_maxp
    └── orient_utils.py            # orient_by_background_knowledge
```

**Supporting modules (root level):**
- `background_knowledge_controls.py` — Streamlit UI widgets for specifying forbidden/required edges and tier ordering
- `random_scm_generation.py` — Generates synthetic SCMs with known ground truth for testing
- `calculate_diagrams.py` — Batch accuracy comparison between FCI and PC algorithms

**R code (`R code/PAGId/`):** The original PAGId R package by Adèle Ribeiro — reference implementation used for cross-validation tests. Not a runtime dependency.

## Key Dependencies

- **causallearn** — Only `causallearn.utils.cit` is used (conditional independence tests: fisherz, chisq, gsq, kci, mv_fisherz). All graph classes are in `causal_discovery/`.
- **rpy2** — Optional, only used in cross-validation tests (`tests/test_idp_cidp.py`). Not a runtime dependency.

## Graph Representation

FCI outputs PAGs (Partial Ancestral Graphs) where each edge endpoint is one of:
- **Circle (o)** — unknown orientation
- **Arrow (>)** — definite arrowhead
- **Tail (-)** — definite tail

Edges carry properties: `dd` (definitely direct, shown green), `pd` (possibly direct), `nl` (no latent confounder), `pl` (possibly latent). These are critical for downstream IDP identification.

The PAGId R code uses pcalg-style adjacency matrices where `amat[i,j]` encodes endpoint type at j on the i-j edge (0=no edge, 1=circle, 2=arrowhead, 3=tail). `GeneralGraph.to_pcalg_matrix()` provides this conversion.

## Important Implementation Details

- **Edge normalization**: The `Edge` class uses `pointing_left` to normalize node order — all FCI rules depend on this. Do not simplify.
- **Tier semantics**: `BackgroundKnowledge.is_forbidden` uses `>` (not `>=`) for tier checks, allowing same-tier edges. This is a deliberate fix from causallearn's behavior.
- **Numpy adjacency matrix**: `GeneralGraph` stores edges as integer endpoint values in a numpy matrix (`graph[i,j]`), not networkx. FCI rules index directly into this matrix.
