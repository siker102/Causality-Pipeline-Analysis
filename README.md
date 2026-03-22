# Data To Discovery — Causal Discovery Pipeline

An end-to-end causal discovery pipeline that takes observational data, discovers causal structure via FCI or PC algorithms, identifies causal effects using IDP/CIDP, and recovers structural equations. All through an interactive Streamlit interface.

## Features

- **Causal discovery** via FCI and PC algorithms with configurable independence tests (Fisher-Z, Chi-Square, G-Square, KCI, missing-value Fisher-Z)
- **Background knowledge**: Specify forbidden/required edges and tier ordering to constrain the search
- **Causal effect identification** via IDP and CIDP algorithms - determines whether P(y|do(x)) or P(y|do(x),z) is identifiable from a PAG and provides the identification formula
- **Numeric causal effect estimation** - evaluates the identification formula against the data using importance-weighted density estimation
- **Structural equation recovery** - recovers full linear structural equations (all parent coefficients simultaneously) using weighted least squares with IDP importance weights, handling latent confounders
- **Graph visualization** with edge property annotations (definitely direct, possibly direct, no latent, possibly latent)
- **Data input** - CSV upload or random SCM generation with known ground truth for testing and validation
- **Ground truth comparison** - when using generated SCMs, compares recovered equations and effects against the true structural coefficients

## Getting Started

### Docker

```sh
docker build -t causality-pipeline .
docker run -p 8501:8501 causality-pipeline
```

Then open `http://localhost:8501`.

### Local

```sh
pip install -r requirements.txt
python -m streamlit run causality_pipeline.py
```

### Tests

```sh
python -m pytest tests/
```

125 tests including IDP/CIDP cross-validation against the original R implementation and WLS structural equation recovery tests.

## Architecture

**Entry point:** `causality_pipeline.py` - Streamlit app with sidebar for data source selection, algorithm configuration, and background knowledge controls.

**Pipeline flow:** Input Data → Background Knowledge (optional) → FCI or PC → Graph Visualization → IDP/CIDP Identification → Numeric Effect Estimation → Structural Equation Recovery

**Core package: `causal_discovery/`** - Standalone causal discovery library, decoupled from causallearn (only `causallearn.utils.cit` remains for CI tests).

```
causal_discovery/
├── graph/           # Graph representation (GeneralGraph, Edge, Node, Endpoint)
├── knowledge/       # BackgroundKnowledge (forbidden/required edges, tiers)
├── utils/           # Visualization (to_pydot), helpers, choice generators
├── fci/             # FCI algorithm, FAS skeleton discovery, orientation rules
├── pc/              # PC algorithm, Meek rules, orientation utilities
└── idp_and_cidp/    # IDP/CIDP identification, do-calculus, PAG utilities,
                     # importance weight extraction, causal effect evaluation
```

**Supporting modules:**
- `background_knowledge_controls.py` - Streamlit UI for specifying background knowledge
- `random_scm_generation.py` - Synthetic SCM generation with known ground truth
- `calculate_diagrams.py` - Batch accuracy comparison between FCI and PC

## Structural Equation Recovery

The pipeline includes a novel approach to recovering structural equations from PAGs in the presence of latent confounders. IDP provides per-data-point importance weights that reweight observational data to match the interventional distribution. Weighted least squares with these weights yields consistent estimates of the true structural coefficients.

See `theory/weighted_structural_equation_recovery.md` for the theoretical details.

## Key Dependencies

- **streamlit** - Web interface
- **causal-learn** - Conditional independence tests (`causallearn.utils.cit`)
- **scipy / scikit-learn** - Density estimation and regression for effect evaluation
- **pydot / graphviz** - Graph visualization
- **pandas / numpy** - Data handling

## Acknowledgements

The IDP and CIDP algorithms in `causal_discovery/idp_and_cidp/` are Python ports of the [PAGId](https://github.com/adele/PAGId) R package by Adele H. Ribeiro. The original R source is included in `R code/PAGId/` for reference and cross-validation testing.
