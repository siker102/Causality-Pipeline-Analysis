# Causality Pipeline

Traditional data analysis methods predominantly rely on observational data, which imposes limits regarding the identifiability of causal structures. Interventional data helps establish a cause-and-effect relationship by breaking the influence of confounding variables. In some cases, counterfactuals pertaining to interventional data can be derived from just observational data. This pipeline can be used to derive counterfactuals from observational data, with user-friendly, descriptive AI to guide users to achieve the best possible results.

## Features

- **Causal discovery** via FCI and PC algorithms with configurable independence tests and background knowledge
- **Causal effect identification** via IDP and CIDP algorithms — determines whether P(y|do(x)) or P(y|do(x),z) is identifiable from a PAG and provides the identification formula
- CSV upload or random SCM generation for testing
- Graph visualization with edge property annotations

## Requirements

- Docker

For local installation: `pip install -r requirements.txt`, then `python -m streamlit run causality_pipeline.py`.

## Getting Started

1. Build the Docker image:
    ```sh
    docker build -t causality-pipeline .
    ```

2. Run the Docker container:
    ```sh
    docker run -p 8501:8501 causality-pipeline
    ```

3. Open your web browser and go to `http://localhost:8501`.

## Acknowledgements

The IDP and CIDP algorithms in `causal_discovery/idp_and_cidp/` are Python ports of the [PAGId](https://github.com/adele/PAGId) R package by Adèle H. Ribeiro. The original R source is included in `R code/PAGId/` for reference and cross-validation testing.
