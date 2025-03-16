# Causality Pipeline

Traditional data analysis methods predominantly rely on observational data, which imposes limits regarding the identifiability of causal structures. Interventional data helps establish a cause-and-effect relationship by breaking the influence of confounding variables. In some cases, counterfactuals pertaining to interventional data can be derived from just observational data. This pipeline can be used to derive counterfactuals from observational data, with user-friendly, descriptive AI to guide users to achieve the best possible results.s

## Requirements

- Docker

For local installation, check the requirements (or run the program and install the dependencies it will tell you it needs)

## Getting Started

1. Build the Docker image:
    ```sh
    docker build -t causality-pipeline .
    ```

2. Run the Docker container:
    ```sh
    docker run -p 8501:8501 causality-pipeline
    ```

3. Open your web browser and go to `http://localhost:8501` to see the application.

## Additional Information

The docker container RIGHT NOW does not support IDP right now, because of difficulties with RPY2, Docker and the PAGId library. To access full functionality download the program and install depenendencies, then start it via "python -m streamlit run causality_pipeline.py".
