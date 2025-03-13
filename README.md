# Project Title

Brief description of the project.

## Requirements

- Docker

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

The docker container RIGHT NOW does not support IDP, because of difficulties with RPY2, Docker and the PAGId library. To access full functionality download the program and install depenendencies, then start it via "python -m streamlit run causality_pipeline.py".
