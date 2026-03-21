# Use the official Python image as the base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    libgraphviz-dev \
    gcc \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pydot \
    pygraphviz

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit runs on 8501 by default
EXPOSE 8501

# Streamlit run command
CMD ["streamlit", "run", "causality_pipeline.py"]
