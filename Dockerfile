# Use the official Python image as the base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libcurl4-gnutls-dev \
    libssl-dev \
    libxml2-dev \
    r-base \
    r-base-dev \
    graphviz \
    libgraphviz-dev \
    gcc \
    gfortran \
    musl-dev \
    make \
    openssl \
    pkg-config \
    libfontconfig1-dev \
    r-cran-devtools \
    libharfbuzz-dev \
    libfribidi-dev \
    libudunits2-dev \
    libgit2-dev \
    libssh2-1-dev \
    git \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pydot \
    pygraphviz

# Set R environment variables
ENV R_HOME=/usr/lib/R
ENV PATH=$PATH:/usr/lib/R/bin
ENV LD_LIBRARY_PATH=/usr/lib/R/lib
#Trying to get the R capacity up
ENV R_MAX_VSIZE=4Gb
ENV R_MAX_NUM_DLLS=150
ENV R_ENABLE_JIT=0

# Install R packages with explicit dependencies
RUN Rscript -e "install.packages(c( \
      'usethis', 'remotes', 'curl', 'rlang', 'cli', 'dagitty', 'pcalg'), repos='https://cloud.r-project.org/')"
RUN Rscript -e 'library(devtools); devtools::install_github("adele/PAGId", dependencies=TRUE)'

#RUN Rscript -e "install.packages('devtools', repos='https://cloud.r-project.org/')" 
#RUN echo "devtools::install_github('adele/PAGId', dependencies=TRUE, upgrade='always')" > /tmp/install_pagid.R \
#    && Rscript /tmp/install_pagid.R 2>&1 | tee /tmp/r_pagid.log

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit runs on 8501 by default
EXPOSE 8501

# Streamlit run command
CMD ["streamlit", "run", "causality_pipeline.py"]