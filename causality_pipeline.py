import streamlit as st
import pandas as pd
import numpy as np
from causallearn.graph.Edge import Edge, Endpoint
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
from typing import List, Optional, Dict, Tuple
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from collections import OrderedDict
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
import networkx as nx
import matplotlib.pyplot as plt
from rpy2.rinterface_lib.embedded import RRuntimeError
import background_knowledge_controls
import random_scm_generation
import IDP_helper_classes
import FCI_helper_classes

# Constants
R_PACKAGES  = ['rlang','cli', 'dagitty', 'pcalg', 'PAGId']
FCI_EDGE_ENCODING = {
    'CIRCLE' : 1,
    'ARROW' : 2,
    'TAIL' : 3,
}

# Streamlit config
st.set_page_config(page_title="Data To Discovery", layout="wide")
st.title('Data To Discovery')

def install_r_packages():
    """Install and load required R packages with proper error handling."""
    try:
        # Install devtools first if missing /done with docker fila but in case someone wants it local
        try:
            ro.r('library(devtools)')
            st.success("devtools loaded successfully")
        except RRuntimeError:
            st.warning("devtools not found. Installing...")
            ro.r('install.packages("devtools")')
            ro.r('library(devtools)')

        # Install core packages
        for pkg in R_PACKAGES:
            try:
                ro.r(f'library({pkg})')
                st.success(f"{pkg} loaded successfully")
            except RRuntimeError:
                st.warning(f"{pkg} not found. Installing...")
                ro.r(f'install.packages("{pkg}")')
                try:
                    ro.r(f'library({pkg})')
                    st.success(f"{pkg} installed and loaded")
                except RRuntimeError as e:
                    st.error(f"Failed to install {pkg}: {str(e)}")
                    raise

        # Install PAGId as well 
        try:
            ro.r('library(PAGId)')
        except RRuntimeError:
            st.warning("PAGId not found. Installing from GitHub...")
            try:
                ro.r('devtools::install_github("adele/PAGId", dependencies=TRUE)')
                ro.r('library(PAGId)')
                st.success("PAGId installed and loaded")
            except RRuntimeError as e:
                st.error(f"Failed to install PAGId: {str(e)}")
                raise

    except RRuntimeError as e:
        st.error(f"Package management failed: {str(e)}")
        raise RuntimeError("R package installation failed") from e

def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Load and preprocess data while preserving original columns."""
    try:
        df = pd.read_csv(uploaded_file)

        if df.isnull().any().any():  # Check if there are any NA values in the entire DataFrame
            st.warning("Warning: The dataset contains missing values (NA). Use the missing-value Fisher Z test for better results.")
        
        # Check for categorical columns and display a warning
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            st.warning("Warning: The dataset contains categorical columns.  Ensure these are appropriately handled for your analysis (use Chi-Square for purely categorical data).")

        # Warn about constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                st.warning(f"There is a constant column: {col}")

        # Find duplicate columns and deal with them
        duplicate_cols = df.T.duplicated()
        # Remove duplicate columns
        if duplicate_cols.any():
            st.warning(f"Removing duplicate columns: {duplicate_cols[duplicate_cols].index.tolist()}")
            df = df.loc[:, ~duplicate_cols]

        # Store original columns before preprocessing
        # original_cols = df.columns.tolist()  
        
        # One-hot encode only categorical columns NOT NEEDED BECAUSE OF CHI-SQUARE TEST
        # categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        # df = pd.get_dummies(df, columns=categorical_cols)
        
        # Convert to numeric and clean
        #  df = df.apply(pd.to_numeric, errors='coerce')
        #
        #  df = df.dropna(axis=1, how='all')  # Remove columns that became all null
        #df = df.dropna()  # Remove rows with missing values
        
        # Show encoding mapping to user
        # if len(categorical_cols) > 0:
        #     with st.expander("Categorical Encoding Details"):
        #         st.write("Encoded columns:", {col: f"{col}_[category]" for col in categorical_cols})
        
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None
    


def hasCycle(graph: GeneralGraph, edges: list[Edge]) -> bool:
    visited = set()
    rec_stack = set()

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)

        for edge in edges:
            if edge.get_node1() == node and edge.get_endpoint1() == Endpoint.TAIL:
                neighbor = edge.get_node2()
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

        rec_stack.remove(node)
        return False

    for node in graph.get_nodes():
        if node not in visited:
            if dfs(node):
                return True

    return False


def display_results(g: GeneralGraph, edges: list[Edge]):
    """Visualize and display FCI results"""
    
    try:
        # Check for cycles in the graph
        if hasCycle(g, edges):
            st.warning("The resulting graph contains cycles. The algorithm may not have processed the data properly.")


        # Debug statements
        # for edge in edges:
        #    st.write(edge.get_node1().get_name(), edge.get_node2().get_name(), edge.get_endpoint1(), edge.get_endpoint2(), edge.properties)

        pdy = GraphUtils.to_pydot(g, edges)
        # Generate graph image in memory
        image_data = pdy.create_png()

        # Display image
        st.image(image_data, caption="FCI Analysis Result, green arrow means definitely direct edge")

    except Exception as e:
        st.error(f"Graph visualization failed: {str(e)}")


    # Model Information
    with st.expander("Model Metadata"):
        st.write(f"Graph type: {type(g).__name__}")
        #st.write(f"Graph data: {g.def_visible()}")
        st.write(f"Number of nodes: {len(g.get_nodes())}")
        st.write(f"Number of edges: {len(edges) if edges else 0}")
        # Remove the curly braces around the condition
        visible_edges = [edge for edge in edges if Edge.Property.dd in edge.properties]
        st.write(f"Definitely direct edges: {', '.join([f'{edge.get_node1().get_name()} -> {edge.get_node2().get_name()}' for edge in visible_edges])}")


def main():
    """Main function for the causal discovery pipeline"""
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):

        # Initialize session state variables for consistent state management
        if 'generated_data' not in st.session_state:
            st.session_state.generated_data = None
        if 'true_graph' not in st.session_state:
            st.session_state.true_graph = None
        
        # Data Source Selector GUI
        st.sidebar.subheader("Data Source Configuration")

        data_source = st.sidebar.radio(
            "Choose data source:",
            ["Upload CSV", "Generate Random SCM"],
            index=0,
            on_change=lambda: background_knowledge_controls.reset_background_knowledge_state()
        )

        dataframe = None
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader('Upload CSV Data', type=['csv'])
            if uploaded_file:
                try:
                    dataframe = load_data(uploaded_file)
                    # Clear generated data when switching to upload mode
                    st.session_state.generated_data = None
                    st.session_state.true_graph = None
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    return
        else:
            # SCM Generation Controls
            with st.sidebar.expander("SCM Parameters"):
                num_vars = st.number_input("Number of variables", 3, 20, 5)
                edge_prob = st.slider("Connection probability", 0.1, 0.5, 0.3)
                noise_level = st.slider("Noise level (σ)", 0.1, 2.0, 1.0)
                num_samples = st.number_input("Number of samples", 100, 10_000, 1000)
                max_mean = st.slider("Max mean value", 0, 1000, 10)
                max_coefficient = st.slider("Max coefficient value for relationship strength", 0, 1000, 10)
            
            if st.sidebar.button("Generate New SCM"):
                # Store the generated SCM in session state
                generated_data, true_graph, equations = random_scm_generation.generate_random_scm(
                    num_vars, edge_prob, noise_level, num_samples, max_mean, max_coefficient
                )
                st.session_state.generated_data = generated_data
                st.session_state.true_graph = true_graph
                st.session_state.scm_equations = equations
                
            # Display generated data 
            if st.session_state.generated_data is not None:
                st.subheader("True SCM Graph")
                pdy = GraphUtils.to_pydot(st.session_state.true_graph)
                image_data = pdy.create_png()
                st.image(image_data, caption="True Causal Structure")
                
                with st.expander("View Generated Data"):
                    st.dataframe(st.session_state.generated_data.head())
                    st.dataframe(st.session_state.generated_data.describe())
        
        
                    st.subheader("Structural Causal Model Equations")
                    for var in st.session_state.generated_data.columns:
                        st.markdown(f"`{st.session_state.scm_equations[var]}`")
                
                dataframe = st.session_state.generated_data

        if dataframe is None:
            st.info("Please configure data source to begin analysis.")
            return

        # Pcik CI test gui
        st.subheader("Analysis Configuration")
        test_type = st.radio(
            "Independence Test Type:",
            options=['fisherz', 'chisq', 'gsq', 'kci', 'mv_fisherz'],
            captions=[
                'Fisher Z-test',
                'Chi-squared test',
                'G-squared test',
                'Kernel-based Conditional Independence test',
                'Missing-value Fisher Z test (use if your data has missing values)'
            ],
            horizontal=True
        )

        # P-value input
        p_value = st.number_input(
            "P-value for conditional dependency calculations of the FCI algorithm:",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            format="%.3f"
        )

        bk = background_knowledge_controls.get_background_knowledge(dataframe)

        run_analysis = st.checkbox('Run FCI Analysis', value=False)
        if run_analysis:
            with st.spinner('Performing FCI Analysis...'):
                try:
                    g, edges = FCI_helper_classes.run_fci_analysis(dataframe, test_type, p_value, bk)
                    display_results(g, edges)
                    with st.expander("IDP Analysis"):
                        if st.checkbox("Run IDP Analysis"):
                            install_r_packages()
                            st.divider()
                            IDP_helper_classes.IDP_call_and_streamlit(g, edges)

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.stop()

if __name__ == "__main__":
    main()

