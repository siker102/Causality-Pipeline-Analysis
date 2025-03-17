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
from rpy2.robjects import globalenv


def clean_r_environment():
    """Clean up R environment and force garbage collection."""
    ro.r('rm(list=ls(all.names=TRUE))')
    ro.r('gc()')


def convert_graph_to_r_matrix(g: GeneralGraph, edges: list[Edge]) -> ro.Matrix:
    """Convert causallearn Graph to R matrix format for PAGId (pcalg-compatible)."""
    n = len(g.get_nodes())
    matrix = np.zeros((n, n), dtype=int)
    node_names = [node.get_name() for node in g.get_nodes()]

    for edge in edges:
        i = node_names.index(edge.get_node1().get_name())
        j = node_names.index(edge.get_node2().get_name())
        ep1 = edge.get_endpoint1()
        ep2 = edge.get_endpoint2()

        # pcalg encoding: 0=none, 1=circle, 2=arrowhead, 3=tail
        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            # X → Y: tail at X (3), arrowhead at Y (2)
            matrix[i, j] = 3
            matrix[j, i] = 2
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            # X ← Y: arrowhead at X (2), tail at Y (3)
            matrix[i, j] = 2
            matrix[j, i] = 3
        elif ep1 == Endpoint.CIRCLE and ep2 == Endpoint.ARROW:
            # X ◯→ Y: circle at X (1), arrowhead at Y (2)
            matrix[i, j] = 1
            matrix[j, i] = 2
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.CIRCLE:
            # X ←◯ Y: arrowhead at X (2), circle at Y (1)
            matrix[i, j] = 2
            matrix[j, i] = 1
        elif ep1 == Endpoint.TAIL and ep2 == Endpoint.TAIL:
            # X — Y: tail at both (3)
            matrix[i, j] = 3
            matrix[j, i] = 3
        elif ep1 == Endpoint.CIRCLE and ep2 == Endpoint.CIRCLE:
            # X ◯—◯ Y: circle at both (1)
            matrix[i, j] = 1
            matrix[j, i] = 1
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.ARROW:
            # X ↔ Y: arrowhead at both (2)
            matrix[i, j] = 2
            matrix[j, i] = 2

    # Convert to R matrix with column-major flattening
    flattened = np.ravel(matrix, order='F').tolist()
    r_matrix = ro.r.matrix(ro.IntVector(flattened), nrow=n, ncol=n)
    return r_matrix

def run_idp_analysis(g: GeneralGraph, edges: list[Edge], treatment: str, outcome: str) -> dict:
    """Run IDP analysis using R's PAGId package"""
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter) and ro.local_context() as lc:
        try:
            #Clear environment
            ro.r('rm(list=ls(all.names=TRUE, envir=.GlobalEnv), envir=.GlobalEnv)')
            ro.r('gc()')

            # Convert graph to R matrix format
            r_pag_matrix = convert_graph_to_r_matrix(g, edges)
            
            #st.write("r_pag_matrix:", r_pag_matrix)
            #st.write("Matrix dimensions:", ro.r.dim(r_pag_matrix))

            #st.write(pagid.IDP.r_repr())
            # Assign the matrix to a variable in R
            
            colnames = [node.get_name() for node in g.get_nodes()]
            #st.write(colnames)

            ro.globalenv['r_pag_matrix'] = r_pag_matrix
            treatment_index = colnames.index(treatment) + 1  # R is 1-indexed
            outcome_index = colnames.index(outcome) + 1  # R is 1-indexed

            # Set row and column names appropriately
            ro.r(f'colnames(r_pag_matrix) <- paste0("X", 1:ncol(r_pag_matrix))')
            ro.r(f'rownames(r_pag_matrix) <- paste0("X", 1:nrow(r_pag_matrix))')
            ro.r(f'treatment <- "X{treatment_index}"')
            ro.r(f'outcome <- "X{outcome_index}"')

            # Verify treatment/outcome exist in graph
            #node_names = [n.get_name() for n in g.get_nodes()]
            #if treatment not in node_names or outcome not in node_names:
            #    raise ValueError(f"Treatment/outcome not in graph nodes: {node_names}")
            
            
            # Execute the IDP function in R
            # ro.r('print("Running IDP... with the following parameters")')
            # ro.r('print(r_pag_matrix)')
            # ro.r('print(treatment)')
            # ro.r('print(outcome)')
            ro.r('result <- IDP(amat=r_pag_matrix, x=treatment, y=outcome, verbose=TRUE)')
            #ro.r('print(result)')

            # Receive the result from R
            result_d = ro.r('result')

            identifiable = bool(result_d['id'][0])  # [0] extracts the first element of the R vector

            query = list(result_d['query'])

            # Extract 'Qop' and 'Qexpr' if identifiable
            Qop = dict(result_d['Qop']) if identifiable else None
            Qexpression = dict(result_d['Qexpr']) if identifiable else None

            # Gather the solution into a new dict
            result_dict = {
                'identifiable': identifiable,
                'query': query,
                'Qop': Qop,
                'Qexpression': Qexpression
            }

            #print(result_dict['identifiable'])
            #print(result_dict)
            return result_dict
            
        except Exception as e:
            ro.r('gc()')
            st.error(f"IDP Analysis failed: {str(e)}")
            st.write("Debug info:")
            st.write(f"Treatment: {treatment}")
            st.write(f"Outcome: {outcome}")
            st.write(f"Error details: {type(e).__name__}: {str(e)}")
            return None
        finally:
            clean_r_environment()

# Update the display_results function to include IDP analysis
def IDP_call_and_streamlit(g: GeneralGraph, edges: list[Edge]) -> None:
    """Visualize and display IDP analysis and its options"""
    try:
        st.subheader("Causal Effect Analysis")
        col1, col2 = st.columns(2)
        with col1:
            treatment = st.selectbox("Select treatment variable:", 
                                   [node.get_name() for node in g.get_nodes()])
        with col2:
            outcome = st.selectbox("Select outcome variable:", 
                                 [node.get_name() for node in g.get_nodes()])
        
        if st.button("Analyze Causal Effect"):
            # Configure R environment (higher stack size)
            ro.r('''
            options(
                expressions=500000,  # Increase stack depth
                Ncpus=2,  # Limit parallelization
                mc.cores=1  # Disable multicore
            )
            invisible(gc(reset=TRUE))  # Force full garbage collection
            ''')
            result: OrderedDict = run_idp_analysis(g=g, edges=edges, treatment=treatment, outcome=outcome)
            if result:
                st.write("### Results")
                st.write(f"Query: {result.get('query')}")
                if result['identifiable'] == True:
                    st.write("✅ The causal effect is identifiable ✅")
                    #Process and display Qexpr
                    Qexpr = result.get('Qexpression')
                    if Qexpr:
                        steps_list = []
                        st.write("### Identification Formula Steps")
                        for step, formula in Qexpr.items():
                            # Extract the LaTeX string from the array
                            if isinstance(formula, np.ndarray):
                                formula = formula[0]  # Extract the first element if it's an array
                            # Add the current step to the steps list
                            steps_list.append(step)

                            # Construct the subscript string for steps
                            steps_subscript = ",".join(steps_list)
                            
                            if step == list(Qexpr.keys())[-1]:  # Check if it's the final step
                                complete_formula = rf"P({outcome}|do({treatment})) = " + formula
                            else:
                                complete_formula = rf"P_{{{steps_subscript}}} = " + formula
                            st.latex(complete_formula)
                    else:
                        st.write("No identification formula available.")
                else:
                    st.write("❌ The causal effect is not identifiable ❌")

                # Save results in session state
                if 'idp_results' not in st.session_state:
                    st.session_state.idp_results = []
                st.session_state.idp_results.append({
                    'treatment': treatment,
                    'outcome': outcome,
                    'identifiable': result.get('identifiable')
                })

        # Display previously queried results
        if 'idp_results' in st.session_state:
            st.write("### Previous Results")
            for res in st.session_state.idp_results:
                st.write(f"Treatment: {res['treatment']}, Outcome: {res['outcome']}, Identifiable: {'✅' if res['identifiable'] else '❌'}")
                    
    except Exception as e:
        st.error(f"Displaying IDP results failed: {str(e)}")