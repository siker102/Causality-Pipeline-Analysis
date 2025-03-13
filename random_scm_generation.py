import pandas as pd
import numpy as np
from typing import Dict, Tuple
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
import networkx as nx

def generate_random_scm(num_vars=5, edge_prob=0.3, noise_level=1.0, num_samples=1000, max_mean=10, max_coefficient=10) -> Tuple[pd.DataFrame, GeneralGraph, Dict[str, str]]: 
    """Generate a random Structural Causal Model (SCM) with Gaussian noise"""
    # Create random DAG adjacency matrix
    while True:
        adj_matrix = np.random.choice([0, 1], size=(num_vars, num_vars), p=[1-edge_prob, edge_prob])
        adj_matrix = np.triu(adj_matrix, k=1)  # Ensure acyclic by zeroing lower triangle
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        if len(list(nx.topological_sort(G))) == num_vars:  # Validate DAG
            break

    # Create coefficients and equation components
    coefficients = (np.random.uniform(-max_coefficient, max_coefficient, size=(num_vars, num_vars)) * adj_matrix.T)
    
    # Sample a fixed offset for each variable (exogenous mean)
    offsets = np.random.uniform(-max_mean, max_mean, size=num_vars)
    
    # Generate equations dictionary
    node_names = [f"X{i+1}" for i in range(num_vars)]  # Start at X1
    equations = {}
    for i in range(num_vars):
        parents = list(G.predecessors(i))
        if parents:
            terms = [f"{coefficients[i, j]:.2f}*{node_names[j]}" for j in parents]
            equation = f"{node_names[i]} = " + " + ".join(terms)
        else:
            equation = f"{node_names[i]} = "
        equation += f" + ε ~ N({offsets[i]:.2f}, {noise_level**2:.2f})"
        equations[node_names[i]] = equation

    # Generate data with noise including the fixed offset
    data = np.zeros((num_samples, num_vars))
    for i in nx.topological_sort(G):
        parents = list(G.predecessors(i))
        if parents:
            linear_comb = np.sum(data[:, parents] * coefficients[i, parents], axis=1)
        else:
            linear_comb = 0
        # Use the fixed offset as the mean for the noise distribution
        data[:, i] = linear_comb + np.random.normal(offsets[i], noise_level, num_samples)
    
    # True graph
    true_graph = GeneralGraph([GraphNode(name) for name in node_names])
    for u, v in G.edges():
        true_graph.add_directed_edge(true_graph.get_nodes()[u], true_graph.get_nodes()[v])
    
    # Return data, the graph object, and the equations
    return pd.DataFrame(data, columns=node_names), true_graph, equations