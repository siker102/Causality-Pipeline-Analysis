import numpy as np
import matplotlib.pyplot as plt
from causal_discovery.fci.fci_algorithm import fci_remake
from causal_discovery.fci.fci_helpers import calculate_accuracy_of_graphs
from causal_discovery.pc.pc_algorithm import pc_remake
import random_scm_generation


def fci_call(generated_data, true_graph, condition, p_value):
    """FCI!!!"""
    if not isinstance(p_value, float) or not (0 < p_value < 1):
        raise ValueError(f"Invalid p_value: {p_value}. It must be a float between 0 and 1.")
    p_value = float(p_value)
    g, edges = fci_remake(dataset=generated_data.to_numpy(), independence_test_method=condition, alpha=p_value)
    correct_percentage, falsely_percentage, partial_percentage = calculate_accuracy_of_graphs(g,true_graph)
    return correct_percentage, falsely_percentage, partial_percentage
    
def PC_call(generated_data, true_graph, condition, p_value):
    """FCI!!!"""
    p_value = float(p_value)
    g = pc_remake(data=generated_data.to_numpy(), indep_test=condition, alpha=p_value)
    correct_percentage, falsely_percentage, partial_percentage = calculate_accuracy_of_graphs(g.G,true_graph)
    return correct_percentage, falsely_percentage, partial_percentage
    

def run_data_collection(N=100, condition1='fisherz', condition2="FCI"):
    """Runs the function N times for each x and calculates the average outcome."""
    x_values = np.arange(0.05, 0.51, 0.01).astype(float)  # Ensure x values are floats
    results_correct_FCI = {x: [] for x in x_values}
    results_false_FCI = {x: [] for x in x_values}
    results_partial_FCI = {x: [] for x in x_values}
    results_correct_PC = {x: [] for x in x_values}
    results_false_PC = {x: [] for x in x_values}
    results_partial_PC = {x: [] for x in x_values}

    # Run the function N times for each x
    for x in x_values:
        for _ in range(N):
            num_vars = 5
            edge_prob = 0.3
            noise_level = 1.0
            num_samples = 100
            max_mean = 10
            max_coefficient = 10
            generated_data, true_graph, equations = random_scm_generation.generate_random_scm(
                    num_vars, edge_prob, noise_level, num_samples, max_mean, max_coefficient
                )
            correct, false, partial = fci_call(generated_data, true_graph, condition1, x)
            results_correct_FCI[x].append(correct)
            results_false_FCI[x].append(false)
            results_partial_FCI[x].append(partial)
            
            correct, false, partial = PC_call(generated_data, true_graph, condition1, x)
            results_correct_PC[x].append(correct)
            results_false_PC[x].append(false)
            results_partial_PC[x].append(partial)

    # Compute the average outcome for each x using PC
    avg_outcomes_correct_FCI = {x: np.mean(results_correct_FCI[x]) for x in x_values}
    avg_outcomes_false_FCI = {x: np.mean(results_false_FCI[x]) for x in x_values}
    avg_outcomes_partial_FCI = {x: np.mean(results_partial_FCI[x]) for x in x_values}
    
    # Compute the average outcome for each x using PC
    avg_outcomes_correct_PC = {x: np.mean(results_correct_PC[x]) for x in x_values}
    avg_outcomes_false_PC = {x: np.mean(results_false_PC[x]) for x in x_values}
    avg_outcomes_partial_PC = {x: np.mean(results_partial_PC[x]) for x in x_values}
    
    return avg_outcomes_correct_FCI, avg_outcomes_false_FCI, avg_outcomes_partial_FCI, avg_outcomes_correct_PC, avg_outcomes_false_PC, avg_outcomes_partial_PC

def plot_results(avg_outcomes_correctFCI, avg_outcomes_falseFCI, avg_outcomes_partialFCI,avg_outcomes_correctPC, avg_outcomes_falsePC, avg_outcomes_partialPC):
    """Plots the average outcomes for two conditions on the same graph."""
    x_values = list(avg_outcomes_correctFCI.keys())

    plt.figure(figsize=(8, 5))
    
    # Correct FCI
    plt.plot(x_values, list(avg_outcomes_correctFCI.values()), marker='o', linestyle='-', label="Perfectly identified edges (FCI)", color="blue", alpha=0.8)
    
    # Correct PC
    plt.plot(x_values, list(avg_outcomes_correctPC.values()), marker='s', linestyle='--', label="Perfectly identified edges (PC)", color="green", alpha=0.8)

    # Partially correct FCI
    plt.plot(x_values, 
             [c + p for c, p in zip(avg_outcomes_correctFCI.values(), avg_outcomes_partialFCI.values())], 
             marker='^', linestyle='-', label="At least partially correct edges (FCI)", color="cyan", alpha=0.8)

    # Partially correct PC
    plt.plot(x_values, 
             [c + p for c, p in zip(avg_outcomes_correctPC.values(), avg_outcomes_partialPC.values())], 
             marker='d', linestyle='--', label="At least partially correct edges (PC)", color="lime", alpha=0.8)

    # False FCI
    plt.plot(x_values, list(avg_outcomes_falseFCI.values()), marker='x', linestyle='-', label="Falsely identified edges (FCI)", color="red", alpha=0.8)
    
    # False PC
    plt.plot(x_values, list(avg_outcomes_falsePC.values()), marker='x', linestyle='--', label="Falsely identified edges(PC)", color="orange", alpha=0.8)

    plt.xlabel("p_value")
    plt.ylabel("Edges / #edges in true graph")
    plt.title("Comparison of PC and FCI in the PC best case (given causal sufficiency)")
    plt.legend()
    plt.grid()

    # Save and display the plot
    plt.savefig("comparison_plot.png")
    print("Plot saved as 'comparison_plot.png'. Open it to view the results.")

if __name__ == "__main__":
    N = 100  # Number of iterations per x value
    avg_outcomes_correctFCI, avg_outcomes_falseFCI, avg_outcomes_partialFCI,avg_outcomes_correctPC, avg_outcomes_falsePC, avg_outcomes_partialPC  = run_data_collection(N, condition1='fisherz', condition2="FCI")

    plot_results(avg_outcomes_correctFCI, avg_outcomes_falseFCI, avg_outcomes_partialFCI,avg_outcomes_correctPC, avg_outcomes_falsePC, avg_outcomes_partialPC)