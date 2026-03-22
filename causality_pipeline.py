import streamlit as st
import pandas as pd
import numpy as np
from causal_discovery.graph.edge import Edge
from causal_discovery.graph.endpoint import Endpoint
from causal_discovery.graph.general_graph import GeneralGraph
from causal_discovery.graph.graph_node import GraphNode
from causal_discovery.graph.causal_graph import CausalGraph
from causal_discovery.knowledge.background_knowledge import BackgroundKnowledge
from causal_discovery.utils.visualization import to_pydot
from causal_discovery.fci.fci_helpers import run_FCI_analysis, calculate_accuracy_of_graphs
from causal_discovery.pc.pc_algorithm import pc_remake
from typing import List, Optional, Dict, Tuple
from collections import OrderedDict
import networkx as nx
import matplotlib.pyplot as plt
import background_knowledge_controls
import random_scm_generation
from causal_discovery.idp_and_cidp.idp import idp
from causal_discovery.idp_and_cidp.cidp import cidp
from causal_discovery.idp_and_cidp.evaluate import evaluate_causal_effect

# Constants
FCI_EDGE_ENCODING = {
    'CIRCLE' : 1,
    'ARROW' : 2,
    'TAIL' : 3,
}

# Streamlit config
st.set_page_config(page_title="Data To Discovery", layout="wide")
st.title('Data To Discovery')

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


def display_FCI_results(g: GeneralGraph, edges: list[Edge]):
    """Visualize and display FCI results"""
    
    try:
        # Check for cycles in the graph
        if hasCycle(g, edges):
            st.warning("The resulting graph contains cycles. The algorithm may not have processed the data properly.")


        # Debug statements
        # for edge in edges:
        #    st.write(edge.get_node1().get_name(), edge.get_node2().get_name(), edge.get_endpoint1(), edge.get_endpoint2(), edge.properties)

    
        pdy = to_pydot(g, edges)
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

def display_PC_results(g: CausalGraph):
    """Visualize and display PC results"""
    
    try:
        # Debug statements
        # for edge in edges:
        #    st.write(edge.get_node1().get_name(), edge.get_node2().get_name(), edge.get_endpoint1(), edge.get_endpoint2(), edge.properties)


        print('here', g.G)
        
        pdy = to_pydot(g.G)
        image_data = pdy.create_png()
        print('here niowe')

        # Display image
        st.image(image_data, caption="PC Analysis Result (CPDAG: undirected edges = orientation undetermined)")

    except Exception as e:
        st.error(f"Graph visualization failed: {str(e)}")


    # Model Information
    #with st.expander("Model Metadata"):
    #    st.write(f"Graph type: {type(g).__name__}")
        #st.write(f"Graph data: {g.def_visible()}")
        #st.write(f"Number of nodes: {len(g.G.get_nodes())}")
        #st.write(f"Number of edges: {len(g.G.edges) if edges else 0}")
        # Remove the curly braces around the condition
        #visible_edges = [edge for edge in edges if Edge.Property.dd in edge.properties]
        #st.write(f"Definitely direct edges: {', '.join([f'{edge.get_node1().get_name()} -> {edge.get_node2().get_name()}' for edge in visible_edges])}")


def _has_possibly_causal_path(amat: np.ndarray, node_names: list[str], source: str, target: str) -> bool:
    """Check if there is any possibly-directed path from source to target in the PAG.

    In a PAG, an edge i-j is possibly directed i->j if:
    - amat[i,j] (endpoint at j) is arrowhead (2) or circle (1)
    - amat[j,i] (endpoint at i) is tail (3) or circle (1)
    An edge with arrowhead at i (amat[j,i]=2) cannot be part of a causal path from i.
    """
    idx = {n: i for i, n in enumerate(node_names)}
    src = idx[source]
    tgt = idx[target]
    n = len(node_names)

    visited = set()
    stack = [src]
    while stack:
        current = stack.pop()
        if current == tgt:
            return True
        if current in visited:
            continue
        visited.add(current)
        for neighbor in range(n):
            if neighbor in visited:
                continue
            ep_at_neighbor = amat[current, neighbor]  # endpoint at neighbor side
            ep_at_current = amat[neighbor, current]    # endpoint at current side
            # Possibly directed current -> neighbor:
            # current side: tail (3) or circle (1), NOT arrowhead (2)
            # neighbor side: arrowhead (2) or circle (1), NOT tail (3) or 0
            if ep_at_neighbor in (1, 2) and ep_at_current in (1, 3):
                stack.append(neighbor)
    return False


def _display_idp_result(result: dict, treatment: str, outcome: str, dataframe: pd.DataFrame):
    """Display identification formula and numeric evaluation for a single IDP/CIDP result."""
    Qexpr = result.get('Qexpr')
    if not Qexpr:
        st.write("No identification formula available.")
        return

    latex_lines = []
    st.write("### Identification Formula Steps")
    for step, formula in Qexpr.items():
        if isinstance(formula, np.ndarray):
            formula = formula[0]

        if step == list(Qexpr.keys())[-1]:
            complete_formula = rf"P({outcome}|do({treatment})) = " + str(formula)
        else:
            complete_formula = rf"P_{{{step}}} = " + str(formula)
        latex_lines.append(complete_formula)
        st.latex(complete_formula)

    full_latex = "\n".join(latex_lines)
    st.code(full_latex, language="latex")
    st.download_button(
        "Download LaTeX",
        data=full_latex,
        file_name="causal_effect.tex",
        mime="text/plain",
    )

    # Numeric evaluation
    st.write("### Numeric Causal Effect Estimate")
    st.caption(f"P({outcome} | do({treatment})): probability of each {outcome} value "
               f"when intervening on {treatment}")
    try:
        has_continuous = any(
            dataframe[c].dtype.kind == 'f' and dataframe[c].nunique() > 20
            for c in dataframe.columns
        )
        treatment_bins = 10
        n_outcome_bins = 10
        if has_continuous:
            bin_col1, bin_col2 = st.columns(2)
            with bin_col1:
                treatment_bins = st.slider(f"Treatment bins ({treatment})", 3, 50, 10)
            with bin_col2:
                n_outcome_bins = st.slider(f"Outcome bins ({outcome})", 3, 50, 5)

        if has_continuous:
            st.caption(
                f"Outcome bins are computed per treatment group using quantile-based "
                f"splitting, so each bin holds roughly the same number of data points. "
                f"Some treatment groups may have fewer than {n_outcome_bins} outcome bins "
                f"if ties at quantile boundaries prevent a clean split."
            )

        effect_table = evaluate_causal_effect(
            result["Qop"], result["query"], dataframe,
            treatment_vars=[treatment], outcome_vars=[outcome],
            treatment_bins=treatment_bins, outcome_bins=n_outcome_bins,
        )
        if len(effect_table) > 0:
            # --- Expected outcome per treatment bin (summary chart) ---
            if treatment in effect_table.columns and outcome in effect_table.columns:
                def _interval_sort_key(x):
                    return x.mid if hasattr(x, "mid") else float(x)

                treat_values_sorted = sorted(effect_table[treatment].unique(), key=_interval_sort_key)
                summary_rows = []
                for t_val in treat_values_sorted:
                    subset = effect_table[effect_table[treatment] == t_val]
                    midpoints = subset[outcome].apply(
                        lambda x: x.mid if hasattr(x, "mid") else float(x)
                    )
                    midpoints = pd.to_numeric(midpoints, errors="coerce")
                    valid = midpoints.notna()
                    if valid.any():
                        probs = subset.loc[valid, "prob"].values
                        mids = midpoints[valid].values
                        expected = (mids * probs).sum()
                        std = np.sqrt((probs * (mids - expected) ** 2).sum())
                        t_mid = float(t_val.mid) if hasattr(t_val, "mid") else float(t_val)
                        summary_rows.append({"treatment_mid": t_mid, "expected_outcome": expected, "std": std})
                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows).sort_values("treatment_mid")
                    x_vals = summary_df["treatment_mid"].values
                    y_vals = summary_df["expected_outcome"].values
                    std_vals = summary_df["std"].values

                    st.write(f"### Expected {outcome} per treatment intervention")
                    st.caption(
                        f"Each point shows the expected value of {outcome} when {treatment} "
                        f"is forced to that range. The shaded band shows ±1 standard "
                        f"deviation of the outcome distribution."
                    )
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.fill_between(x_vals, y_vals - std_vals, y_vals + std_vals, alpha=0.2, label="±1 std")
                    ax.plot(x_vals, y_vals, marker="o", linewidth=2, label=f"E[{outcome}]")
                    ax.set_xlabel(f"{treatment} (bin midpoint)")
                    ax.set_ylabel(f"E[{outcome}]")
                    ax.set_title(f"Causal Effect: {treatment} → {outcome}")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)

            # --- Full probability table and per-bin charts ---
            st.write("### Per-bin outcome distributions")
            display_table = effect_table.copy()
            for col in display_table.columns:
                if col != "prob":
                    display_table[col] = display_table[col].astype(str)

            st.dataframe(display_table.style.format({"prob": "{:.4f}"}))

            if treatment in display_table.columns and outcome in display_table.columns:
                treat_values = [str(t) for t in treat_values_sorted]
                cols = st.columns(min(len(treat_values), 4))
                for i, t_val in enumerate(treat_values):
                    subset = display_table[display_table[treatment] == t_val].sort_values(outcome, key=lambda s: s.astype(str))
                    with cols[i % len(cols)]:
                        st.write(f"**do({treatment}={t_val})**")
                        chart = subset[[outcome, "prob"]].copy()
                        chart[outcome] = chart[outcome].astype(str)
                        st.line_chart(chart.set_index(outcome)["prob"])
            else:
                var_cols = [c for c in display_table.columns if c != "prob"]
                if var_cols:
                    chart_data = display_table.copy()
                    chart_data["label"] = chart_data[var_cols].astype(str).agg(", ".join, axis=1)
                    st.line_chart(chart_data.set_index("label")["prob"])
        else:
            st.warning("No data rows matched the evaluation.")
    except Exception as eval_err:
        st.warning(f"Numeric evaluation failed: {eval_err}")


def _run_idp_cidp_ui(g: GeneralGraph, edges: list[Edge], dataframe: pd.DataFrame):
    """Streamlit UI for IDP/CIDP causal effect identification."""
    st.subheader("Causal Effect Identification")

    node_names = [node.get_name() for node in g.get_nodes()]

    mode = st.radio(
        "Mode:",
        ["Find all identifiable effects (IDP)", "Single query (IDP/CIDP)"],
        horizontal=True,
    )

    if mode == "Find all identifiable effects (IDP)":
        _run_idp_batch_ui(g, node_names, dataframe)
    else:
        _run_idp_single_ui(g, node_names, dataframe)


def _run_idp_batch_ui(g: GeneralGraph, node_names: list[str], dataframe: pd.DataFrame):
    """Run IDP on all treatment-outcome pairs, show identifiable ones, let user pick."""
    if st.button("Find all identifiable effects"):
        amat = g.to_pcalg_matrix()
        all_results = {}
        identifiable = []
        trivial = []
        not_identifiable = []

        progress = st.progress(0)
        pairs = [(x, y) for x in node_names for y in node_names if x != y]
        for i, (x, y) in enumerate(pairs):
            try:
                result = idp(amat, [x], [y], node_names, verbose=False)
                key = f"P({y}|do({x}))"
                is_trivial = not _has_possibly_causal_path(amat, node_names, x, y)
                all_results[key] = {
                    'result': result,
                    'treatment': x,
                    'outcome': y,
                    'trivial': is_trivial,
                }
                if result['id']:
                    if is_trivial:
                        trivial.append(key)
                    else:
                        identifiable.append(key)
                else:
                    not_identifiable.append(key)
            except Exception:
                not_identifiable.append(f"P({y}|do({x}))")
            progress.progress((i + 1) / len(pairs))
        progress.empty()

        st.session_state.idp_batch_results = all_results
        st.session_state.idp_batch_identifiable = identifiable
        st.session_state.idp_batch_trivial = trivial
        st.session_state.idp_batch_not_identifiable = not_identifiable

    # Display results (persists across reruns)
    if 'idp_batch_results' not in st.session_state:
        return

    identifiable = st.session_state.idp_batch_identifiable
    trivial = st.session_state.idp_batch_trivial
    not_identifiable = st.session_state.idp_batch_not_identifiable

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**{len(identifiable)} identifiable**")
        for key in identifiable:
            st.write(f"- {key}")
    with col2:
        if not_identifiable:
            st.error(f"**{len(not_identifiable)} not identifiable**")
            for key in not_identifiable:
                st.write(f"- {key}")

    show_trivial = False
    if trivial:
        show_trivial = st.checkbox(
            f"Show {len(trivial)} trivial effects (no causal path, P(Y|do(X)) = P(Y))",
            value=False,
        )
        if show_trivial:
            st.info(f"**{len(trivial)} trivial** (no causal path)")
            for key in trivial:
                st.write(f"- {key} = P({key.split('(')[1].split('|')[0]})")

    options = list(identifiable)
    if show_trivial:
        for key in trivial:
            options.append(f"{key} (trivial)")

    if not options:
        st.warning("No identifiable causal effects found.")
        return

    selected_label = st.selectbox("Select an effect to evaluate:", options)

    if selected_label:
        selected_key = selected_label.replace(" (trivial)", "")
        entry = st.session_state.idp_batch_results[selected_key]
        if entry.get('trivial'):
            st.info(f"No causal path exists from {entry['treatment']} to {entry['outcome']}. "
                    f"P({entry['outcome']}|do({entry['treatment']})) = P({entry['outcome']})")
        _display_idp_result(entry['result'], entry['treatment'], entry['outcome'], dataframe)


def _run_idp_single_ui(g: GeneralGraph, node_names: list[str], dataframe: pd.DataFrame):
    """Single query mode for IDP or CIDP."""
    col1, col2 = st.columns(2)
    with col1:
        treatment = st.selectbox("Treatment variable (X):", node_names)
    with col2:
        outcome = st.selectbox("Outcome variable (Y):", node_names)

    use_cidp = st.checkbox("Condition on variables (CIDP)", value=False)
    conditioning = []
    if use_cidp:
        available_z = [n for n in node_names if n != treatment and n != outcome]
        conditioning = st.multiselect("Conditioning variables (Z):", available_z)

    if st.button("Identify Causal Effect"):
        amat = g.to_pcalg_matrix()
        try:
            if use_cidp and len(conditioning) > 0:
                result = cidp(amat, [treatment], [outcome], conditioning, node_names, verbose=False)
            else:
                result = idp(amat, [treatment], [outcome], node_names, verbose=False)

            st.session_state.idp_current_result = result
            st.session_state.idp_current_treatment = treatment
            st.session_state.idp_current_outcome = outcome
        except Exception as e:
            st.error(f"Identification failed: {str(e)}")

    if 'idp_current_result' in st.session_state:
        result = st.session_state.idp_current_result
        treatment = st.session_state.idp_current_treatment
        outcome = st.session_state.idp_current_outcome

        st.write("### Results")
        st.write(f"Query: {result.get('query')}")

        if result['id']:
            st.success("The causal effect is identifiable!")
            _display_idp_result(result, treatment, outcome, dataframe)
        else:
            st.error("The causal effect is not identifiable.")


def main():
    """Main function for the causal discovery pipeline"""

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
                with st.expander("View Uploaded Data"):
                    st.dataframe(dataframe.head())
                    st.dataframe(dataframe.describe())
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
            #Reset background knowledge
            background_knowledge_controls.reset_background_knowledge_state()
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
            pdy = to_pydot(st.session_state.true_graph)
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

    # Pick CI test gui
    st.subheader("Analysis Configuration")
    test_algorithm = st.radio(
        "Algorithm Type:",
        options=['FCI', 'PC'],
        captions=[
            'When you expect unobserved latent confounding.',
            'When you have causal sufficiency',
        ],
        horizontal=True
    )
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

    run_analysis = st.checkbox('Run FCI Analysis', value=False, key='run_analysis')
    if run_analysis:
        # Build a cache key from the analysis parameters
        bk_forbidden = frozenset((str(a), str(b)) for a, b in bk.forbidden_rules_specs)
        bk_required = frozenset((str(a), str(b)) for a, b in bk.required_rules_specs)
        bk_tiers = frozenset((k, frozenset(str(n) for n in v)) for k, v in bk.tier_map.items())
        cache_key = (
            test_algorithm,
            test_type,
            p_value,
            id(dataframe),
            bk_forbidden,
            bk_required,
            bk_tiers,
        )

        # Only recompute if inputs changed
        if st.session_state.get('_analysis_cache_key') != cache_key:
            with st.spinner('Performing FCI Analysis...'):
                try:
                    if test_algorithm == "FCI":
                        g, edges = run_FCI_analysis(dataframe, test_type, p_value, bk)
                        st.session_state['_cached_fci_result'] = (g, edges)
                    else:
                        mvpc = dataframe.isnull().values.any()
                        data = dataframe.to_numpy()
                        node_names = dataframe.columns.tolist()
                        g = pc_remake(data, indep_test=test_type, alpha=p_value, background_knowledge=bk, mvpc=mvpc, node_names=node_names)
                        st.session_state['_cached_pc_result'] = g
                    st.session_state['_analysis_cache_key'] = cache_key
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.stop()

        try:
            if test_algorithm == "FCI":
                g, edges = st.session_state['_cached_fci_result']
                display_FCI_results(g, edges)
                if st.session_state.generated_data is not None:
                    accuracy = calculate_accuracy_of_graphs(g=g, true_graph=st.session_state.true_graph)
                    st.subheader("Graph Accuracy Results")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric(label="Correctly identified", value=f"{accuracy['correct']}/{accuracy['total_true']}")
                    col2.metric(label="Partially oriented", value=accuracy['partial'])
                    col3.metric(label="Wrongly oriented", value=accuracy['wrong_orient'])
                    col4.metric(label="Missing edges", value=accuracy['missing'])
                    col5.metric(label="Spurious edges", value=accuracy['spurious'])
            else:
                g = st.session_state['_cached_pc_result']
                display_PC_results(g)
                if st.session_state.generated_data is not None:
                    accuracy = calculate_accuracy_of_graphs(g=g.G, true_graph=st.session_state.true_graph)
                    st.subheader("Graph Accuracy Results")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric(label="Correctly identified", value=f"{accuracy['correct']}/{accuracy['total_true']}")
                    col2.metric(label="Partially oriented", value=accuracy['partial'])
                    col3.metric(label="Wrongly oriented", value=accuracy['wrong_orient'])
                    col4.metric(label="Missing edges", value=accuracy['missing'])
                    col5.metric(label="Spurious edges", value=accuracy['spurious'])

            # IDP/CIDP Analysis (only for FCI which produces PAGs)
            if test_algorithm == "FCI":
                _run_idp_cidp_ui(g, edges, dataframe)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()

